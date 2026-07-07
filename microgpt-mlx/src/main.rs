//! microgpt on the Apple Silicon GPU: the same algorithm as microgpt-rs, re-expressed
//! as tensor operations so MLX can differentiate and execute it on Metal.
//!
//! The hand-rolled scalar autograd tape is gone -- `value_and_grad` differentiates the
//! loss function for us. Scalars become (T, C) arrays, the training-time KV cache
//! becomes causal-masked attention over the whole sequence, and parameters live in
//! unified memory where the GPU reads them in place.

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::{eval, value_and_grad_with_argnums};
use mlx_rs::{nn, ops, random, Array};

// Same tiny RNG as microgpt-rs, for the host-side pieces (shuffle, init)
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let u1 = loop {
            let u = self.uniform();
            if u > 0.0 {
                break u;
            }
        };
        let u2 = self.uniform();
        mu + sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            xs.swap(i, j);
        }
    }
}

const N_LAYER: usize = 1;
const N_EMBD: i32 = 16;
const BLOCK_SIZE: usize = 16;
const N_HEAD: i32 = 4;
const HEAD_DIM: i32 = N_EMBD / N_HEAD;

// Parameter list layout (mirrors the Python state_dict, flattened):
// [wte, wpe, lm_head, then per layer: wq, wk, wv, wo, fc1, fc2]
const P_WTE: usize = 0;
const P_WPE: usize = 1;
const P_LM_HEAD: usize = 2;
const P_LAYER0: usize = 3;
const PARAMS_PER_LAYER: usize = 6;

fn init_params(rng: &mut Rng, vocab_size: i32) -> Vec<Array> {
    let matrix = |rng: &mut Rng, nout: i32, nin: i32| {
        let data: Vec<f32> =
            (0..nout * nin).map(|_| rng.gauss(0.0, 0.08) as f32).collect();
        Array::from_slice(&data, &[nout, nin])
    };
    let mut params = vec![
        matrix(rng, vocab_size, N_EMBD),          // wte
        matrix(rng, BLOCK_SIZE as i32, N_EMBD),   // wpe
        matrix(rng, vocab_size, N_EMBD),          // lm_head
    ];
    for _ in 0..N_LAYER {
        params.push(matrix(rng, N_EMBD, N_EMBD));     // attn_wq
        params.push(matrix(rng, N_EMBD, N_EMBD));     // attn_wk
        params.push(matrix(rng, N_EMBD, N_EMBD));     // attn_wv
        params.push(matrix(rng, N_EMBD, N_EMBD));     // attn_wo
        params.push(matrix(rng, 4 * N_EMBD, N_EMBD)); // mlp_fc1
        params.push(matrix(rng, N_EMBD, 4 * N_EMBD)); // mlp_fc2
    }
    params
}

/// x * rsqrt(mean(x^2, last axis) + 1e-5), the tensor form of the scalar loop
fn rmsnorm(x: &Array) -> Array {
    let ms = x.multiply(x).unwrap().mean_axis(-1, true).unwrap();
    x.multiply(ops::rsqrt(&ms.add(&Array::from_f32(1e-5)).unwrap()).unwrap()).unwrap()
}

/// Python's `linear(x, w)` was y_o = sum_i w_oi * x_i; in tensor form that is x @ w^T
fn linear(x: &Array, w: &Array) -> Array {
    x.matmul(&w.transpose().unwrap()).unwrap()
}

/// Forward the whole token sequence at once: (T,) token ids -> (T, vocab) logits.
/// Causality comes from a mask instead of a KV cache -- every position attends to
/// positions <= itself, all computed in one batch of matmuls.
fn gpt(params: &[Array], token_ids: &Array) -> Array {
    let t = token_ids.shape()[0];
    let tok_emb = params[P_WTE].index((token_ids, ..)); // (T, C)
    let pos_emb = params[P_WPE].index((..t, ..)); // (T, C)
    let mut x = rmsnorm(&tok_emb.add(&pos_emb).unwrap());

    for li in 0..N_LAYER {
        let lp = P_LAYER0 + li * PARAMS_PER_LAYER;
        let (wq, wk, wv, wo) = (&params[lp], &params[lp + 1], &params[lp + 2], &params[lp + 3]);
        let (fc1, fc2) = (&params[lp + 4], &params[lp + 5]);

        // 1) Multi-head Attention block
        let x_residual = &x;
        let xn = rmsnorm(&x);
        // (T, C) -> (T, H, D) -> (H, T, D)
        let heads = |a: &Array| {
            a.reshape(&[t, N_HEAD, HEAD_DIM]).unwrap().transpose_axes(&[1, 0, 2]).unwrap()
        };
        let q = heads(&linear(&xn, wq));
        let k = heads(&linear(&xn, wk));
        let v = heads(&linear(&xn, wv));
        let scores = q
            .matmul(&k.transpose_axes(&[0, 2, 1]).unwrap())
            .unwrap()
            .divide(&Array::from_f32((HEAD_DIM as f32).sqrt()))
            .unwrap(); // (H, T, T)
        let causal = ops::tril(&Array::ones::<bool>(&[t, t]).unwrap(), None).unwrap();
        let masked = ops::r#where(&causal, &scores, &Array::from_f32(-1e9)).unwrap();
        let attn = ops::softmax_axis(&masked, -1, None).unwrap();
        let out = attn.matmul(&v).unwrap(); // (H, T, D)
        let out = out.transpose_axes(&[1, 0, 2]).unwrap().reshape(&[t, N_EMBD]).unwrap();
        x = x_residual.add(&linear(&out, wo)).unwrap();

        // 2) MLP block
        let x_residual = &x;
        let xn = rmsnorm(&x);
        let h = nn::relu(linear(&xn, fc1)).unwrap();
        x = x_residual.add(&linear(&h, fc2)).unwrap();
    }

    linear(&x, &params[P_LM_HEAD]) // (T, vocab) logits
}

/// Mean cross-entropy: logsumexp(logits) - logits[target], averaged over positions
fn loss_fn(params: &[Array], token_ids: &Array, target_ids: &Array) -> Array {
    let logits = gpt(params, token_ids);
    let lse = logits.logsumexp_axis(-1, true).unwrap(); // (T, 1)
    let t = target_ids.shape()[0];
    let picked = ops::indexing::take_along_axis(
        &logits,
        &target_ids.reshape(&[t, 1]).unwrap(),
        -1,
    )
    .unwrap(); // (T, 1)
    lse.subtract(&picked).unwrap().mean(None).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(42);
    random::seed(42)?;

    // Dataset and tokenizer, identical to microgpt-rs
    if !std::path::Path::new("input.txt").exists() {
        let url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt";
        eprintln!("downloading {url} ...");
        let ok = std::process::Command::new("curl")
            .args(["-fsSL", "-o", "input.txt", url])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            eprintln!("download failed; fetch it manually:\n  curl -o input.txt {url}");
            std::process::exit(1);
        }
    }
    let text = std::fs::read_to_string("input.txt")?;
    let mut docs: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    rng.shuffle(&mut docs);
    println!("num docs: {}", docs.len());

    let uchars: Vec<char> = docs
        .iter()
        .flat_map(|d| d.chars())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let stoi: std::collections::HashMap<char, usize> =
        uchars.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {vocab_size}");

    let mut params = init_params(&mut rng, vocab_size as i32);
    let num_params: i32 = params.iter().map(|p| p.size() as i32).sum();
    println!("num params: {num_params}");

    // Adam, with per-parameter-matrix moment buffers this time
    let (learning_rate, beta1, beta2, eps_adam) = (0.01f32, 0.85f32, 0.99f32, 1e-8f32);
    let mut m: Vec<Array> = params.iter().map(|p| ops::zeros_like(p).unwrap()).collect();
    let mut v: Vec<Array> = params.iter().map(|p| ops::zeros_like(p).unwrap()).collect();
    let argnums: Vec<i32> = (0..params.len() as i32).collect();

    let num_steps = 1000;
    let t_start = std::time::Instant::now();
    for step in 0..num_steps {
        let doc = docs[step % docs.len()];
        let mut tokens = vec![bos as i32];
        tokens.extend(doc.chars().map(|c| stoi[&c] as i32));
        tokens.push(bos as i32);
        let n = BLOCK_SIZE.min(tokens.len() - 1);
        let inputs = Array::from_slice(&tokens[..n], &[n as i32]);
        let targets = Array::from_slice(&tokens[1..n + 1], &[n as i32]);

        // One call gives the loss and d(loss)/d(every parameter matrix)
        let f = |ps: &[Array]| -> Vec<Array> { vec![loss_fn(ps, &inputs, &targets)] };
        let (loss, grads) = value_and_grad_with_argnums(f, &argnums[..])(&params)?;

        let lr_t = learning_rate * (1.0 - step as f32 / num_steps as f32);
        let bc1 = 1.0 - beta1.powi(step as i32 + 1);
        let bc2 = 1.0 - beta2.powi(step as i32 + 1);
        for i in 0..params.len() {
            let g = &grads[i];
            m[i] = m[i].multiply(&Array::from_f32(beta1))?.add(&g.multiply(&Array::from_f32(1.0 - beta1))?)?;
            v[i] = v[i].multiply(&Array::from_f32(beta2))?.add(&g.multiply(g)?.multiply(&Array::from_f32(1.0 - beta2))?)?;
            let m_hat = m[i].divide(&Array::from_f32(bc1))?;
            let v_hat = v[i].divide(&Array::from_f32(bc2))?;
            let update = m_hat
                .multiply(&Array::from_f32(lr_t))?
                .divide(&v_hat.sqrt()?.add(&Array::from_f32(eps_adam))?)?;
            params[i] = params[i].subtract(&update)?;
        }
        // MLX is lazy: force this step's graph to actually run, so it can't grow across steps
        eval(params.iter().chain(m.iter()).chain(v.iter()))?;

        print!("step {:4} / {:4} | loss {:.4}\r", step + 1, num_steps, loss[0].item::<f32>());
        use std::io::Write;
        std::io::stdout().flush()?;
    }
    let elapsed = t_start.elapsed();
    println!("\ntrain time: {:.2}s ({:.1} steps/s)", elapsed.as_secs_f64(), num_steps as f64 / elapsed.as_secs_f64());

    // Inference. No KV cache here: at T <= 16 we simply re-forward the whole prefix
    // per character and read the last position's logits.
    let temperature = 0.5f32;
    println!("--- inference (new, hallucinated names) ---");
    for sample_idx in 0..20 {
        let mut ctx = vec![bos as i32];
        let mut sample = String::new();
        for _ in 0..BLOCK_SIZE {
            let inputs = Array::from_slice(&ctx, &[ctx.len() as i32]);
            let logits = gpt(&params, &inputs);
            let last = logits.index((ctx.len() as i32 - 1, ..));
            let scaled = last.divide(&Array::from_f32(temperature))?;
            let tok = random::categorical(&scaled, None, None, None)?.item::<u32>() as i32;
            if tok == bos as i32 {
                break;
            }
            sample.push(uchars[tok as usize]);
            ctx.push(tok);
        }
        println!("sample {:2}: {}", sample_idx + 1, sample);
    }
    Ok(())
}
