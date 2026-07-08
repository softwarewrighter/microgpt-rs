//! microgpt on the Apple Silicon GPU: the same algorithm as microgpt-rs, re-expressed
//! as tensor operations so MLX can differentiate and execute it on Metal.
//!
//! The hand-rolled scalar autograd tape is gone -- `value_and_grad` differentiates the
//! loss function for us. Scalars become (B, T, C) arrays, the training-time KV cache
//! becomes causal-masked attention over the whole sequence, and parameters live in
//! unified memory where the GPU reads them in place.
//!
//! Two configs, same code path (mirroring microgpt-cuda): the default is Karpathy's
//! original 4,192-parameter parity config; `--scale` is the 795K-parameter batched
//! config shared with microgpt-cuda/microgpt-scale.

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::{eval, value_and_grad_with_argnums};
use mlx_rs::{nn, ops, Array};

// Same tiny RNG as every sibling crate, for the host-side pieces (shuffle, init, sampling)
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

    fn choices(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut r = self.uniform() * total;
        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

#[derive(Clone, Copy)]
struct Cfg {
    n_layer: usize,
    n_embd: i32,
    n_head: i32,
    block_size: usize,
    batch: usize,
    num_steps: usize,
    init_std: f64,
    learning_rate: f32,
}

const PARITY: Cfg = Cfg {
    n_layer: 1,
    n_embd: 16,
    n_head: 4,
    block_size: 16,
    batch: 1,
    num_steps: 1000,
    init_std: 0.08,
    learning_rate: 0.01,
};

const SCALE: Cfg = Cfg {
    n_layer: 4,
    n_embd: 128,
    n_head: 8,
    block_size: 16,
    batch: 32,
    num_steps: 1000,
    init_std: 0.02,
    learning_rate: 0.001,
};

// Parameter list layout (same order as every sibling crate):
// [wte, wpe, lm_head, then per layer: wq, wk, wv, wo, fc1, fc2]
const P_WTE: usize = 0;
const P_WPE: usize = 1;
const P_LM_HEAD: usize = 2;
const P_LAYER0: usize = 3;
const PARAMS_PER_LAYER: usize = 6;

fn init_params(rng: &mut Rng, cfg: &Cfg, vocab_size: i32) -> Vec<Array> {
    let matrix = |rng: &mut Rng, nout: i32, nin: i32| {
        let data: Vec<f32> =
            (0..nout * nin).map(|_| rng.gauss(0.0, cfg.init_std) as f32).collect();
        Array::from_slice(&data, &[nout, nin])
    };
    let c = cfg.n_embd;
    let mut params = vec![
        matrix(rng, vocab_size, c),            // wte
        matrix(rng, cfg.block_size as i32, c), // wpe
        matrix(rng, vocab_size, c),            // lm_head
    ];
    for _ in 0..cfg.n_layer {
        params.push(matrix(rng, c, c));     // attn_wq
        params.push(matrix(rng, c, c));     // attn_wk
        params.push(matrix(rng, c, c));     // attn_wv
        params.push(matrix(rng, c, c));     // attn_wo
        params.push(matrix(rng, 4 * c, c)); // mlp_fc1
        params.push(matrix(rng, c, 4 * c)); // mlp_fc2
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

/// Forward a whole batch of sequences at once: (B, T) token ids -> (B, T, vocab)
/// logits. Causality comes from a mask instead of a KV cache -- every position
/// attends to positions <= itself within its own row, all in one batch of matmuls.
fn gpt(params: &[Array], cfg: &Cfg, token_ids: &Array) -> Array {
    let (b, t) = (token_ids.shape()[0], token_ids.shape()[1]);
    let tok_emb = params[P_WTE].take_axis(token_ids, 0).unwrap(); // (B, T, C)
    let pos_emb = params[P_WPE].index((..t, ..)); // (T, C), broadcasts over B
    let mut x = rmsnorm(&tok_emb.add(&pos_emb).unwrap());

    for li in 0..cfg.n_layer {
        let lp = P_LAYER0 + li * PARAMS_PER_LAYER;
        let (wq, wk, wv, wo) = (&params[lp], &params[lp + 1], &params[lp + 2], &params[lp + 3]);
        let (fc1, fc2) = (&params[lp + 4], &params[lp + 5]);

        // 1) Multi-head Attention block
        let x_residual = &x;
        let xn = rmsnorm(&x);
        // (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        let heads = |a: &Array| {
            a.reshape(&[b, t, cfg.n_head, cfg.n_embd / cfg.n_head])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap()
        };
        let q = heads(&linear(&xn, wq));
        let k = heads(&linear(&xn, wk));
        let v = heads(&linear(&xn, wv));
        let head_dim = (cfg.n_embd / cfg.n_head) as f32;
        let scores = q
            .matmul(&k.transpose_axes(&[0, 1, 3, 2]).unwrap())
            .unwrap()
            .divide(&Array::from_f32(head_dim.sqrt()))
            .unwrap(); // (B, H, T, T)
        let causal = ops::tril(&Array::ones::<bool>(&[t, t]).unwrap(), None).unwrap();
        let masked = ops::r#where(&causal, &scores, &Array::from_f32(-1e9)).unwrap();
        let attn = ops::softmax_axis(&masked, -1, None).unwrap();
        let out = attn.matmul(&v).unwrap(); // (B, H, T, D)
        let out =
            out.transpose_axes(&[0, 2, 1, 3]).unwrap().reshape(&[b, t, cfg.n_embd]).unwrap();
        x = x_residual.add(&linear(&out, wo)).unwrap();

        // 2) MLP block
        let x_residual = &x;
        let xn = rmsnorm(&x);
        let h = nn::relu(linear(&xn, fc1)).unwrap();
        x = x_residual.add(&linear(&h, fc2)).unwrap();
    }

    linear(&x, &params[P_LM_HEAD]) // (B, T, vocab) logits
}

/// Cross-entropy averaged over the valid (unpadded) positions:
/// sum over mask of (logsumexp(logits) - logits[target]), divided by the valid count.
fn loss_fn(
    params: &[Array],
    cfg: &Cfg,
    token_ids: &Array,  // (B, T) i32
    target_ids: &Array, // (B, T) i32, padding clamped to 0
    mask: &Array,       // (B, T, 1) f32, 1.0 on valid positions
    n_valid: f32,
) -> Array {
    let logits = gpt(params, cfg, token_ids);
    let lse = logits.logsumexp_axis(-1, true).unwrap(); // (B, T, 1)
    let (b, t) = (token_ids.shape()[0], token_ids.shape()[1]);
    let picked =
        ops::indexing::take_along_axis(&logits, &target_ids.reshape(&[b, t, 1]).unwrap(), -1)
            .unwrap(); // (B, T, 1)
    lse.subtract(&picked)
        .unwrap()
        .multiply(mask)
        .unwrap()
        .sum(None)
        .unwrap()
        .divide(&Array::from_f32(n_valid))
        .unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scale = std::env::args().any(|a| a == "--scale");
    let cfg = if scale { SCALE } else { PARITY };
    let mut rng = Rng::new(42);

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

    let mut params = init_params(&mut rng, &cfg, vocab_size as i32);
    let num_params: usize = params.iter().map(|p| p.size()).sum();
    println!("num params: {num_params}");
    if scale {
        println!(
            "mode: scale ({} layers, {}-dim, {} heads, batch {})",
            cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.batch
        );
    }
    println!("device: Apple Silicon GPU (MLX)");

    // Adam, with per-parameter-matrix moment buffers
    let (beta1, beta2, eps_adam) = (0.85f32, 0.99f32, 1e-8f32);
    let mut m: Vec<Array> = params.iter().map(|p| ops::zeros_like(p).unwrap()).collect();
    let mut v: Vec<Array> = params.iter().map(|p| ops::zeros_like(p).unwrap()).collect();
    let argnums: Vec<i32> = (0..params.len() as i32).collect();

    let t_start = std::time::Instant::now();
    for step in 0..cfg.num_steps {
        // Assemble a batch: `cfg.batch` documents, each BOS-wrapped and clipped to the
        // block size; short ones padded with BOS tokens and masked-out targets.
        let t_max = cfg.block_size;
        let mut tokens = vec![bos as i32; cfg.batch * t_max];
        let mut targets = vec![0i32; cfg.batch * t_max]; // padding clamped to 0 for the gather
        let mut mask = vec![0.0f32; cfg.batch * t_max];
        let mut n_last = 0;
        for bi in 0..cfg.batch {
            let doc = docs[(step * cfg.batch + bi) % docs.len()];
            let mut toks = vec![bos as i32];
            toks.extend(doc.chars().map(|c| stoi[&c] as i32));
            toks.push(bos as i32);
            let n = t_max.min(toks.len() - 1);
            tokens[bi * t_max..bi * t_max + n].copy_from_slice(&toks[..n]);
            targets[bi * t_max..bi * t_max + n].copy_from_slice(&toks[1..n + 1]);
            mask[bi * t_max..bi * t_max + n].fill(1.0);
            n_last = n;
        }
        // A batch of one (parity mode) is sliced to the document length, unpadded,
        // so its loss is directly comparable with the tape crate's per-document mean.
        let (b, t) = if cfg.batch == 1 { (1, n_last) } else { (cfg.batch, t_max) };
        let rows = b * t;
        let inputs = Array::from_slice(&tokens[..rows], &[b as i32, t as i32]);
        let tgts = Array::from_slice(&targets[..rows], &[b as i32, t as i32]);
        let n_valid: f32 = mask[..rows].iter().sum();
        let msk = Array::from_slice(&mask[..rows], &[b as i32, t as i32, 1]);

        // One call gives the loss and d(loss)/d(every parameter matrix)
        let f = |ps: &[Array]| -> Vec<Array> {
            vec![loss_fn(ps, &cfg, &inputs, &tgts, &msk, n_valid)]
        };
        let (loss, grads) = value_and_grad_with_argnums(f, &argnums[..])(&params)?;

        let lr_t = cfg.learning_rate * (1.0 - step as f32 / cfg.num_steps as f32);
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

        print!("step {:4} / {:4} | loss {:.4}\r", step + 1, cfg.num_steps, loss[0].item::<f32>());
        use std::io::Write;
        std::io::stdout().flush()?;
    }
    let elapsed = t_start.elapsed();
    println!(
        "\ntrain time: {:.2}s ({:.1} steps/s)",
        elapsed.as_secs_f64(),
        cfg.num_steps as f64 / elapsed.as_secs_f64()
    );

    // Inference: re-forward the whole prefix per character and sample with the host
    // RNG, exactly like microgpt-scale/microgpt-cuda -- so given equal logits, all
    // the crates print the same names.
    let temperature = 0.5f64;
    println!("--- inference (new, hallucinated names) ---");
    for sample_idx in 0..20 {
        let mut ctx = vec![bos as i32];
        let mut sample = String::new();
        for _ in 0..cfg.block_size {
            let inputs = Array::from_slice(&ctx, &[1, ctx.len() as i32]);
            let logits = gpt(&params, &cfg, &inputs);
            let last = logits.index((0, ctx.len() as i32 - 1, ..));
            eval([&last])?;
            let logits_host = last.as_slice::<f32>();
            let maxv = logits_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let weights: Vec<f64> =
                logits_host.iter().map(|&l| ((l as f64 - maxv) / temperature).exp()).collect();
            let tok = rng.choices(&weights);
            if tok == bos {
                break;
            }
            sample.push(uchars[tok]);
            ctx.push(tok as i32);
        }
        println!("sample {:2}: {}", sample_idx + 1, sample);
    }
    Ok(())
}
