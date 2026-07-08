//! microgpt at scale on the CPU: the batched, multi-layer configuration that the
//! scalar autograd tape cannot reach, with the tape replaced by hand-written
//! per-layer forward and backward passes over flat `f32` buffers.
//!
//! Why this crate exists: microgpt-rs materializes one tape node per scalar
//! operation. At the scale config (~800K parameters, 32 documents per step) a single
//! training step is ~2.4 GFLOP -- billions of tape nodes, over a hundred gigabytes --
//! so the tape physically cannot follow. This crate is the CPU's honest entry at
//! that scale: the same per-layer backward derivation as microgpt-cuda, the same
//! flat parameter buffer, the same batching and loss masking, executed by CPU cores
//! instead of CUDA kernels. Each function here mirrors one CUDA kernel one-to-one,
//! down to the per-element summation order, so the two crates' loss curves track
//! each other and both are bit-deterministic run to run.
//!
//! Still zero dependencies: parallelism is `std::thread::scope` splitting each
//! output buffer into disjoint row chunks -- which changes nothing about any single
//! element's arithmetic, so threading does not perturb determinism.
//!
//! `--parity` runs the original 4,192-parameter config instead, as a cross-check
//! against microgpt-rs / microgpt-cuda.

use std::collections::{BTreeSet, HashMap};
use std::io::Write;

// Same tiny RNG as the sibling crates: identical shuffle, identical initialization.
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

/// Model + training shape; the same numbers as microgpt-cuda's two configs.
#[derive(Clone, Copy)]
struct Cfg {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    block_size: usize,
    batch: usize,
    num_steps: usize,
    init_std: f64,
    learning_rate: f32,
}

impl Cfg {
    fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    fn max_rows(&self) -> usize {
        self.batch * self.block_size
    }
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

// ---- the "kernels": each function mirrors one microgpt-cuda CUDA kernel ----

/// Run `f(row_index, row)` over every `row_len`-sized row of `out`, splitting the
/// rows across threads. Each output element is still computed by exactly one
/// closure call with an unchanged summation order, so results are bit-identical
/// to a serial loop. Small buffers stay serial -- thread spawns would dominate.
fn par_rows(out: &mut [f32], row_len: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = out.len() / row_len;
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    if out.len() < (1 << 15) || workers < 2 {
        for (r, row) in out.chunks_mut(row_len).enumerate() {
            f(r, row);
        }
        return;
    }
    let per = rows.div_ceil(workers);
    std::thread::scope(|s| {
        for (w, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (r, row) in chunk.chunks_mut(row_len).enumerate() {
                    f(w * per + r, row);
                }
            });
        }
    });
}

/// x[r] = wte[token[r]] + wpe[r mod t_len]
fn embed_fwd(x: &mut [f32], wte: &[f32], wpe: &[f32], tokens: &[i32], t_len: usize, c: usize) {
    par_rows(x, c, |r, row| {
        let tok = tokens[r] as usize;
        let pos = r % t_len;
        for j in 0..c {
            row[j] = wte[tok * c + j] + wpe[pos * c + j];
        }
    });
}

/// dwte[v] += sum over rows with tokens[r] == v of dx[r] -- a gather, like the kernel
fn embed_bwd_wte(dwte: &mut [f32], dx: &[f32], tokens: &[i32], c: usize) {
    par_rows(dwte, c, |v, row| {
        for (r, &tok) in tokens.iter().enumerate() {
            if tok as usize == v {
                for j in 0..c {
                    row[j] += dx[r * c + j];
                }
            }
        }
    });
}

fn embed_bwd_wpe(dwpe: &mut [f32], dx: &[f32], rows: usize, t_len: usize, c: usize) {
    for t in 0..t_len {
        for r in (t..rows).step_by(t_len) {
            for j in 0..c {
                dwpe[t * c + j] += dx[r * c + j];
            }
        }
    }
}

/// y = x * r, r = rsqrt(mean(x^2) + 1e-5), saving r per row for the backward pass
fn rmsnorm_fwd(y: &mut [f32], rinv: &mut [f32], x: &[f32], c: usize) {
    for (t, r) in rinv.iter_mut().enumerate() {
        let ms: f32 = x[t * c..(t + 1) * c].iter().map(|v| v * v).sum::<f32>() / c as f32;
        *r = 1.0 / (ms + 1e-5).sqrt();
    }
    par_rows(y, c, |t, row| {
        let r = rinv[t];
        for j in 0..c {
            row[j] = x[t * c + j] * r;
        }
    });
}

/// dx_j = r * dy_j - (r^3 / c) * x_j * dot(dy, x)
fn rmsnorm_bwd(dx: &mut [f32], x: &[f32], rinv: &[f32], dy: &[f32], c: usize, accum: bool) {
    par_rows(dx, c, |t, row| {
        let dot: f32 = (0..c).map(|j| dy[t * c + j] * x[t * c + j]).sum();
        let r = rinv[t];
        let k = r * r * r * dot / c as f32;
        for j in 0..c {
            let g = r * dy[t * c + j] - k * x[t * c + j];
            row[j] = if accum { row[j] + g } else { g };
        }
    });
}

/// y = x @ w^T,  x (m,k), w (n,k), y (m,n) -- the forward of linear(x, w)
fn matmul_nt(y: &mut [f32], x: &[f32], w: &[f32], n: usize, k: usize, accum: bool) {
    par_rows(y, n, |r, row| {
        let xr = &x[r * k..(r + 1) * k];
        for (col, out) in row.iter_mut().enumerate() {
            let wr = &w[col * k..(col + 1) * k];
            let acc: f32 = xr.iter().zip(wr).map(|(a, b)| a * b).sum();
            *out = if accum { *out + acc } else { acc };
        }
    });
}

/// dx = dy @ w,  dy (m,k), w (k,n), dx (m,n) -- backward wrt a linear's input.
/// Loop-interchanged: p outer, col inner, so both w and the output row are walked
/// unit-stride (which lets the compiler vectorize -- a strided reduction does not).
/// Each element still sums its k products in ascending-p order from zero and lands
/// in the output with a single add, exactly like the CUDA kernel's register
/// accumulator -- so the interchange is bit-identical to the naive loop.
fn matmul_nn(y: &mut [f32], x: &[f32], w: &[f32], n: usize, k: usize, accum: bool) {
    par_rows(y, n, |r, row| {
        let mut acc = vec![0.0f32; n];
        for p in 0..k {
            let s = x[r * k + p];
            let wr = &w[p * n..(p + 1) * n];
            for (out, &wv) in acc.iter_mut().zip(wr) {
                *out += s * wv;
            }
        }
        for (out, &a) in row.iter_mut().zip(&acc) {
            *out = if accum { *out + a } else { a };
        }
    });
}

/// dw += dy^T @ x,  dy (k,m), x (k,n), dw (m,n) -- backward wrt a linear's weights.
/// Same interchange and single-add accumulation as matmul_nn.
fn matmul_tn(dw: &mut [f32], dy: &[f32], x: &[f32], m: usize, n: usize, k: usize) {
    par_rows(dw, n, |r, row| {
        let mut acc = vec![0.0f32; n];
        for p in 0..k {
            let s = dy[p * m + r];
            let xr = &x[p * n..(p + 1) * n];
            for (out, &xv) in acc.iter_mut().zip(xr) {
                *out += s * xv;
            }
        }
        for (out, &a) in row.iter_mut().zip(&acc) {
            *out += a;
        }
    });
}

/// probs[b,h,t,u] = softmax_u(q_bt . k_bu / sqrt(d)) for u <= t, 0 above the diagonal
fn attn_softmax_fwd(att: &mut [f32], q: &[f32], k: &[f32], h_n: usize, t_len: usize, d: usize) {
    let c = h_n * d;
    par_rows(att, t_len, |i, row| {
        let (b, h, t) = (i / (h_n * t_len), (i / t_len) % h_n, i % t_len);
        let scale = 1.0 / (d as f32).sqrt();
        let mut s = [0f32; 16]; // MAX_T
        let mut maxv = f32::NEG_INFINITY;
        for u in 0..=t {
            let dot: f32 = (0..d)
                .map(|j| q[(b * t_len + t) * c + h * d + j] * k[(b * t_len + u) * c + h * d + j])
                .sum();
            s[u] = dot * scale;
            maxv = maxv.max(s[u]);
        }
        let mut tot = 0.0;
        for su in s.iter_mut().take(t + 1) {
            *su = (*su - maxv).exp();
            tot += *su;
        }
        for (u, out) in row.iter_mut().enumerate() {
            *out = if u <= t { s[u] / tot } else { 0.0 };
        }
    });
}

/// ds_u = p_u * (dp_u - dot(dp, p)), per causal row
fn attn_softmax_bwd(ds: &mut [f32], att: &[f32], dp: &[f32], t_len: usize) {
    par_rows(ds, t_len, |i, row| {
        let t = i % t_len;
        let base = i * t_len;
        let dot: f32 = (0..=t).map(|u| dp[base + u] * att[base + u]).sum();
        for (u, out) in row.iter_mut().enumerate() {
            *out = if u <= t { att[base + u] * (dp[base + u] - dot) } else { 0.0 };
        }
    });
}

/// dq_t = sum_u ds[t,u] * k_u / sqrt(d)
fn attn_scores_bwd_dq(dq: &mut [f32], ds: &[f32], k: &[f32], h_n: usize, t_len: usize, d: usize) {
    let c = h_n * d;
    par_rows(dq, c, |r, row| {
        let (b, t) = (r / t_len, r % t_len);
        let scale = 1.0 / (d as f32).sqrt();
        for (col, out) in row.iter_mut().enumerate() {
            let h = col / d;
            let acc: f32 = (0..=t)
                .map(|u| ds[((b * h_n + h) * t_len + t) * t_len + u] * k[(b * t_len + u) * c + col])
                .sum();
            *out = acc * scale;
        }
    });
}

/// dk_u = sum_t ds[t,u] * q_t / sqrt(d)
fn attn_scores_bwd_dk(dk: &mut [f32], ds: &[f32], q: &[f32], h_n: usize, t_len: usize, d: usize) {
    let c = h_n * d;
    par_rows(dk, c, |r, row| {
        let (b, u) = (r / t_len, r % t_len);
        let scale = 1.0 / (d as f32).sqrt();
        for (col, out) in row.iter_mut().enumerate() {
            let h = col / d;
            let acc: f32 = (u..t_len)
                .map(|t| ds[((b * h_n + h) * t_len + t) * t_len + u] * q[(b * t_len + t) * c + col])
                .sum();
            *out = acc * scale;
        }
    });
}

/// y[b,t] = sum_u probs[b,h,t,u] * v[b,u], per head
fn attn_mix_fwd(y: &mut [f32], att: &[f32], v: &[f32], h_n: usize, t_len: usize, d: usize) {
    let c = h_n * d;
    par_rows(y, c, |r, row| {
        let (b, t) = (r / t_len, r % t_len);
        for (col, out) in row.iter_mut().enumerate() {
            let h = col / d;
            let acc: f32 = (0..=t)
                .map(|u| att[((b * h_n + h) * t_len + t) * t_len + u] * v[(b * t_len + u) * c + col])
                .sum();
            *out = acc;
        }
    });
}

fn attn_mix_bwd_dp(dp: &mut [f32], dy: &[f32], v: &[f32], h_n: usize, t_len: usize, d: usize) {
    let c = h_n * d;
    par_rows(dp, t_len, |i, row| {
        let (b, h, t) = (i / (h_n * t_len), (i / t_len) % h_n, i % t_len);
        for (u, out) in row.iter_mut().enumerate() {
            *out = if u <= t {
                (0..d)
                    .map(|j| dy[(b * t_len + t) * c + h * d + j] * v[(b * t_len + u) * c + h * d + j])
                    .sum()
            } else {
                0.0
            };
        }
    });
}

fn attn_mix_bwd_dv(dv: &mut [f32], att: &[f32], dy: &[f32], h_n: usize, t_len: usize, d: usize) {
    let c = h_n * d;
    par_rows(dv, c, |r, row| {
        let (b, u) = (r / t_len, r % t_len);
        for (col, out) in row.iter_mut().enumerate() {
            let h = col / d;
            let acc: f32 = (u..t_len)
                .map(|t| att[((b * h_n + h) * t_len + t) * t_len + u] * dy[(b * t_len + t) * c + col])
                .sum();
            *out = acc;
        }
    });
}

fn relu_fwd(y: &mut [f32], x: &[f32]) {
    for (out, &v) in y.iter_mut().zip(x) {
        *out = v.max(0.0);
    }
}

fn relu_bwd_inplace(d: &mut [f32], x: &[f32]) {
    for (out, &v) in d.iter_mut().zip(x) {
        if v <= 0.0 {
            *out = 0.0;
        }
    }
}

fn add_fwd(y: &mut [f32], a: &[f32], b: &[f32]) {
    for ((out, &x1), &x2) in y.iter_mut().zip(a).zip(b) {
        *out = x1 + x2;
    }
}

fn add_inplace(y: &mut [f32], x: &[f32]) {
    for (out, &v) in y.iter_mut().zip(x) {
        *out += v;
    }
}

/// Softmax + negative log-likelihood per row; a target of -1 marks padding
fn ce_fwd(losses: &mut [f32], probs: &mut [f32], logits: &[f32], targets: &[i32], vocab: usize) {
    for (t, loss) in losses.iter_mut().enumerate() {
        if targets[t] < 0 {
            *loss = 0.0;
            continue;
        }
        let row = &logits[t * vocab..(t + 1) * vocab];
        let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut tot = 0.0;
        for j in 0..vocab {
            probs[t * vocab + j] = (row[j] - maxv).exp();
            tot += probs[t * vocab + j];
        }
        for j in 0..vocab {
            probs[t * vocab + j] /= tot;
        }
        *loss = -probs[t * vocab + targets[t] as usize].ln();
    }
}

/// d(mean loss)/dlogits = (probs - onehot(target)) / valid, zero on padding
fn ce_bwd(dlogits: &mut [f32], probs: &[f32], targets: &[i32], vocab: usize, valid: usize) {
    par_rows(dlogits, vocab, |t, row| {
        for (j, out) in row.iter_mut().enumerate() {
            *out = if targets[t] < 0 {
                0.0
            } else {
                (probs[t * vocab + j] - if j as i32 == targets[t] { 1.0 } else { 0.0 })
                    / valid as f32
            };
        }
    });
}

#[derive(Clone, Copy)]
struct LayerOff {
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    fc1: usize,
    fc2: usize,
}

/// Offsets into the single flat parameter buffer; same order as every sibling crate.
#[derive(Clone)]
struct Layout {
    wte: usize,
    wpe: usize,
    lm_head: usize,
    layers: Vec<LayerOff>,
    total: usize,
}

impl Layout {
    fn new(cfg: &Cfg, vocab: usize) -> Layout {
        let (c, c4) = (cfg.n_embd, 4 * cfg.n_embd);
        let mut off = 0;
        let mut take = |n: usize| {
            let o = off;
            off += n;
            o
        };
        Layout {
            wte: take(vocab * c),
            wpe: take(cfg.block_size * c),
            lm_head: take(vocab * c),
            layers: (0..cfg.n_layer)
                .map(|_| LayerOff {
                    wq: take(c * c),
                    wk: take(c * c),
                    wv: take(c * c),
                    wo: take(c * c),
                    fc1: take(c4 * c),
                    fc2: take(c * c4),
                })
                .collect(),
            total: off,
        }
    }
}

fn init_params(rng: &mut Rng, cfg: &Cfg, vocab: usize) -> Vec<f32> {
    let l = Layout::new(cfg, vocab);
    (0..l.total).map(|_| rng.gauss(0.0, cfg.init_std) as f32).collect()
}

/// One transformer layer's forward activations, saved for the backward pass.
struct LayerActs {
    xn: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    atty: Vec<f32>,
    x_mid: Vec<f32>,
    xn2: Vec<f32>,
    hpre: Vec<f32>,
    h: Vec<f32>,
    x_out: Vec<f32>,
    r1: Vec<f32>,
    r2: Vec<f32>,
}

/// The whole model in host memory: the CPU mirror of microgpt-cuda's `Gpu` struct.
struct Model {
    cfg: Cfg,
    vocab: usize,
    l: Layout,
    t_len: usize,
    rows: usize,
    valid: usize,

    params: Vec<f32>,
    grads: Vec<f32>,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,

    tokens: Vec<i32>,
    targets: Vec<i32>,
    x0: Vec<f32>,
    x1: Vec<f32>,
    r0: Vec<f32>,
    layers: Vec<LayerActs>,
    proj: Vec<f32>,
    mlpo: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
    losses: Vec<f32>,
    d_logits: Vec<f32>,
    d_x: Vec<f32>,
    d_xin: Vec<f32>,
    d_h: Vec<f32>,
    d_xn2: Vec<f32>,
    d_x2: Vec<f32>,
    d_atty: Vec<f32>,
    d_att: Vec<f32>,
    d_scores: Vec<f32>,
    d_q: Vec<f32>,
    d_k: Vec<f32>,
    d_v: Vec<f32>,
    d_xn: Vec<f32>,
    d_x0: Vec<f32>,
}

impl Model {
    fn new(cfg: Cfg, vocab: usize, params: Vec<f32>) -> Model {
        assert!(cfg.block_size <= 16, "the softmax stack row is sized for block_size <= 16");
        let l = Layout::new(&cfg, vocab);
        assert_eq!(params.len(), l.total);
        let (r, c, c4) = (cfg.max_rows(), cfg.n_embd, 4 * cfg.n_embd);
        let att_len = cfg.batch * cfg.n_head * cfg.block_size * cfg.block_size;
        let layers = (0..cfg.n_layer)
            .map(|_| LayerActs {
                xn: vec![0.0; r * c],
                q: vec![0.0; r * c],
                k: vec![0.0; r * c],
                v: vec![0.0; r * c],
                att: vec![0.0; att_len],
                atty: vec![0.0; r * c],
                x_mid: vec![0.0; r * c],
                xn2: vec![0.0; r * c],
                hpre: vec![0.0; r * c4],
                h: vec![0.0; r * c4],
                x_out: vec![0.0; r * c],
                r1: vec![0.0; r],
                r2: vec![0.0; r],
            })
            .collect();
        Model {
            vocab,
            t_len: 0,
            rows: 0,
            valid: 0,
            grads: vec![0.0; l.total],
            adam_m: vec![0.0; l.total],
            adam_v: vec![0.0; l.total],
            tokens: vec![0; r],
            targets: vec![0; r],
            x0: vec![0.0; r * c],
            x1: vec![0.0; r * c],
            r0: vec![0.0; r],
            layers,
            proj: vec![0.0; r * c],
            mlpo: vec![0.0; r * c],
            logits: vec![0.0; r * vocab],
            probs: vec![0.0; r * vocab],
            losses: vec![0.0; r],
            d_logits: vec![0.0; r * vocab],
            d_x: vec![0.0; r * c],
            d_xin: vec![0.0; r * c],
            d_h: vec![0.0; r * c4],
            d_xn2: vec![0.0; r * c],
            d_x2: vec![0.0; r * c],
            d_atty: vec![0.0; r * c],
            d_att: vec![0.0; att_len],
            d_scores: vec![0.0; att_len],
            d_q: vec![0.0; r * c],
            d_k: vec![0.0; r * c],
            d_v: vec![0.0; r * c],
            d_xn: vec![0.0; r * c],
            d_x0: vec![0.0; r * c],
            l,
            cfg,
            params,
        }
    }

    /// Forward `tokens` -- `batch` documents of `t_len` positions each, concatenated,
    /// short documents padded -- leaving logits (rows, V) in `self.logits`.
    fn forward(&mut self, tokens: &[i32], t_len: usize) {
        assert!(t_len > 0 && t_len <= self.cfg.block_size && tokens.len() % t_len == 0);
        assert!(tokens.len() <= self.cfg.max_rows());
        self.t_len = t_len;
        self.rows = tokens.len();
        let (rows, c, c4) = (self.rows, self.cfg.n_embd, 4 * self.cfg.n_embd);
        let (b_n, h_n, d, v) = (rows / t_len, self.cfg.n_head, self.cfg.head_dim(), self.vocab);
        self.tokens[..rows].copy_from_slice(tokens);

        let p = &self.params;
        embed_fwd(
            &mut self.x0[..rows * c],
            &p[self.l.wte..self.l.wpe],
            &p[self.l.wpe..self.l.lm_head],
            &self.tokens[..rows],
            t_len,
            c,
        );
        rmsnorm_fwd(&mut self.x1[..rows * c], &mut self.r0[..rows], &self.x0[..rows * c], c);

        for li in 0..self.cfg.n_layer {
            let lo = self.l.layers[li];
            let (done, rest) = self.layers.split_at_mut(li);
            let xin = if li == 0 { &self.x1[..rows * c] } else { &done[li - 1].x_out[..rows * c] };
            let la = &mut rest[0];

            rmsnorm_fwd(&mut la.xn[..rows * c], &mut la.r1[..rows], xin, c);
            matmul_nt(&mut la.q[..rows * c], &la.xn, &p[lo.wq..lo.wk], c, c, false);
            matmul_nt(&mut la.k[..rows * c], &la.xn, &p[lo.wk..lo.wv], c, c, false);
            matmul_nt(&mut la.v[..rows * c], &la.xn, &p[lo.wv..lo.wo], c, c, false);
            attn_softmax_fwd(
                &mut la.att[..b_n * h_n * t_len * t_len],
                &la.q,
                &la.k,
                h_n,
                t_len,
                d,
            );
            attn_mix_fwd(&mut la.atty[..rows * c], &la.att, &la.v, h_n, t_len, d);
            matmul_nt(&mut self.proj[..rows * c], &la.atty, &p[lo.wo..lo.fc1], c, c, false);
            add_fwd(&mut la.x_mid[..rows * c], xin, &self.proj[..rows * c]);

            rmsnorm_fwd(&mut la.xn2[..rows * c], &mut la.r2[..rows], &la.x_mid[..rows * c], c);
            matmul_nt(&mut la.hpre[..rows * c4], &la.xn2, &p[lo.fc1..lo.fc2], c4, c, false);
            relu_fwd(&mut la.h[..rows * c4], &la.hpre[..rows * c4]);
            matmul_nt(&mut self.mlpo[..rows * c], &la.h, &p[lo.fc2..lo.fc2 + c * c4], c, c4, false);
            add_fwd(&mut la.x_out[..rows * c], &la.x_mid[..rows * c], &self.mlpo[..rows * c]);
        }

        let x_final = &self.layers[self.cfg.n_layer - 1].x_out;
        matmul_nt(
            &mut self.logits[..rows * v],
            x_final,
            &p[self.l.lm_head..self.l.lm_head + v * c],
            v,
            c,
            false,
        );
    }

    /// Cross-entropy against `targets` (-1 marks padding), averaged over valid positions.
    fn loss(&mut self, targets: &[i32]) -> f32 {
        assert_eq!(targets.len(), self.rows);
        self.valid = targets.iter().filter(|&&t| t >= 0).count();
        self.targets[..self.rows].copy_from_slice(targets);
        ce_fwd(
            &mut self.losses[..self.rows],
            &mut self.probs[..self.rows * self.vocab],
            &self.logits[..self.rows * self.vocab],
            targets,
            self.vocab,
        );
        self.losses[..self.rows].iter().sum::<f32>() / self.valid as f32
    }

    /// The chain rule, unrolled by hand from the loss back to every parameter --
    /// the same sequence of steps as microgpt-cuda's backward(), on CPU buffers.
    fn backward(&mut self) {
        let (rows, t_len, c, c4) = (self.rows, self.t_len, self.cfg.n_embd, 4 * self.cfg.n_embd);
        let (b_n, h_n, d, v) = (rows / t_len, self.cfg.n_head, self.cfg.head_dim(), self.vocab);
        let att_n = b_n * h_n * t_len * t_len;

        ce_bwd(
            &mut self.d_logits[..rows * v],
            &self.probs[..rows * v],
            &self.targets[..rows],
            v,
            self.valid,
        );

        // logits = x_out[last] @ lm_head^T
        let x_final = &self.layers[self.cfg.n_layer - 1].x_out;
        matmul_tn(
            &mut self.grads[self.l.lm_head..self.l.lm_head + v * c],
            &self.d_logits[..rows * v],
            &x_final[..rows * c],
            v,
            c,
            rows,
        );
        matmul_nn(
            &mut self.d_x[..rows * c],
            &self.d_logits[..rows * v],
            &self.params[self.l.lm_head..self.l.lm_head + v * c],
            c,
            v,
            false,
        );

        for li in (0..self.cfg.n_layer).rev() {
            let lo = self.l.layers[li];
            let xin =
                if li == 0 { &self.x1[..rows * c] } else { &self.layers[li - 1].x_out[..rows * c] };
            let la = &self.layers[li];
            let p = &self.params;

            // x_out = x_mid + relu(xn2 @ fc1^T) @ fc2^T;  xn2 = rmsnorm(x_mid)
            matmul_tn(
                &mut self.grads[lo.fc2..lo.fc2 + c * c4],
                &self.d_x[..rows * c],
                &la.h[..rows * c4],
                c,
                c4,
                rows,
            );
            matmul_nn(
                &mut self.d_h[..rows * c4],
                &self.d_x[..rows * c],
                &p[lo.fc2..lo.fc2 + c * c4],
                c4,
                c,
                false,
            );
            relu_bwd_inplace(&mut self.d_h[..rows * c4], &la.hpre[..rows * c4]);
            matmul_tn(
                &mut self.grads[lo.fc1..lo.fc2],
                &self.d_h[..rows * c4],
                &la.xn2[..rows * c],
                c4,
                c,
                rows,
            );
            matmul_nn(&mut self.d_xn2[..rows * c], &self.d_h[..rows * c4], &p[lo.fc1..lo.fc2], c, c4, false);
            rmsnorm_bwd(
                &mut self.d_x2[..rows * c],
                &la.x_mid[..rows * c],
                &la.r2[..rows],
                &self.d_xn2[..rows * c],
                c,
                false,
            );
            add_inplace(&mut self.d_x2[..rows * c], &self.d_x[..rows * c]); // residual

            // x_mid = x_in + atty @ wo^T
            matmul_tn(&mut self.grads[lo.wo..lo.fc1], &self.d_x2[..rows * c], &la.atty[..rows * c], c, c, rows);
            matmul_nn(&mut self.d_atty[..rows * c], &self.d_x2[..rows * c], &p[lo.wo..lo.fc1], c, c, false);

            // atty = att @ v;  att = causal_softmax(q @ k^T / sqrt(d))
            attn_mix_bwd_dp(&mut self.d_att[..att_n], &self.d_atty[..rows * c], &la.v[..rows * c], h_n, t_len, d);
            attn_mix_bwd_dv(&mut self.d_v[..rows * c], &la.att[..att_n], &self.d_atty[..rows * c], h_n, t_len, d);
            attn_softmax_bwd(&mut self.d_scores[..att_n], &la.att[..att_n], &self.d_att[..att_n], t_len);
            attn_scores_bwd_dq(&mut self.d_q[..rows * c], &self.d_scores[..att_n], &la.k[..rows * c], h_n, t_len, d);
            attn_scores_bwd_dk(&mut self.d_k[..rows * c], &self.d_scores[..att_n], &la.q[..rows * c], h_n, t_len, d);

            // q/k/v = xn @ w{q,k,v}^T;  xn = rmsnorm(x_in)
            matmul_tn(&mut self.grads[lo.wq..lo.wk], &self.d_q[..rows * c], &la.xn[..rows * c], c, c, rows);
            matmul_tn(&mut self.grads[lo.wk..lo.wv], &self.d_k[..rows * c], &la.xn[..rows * c], c, c, rows);
            matmul_tn(&mut self.grads[lo.wv..lo.wo], &self.d_v[..rows * c], &la.xn[..rows * c], c, c, rows);
            matmul_nn(&mut self.d_xn[..rows * c], &self.d_q[..rows * c], &p[lo.wq..lo.wk], c, c, false);
            matmul_nn(&mut self.d_xn[..rows * c], &self.d_k[..rows * c], &p[lo.wk..lo.wv], c, c, true);
            matmul_nn(&mut self.d_xn[..rows * c], &self.d_v[..rows * c], &p[lo.wv..lo.wo], c, c, true);
            rmsnorm_bwd(&mut self.d_xin[..rows * c], xin, &la.r1[..rows], &self.d_xn[..rows * c], c, false);
            add_inplace(&mut self.d_xin[..rows * c], &self.d_x2[..rows * c]); // residual

            std::mem::swap(&mut self.d_x, &mut self.d_xin); // d_x now feeds layer li-1
        }

        // x1 = rmsnorm(x0);  x0 = wte[token] + wpe[pos]
        rmsnorm_bwd(&mut self.d_x0[..rows * c], &self.x0[..rows * c], &self.r0[..rows], &self.d_x[..rows * c], c, false);
        embed_bwd_wte(&mut self.grads[self.l.wte..self.l.wpe], &self.d_x0[..rows * c], &self.tokens[..rows], c);
        embed_bwd_wpe(&mut self.grads[self.l.wpe..self.l.lm_head], &self.d_x0[..rows * c], rows, t_len, c);
    }

    /// The same fused Adam update as the CUDA kernel; consumes and zeroes the grads.
    fn adam(&mut self, step: usize, lr_t: f32) {
        let (beta1, beta2, eps) = (0.85f32, 0.99f32, 1e-8f32);
        let bc1 = (1.0 - 0.85f64.powi(step as i32 + 1)) as f32;
        let bc2 = (1.0 - 0.99f64.powi(step as i32 + 1)) as f32;
        for i in 0..self.l.total {
            let gi = self.grads[i];
            self.adam_m[i] = beta1 * self.adam_m[i] + (1.0 - beta1) * gi;
            self.adam_v[i] = beta2 * self.adam_v[i] + (1.0 - beta2) * gi * gi;
            let mh = self.adam_m[i] / bc1;
            let vh = self.adam_v[i] / bc2;
            self.params[i] -= lr_t * mh / (vh.sqrt() + eps);
            self.grads[i] = 0.0;
        }
    }

    fn last_logits(&self) -> &[f32] {
        &self.logits[(self.rows - 1) * self.vocab..self.rows * self.vocab]
    }
}

fn main() {
    let parity = std::env::args().any(|a| a == "--parity");
    let cfg = if parity { PARITY } else { SCALE };
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
    let text = std::fs::read_to_string("input.txt").expect("read input.txt");
    let mut docs: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    rng.shuffle(&mut docs);
    println!("num docs: {}", docs.len());

    let uchars: Vec<char> =
        docs.iter().flat_map(|d| d.chars()).collect::<BTreeSet<_>>().into_iter().collect();
    let stoi: HashMap<char, usize> = uchars.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {vocab_size}");

    let host_params = init_params(&mut rng, &cfg, vocab_size);
    println!("num params: {}", host_params.len());
    if !parity {
        println!(
            "mode: scale ({} layers, {}-dim, {} heads, batch {})",
            cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.batch
        );
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("device: CPU, {threads} threads");
    let mut model = Model::new(cfg, vocab_size, host_params);

    let t_start = std::time::Instant::now();
    for step in 0..cfg.num_steps {
        // Assemble a batch: `cfg.batch` documents, each BOS-wrapped and clipped to the
        // block size; short ones padded with BOS tokens and -1 (masked) targets.
        let t_max = cfg.block_size;
        let mut tokens = vec![bos as i32; cfg.batch * t_max];
        let mut targets = vec![-1i32; cfg.batch * t_max];
        let mut n_last = 0;
        for b in 0..cfg.batch {
            let doc = docs[(step * cfg.batch + b) % docs.len()];
            let mut toks = vec![bos as i32];
            toks.extend(doc.chars().map(|c| stoi[&c] as i32));
            toks.push(bos as i32);
            let n = t_max.min(toks.len() - 1);
            tokens[b * t_max..b * t_max + n].copy_from_slice(&toks[..n]);
            targets[b * t_max..b * t_max + n].copy_from_slice(&toks[1..n + 1]);
            n_last = n;
        }
        let (tok_slice, tgt_slice, t_len) = if cfg.batch == 1 {
            (&tokens[..n_last], &targets[..n_last], n_last)
        } else {
            (&tokens[..], &targets[..], t_max)
        };

        model.forward(tok_slice, t_len);
        let loss = model.loss(tgt_slice);
        model.backward();
        let lr_t = cfg.learning_rate * (1.0 - step as f32 / cfg.num_steps as f32);
        model.adam(step, lr_t);

        print!("step {:4} / {:4} | loss {:.4}\r", step + 1, cfg.num_steps, loss);
        std::io::stdout().flush().unwrap();
    }
    let elapsed = t_start.elapsed();
    println!(
        "\ntrain time: {:.2}s ({:.1} steps/s)",
        elapsed.as_secs_f64(),
        cfg.num_steps as f64 / elapsed.as_secs_f64()
    );

    // Inference: re-forward the whole prefix per character, sample with the host RNG
    let temperature = 0.5f64;
    println!("--- inference (new, hallucinated names) ---");
    for sample_idx in 0..20 {
        let mut ctx = vec![bos as i32];
        let mut sample = String::new();
        for _ in 0..cfg.block_size {
            model.forward(&ctx, ctx.len());
            let logits = model.last_logits();
            let maxv = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let weights: Vec<f64> =
                logits.iter().map(|&l| ((l as f64 - maxv) / temperature).exp()).collect();
            let tok = rng.choices(&weights);
            if tok == bos {
                break;
            }
            sample.push(uchars[tok]);
            ctx.push(tok as i32);
        }
        println!("sample {:2}: {}", sample_idx + 1, sample);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(analytic: f32, numeric: f32, what: &str) {
        let tol = 1e-3 + 0.02 * numeric.abs();
        assert!(
            (analytic - numeric).abs() < tol,
            "{what}: analytic {analytic} vs numeric {numeric}"
        );
    }

    #[test]
    fn untrained_parity_loss_is_near_uniform() {
        let mut rng = Rng::new(42);
        let vocab = 27;
        let params = init_params(&mut rng, &PARITY, vocab);
        let mut model = Model::new(PARITY, vocab, params);
        let tokens: Vec<i32> = vec![26, 4, 12, 12, 0]; // BOS e m m a
        let targets: Vec<i32> = vec![4, 12, 12, 0, 26];
        model.forward(&tokens, tokens.len());
        let loss = model.loss(&targets);
        let uniform = (vocab as f32).ln();
        assert!((loss - uniform).abs() < 0.5, "untrained loss {loss} vs ln(V) {uniform}");
    }

    #[test]
    fn multilayer_batched_gradients_match_finite_differences() {
        // Same tiny scale-shaped config as microgpt-cuda's test: multiple layers, a
        // batch of two documents of different lengths, padding and loss mask active.
        let tiny = Cfg {
            n_layer: 2,
            n_embd: 8,
            n_head: 2,
            block_size: 8,
            batch: 2,
            num_steps: 0,
            init_std: 0.08,
            learning_rate: 0.01,
        };
        let mut rng = Rng::new(7);
        let vocab = 11;
        let host_params = init_params(&mut rng, &tiny, vocab);
        let mut model = Model::new(tiny, vocab, host_params.clone());
        let l = model.l.clone();

        let tokens: Vec<i32> = vec![
            10, 3, 1, 4, 1, 5, 9, 2, // doc 0: 8 positions
            10, 2, 7, 1, 10, 10, 10, 10, // doc 1: 4 valid + BOS padding
        ];
        let targets: Vec<i32> = vec![
            3, 1, 4, 1, 5, 9, 2, 6, //
            2, 7, 1, 10, -1, -1, -1, -1,
        ];

        model.forward(&tokens, 8);
        model.loss(&targets);
        model.backward();
        let grads = model.grads.clone();

        let probes: Vec<usize> = vec![
            l.wte + 10 * 8 + 2,
            l.wte + 3 * 8 + 5,
            l.wpe + 6 * 8 + 1,
            l.lm_head + 2 * 8 + 4,
            l.layers[0].wq + 13,
            l.layers[0].wo + 40,
            l.layers[0].fc1 + 99,
            l.layers[1].wk + 21,
            l.layers[1].wv + 50,
            l.layers[1].fc2 + 123,
        ];

        let h = 1e-2f32;
        for &i in &probes {
            let mut eval = |delta: f32| -> f32 {
                model.params.copy_from_slice(&host_params);
                model.params[i] += delta;
                model.forward(&tokens, 8);
                model.loss(&targets)
            };
            let numeric = (eval(h) - eval(-h)) / (2.0 * h);
            assert_close(grads[i], numeric, &format!("multilayer dparam[{i}]"));
        }
    }

    #[test]
    fn threaded_ops_match_serial_reference() {
        // par_rows must not change any element's math: compare a threaded matmul
        // against a plain triple loop.
        let mut rng = Rng::new(9);
        let (m, n, k) = (67usize, 33usize, 41usize); // odd sizes, forces ragged chunks
        let x: Vec<f32> = (0..m * k).map(|_| rng.gauss(0.0, 1.0) as f32).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.gauss(0.0, 1.0) as f32).collect();
        let mut y = vec![0.0f32; m * n];
        matmul_nt(&mut y, &x, &w, n, k, false);
        for r in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|p| x[r * k + p] * w[col * k + p]).sum();
                assert_eq!(y[r * n + col].to_bits(), want.to_bits(), "matmul_nt [{r},{col}]");
            }
        }
    }
}
