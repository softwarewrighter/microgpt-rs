//! microgpt on an NVIDIA GPU: the same algorithm as microgpt-rs, with the autograd
//! removed entirely. Every layer implements forward *and* backward as a hand-written
//! CUDA kernel -- a micro-sized, Rust-flavored llm.c.
//!
//! Where microgpt-rs derives gradients per-scalar by machine (the tape) and
//! microgpt-mlx asks a framework for them (`value_and_grad`), here the chain rule is
//! applied per-layer on paper and transcribed into kernels: rmsnorm backward, causal
//! softmax backward, matmul backward, cross-entropy backward. The contrast with the
//! Mac's unified memory is deliberate too -- parameters and activations live in GPU
//! memory, and every number the host sees (the loss, the sampled logits) crosses the
//! PCIe bus in an explicit device-to-host copy.
//!
//! Kernels are CUDA C strings compiled at startup with NVRTC (via `cudarc`), launched
//! from safe Rust. No atomics anywhere -- every kernel writes each output element from
//! exactly one thread in a fixed order -- so training is bit-deterministic: run it
//! twice, get identical bits, same as the CPU crate.

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::sync::Arc;

// Same tiny RNG as microgpt-rs (splitmix64 + Box-Muller), so the two crates shuffle
// the dataset identically and initialize bit-identical parameters (modulo f32 cast).
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

#[allow(dead_code)] // kept for parity with the sibling crates; the single layer is unrolled below
const N_LAYER: usize = 1;
const N_EMBD: usize = 16;
const BLOCK_SIZE: usize = 16;
const N_HEAD: usize = 4;
const HEAD_DIM: usize = N_EMBD / N_HEAD;

// The complete GPU-side algorithm: every forward kernel below has its hand-derived
// backward next to it. Indexing is row-major throughout. `i` is always this thread's
// output element -- one thread per output, no atomics, so every run is bit-identical.
const KERNELS: &str = r#"
#define MAX_T 16  // BLOCK_SIZE upper bound, for stack rows in the softmax kernels

// ---- embedding: x[t] = wte[token[t]] + wpe[t] ----
extern "C" __global__ void embed_fwd(float *x, const float *params, const int *tokens,
                                     int wte_off, int wpe_off, int t_len, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t_len * c) return;
    int t = i / c, j = i % c;
    x[i] = params[wte_off + tokens[t] * c + j] + params[wpe_off + t * c + j];
}

// dwte[v][j] = sum over positions t where tokens[t] == v of dx[t][j].
// A gather per vocab row instead of a scatter with atomicAdd: deterministic.
extern "C" __global__ void embed_bwd_wte(float *grads, const float *dx, const int *tokens,
                                         int wte_off, int vocab, int t_len, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab * c) return;
    int v = i / c, j = i % c;
    float acc = 0.0f;
    for (int t = 0; t < t_len; t++)
        if (tokens[t] == v) acc += dx[t * c + j];
    grads[wte_off + i] += acc;
}

extern "C" __global__ void embed_bwd_wpe(float *grads, const float *dx,
                                         int wpe_off, int t_len, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t_len * c) return;
    grads[wpe_off + i] += dx[i];
}

// ---- rmsnorm: y = x * r, r = rsqrt(mean(x^2) + 1e-5), one thread per row ----
extern "C" __global__ void rmsnorm_fwd(float *y, float *rinv, const float *x,
                                       int t_len, int c) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= t_len) return;
    float ms = 0.0f;
    for (int j = 0; j < c; j++) ms += x[t * c + j] * x[t * c + j];
    float r = rsqrtf(ms / c + 1e-5f);
    rinv[t] = r;
    for (int j = 0; j < c; j++) y[t * c + j] = x[t * c + j] * r;
}

// dy_i/dx_j = r * delta_ij - x_i * x_j * r^3 / c, so
// dx_j = r * dy_j - (r^3 / c) * x_j * dot(dy, x)
extern "C" __global__ void rmsnorm_bwd(float *dx, const float *x, const float *rinv,
                                       const float *dy, int t_len, int c, int accum) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= t_len) return;
    float dot = 0.0f;
    for (int j = 0; j < c; j++) dot += dy[t * c + j] * x[t * c + j];
    float r = rinv[t];
    float k = r * r * r * dot / c;
    for (int j = 0; j < c; j++) {
        float g = r * dy[t * c + j] - k * x[t * c + j];
        dx[t * c + j] = accum ? dx[t * c + j] + g : g;
    }
}

// ---- matmul, naive: one thread per output element (K <= 64 here) ----
// forward of linear(x, w): y = x @ w^T,  x (m,k), w (n,k), y (m,n)
extern "C" __global__ void matmul_nt(float *y, const float *x, const float *w,
                                     int m, int n, int k, int accum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * n) return;
    int r = i / n, col = i % n;
    float acc = 0.0f;
    for (int p = 0; p < k; p++) acc += x[r * k + p] * w[col * k + p];
    y[i] = accum ? y[i] + acc : acc;
}

// backward wrt the input: dx = dy @ w,  dy (m,k), w (k,n), dx (m,n)
extern "C" __global__ void matmul_nn(float *y, const float *x, const float *w,
                                     int m, int n, int k, int accum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * n) return;
    int r = i / n, col = i % n;
    float acc = 0.0f;
    for (int p = 0; p < k; p++) acc += x[r * k + p] * w[p * n + col];
    y[i] = accum ? y[i] + acc : acc;
}

// backward wrt the weights: dw += dy^T @ x,  dy (k,m), x (k,n), dw (m,n)
extern "C" __global__ void matmul_tn(float *dw, const float *dy, const float *x,
                                     int m, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * n) return;
    int r = i / n, col = i % n;
    float acc = 0.0f;
    for (int p = 0; p < k; p++) acc += dy[p * m + r] * x[p * n + col];
    dw[i] += acc;
}

// ---- causal attention softmax: probs[h,t,u] = softmax_u(q_t . k_u / sqrt(d)), u <= t ----
extern "C" __global__ void attn_softmax_fwd(float *probs, const float *q, const float *k,
                                            int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h_n * t_len) return;
    int h = i / t_len, t = i % t_len, c = h_n * d;
    float scale = rsqrtf((float)d);
    float s[MAX_T];
    float maxv = -1e30f;
    for (int u = 0; u <= t; u++) {
        float dot = 0.0f;
        for (int j = 0; j < d; j++) dot += q[t * c + h * d + j] * k[u * c + h * d + j];
        s[u] = dot * scale;
        maxv = fmaxf(maxv, s[u]);
    }
    float tot = 0.0f;
    for (int u = 0; u <= t; u++) { s[u] = expf(s[u] - maxv); tot += s[u]; }
    float *row = probs + (h * t_len + t) * t_len;
    for (int u = 0; u < t_len; u++) row[u] = (u <= t) ? s[u] / tot : 0.0f;
}

// softmax backward per causal row: ds_u = p_u * (dp_u - dot(dp, p))
extern "C" __global__ void attn_softmax_bwd(float *ds, const float *probs, const float *dp,
                                            int h_n, int t_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h_n * t_len) return;
    int row = i * t_len, t = i % t_len;
    float dot = 0.0f;
    for (int u = 0; u <= t; u++) dot += dp[row + u] * probs[row + u];
    for (int u = 0; u < t_len; u++)
        ds[row + u] = (u <= t) ? probs[row + u] * (dp[row + u] - dot) : 0.0f;
}

// scores backward: dq_t += ds[t,u] * k_u / sqrt(d);  dk_u += ds[t,u] * q_t / sqrt(d)
extern "C" __global__ void attn_scores_bwd_dq(float *dq, const float *ds, const float *k,
                                              int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = h_n * d;
    if (i >= t_len * c) return;
    int t = i / c, h = (i % c) / d;
    float acc = 0.0f;
    for (int u = 0; u <= t; u++) acc += ds[(h * t_len + t) * t_len + u] * k[u * c + i % c];
    dq[i] = acc * rsqrtf((float)d);
}

extern "C" __global__ void attn_scores_bwd_dk(float *dk, const float *ds, const float *q,
                                              int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = h_n * d;
    if (i >= t_len * c) return;
    int u = i / c, h = (i % c) / d;
    float acc = 0.0f;
    for (int t = u; t < t_len; t++) acc += ds[(h * t_len + t) * t_len + u] * q[t * c + i % c];
    dk[i] = acc * rsqrtf((float)d);
}

// ---- attention mix: y[t] = sum_u probs[h,t,u] * v[u], per head ----
extern "C" __global__ void attn_mix_fwd(float *y, const float *probs, const float *v,
                                        int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = h_n * d;
    if (i >= t_len * c) return;
    int t = i / c, h = (i % c) / d;
    float acc = 0.0f;
    for (int u = 0; u <= t; u++) acc += probs[(h * t_len + t) * t_len + u] * v[u * c + i % c];
    y[i] = acc;
}

extern "C" __global__ void attn_mix_bwd_dp(float *dp, const float *dy, const float *v,
                                           int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h_n * t_len * t_len) return;
    int h = i / (t_len * t_len), t = (i / t_len) % t_len, u = i % t_len, c = h_n * d;
    float acc = 0.0f;
    if (u <= t)
        for (int j = 0; j < d; j++) acc += dy[t * c + h * d + j] * v[u * c + h * d + j];
    dp[i] = acc;
}

extern "C" __global__ void attn_mix_bwd_dv(float *dv, const float *probs, const float *dy,
                                           int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = h_n * d;
    if (i >= t_len * c) return;
    int u = i / c, h = (i % c) / d;
    float acc = 0.0f;
    for (int t = u; t < t_len; t++) acc += probs[(h * t_len + t) * t_len + u] * dy[t * c + i % c];
    dv[i] = acc;
}

// ---- elementwise ----
extern "C" __global__ void relu_fwd(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(x[i], 0.0f);
}

extern "C" __global__ void relu_bwd_inplace(float *d, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] <= 0.0f) d[i] = 0.0f;
}

extern "C" __global__ void add_fwd(float *y, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

extern "C" __global__ void add_inplace(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

// ---- cross-entropy: softmax + negative log-likelihood, one thread per row ----
extern "C" __global__ void ce_fwd(float *losses, float *probs, const float *logits,
                                  const int *targets, int t_len, int vocab) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= t_len) return;
    const float *row = logits + t * vocab;
    float maxv = -1e30f;
    for (int j = 0; j < vocab; j++) maxv = fmaxf(maxv, row[j]);
    float tot = 0.0f;
    for (int j = 0; j < vocab; j++) { probs[t * vocab + j] = expf(row[j] - maxv); tot += probs[t * vocab + j]; }
    for (int j = 0; j < vocab; j++) probs[t * vocab + j] /= tot;
    losses[t] = -logf(probs[t * vocab + targets[t]]);
}

// d(mean loss)/dlogits = (probs - onehot(target)) / t_len
extern "C" __global__ void ce_bwd(float *dlogits, const float *probs, const int *targets,
                                  int t_len, int vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t_len * vocab) return;
    int t = i / vocab, j = i % vocab;
    dlogits[i] = (probs[i] - (j == targets[t] ? 1.0f : 0.0f)) / t_len;
}

// ---- fused Adam step over the flat parameter buffer; zeroes the grad it consumed ----
extern "C" __global__ void adam_step(float *p, float *g, float *m, float *v, int n,
                                     float lr, float b1, float b2, float bc1, float bc2,
                                     float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    m[i] = b1 * m[i] + (1.0f - b1) * gi;
    v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
    float mh = m[i] / bc1, vh = v[i] / bc2;
    p[i] -= lr * mh / (sqrtf(vh) + eps);
    g[i] = 0.0f;
}
"#;

const KERNEL_NAMES: &[&str] = &[
    "embed_fwd", "embed_bwd_wte", "embed_bwd_wpe",
    "rmsnorm_fwd", "rmsnorm_bwd",
    "matmul_nt", "matmul_nn", "matmul_tn",
    "attn_softmax_fwd", "attn_softmax_bwd",
    "attn_scores_bwd_dq", "attn_scores_bwd_dk",
    "attn_mix_fwd", "attn_mix_bwd_dp", "attn_mix_bwd_dv",
    "relu_fwd", "relu_bwd_inplace", "add_fwd", "add_inplace",
    "ce_fwd", "ce_bwd", "adam_step",
];

/// `launch!(kernels, "name", n_threads; &arg, &mut arg, ...)` -- one thread per output
/// element, `n_threads` of them.
macro_rules! launch {
    ($k:expr, $name:literal, $n:expr; $($arg:expr),* $(,)?) => {{
        let mut b = $k.stream.launch_builder(&$k.funcs[$name]);
        $( b.arg($arg); )*
        unsafe { b.launch(cudarc::driver::LaunchConfig::for_num_elems($n as u32)) }
    }};
}

struct Kernels {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    funcs: HashMap<&'static str, CudaFunction>,
}

impl Kernels {
    fn new() -> Result<Kernels, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(compile_ptx(KERNELS)?)?;
        let mut funcs = HashMap::new();
        for name in KERNEL_NAMES {
            funcs.insert(*name, module.load_function(name)?);
        }
        Ok(Kernels { ctx, stream, funcs })
    }
}

/// Byte offsets of each weight matrix inside the single flat parameter buffer.
/// Order matches microgpt-rs / microgpt-mlx exactly, so the same RNG stream
/// produces the same initialization in all three crates.
#[derive(Clone, Copy)]
struct Layout {
    wte: usize,
    wpe: usize,
    lm_head: usize,
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    fc1: usize,
    fc2: usize,
    total: usize,
}

impl Layout {
    fn new(vocab: usize) -> Layout {
        let (c, c4) = (N_EMBD, 4 * N_EMBD);
        let mut off = 0;
        let mut take = |n: usize| {
            let o = off;
            off += n;
            o
        };
        Layout {
            wte: take(vocab * c),
            wpe: take(BLOCK_SIZE * c),
            lm_head: take(vocab * c),
            wq: take(c * c),
            wk: take(c * c),
            wv: take(c * c),
            wo: take(c * c),
            fc1: take(c4 * c),
            fc2: take(c * c4),
            total: off,
        }
    }
}

fn init_params(rng: &mut Rng, vocab: usize) -> Vec<f32> {
    let l = Layout::new(vocab);
    (0..l.total).map(|_| rng.gauss(0.0, 0.08) as f32).collect()
}

/// The whole model on the device: parameters, Adam state, and every activation
/// (and its gradient) that the backward pass needs, preallocated at BLOCK_SIZE.
struct Gpu {
    kern: Kernels,
    vocab: usize,
    l: Layout,
    t_len: usize, // sequence length of the last forward() call

    params: CudaSlice<f32>,
    grads: CudaSlice<f32>,
    adam_m: CudaSlice<f32>,
    adam_v: CudaSlice<f32>,

    tokens: CudaSlice<i32>,
    targets: CudaSlice<i32>,
    // forward activations, saved for backward
    x0: CudaSlice<f32>,   // token + position embeddings         (T, C)
    x1: CudaSlice<f32>,   // rmsnorm(x0)                          (T, C)
    xn: CudaSlice<f32>,   // rmsnorm(x1), input to q/k/v          (T, C)
    q: CudaSlice<f32>,    //                                      (T, C)
    k: CudaSlice<f32>,    //                                      (T, C)
    v: CudaSlice<f32>,    //                                      (T, C)
    att: CudaSlice<f32>,  // causal softmax probabilities         (H, T, T)
    atty: CudaSlice<f32>, // attention-weighted values            (T, C)
    proj: CudaSlice<f32>, // atty @ wo^T                          (T, C)
    x2: CudaSlice<f32>,   // x1 + proj (attention residual)       (T, C)
    xn2: CudaSlice<f32>,  // rmsnorm(x2), input to the MLP        (T, C)
    hpre: CudaSlice<f32>, // xn2 @ fc1^T                          (T, 4C)
    h: CudaSlice<f32>,    // relu(hpre)                           (T, 4C)
    mlpo: CudaSlice<f32>, // h @ fc2^T                            (T, C)
    x3: CudaSlice<f32>,   // x2 + mlpo (MLP residual)             (T, C)
    logits: CudaSlice<f32>, //                                    (T, V)
    probs: CudaSlice<f32>,  // softmax(logits)                    (T, V)
    losses: CudaSlice<f32>, //                                    (T,)
    r0: CudaSlice<f32>,   // saved 1/rms per row, for each rmsnorm (T,)
    r1: CudaSlice<f32>,
    r2: CudaSlice<f32>,
    // gradient buffers, one per activation the chain rule flows through
    d_logits: CudaSlice<f32>,
    d_x3: CudaSlice<f32>,
    d_h: CudaSlice<f32>,
    d_xn2: CudaSlice<f32>,
    d_x2: CudaSlice<f32>,
    d_atty: CudaSlice<f32>,
    d_att: CudaSlice<f32>,
    d_scores: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    d_k: CudaSlice<f32>,
    d_v: CudaSlice<f32>,
    d_xn: CudaSlice<f32>,
    d_x1: CudaSlice<f32>,
    d_x0: CudaSlice<f32>,
}

type Res<T> = Result<T, Box<dyn std::error::Error>>;

impl Gpu {
    fn new(vocab: usize, host_params: &[f32]) -> Res<Gpu> {
        let kern = Kernels::new()?;
        let l = Layout::new(vocab);
        assert_eq!(host_params.len(), l.total);
        let s = &kern.stream;
        let (t, c, c4, h_n) = (BLOCK_SIZE, N_EMBD, 4 * N_EMBD, N_HEAD);
        let params = s.clone_htod(host_params)?; // the one host-to-device copy of the weights
        let gpu = Gpu {
            vocab,
            l,
            t_len: 0,
            grads: s.alloc_zeros(l.total)?,
            adam_m: s.alloc_zeros(l.total)?,
            adam_v: s.alloc_zeros(l.total)?,
            tokens: s.alloc_zeros(t)?,
            targets: s.alloc_zeros(t)?,
            x0: s.alloc_zeros(t * c)?,
            x1: s.alloc_zeros(t * c)?,
            xn: s.alloc_zeros(t * c)?,
            q: s.alloc_zeros(t * c)?,
            k: s.alloc_zeros(t * c)?,
            v: s.alloc_zeros(t * c)?,
            att: s.alloc_zeros(h_n * t * t)?,
            atty: s.alloc_zeros(t * c)?,
            proj: s.alloc_zeros(t * c)?,
            x2: s.alloc_zeros(t * c)?,
            xn2: s.alloc_zeros(t * c)?,
            hpre: s.alloc_zeros(t * c4)?,
            h: s.alloc_zeros(t * c4)?,
            mlpo: s.alloc_zeros(t * c)?,
            x3: s.alloc_zeros(t * c)?,
            logits: s.alloc_zeros(t * vocab)?,
            probs: s.alloc_zeros(t * vocab)?,
            losses: s.alloc_zeros(t)?,
            r0: s.alloc_zeros(t)?,
            r1: s.alloc_zeros(t)?,
            r2: s.alloc_zeros(t)?,
            d_logits: s.alloc_zeros(t * vocab)?,
            d_x3: s.alloc_zeros(t * c)?,
            d_h: s.alloc_zeros(t * c4)?,
            d_xn2: s.alloc_zeros(t * c)?,
            d_x2: s.alloc_zeros(t * c)?,
            d_atty: s.alloc_zeros(t * c)?,
            d_att: s.alloc_zeros(h_n * t * t)?,
            d_scores: s.alloc_zeros(h_n * t * t)?,
            d_q: s.alloc_zeros(t * c)?,
            d_k: s.alloc_zeros(t * c)?,
            d_v: s.alloc_zeros(t * c)?,
            d_xn: s.alloc_zeros(t * c)?,
            d_x1: s.alloc_zeros(t * c)?,
            d_x0: s.alloc_zeros(t * c)?,
            params,
            kern,
        };
        Ok(gpu)
    }

    /// Forward `tokens` through the model, leaving logits (T, V) on the device.
    fn forward(&mut self, tokens: &[i32]) -> Res<()> {
        assert!(!tokens.is_empty() && tokens.len() <= BLOCK_SIZE);
        self.t_len = tokens.len();
        let (t, c, c4, h_n, d, v) = (
            tokens.len() as i32,
            N_EMBD as i32,
            4 * N_EMBD as i32,
            N_HEAD as i32,
            HEAD_DIM as i32,
            self.vocab as i32,
        );
        let (wte_off, wpe_off) = (self.l.wte as i32, self.l.wpe as i32);
        let (tc, tc4) = (t * c, t * c4);
        let k = &self.kern;
        let s = &k.stream;
        let zero = 0i32;

        let mut tok_view = self.tokens.slice_mut(0..tokens.len());
        s.memcpy_htod(tokens, &mut tok_view)?;
        drop(tok_view);

        launch!(k, "embed_fwd", t * c;
            &mut self.x0, &self.params, &self.tokens, &wte_off, &wpe_off, &t, &c)?;
        launch!(k, "rmsnorm_fwd", t; &mut self.x1, &mut self.r0, &self.x0, &t, &c)?;

        // the transformer layer (N_LAYER == 1)
        launch!(k, "rmsnorm_fwd", t; &mut self.xn, &mut self.r1, &self.x1, &t, &c)?;
        let wq = self.params.slice(self.l.wq..self.l.wk);
        let wk = self.params.slice(self.l.wk..self.l.wv);
        let wv = self.params.slice(self.l.wv..self.l.wo);
        launch!(k, "matmul_nt", t * c; &mut self.q, &self.xn, &wq, &t, &c, &c, &zero)?;
        launch!(k, "matmul_nt", t * c; &mut self.k, &self.xn, &wk, &t, &c, &c, &zero)?;
        launch!(k, "matmul_nt", t * c; &mut self.v, &self.xn, &wv, &t, &c, &c, &zero)?;
        launch!(k, "attn_softmax_fwd", h_n * t; &mut self.att, &self.q, &self.k, &h_n, &t, &d)?;
        launch!(k, "attn_mix_fwd", t * c; &mut self.atty, &self.att, &self.v, &h_n, &t, &d)?;
        let wo = self.params.slice(self.l.wo..self.l.fc1);
        launch!(k, "matmul_nt", t * c; &mut self.proj, &self.atty, &wo, &t, &c, &c, &zero)?;
        launch!(k, "add_fwd", t * c; &mut self.x2, &self.x1, &self.proj, &tc)?;

        launch!(k, "rmsnorm_fwd", t; &mut self.xn2, &mut self.r2, &self.x2, &t, &c)?;
        let fc1 = self.params.slice(self.l.fc1..self.l.fc2);
        let fc2 = self.params.slice(self.l.fc2..self.l.total);
        launch!(k, "matmul_nt", t * c4; &mut self.hpre, &self.xn2, &fc1, &t, &c4, &c, &zero)?;
        launch!(k, "relu_fwd", t * c4; &mut self.h, &self.hpre, &tc4)?;
        launch!(k, "matmul_nt", t * c; &mut self.mlpo, &self.h, &fc2, &t, &c, &c4, &zero)?;
        launch!(k, "add_fwd", t * c; &mut self.x3, &self.x2, &self.mlpo, &tc)?;

        let lm_head = self.params.slice(self.l.lm_head..self.l.wq);
        launch!(k, "matmul_nt", t * v; &mut self.logits, &self.x3, &lm_head, &t, &v, &c, &zero)?;
        Ok(())
    }

    /// Cross-entropy of the last forward() against `targets`; copies the per-position
    /// losses back to the host (an explicit D2H sync, once per step) and averages.
    fn loss(&mut self, targets: &[i32]) -> Res<f32> {
        assert_eq!(targets.len(), self.t_len);
        let (t, v) = (self.t_len as i32, self.vocab as i32);
        let k = &self.kern;
        let mut tgt_view = self.targets.slice_mut(0..targets.len());
        k.stream.memcpy_htod(targets, &mut tgt_view)?;
        drop(tgt_view);
        launch!(k, "ce_fwd", t;
            &mut self.losses, &mut self.probs, &self.logits, &self.targets, &t, &v)?;
        let losses = k.stream.clone_dtoh(&self.losses)?;
        Ok(losses[..self.t_len].iter().sum::<f32>() / self.t_len as f32)
    }

    /// The chain rule, unrolled by hand from the loss back to every parameter.
    /// Accumulates into `grads` (which adam() zeroes after each use).
    fn backward(&mut self) -> Res<()> {
        let (t, c, c4, h_n, d, v) = (
            self.t_len as i32,
            N_EMBD as i32,
            4 * N_EMBD as i32,
            N_HEAD as i32,
            HEAD_DIM as i32,
            self.vocab as i32,
        );
        let (tc, tc4) = (t * c, t * c4);
        let k = &self.kern;
        let (zero, one) = (0i32, 1i32);

        launch!(k, "ce_bwd", t * v; &mut self.d_logits, &self.probs, &self.targets, &t, &v)?;

        // logits = x3 @ lm_head^T
        let mut g = self.grads.slice_mut(self.l.lm_head..self.l.wq);
        launch!(k, "matmul_tn", v * c; &mut g, &self.d_logits, &self.x3, &v, &c, &t)?;
        drop(g);
        let lm_head = self.params.slice(self.l.lm_head..self.l.wq);
        launch!(k, "matmul_nn", t * c; &mut self.d_x3, &self.d_logits, &lm_head, &t, &c, &v, &zero)?;
        drop(lm_head);

        // x3 = x2 + mlpo;  mlpo = relu(xn2 @ fc1^T) @ fc2^T;  xn2 = rmsnorm(x2)
        let mut g = self.grads.slice_mut(self.l.fc2..self.l.total);
        launch!(k, "matmul_tn", c * c4; &mut g, &self.d_x3, &self.h, &c, &c4, &t)?;
        drop(g);
        let fc2 = self.params.slice(self.l.fc2..self.l.total);
        launch!(k, "matmul_nn", t * c4; &mut self.d_h, &self.d_x3, &fc2, &t, &c4, &c, &zero)?;
        drop(fc2);
        launch!(k, "relu_bwd_inplace", t * c4; &mut self.d_h, &self.hpre, &tc4)?;
        let mut g = self.grads.slice_mut(self.l.fc1..self.l.fc2);
        launch!(k, "matmul_tn", c4 * c; &mut g, &self.d_h, &self.xn2, &c4, &c, &t)?;
        drop(g);
        let fc1 = self.params.slice(self.l.fc1..self.l.fc2);
        launch!(k, "matmul_nn", t * c; &mut self.d_xn2, &self.d_h, &fc1, &t, &c, &c4, &zero)?;
        drop(fc1);
        launch!(k, "rmsnorm_bwd", t; &mut self.d_x2, &self.x2, &self.r2, &self.d_xn2, &t, &c, &zero)?;
        launch!(k, "add_inplace", t * c; &mut self.d_x2, &self.d_x3, &tc)?; // residual

        // x2 = x1 + atty @ wo^T
        let mut g = self.grads.slice_mut(self.l.wo..self.l.fc1);
        launch!(k, "matmul_tn", c * c; &mut g, &self.d_x2, &self.atty, &c, &c, &t)?;
        drop(g);
        let wo = self.params.slice(self.l.wo..self.l.fc1);
        launch!(k, "matmul_nn", t * c; &mut self.d_atty, &self.d_x2, &wo, &t, &c, &c, &zero)?;
        drop(wo);

        // atty = att @ v;  att = causal_softmax(q @ k^T / sqrt(d))
        launch!(k, "attn_mix_bwd_dp", h_n * t * t; &mut self.d_att, &self.d_atty, &self.v, &h_n, &t, &d)?;
        launch!(k, "attn_mix_bwd_dv", t * c; &mut self.d_v, &self.att, &self.d_atty, &h_n, &t, &d)?;
        launch!(k, "attn_softmax_bwd", h_n * t; &mut self.d_scores, &self.att, &self.d_att, &h_n, &t)?;
        launch!(k, "attn_scores_bwd_dq", t * c; &mut self.d_q, &self.d_scores, &self.k, &h_n, &t, &d)?;
        launch!(k, "attn_scores_bwd_dk", t * c; &mut self.d_k, &self.d_scores, &self.q, &h_n, &t, &d)?;

        // q/k/v = xn @ w{q,k,v}^T;  xn = rmsnorm(x1)
        let mut g = self.grads.slice_mut(self.l.wq..self.l.wk);
        launch!(k, "matmul_tn", c * c; &mut g, &self.d_q, &self.xn, &c, &c, &t)?;
        drop(g);
        let mut g = self.grads.slice_mut(self.l.wk..self.l.wv);
        launch!(k, "matmul_tn", c * c; &mut g, &self.d_k, &self.xn, &c, &c, &t)?;
        drop(g);
        let mut g = self.grads.slice_mut(self.l.wv..self.l.wo);
        launch!(k, "matmul_tn", c * c; &mut g, &self.d_v, &self.xn, &c, &c, &t)?;
        drop(g);
        let wq = self.params.slice(self.l.wq..self.l.wk);
        launch!(k, "matmul_nn", t * c; &mut self.d_xn, &self.d_q, &wq, &t, &c, &c, &zero)?;
        drop(wq);
        let wk = self.params.slice(self.l.wk..self.l.wv);
        launch!(k, "matmul_nn", t * c; &mut self.d_xn, &self.d_k, &wk, &t, &c, &c, &one)?;
        drop(wk);
        let wv = self.params.slice(self.l.wv..self.l.wo);
        launch!(k, "matmul_nn", t * c; &mut self.d_xn, &self.d_v, &wv, &t, &c, &c, &one)?;
        drop(wv);
        launch!(k, "rmsnorm_bwd", t; &mut self.d_x1, &self.x1, &self.r1, &self.d_xn, &t, &c, &zero)?;
        launch!(k, "add_inplace", t * c; &mut self.d_x1, &self.d_x2, &tc)?; // residual

        // x1 = rmsnorm(x0);  x0 = wte[token] + wpe[pos]
        launch!(k, "rmsnorm_bwd", t; &mut self.d_x0, &self.x0, &self.r0, &self.d_x1, &t, &c, &zero)?;
        let (wte_off, wpe_off) = (self.l.wte as i32, self.l.wpe as i32);
        launch!(k, "embed_bwd_wte", v * c;
            &mut self.grads, &self.d_x0, &self.tokens, &wte_off, &v, &t, &c)?;
        launch!(k, "embed_bwd_wpe", t * c; &mut self.grads, &self.d_x0, &wpe_off, &t, &c)?;
        Ok(())
    }

    /// One fused Adam update over all parameters; consumes and zeroes the gradients.
    fn adam(&mut self, step: usize, lr_t: f32) -> Res<()> {
        let (beta1, beta2, eps) = (0.85f32, 0.99f32, 1e-8f32);
        let bc1 = (1.0 - 0.85f64.powi(step as i32 + 1)) as f32;
        let bc2 = (1.0 - 0.99f64.powi(step as i32 + 1)) as f32;
        let n = self.l.total as i32;
        let k = &self.kern;
        launch!(k, "adam_step", n;
            &mut self.params, &mut self.grads, &mut self.adam_m, &mut self.adam_v,
            &n, &lr_t, &beta1, &beta2, &bc1, &bc2, &eps)?;
        Ok(())
    }

    /// Copy the last position's logits back to the host (explicit D2H, per sampled char).
    fn last_logits(&self) -> Res<Vec<f32>> {
        let view = self.logits.slice((self.t_len - 1) * self.vocab..self.t_len * self.vocab);
        Ok(self.kern.stream.clone_dtoh(&view)?)
    }
}

fn main() -> Res<()> {
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

    let uchars: Vec<char> =
        docs.iter().flat_map(|d| d.chars()).collect::<BTreeSet<_>>().into_iter().collect();
    let stoi: HashMap<char, usize> = uchars.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {vocab_size}");

    let host_params = init_params(&mut rng, vocab_size);
    println!("num params: {}", host_params.len());
    let mut gpu = Gpu::new(vocab_size, &host_params)?;
    println!("device: {}", gpu.kern.ctx.name()?);

    let (learning_rate, num_steps) = (0.01f32, 1000usize);
    let t_start = std::time::Instant::now();
    for step in 0..num_steps {
        let doc = docs[step % docs.len()];
        let mut tokens = vec![bos as i32];
        tokens.extend(doc.chars().map(|c| stoi[&c] as i32));
        tokens.push(bos as i32);
        let n = BLOCK_SIZE.min(tokens.len() - 1);

        gpu.forward(&tokens[..n])?;
        let loss = gpu.loss(&tokens[1..n + 1])?;
        gpu.backward()?;
        let lr_t = learning_rate * (1.0 - step as f32 / num_steps as f32);
        gpu.adam(step, lr_t)?;

        print!("step {:4} / {:4} | loss {:.4}\r", step + 1, num_steps, loss);
        std::io::stdout().flush()?;
    }
    let elapsed = t_start.elapsed();
    println!(
        "\ntrain time: {:.2}s ({:.1} steps/s)",
        elapsed.as_secs_f64(),
        num_steps as f64 / elapsed.as_secs_f64()
    );

    // Inference: re-forward the whole prefix per character (T <= 16), sample on the host
    let temperature = 0.5f64;
    println!("--- inference (new, hallucinated names) ---");
    for sample_idx in 0..20 {
        let mut ctx = vec![bos as i32];
        let mut sample = String::new();
        for _ in 0..BLOCK_SIZE {
            gpu.forward(&ctx)?;
            let logits = gpu.last_logits()?;
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernels() -> Kernels {
        Kernels::new().expect("CUDA init")
    }

    /// |analytic - numeric| must be small in absolute AND relative terms (f32 math).
    fn assert_close(analytic: f32, numeric: f32, what: &str) {
        let tol = 1e-3 + 0.02 * numeric.abs();
        assert!(
            (analytic - numeric).abs() < tol,
            "{what}: analytic {analytic} vs numeric {numeric}"
        );
    }

    fn rand_vec(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.gauss(0.0, 1.0) as f32).collect()
    }

    #[test]
    fn matmul_kernels_match_host_reference() {
        let k = kernels();
        let mut rng = Rng::new(1);
        let (m, n, kk) = (5usize, 7usize, 11usize);
        let a = rand_vec(&mut rng, m * kk);
        let b = rand_vec(&mut rng, n * kk); // used as (n,kk) for nt, (kk,n) for nn/tn
        let s = &k.stream;
        let da = s.clone_htod(&a).unwrap();
        let db = s.clone_htod(&b).unwrap();
        let mut dy = s.alloc_zeros::<f32>(m * n).unwrap();
        let (mi, ni, ki, zero) = (m as i32, n as i32, kk as i32, 0i32);

        launch!(k, "matmul_nt", m * n; &mut dy, &da, &db, &mi, &ni, &ki, &zero).unwrap();
        let y = s.clone_dtoh(&dy).unwrap();
        for r in 0..m {
            for c in 0..n {
                let want: f32 = (0..kk).map(|p| a[r * kk + p] * b[c * kk + p]).sum();
                assert!((y[r * n + c] - want).abs() < 1e-4, "matmul_nt [{r},{c}]");
            }
        }

        launch!(k, "matmul_nn", m * n; &mut dy, &da, &db, &mi, &ni, &ki, &zero).unwrap();
        let y = s.clone_dtoh(&dy).unwrap();
        for r in 0..m {
            for c in 0..n {
                let want: f32 = (0..kk).map(|p| a[r * kk + p] * b[p * n + c]).sum();
                assert!((y[r * n + c] - want).abs() < 1e-4, "matmul_nn [{r},{c}]");
            }
        }

        // tn: dw (kk_out?) -- here dw is (mi2, ni2) with sum over rows of a2 (K, M) and b2 (K, N)
        let (m2, n2, k2) = (4usize, 6usize, 9usize);
        let a2 = rand_vec(&mut rng, k2 * m2);
        let b2 = rand_vec(&mut rng, k2 * n2);
        let da2 = s.clone_htod(&a2).unwrap();
        let db2 = s.clone_htod(&b2).unwrap();
        let mut dw = s.alloc_zeros::<f32>(m2 * n2).unwrap();
        let (mi2, ni2, ki2) = (m2 as i32, n2 as i32, k2 as i32);
        launch!(k, "matmul_tn", m2 * n2; &mut dw, &da2, &db2, &mi2, &ni2, &ki2).unwrap();
        let w = s.clone_dtoh(&dw).unwrap();
        for r in 0..m2 {
            for c in 0..n2 {
                let want: f32 = (0..k2).map(|p| a2[p * m2 + r] * b2[p * n2 + c]).sum();
                assert!((w[r * n2 + c] - want).abs() < 1e-4, "matmul_tn [{r},{c}]");
            }
        }
    }

    #[test]
    fn rmsnorm_backward_matches_finite_differences() {
        let k = kernels();
        let s = &k.stream;
        let mut rng = Rng::new(2);
        let (t, c) = (3usize, 8usize);
        let x = rand_vec(&mut rng, t * c);
        let w = rand_vec(&mut rng, t * c); // loss = dot(w, rmsnorm(x)), so dy = w

        let fwd = |x: &[f32]| -> f32 {
            let dx = s.clone_htod(x).unwrap();
            let mut dy = s.alloc_zeros::<f32>(t * c).unwrap();
            let mut dr = s.alloc_zeros::<f32>(t).unwrap();
            let (ti, ci) = (t as i32, c as i32);
            launch!(k, "rmsnorm_fwd", t; &mut dy, &mut dr, &dx, &ti, &ci).unwrap();
            let y = s.clone_dtoh(&dy).unwrap();
            y.iter().zip(&w).map(|(a, b)| a * b).sum()
        };

        // analytic via the kernel
        let dx_dev = s.clone_htod(&x).unwrap();
        let mut y_dev = s.alloc_zeros::<f32>(t * c).unwrap();
        let mut r_dev = s.alloc_zeros::<f32>(t).unwrap();
        let (ti, ci, zero) = (t as i32, c as i32, 0i32);
        launch!(k, "rmsnorm_fwd", t; &mut y_dev, &mut r_dev, &dx_dev, &ti, &ci).unwrap();
        let dy_dev = s.clone_htod(&w).unwrap();
        let mut dgrad = s.alloc_zeros::<f32>(t * c).unwrap();
        launch!(k, "rmsnorm_bwd", t; &mut dgrad, &dx_dev, &r_dev, &dy_dev, &ti, &ci, &zero)
            .unwrap();
        let analytic = s.clone_dtoh(&dgrad).unwrap();

        let h = 1e-2f32;
        for i in 0..t * c {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += h;
            xm[i] -= h;
            let numeric = (fwd(&xp) - fwd(&xm)) / (2.0 * h);
            assert_close(analytic[i], numeric, &format!("rmsnorm dx[{i}]"));
        }
    }

    #[test]
    fn attention_backward_matches_finite_differences() {
        let k = kernels();
        let s = &k.stream;
        let mut rng = Rng::new(3);
        let (h_n, t, d) = (2usize, 5usize, 3usize);
        let c = h_n * d;
        let q = rand_vec(&mut rng, t * c);
        let kv = rand_vec(&mut rng, t * c);
        let v = rand_vec(&mut rng, t * c);
        let w = rand_vec(&mut rng, t * c); // loss = dot(w, attention(q,k,v))

        let fwd = |q: &[f32], kk: &[f32], v: &[f32]| -> f32 {
            let (dq, dk, dv) =
                (s.clone_htod(q).unwrap(), s.clone_htod(kk).unwrap(), s.clone_htod(v).unwrap());
            let mut att = s.alloc_zeros::<f32>(h_n * t * t).unwrap();
            let mut y = s.alloc_zeros::<f32>(t * c).unwrap();
            let (hi, ti, di) = (h_n as i32, t as i32, d as i32);
            launch!(k, "attn_softmax_fwd", h_n * t; &mut att, &dq, &dk, &hi, &ti, &di).unwrap();
            launch!(k, "attn_mix_fwd", t * c; &mut y, &att, &dv, &hi, &ti, &di).unwrap();
            let y = s.clone_dtoh(&y).unwrap();
            y.iter().zip(&w).map(|(a, b)| a * b).sum()
        };

        // analytic: chain the four backward kernels
        let (hi, ti, di) = (h_n as i32, t as i32, d as i32);
        let (dq_in, dk_in, dv_in) =
            (s.clone_htod(&q).unwrap(), s.clone_htod(&kv).unwrap(), s.clone_htod(&v).unwrap());
        let mut att = s.alloc_zeros::<f32>(h_n * t * t).unwrap();
        launch!(k, "attn_softmax_fwd", h_n * t; &mut att, &dq_in, &dk_in, &hi, &ti, &di).unwrap();
        let dy = s.clone_htod(&w).unwrap();
        let mut d_att = s.alloc_zeros::<f32>(h_n * t * t).unwrap();
        let mut d_scores = s.alloc_zeros::<f32>(h_n * t * t).unwrap();
        let mut gq = s.alloc_zeros::<f32>(t * c).unwrap();
        let mut gk = s.alloc_zeros::<f32>(t * c).unwrap();
        let mut gv = s.alloc_zeros::<f32>(t * c).unwrap();
        launch!(k, "attn_mix_bwd_dp", h_n * t * t; &mut d_att, &dy, &dv_in, &hi, &ti, &di).unwrap();
        launch!(k, "attn_mix_bwd_dv", t * c; &mut gv, &att, &dy, &hi, &ti, &di).unwrap();
        launch!(k, "attn_softmax_bwd", h_n * t; &mut d_scores, &att, &d_att, &hi, &ti).unwrap();
        launch!(k, "attn_scores_bwd_dq", t * c; &mut gq, &d_scores, &dk_in, &hi, &ti, &di).unwrap();
        launch!(k, "attn_scores_bwd_dk", t * c; &mut gk, &d_scores, &dq_in, &hi, &ti, &di).unwrap();
        let (gq, gk, gv) = (
            s.clone_dtoh(&gq).unwrap(),
            s.clone_dtoh(&gk).unwrap(),
            s.clone_dtoh(&gv).unwrap(),
        );

        let h = 1e-2f32;
        for i in 0..t * c {
            let perturb = |base: &[f32], delta: f32| {
                let mut p = base.to_vec();
                p[i] += delta;
                p
            };
            let nq = (fwd(&perturb(&q, h), &kv, &v) - fwd(&perturb(&q, -h), &kv, &v)) / (2.0 * h);
            let nk = (fwd(&q, &perturb(&kv, h), &v) - fwd(&q, &perturb(&kv, -h), &v)) / (2.0 * h);
            let nv = (fwd(&q, &kv, &perturb(&v, h)) - fwd(&q, &kv, &perturb(&v, -h))) / (2.0 * h);
            assert_close(gq[i], nq, &format!("attn dq[{i}]"));
            assert_close(gk[i], nk, &format!("attn dk[{i}]"));
            assert_close(gv[i], nv, &format!("attn dv[{i}]"));
        }
    }

    #[test]
    fn cross_entropy_backward_matches_finite_differences() {
        let k = kernels();
        let s = &k.stream;
        let mut rng = Rng::new(4);
        let (t, v) = (3usize, 7usize);
        let logits = rand_vec(&mut rng, t * v);
        let targets: Vec<i32> = vec![2, 0, 5];
        let dtgt = s.clone_htod(&targets).unwrap();

        let fwd = |logits: &[f32]| -> f32 {
            let dl = s.clone_htod(logits).unwrap();
            let mut dloss = s.alloc_zeros::<f32>(t).unwrap();
            let mut dprobs = s.alloc_zeros::<f32>(t * v).unwrap();
            let (ti, vi) = (t as i32, v as i32);
            launch!(k, "ce_fwd", t; &mut dloss, &mut dprobs, &dl, &dtgt, &ti, &vi).unwrap();
            let l = s.clone_dtoh(&dloss).unwrap();
            l.iter().sum::<f32>() / t as f32
        };

        let dl = s.clone_htod(&logits).unwrap();
        let mut dloss = s.alloc_zeros::<f32>(t).unwrap();
        let mut dprobs = s.alloc_zeros::<f32>(t * v).unwrap();
        let mut dgrad = s.alloc_zeros::<f32>(t * v).unwrap();
        let (ti, vi) = (t as i32, v as i32);
        launch!(k, "ce_fwd", t; &mut dloss, &mut dprobs, &dl, &dtgt, &ti, &vi).unwrap();
        launch!(k, "ce_bwd", t * v; &mut dgrad, &dprobs, &dtgt, &ti, &vi).unwrap();
        let analytic = s.clone_dtoh(&dgrad).unwrap();

        let h = 1e-2f32;
        for i in 0..t * v {
            let mut lp = logits.clone();
            let mut lm = logits.clone();
            lp[i] += h;
            lm[i] -= h;
            let numeric = (fwd(&lp) - fwd(&lm)) / (2.0 * h);
            assert_close(analytic[i], numeric, &format!("ce dlogits[{i}]"));
        }
    }

    #[test]
    fn end_to_end_gradients_match_finite_differences() {
        let mut rng = Rng::new(42);
        let vocab = 27;
        let host_params = init_params(&mut rng, vocab);
        let mut gpu = Gpu::new(vocab, &host_params).unwrap();
        let l = gpu.l;

        let tokens: Vec<i32> = vec![26, 4, 12, 12, 0]; // BOS e m m a
        let targets: Vec<i32> = vec![4, 12, 12, 0, 26];

        let loss0 = {
            gpu.forward(&tokens).unwrap();
            gpu.loss(&targets).unwrap()
        };
        // untrained loss should sit near the uniform floor ln(27) ~ 3.296
        let uniform = (vocab as f32).ln();
        assert!((loss0 - uniform).abs() < 0.5, "untrained loss {loss0} vs ln(V) {uniform}");

        gpu.backward().unwrap();
        let grads = gpu.kern.stream.clone_dtoh(&gpu.grads).unwrap();

        // probe a few parameters from every matrix in the model
        let probes: Vec<usize> = [
            l.wte + 26 * N_EMBD + 3, // a token actually used (BOS row)
            l.wte + 4 * N_EMBD + 7,  // 'e' row
            l.wpe + 5,
            l.wpe + 2 * N_EMBD + 1,
            l.lm_head + 4 * N_EMBD + 2,
            l.lm_head + 26 * N_EMBD + 9,
            l.wq, l.wq + 100, l.wk + 37, l.wv + 200, l.wo + 141,
            l.fc1 + 11, l.fc1 + 500, l.fc2 + 300, l.fc2 + 777,
        ]
        .to_vec();

        let h = 1e-2f32;
        for &i in &probes {
            let mut eval = |delta: f32| -> f32 {
                let mut p = host_params.clone();
                p[i] += delta;
                let mut view = gpu.params.slice_mut(..);
                gpu.kern.stream.memcpy_htod(&p, &mut view).unwrap();
                drop(view);
                gpu.forward(&tokens).unwrap();
                gpu.loss(&targets).unwrap()
            };
            let numeric = (eval(h) - eval(-h)) / (2.0 * h);
            assert_close(grads[i], numeric, &format!("end-to-end dparam[{i}]"));
        }
    }

    #[test]
    fn training_is_bit_deterministic() {
        let run = || -> (f32, Vec<f32>) {
            let mut rng = Rng::new(42);
            let vocab = 27;
            let host_params = init_params(&mut rng, vocab);
            let mut gpu = Gpu::new(vocab, &host_params).unwrap();
            let tokens: Vec<i32> = vec![26, 4, 12, 12, 0];
            let targets: Vec<i32> = vec![4, 12, 12, 0, 26];
            let mut last = 0.0;
            for step in 0..20 {
                gpu.forward(&tokens).unwrap();
                last = gpu.loss(&targets).unwrap();
                gpu.backward().unwrap();
                gpu.adam(step, 0.01).unwrap();
            }
            (last, gpu.kern.stream.clone_dtoh(&gpu.params).unwrap())
        };
        let (l1, p1) = run();
        let (l2, p2) = run();
        assert_eq!(l1.to_bits(), l2.to_bits(), "loss must be bit-identical");
        assert!(
            p1.iter().zip(&p2).all(|(a, b)| a.to_bits() == b.to_bits()),
            "params must be bit-identical"
        );
        assert!(l1 < 3.0, "20 steps on one doc should reduce the loss below start ({l1})");
    }
}
