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
//!
//! Two modes, one code path:
//! - default (parity): the 4,192-parameter microgpt config, one document per step,
//!   reproducing the CPU crate's training run to the printed decimal;
//! - `--scale`: ~800K parameters (4 layers, 128-dim, 8 heads), 32 documents per
//!   batched step, padded to the block size with the loss masked on the padding.
//!   Same kernels -- parity is just batch=1, layers=1. This is where the GPU stops
//!   measuring launch overhead and starts doing arithmetic; it is also a scale the
//!   CPU crate's scalar tape could not follow (one tape node per scalar op would
//!   mean billions of nodes -- over 100 GB -- per step).

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

/// Model + training shape. Everything else derives from these six numbers.
#[derive(Clone, Copy)]
struct Cfg {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    block_size: usize,
    batch: usize, // documents per training step
    num_steps: usize,
    init_std: f64, // 0.08 suits the 16-dim model; a 4-layer 256-dim one needs cooler weights
    learning_rate: f32, // likewise: Adam at 0.01 is stable for 4K params, explosive for 3M
}

impl Cfg {
    fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Activation rows per training step: `batch` documents of `block_size` positions.
    fn max_rows(&self) -> usize {
        self.batch * self.block_size
    }
}

/// The original microgpt: identical numbers to microgpt-rs / microgpt-mlx.
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

/// ~800K parameters, 32 documents per step -- compute-bound instead of launch-bound,
/// and sized so the CPU control run (microgpt-scale) stays a few-minute demo.
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

// The complete GPU-side algorithm: every forward kernel below has its hand-derived
// backward next to it. Indexing is row-major throughout; activations are (rows, C)
// where rows = batch * t_len, and attention tensors are (batch, heads, t_len, t_len).
// `i` is always this thread's output element -- one thread per output, no atomics,
// so every run is bit-identical. Padded positions need no special casing outside the
// loss: they sit at the tail of each document, so causality already keeps every valid
// position from attending to them, and the masked loss feeds them exactly-zero
// gradients that stay zero through every backward kernel.
const KERNELS: &str = r#"
#define MAX_T 16  // block_size upper bound, for stack rows in the softmax kernels

// ---- embedding: x[b,t] = wte[token[b,t]] + wpe[t] ----
extern "C" __global__ void embed_fwd(float *x, const float *params, const int *tokens,
                                     int wte_off, int wpe_off, int rows, int t_len, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * c) return;
    int t = (i / c) % t_len, j = i % c;
    x[i] = params[wte_off + tokens[i / c] * c + j] + params[wpe_off + t * c + j];
}

// dwte[v][j] = sum over positions where tokens[r] == v of dx[r][j].
// A gather per vocab row instead of a scatter with atomicAdd: deterministic.
extern "C" __global__ void embed_bwd_wte(float *grads, const float *dx, const int *tokens,
                                         int wte_off, int vocab, int rows, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab * c) return;
    int v = i / c, j = i % c;
    float acc = 0.0f;
    for (int r = 0; r < rows; r++)
        if (tokens[r] == v) acc += dx[r * c + j];
    grads[wte_off + i] += acc;
}

// dwpe[t][j] = sum over batch of dx[b,t][j]
extern "C" __global__ void embed_bwd_wpe(float *grads, const float *dx,
                                         int wpe_off, int rows, int t_len, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t_len * c) return;
    int t = i / c, j = i % c;
    float acc = 0.0f;
    for (int r = t; r < rows; r += t_len) acc += dx[r * c + j];
    grads[wpe_off + i] += acc;
}

// ---- rmsnorm: y = x * r, r = rsqrt(mean(x^2) + 1e-5), one thread per row ----
extern "C" __global__ void rmsnorm_fwd(float *y, float *rinv, const float *x,
                                       int rows, int c) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows) return;
    float ms = 0.0f;
    for (int j = 0; j < c; j++) ms += x[t * c + j] * x[t * c + j];
    float r = rsqrtf(ms / c + 1e-5f);
    rinv[t] = r;
    for (int j = 0; j < c; j++) y[t * c + j] = x[t * c + j] * r;
}

// dy_i/dx_j = r * delta_ij - x_i * x_j * r^3 / c, so
// dx_j = r * dy_j - (r^3 / c) * x_j * dot(dy, x)
extern "C" __global__ void rmsnorm_bwd(float *dx, const float *x, const float *rinv,
                                       const float *dy, int rows, int c, int accum) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows) return;
    float dot = 0.0f;
    for (int j = 0; j < c; j++) dot += dy[t * c + j] * x[t * c + j];
    float r = rinv[t];
    float k = r * r * r * dot / c;
    for (int j = 0; j < c; j++) {
        float g = r * dy[t * c + j] - k * x[t * c + j];
        dx[t * c + j] = accum ? dx[t * c + j] + g : g;
    }
}

// ---- matmul, tiled: 16x16 output tiles staged through shared memory ----
// The first version of these kernels was one naive thread per output element; that
// left scale mode at 17 steps/s because every warp scattered its weight reads across
// 32 cache lines. Tiling loads each 16x16 patch of x and w once, coalesced, into
// shared memory. Each output element still sums its k-products in ascending order,
// so the results are bit-identical to the naive kernels (padding tiles contribute
// exact +0.0f terms).
#define TILE 16

// forward of linear(x, w): y = x @ w^T,  x (m,k), w (n,k), y (m,n)
extern "C" __global__ void matmul_nt(float *y, const float *x, const float *w,
                                     int m, int n, int k, int accum) {
    __shared__ float xs[TILE][TILE], ws[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y; // m index
    int col = blockIdx.x * TILE + threadIdx.x; // n index
    float acc = 0.0f;
    for (int p0 = 0; p0 < k; p0 += TILE) {
        int p = p0 + threadIdx.x;
        int wrow = blockIdx.x * TILE + threadIdx.y;
        xs[threadIdx.y][threadIdx.x] = (row < m && p < k) ? x[row * k + p] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (wrow < n && p < k) ? w[wrow * k + p] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += xs[threadIdx.y][i] * ws[threadIdx.x][i];
        __syncthreads();
    }
    if (row < m && col < n) y[row * n + col] = accum ? y[row * n + col] + acc : acc;
}

// backward wrt the input: dx = dy @ w,  dy (m,k), w (k,n), dx (m,n)
extern "C" __global__ void matmul_nn(float *y, const float *x, const float *w,
                                     int m, int n, int k, int accum) {
    __shared__ float xs[TILE][TILE], ws[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int p0 = 0; p0 < k; p0 += TILE) {
        int px = p0 + threadIdx.x, py = p0 + threadIdx.y;
        xs[threadIdx.y][threadIdx.x] = (row < m && px < k) ? x[row * k + px] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (py < k && col < n) ? w[py * n + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += xs[threadIdx.y][i] * ws[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) y[row * n + col] = accum ? y[row * n + col] + acc : acc;
}

// backward wrt the weights: dw += dy^T @ x,  dy (k,m), x (k,n), dw (m,n)
extern "C" __global__ void matmul_tn(float *dw, const float *dy, const float *x,
                                     int m, int n, int k) {
    __shared__ float dys[TILE][TILE], xs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int p0 = 0; p0 < k; p0 += TILE) {
        int py = p0 + threadIdx.y;
        int drow = blockIdx.y * TILE + threadIdx.x;
        dys[threadIdx.y][threadIdx.x] = (py < k && drow < m) ? dy[py * m + drow] : 0.0f;
        xs[threadIdx.y][threadIdx.x] = (py < k && col < n) ? x[py * n + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += dys[i][threadIdx.y] * xs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) dw[row * n + col] += acc;
}

// ---- causal attention softmax: probs[b,h,t,u] = softmax_u(q_bt . k_bu / sqrt(d)), u <= t ----
extern "C" __global__ void attn_softmax_fwd(float *probs, const float *q, const float *k,
                                            int b_n, int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b_n * h_n * t_len) return;
    int b = i / (h_n * t_len), h = (i / t_len) % h_n, t = i % t_len, c = h_n * d;
    float scale = rsqrtf((float)d);
    float s[MAX_T];
    float maxv = -1e30f;
    for (int u = 0; u <= t; u++) {
        float dot = 0.0f;
        for (int j = 0; j < d; j++)
            dot += q[(b * t_len + t) * c + h * d + j] * k[(b * t_len + u) * c + h * d + j];
        s[u] = dot * scale;
        maxv = fmaxf(maxv, s[u]);
    }
    float tot = 0.0f;
    for (int u = 0; u <= t; u++) { s[u] = expf(s[u] - maxv); tot += s[u]; }
    float *row = probs + ((b * h_n + h) * t_len + t) * t_len;
    for (int u = 0; u < t_len; u++) row[u] = (u <= t) ? s[u] / tot : 0.0f;
}

// softmax backward per causal row: ds_u = p_u * (dp_u - dot(dp, p))
extern "C" __global__ void attn_softmax_bwd(float *ds, const float *probs, const float *dp,
                                            int b_n, int h_n, int t_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b_n * h_n * t_len) return;
    int row = i * t_len, t = i % t_len;
    float dot = 0.0f;
    for (int u = 0; u <= t; u++) dot += dp[row + u] * probs[row + u];
    for (int u = 0; u < t_len; u++)
        ds[row + u] = (u <= t) ? probs[row + u] * (dp[row + u] - dot) : 0.0f;
}

// scores backward: dq_t += ds[t,u] * k_u / sqrt(d);  dk_u += ds[t,u] * q_t / sqrt(d)
extern "C" __global__ void attn_scores_bwd_dq(float *dq, const float *ds, const float *k,
                                              int b_n, int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = h_n * d;
    if (i >= b_n * t_len * c) return;
    int b = i / (t_len * c), t = (i / c) % t_len, h = (i % c) / d;
    float acc = 0.0f;
    for (int u = 0; u <= t; u++)
        acc += ds[((b * h_n + h) * t_len + t) * t_len + u] * k[(b * t_len + u) * c + i % c];
    dq[i] = acc * rsqrtf((float)d);
}

extern "C" __global__ void attn_scores_bwd_dk(float *dk, const float *ds, const float *q,
                                              int b_n, int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = h_n * d;
    if (i >= b_n * t_len * c) return;
    int b = i / (t_len * c), u = (i / c) % t_len, h = (i % c) / d;
    float acc = 0.0f;
    for (int t = u; t < t_len; t++)
        acc += ds[((b * h_n + h) * t_len + t) * t_len + u] * q[(b * t_len + t) * c + i % c];
    dk[i] = acc * rsqrtf((float)d);
}

// ---- attention mix: y[b,t] = sum_u probs[b,h,t,u] * v[b,u], per head ----
extern "C" __global__ void attn_mix_fwd(float *y, const float *probs, const float *v,
                                        int b_n, int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = h_n * d;
    if (i >= b_n * t_len * c) return;
    int b = i / (t_len * c), t = (i / c) % t_len, h = (i % c) / d;
    float acc = 0.0f;
    for (int u = 0; u <= t; u++)
        acc += probs[((b * h_n + h) * t_len + t) * t_len + u] * v[(b * t_len + u) * c + i % c];
    y[i] = acc;
}

extern "C" __global__ void attn_mix_bwd_dp(float *dp, const float *dy, const float *v,
                                           int b_n, int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b_n * h_n * t_len * t_len) return;
    int b = i / (h_n * t_len * t_len), h = (i / (t_len * t_len)) % h_n;
    int t = (i / t_len) % t_len, u = i % t_len, c = h_n * d;
    float acc = 0.0f;
    if (u <= t)
        for (int j = 0; j < d; j++)
            acc += dy[(b * t_len + t) * c + h * d + j] * v[(b * t_len + u) * c + h * d + j];
    dp[i] = acc;
}

extern "C" __global__ void attn_mix_bwd_dv(float *dv, const float *probs, const float *dy,
                                           int b_n, int h_n, int t_len, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = h_n * d;
    if (i >= b_n * t_len * c) return;
    int b = i / (t_len * c), u = (i / c) % t_len, h = (i % c) / d;
    float acc = 0.0f;
    for (int t = u; t < t_len; t++)
        acc += probs[((b * h_n + h) * t_len + t) * t_len + u] * dy[(b * t_len + t) * c + i % c];
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
// A target of -1 marks padding: its loss is zero and (in ce_bwd) so is its gradient.
extern "C" __global__ void ce_fwd(float *losses, float *probs, const float *logits,
                                  const int *targets, int rows, int vocab) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows) return;
    if (targets[t] < 0) { losses[t] = 0.0f; return; }
    const float *row = logits + t * vocab;
    float maxv = -1e30f;
    for (int j = 0; j < vocab; j++) maxv = fmaxf(maxv, row[j]);
    float tot = 0.0f;
    for (int j = 0; j < vocab; j++) { probs[t * vocab + j] = expf(row[j] - maxv); tot += probs[t * vocab + j]; }
    for (int j = 0; j < vocab; j++) probs[t * vocab + j] /= tot;
    losses[t] = -logf(probs[t * vocab + targets[t]]);
}

// d(mean loss)/dlogits = (probs - onehot(target)) / valid, zero on padding
extern "C" __global__ void ce_bwd(float *dlogits, const float *probs, const int *targets,
                                  int rows, int vocab, int valid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * vocab) return;
    int t = i / vocab, j = i % vocab;
    dlogits[i] =
        targets[t] < 0 ? 0.0f : (probs[i] - (j == targets[t] ? 1.0f : 0.0f)) / valid;
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

/// `launch2d!(kernels, "matmul_*", m, n; ...)` -- a 16x16-threaded block per 16x16
/// tile of the (m, n) output.
macro_rules! launch2d {
    ($k:expr, $name:literal, $m:expr, $n:expr; $($arg:expr),* $(,)?) => {{
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (($n as u32).div_ceil(16), ($m as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let mut b = $k.stream.launch_builder(&$k.funcs[$name]);
        $( b.arg($arg); )*
        unsafe { b.launch(cfg) }
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

#[derive(Clone, Copy)]
struct LayerOff {
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    fc1: usize,
    fc2: usize,
}

/// Offsets of each weight matrix inside the single flat parameter buffer.
/// Order matches microgpt-rs / microgpt-mlx exactly, so the same RNG stream
/// produces the same initialization in all three crates.
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
    xn: CudaSlice<f32>,    // rmsnorm(x_in), input to q/k/v          (R, C)
    q: CudaSlice<f32>,     //                                        (R, C)
    k: CudaSlice<f32>,     //                                        (R, C)
    v: CudaSlice<f32>,     //                                        (R, C)
    att: CudaSlice<f32>,   // causal softmax probabilities           (B, H, T, T)
    atty: CudaSlice<f32>,  // attention-weighted values              (R, C)
    x_mid: CudaSlice<f32>, // x_in + atty @ wo^T (attention residual) (R, C)
    xn2: CudaSlice<f32>,   // rmsnorm(x_mid), input to the MLP       (R, C)
    hpre: CudaSlice<f32>,  // xn2 @ fc1^T                            (R, 4C)
    h: CudaSlice<f32>,     // relu(hpre)                             (R, 4C)
    x_out: CudaSlice<f32>, // x_mid + h @ fc2^T (MLP residual)       (R, C)
    r1: CudaSlice<f32>,    // saved 1/rms per row, per rmsnorm       (R,)
    r2: CudaSlice<f32>,
}

/// The whole model on the device: parameters, Adam state, and every activation
/// (and its gradient) that the backward pass needs, preallocated at max_rows.
struct Gpu {
    kern: Kernels,
    cfg: Cfg,
    vocab: usize,
    l: Layout,
    t_len: usize, // positions per document of the last forward() call
    rows: usize,  // batch * t_len of the last forward() call
    valid: usize, // non-padding targets of the last loss() call

    params: CudaSlice<f32>,
    grads: CudaSlice<f32>,
    adam_m: CudaSlice<f32>,
    adam_v: CudaSlice<f32>,

    tokens: CudaSlice<i32>,
    targets: CudaSlice<i32>,
    x0: CudaSlice<f32>, // token + position embeddings (R, C)
    x1: CudaSlice<f32>, // rmsnorm(x0), input to layer 0 (R, C)
    r0: CudaSlice<f32>,
    layers: Vec<LayerActs>,
    proj: CudaSlice<f32>, // per-layer scratch, not needed in backward
    mlpo: CudaSlice<f32>,
    logits: CudaSlice<f32>, //                  (R, V)
    probs: CudaSlice<f32>,  // softmax(logits) (R, V)
    losses: CudaSlice<f32>, //                 (R,)
    // gradient buffers, shared across layers (backward is sequential)
    d_logits: CudaSlice<f32>,
    d_x: CudaSlice<f32>,   // upstream dL/d(x_out) of the layer being backpropped
    d_xin: CudaSlice<f32>, // dL/d(x_in) of that layer, swapped into d_x for the next
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
    d_x0: CudaSlice<f32>,
}

type Res<T> = Result<T, Box<dyn std::error::Error>>;

impl Gpu {
    fn new(cfg: Cfg, vocab: usize, host_params: &[f32]) -> Res<Gpu> {
        assert!(cfg.block_size <= 16, "MAX_T in the kernel source is 16");
        let kern = Kernels::new()?;
        let l = Layout::new(&cfg, vocab);
        assert_eq!(host_params.len(), l.total);
        let s = &kern.stream;
        let (r, c, c4) = (cfg.max_rows(), cfg.n_embd, 4 * cfg.n_embd);
        let att_len = cfg.batch * cfg.n_head * cfg.block_size * cfg.block_size;
        let params = s.clone_htod(host_params)?; // the one host-to-device copy of the weights
        let layers = (0..cfg.n_layer)
            .map(|_| -> Res<LayerActs> {
                Ok(LayerActs {
                    xn: s.alloc_zeros(r * c)?,
                    q: s.alloc_zeros(r * c)?,
                    k: s.alloc_zeros(r * c)?,
                    v: s.alloc_zeros(r * c)?,
                    att: s.alloc_zeros(att_len)?,
                    atty: s.alloc_zeros(r * c)?,
                    x_mid: s.alloc_zeros(r * c)?,
                    xn2: s.alloc_zeros(r * c)?,
                    hpre: s.alloc_zeros(r * c4)?,
                    h: s.alloc_zeros(r * c4)?,
                    x_out: s.alloc_zeros(r * c)?,
                    r1: s.alloc_zeros(r)?,
                    r2: s.alloc_zeros(r)?,
                })
            })
            .collect::<Res<Vec<_>>>()?;
        let gpu = Gpu {
            cfg,
            vocab,
            l: l.clone(),
            t_len: 0,
            rows: 0,
            valid: 0,
            grads: s.alloc_zeros(l.total)?,
            adam_m: s.alloc_zeros(l.total)?,
            adam_v: s.alloc_zeros(l.total)?,
            tokens: s.alloc_zeros(r)?,
            targets: s.alloc_zeros(r)?,
            x0: s.alloc_zeros(r * c)?,
            x1: s.alloc_zeros(r * c)?,
            r0: s.alloc_zeros(r)?,
            layers,
            proj: s.alloc_zeros(r * c)?,
            mlpo: s.alloc_zeros(r * c)?,
            logits: s.alloc_zeros(r * vocab)?,
            probs: s.alloc_zeros(r * vocab)?,
            losses: s.alloc_zeros(r)?,
            d_logits: s.alloc_zeros(r * vocab)?,
            d_x: s.alloc_zeros(r * c)?,
            d_xin: s.alloc_zeros(r * c)?,
            d_h: s.alloc_zeros(r * c4)?,
            d_xn2: s.alloc_zeros(r * c)?,
            d_x2: s.alloc_zeros(r * c)?,
            d_atty: s.alloc_zeros(r * c)?,
            d_att: s.alloc_zeros(att_len)?,
            d_scores: s.alloc_zeros(att_len)?,
            d_q: s.alloc_zeros(r * c)?,
            d_k: s.alloc_zeros(r * c)?,
            d_v: s.alloc_zeros(r * c)?,
            d_xn: s.alloc_zeros(r * c)?,
            d_x0: s.alloc_zeros(r * c)?,
            params,
            kern,
        };
        Ok(gpu)
    }

    /// Forward `tokens` -- `batch` documents of `t_len` positions each, concatenated,
    /// short documents padded (padding is neutralized by causality + the loss mask) --
    /// leaving logits (rows, V) on the device.
    fn forward(&mut self, tokens: &[i32], t_len: usize) -> Res<()> {
        assert!(t_len > 0 && t_len <= self.cfg.block_size && tokens.len() % t_len == 0);
        assert!(tokens.len() <= self.cfg.max_rows());
        self.t_len = t_len;
        self.rows = tokens.len();
        let (rows, t, c, c4, h_n, d, v) = (
            self.rows as i32,
            t_len as i32,
            self.cfg.n_embd as i32,
            4 * self.cfg.n_embd as i32,
            self.cfg.n_head as i32,
            self.cfg.head_dim() as i32,
            self.vocab as i32,
        );
        let b_n = rows / t;
        let (wte_off, wpe_off) = (self.l.wte as i32, self.l.wpe as i32);
        let (rc, rc4) = (rows * c, rows * c4);
        let k = &self.kern;
        let s = &k.stream;
        let zero = 0i32;

        let mut tok_view = self.tokens.slice_mut(0..tokens.len());
        s.memcpy_htod(tokens, &mut tok_view)?;
        drop(tok_view);

        launch!(k, "embed_fwd", rc;
            &mut self.x0, &self.params, &self.tokens, &wte_off, &wpe_off, &rows, &t, &c)?;
        launch!(k, "rmsnorm_fwd", rows; &mut self.x1, &mut self.r0, &self.x0, &rows, &c)?;

        for li in 0..self.cfg.n_layer {
            let lo = self.l.layers[li];
            let (done, rest) = self.layers.split_at_mut(li);
            let xin = if li == 0 { &self.x1 } else { &done[li - 1].x_out };
            let la = &mut rest[0];

            launch!(k, "rmsnorm_fwd", rows; &mut la.xn, &mut la.r1, xin, &rows, &c)?;
            let wq = self.params.slice(lo.wq..lo.wk);
            let wk = self.params.slice(lo.wk..lo.wv);
            let wv = self.params.slice(lo.wv..lo.wo);
            launch2d!(k, "matmul_nt", rows, c; &mut la.q, &la.xn, &wq, &rows, &c, &c, &zero)?;
            launch2d!(k, "matmul_nt", rows, c; &mut la.k, &la.xn, &wk, &rows, &c, &c, &zero)?;
            launch2d!(k, "matmul_nt", rows, c; &mut la.v, &la.xn, &wv, &rows, &c, &c, &zero)?;
            launch!(k, "attn_softmax_fwd", b_n * h_n * t;
                &mut la.att, &la.q, &la.k, &b_n, &h_n, &t, &d)?;
            launch!(k, "attn_mix_fwd", rc; &mut la.atty, &la.att, &la.v, &b_n, &h_n, &t, &d)?;
            let wo = self.params.slice(lo.wo..lo.fc1);
            launch2d!(k, "matmul_nt", rows, c; &mut self.proj, &la.atty, &wo, &rows, &c, &c, &zero)?;
            launch!(k, "add_fwd", rc; &mut la.x_mid, xin, &self.proj, &rc)?;

            launch!(k, "rmsnorm_fwd", rows; &mut la.xn2, &mut la.r2, &la.x_mid, &rows, &c)?;
            let fc1 = self.params.slice(lo.fc1..lo.fc2);
            let fc2 = self.params.slice(lo.fc2..lo.fc2 + (c * c4) as usize);
            launch2d!(k, "matmul_nt", rows, c4; &mut la.hpre, &la.xn2, &fc1, &rows, &c4, &c, &zero)?;
            launch!(k, "relu_fwd", rc4; &mut la.h, &la.hpre, &rc4)?;
            launch2d!(k, "matmul_nt", rows, c; &mut self.mlpo, &la.h, &fc2, &rows, &c, &c4, &zero)?;
            launch!(k, "add_fwd", rc; &mut la.x_out, &la.x_mid, &self.mlpo, &rc)?;
        }

        let x_final = &self.layers[self.cfg.n_layer - 1].x_out;
        let lm_head = self.params.slice(self.l.lm_head..self.l.lm_head + (v * c) as usize);
        launch2d!(k, "matmul_nt", rows, v;
            &mut self.logits, x_final, &lm_head, &rows, &v, &c, &zero)?;
        Ok(())
    }

    /// Cross-entropy of the last forward() against `targets` (-1 marks padding);
    /// copies the per-position losses back to the host (an explicit D2H sync, once
    /// per step) and averages over the non-padding positions.
    fn loss(&mut self, targets: &[i32]) -> Res<f32> {
        assert_eq!(targets.len(), self.rows);
        self.valid = targets.iter().filter(|&&t| t >= 0).count();
        let (rows, v) = (self.rows as i32, self.vocab as i32);
        let k = &self.kern;
        let mut tgt_view = self.targets.slice_mut(0..targets.len());
        k.stream.memcpy_htod(targets, &mut tgt_view)?;
        drop(tgt_view);
        launch!(k, "ce_fwd", rows;
            &mut self.losses, &mut self.probs, &self.logits, &self.targets, &rows, &v)?;
        let losses = k.stream.clone_dtoh(&self.losses)?;
        Ok(losses[..self.rows].iter().sum::<f32>() / self.valid as f32)
    }

    /// The chain rule, unrolled by hand from the loss back to every parameter.
    /// Accumulates into `grads` (which adam() zeroes after each use).
    fn backward(&mut self) -> Res<()> {
        let (rows, t, c, c4, h_n, d, v) = (
            self.rows as i32,
            self.t_len as i32,
            self.cfg.n_embd as i32,
            4 * self.cfg.n_embd as i32,
            self.cfg.n_head as i32,
            self.cfg.head_dim() as i32,
            self.vocab as i32,
        );
        let b_n = rows / t;
        let (rc, rc4) = (rows * c, rows * c4);
        let valid = self.valid as i32;
        let k = &self.kern;
        let (zero, one) = (0i32, 1i32);

        launch!(k, "ce_bwd", rows * v;
            &mut self.d_logits, &self.probs, &self.targets, &rows, &v, &valid)?;

        // logits = x_out[last] @ lm_head^T
        let x_final = &self.layers[self.cfg.n_layer - 1].x_out;
        let mut g = self.grads.slice_mut(self.l.lm_head..self.l.lm_head + (v * c) as usize);
        launch2d!(k, "matmul_tn", v, c; &mut g, &self.d_logits, x_final, &v, &c, &rows)?;
        drop(g);
        let lm_head = self.params.slice(self.l.lm_head..self.l.lm_head + (v * c) as usize);
        launch2d!(k, "matmul_nn", rows, c; &mut self.d_x, &self.d_logits, &lm_head, &rows, &c, &v, &zero)?;
        drop(lm_head);

        for li in (0..self.cfg.n_layer).rev() {
            let lo = self.l.layers[li];
            let xin = if li == 0 { &self.x1 } else { &self.layers[li - 1].x_out };
            let la = &self.layers[li];

            // x_out = x_mid + relu(xn2 @ fc1^T) @ fc2^T;  xn2 = rmsnorm(x_mid)
            let mut g = self.grads.slice_mut(lo.fc2..lo.fc2 + (c * c4) as usize);
            launch2d!(k, "matmul_tn", c, c4; &mut g, &self.d_x, &la.h, &c, &c4, &rows)?;
            drop(g);
            let fc2 = self.params.slice(lo.fc2..lo.fc2 + (c * c4) as usize);
            launch2d!(k, "matmul_nn", rows, c4; &mut self.d_h, &self.d_x, &fc2, &rows, &c4, &c, &zero)?;
            drop(fc2);
            launch!(k, "relu_bwd_inplace", rc4; &mut self.d_h, &la.hpre, &rc4)?;
            let mut g = self.grads.slice_mut(lo.fc1..lo.fc2);
            launch2d!(k, "matmul_tn", c4, c; &mut g, &self.d_h, &la.xn2, &c4, &c, &rows)?;
            drop(g);
            let fc1 = self.params.slice(lo.fc1..lo.fc2);
            launch2d!(k, "matmul_nn", rows, c; &mut self.d_xn2, &self.d_h, &fc1, &rows, &c, &c4, &zero)?;
            drop(fc1);
            launch!(k, "rmsnorm_bwd", rows;
                &mut self.d_x2, &la.x_mid, &la.r2, &self.d_xn2, &rows, &c, &zero)?;
            launch!(k, "add_inplace", rc; &mut self.d_x2, &self.d_x, &rc)?; // residual

            // x_mid = x_in + atty @ wo^T
            let mut g = self.grads.slice_mut(lo.wo..lo.fc1);
            launch2d!(k, "matmul_tn", c, c; &mut g, &self.d_x2, &la.atty, &c, &c, &rows)?;
            drop(g);
            let wo = self.params.slice(lo.wo..lo.fc1);
            launch2d!(k, "matmul_nn", rows, c; &mut self.d_atty, &self.d_x2, &wo, &rows, &c, &c, &zero)?;
            drop(wo);

            // atty = att @ v;  att = causal_softmax(q @ k^T / sqrt(d))
            launch!(k, "attn_mix_bwd_dp", b_n * h_n * t * t;
                &mut self.d_att, &self.d_atty, &la.v, &b_n, &h_n, &t, &d)?;
            launch!(k, "attn_mix_bwd_dv", rc;
                &mut self.d_v, &la.att, &self.d_atty, &b_n, &h_n, &t, &d)?;
            launch!(k, "attn_softmax_bwd", b_n * h_n * t;
                &mut self.d_scores, &la.att, &self.d_att, &b_n, &h_n, &t)?;
            launch!(k, "attn_scores_bwd_dq", rc;
                &mut self.d_q, &self.d_scores, &la.k, &b_n, &h_n, &t, &d)?;
            launch!(k, "attn_scores_bwd_dk", rc;
                &mut self.d_k, &self.d_scores, &la.q, &b_n, &h_n, &t, &d)?;

            // q/k/v = xn @ w{q,k,v}^T;  xn = rmsnorm(x_in)
            let mut g = self.grads.slice_mut(lo.wq..lo.wk);
            launch2d!(k, "matmul_tn", c, c; &mut g, &self.d_q, &la.xn, &c, &c, &rows)?;
            drop(g);
            let mut g = self.grads.slice_mut(lo.wk..lo.wv);
            launch2d!(k, "matmul_tn", c, c; &mut g, &self.d_k, &la.xn, &c, &c, &rows)?;
            drop(g);
            let mut g = self.grads.slice_mut(lo.wv..lo.wo);
            launch2d!(k, "matmul_tn", c, c; &mut g, &self.d_v, &la.xn, &c, &c, &rows)?;
            drop(g);
            let wq = self.params.slice(lo.wq..lo.wk);
            launch2d!(k, "matmul_nn", rows, c; &mut self.d_xn, &self.d_q, &wq, &rows, &c, &c, &zero)?;
            drop(wq);
            let wk = self.params.slice(lo.wk..lo.wv);
            launch2d!(k, "matmul_nn", rows, c; &mut self.d_xn, &self.d_k, &wk, &rows, &c, &c, &one)?;
            drop(wk);
            let wv = self.params.slice(lo.wv..lo.wo);
            launch2d!(k, "matmul_nn", rows, c; &mut self.d_xn, &self.d_v, &wv, &rows, &c, &c, &one)?;
            drop(wv);
            launch!(k, "rmsnorm_bwd", rows;
                &mut self.d_xin, xin, &la.r1, &self.d_xn, &rows, &c, &zero)?;
            launch!(k, "add_inplace", rc; &mut self.d_xin, &self.d_x2, &rc)?; // residual

            std::mem::swap(&mut self.d_x, &mut self.d_xin); // d_x now feeds layer li-1
        }

        // x1 = rmsnorm(x0);  x0 = wte[token] + wpe[pos]
        launch!(k, "rmsnorm_bwd", rows; &mut self.d_x0, &self.x0, &self.r0, &self.d_x, &rows, &c, &zero)?;
        let (wte_off, wpe_off) = (self.l.wte as i32, self.l.wpe as i32);
        launch!(k, "embed_bwd_wte", v * c;
            &mut self.grads, &self.d_x0, &self.tokens, &wte_off, &v, &rows, &c)?;
        launch!(k, "embed_bwd_wpe", t * c; &mut self.grads, &self.d_x0, &wpe_off, &rows, &t, &c)?;
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
        let view = self.logits.slice((self.rows - 1) * self.vocab..self.rows * self.vocab);
        Ok(self.kern.stream.clone_dtoh(&view)?)
    }
}

fn main() -> Res<()> {
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

    let uchars: Vec<char> =
        docs.iter().flat_map(|d| d.chars()).collect::<BTreeSet<_>>().into_iter().collect();
    let stoi: HashMap<char, usize> = uchars.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {vocab_size}");

    let host_params = init_params(&mut rng, &cfg, vocab_size);
    println!("num params: {}", host_params.len());
    if scale {
        println!(
            "mode: scale ({} layers, {}-dim, {} heads, batch {})",
            cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.batch
        );
    }
    let mut gpu = Gpu::new(cfg, vocab_size, &host_params)?;
    println!("device: {}", gpu.kern.ctx.name()?);

    let learning_rate = cfg.learning_rate;
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
        // Parity mode forwards exactly the document's length, like the CPU crate does
        let (tok_slice, tgt_slice, t_len) = if cfg.batch == 1 {
            (&tokens[..n_last], &targets[..n_last], n_last)
        } else {
            (&tokens[..], &targets[..], t_max)
        };

        gpu.forward(tok_slice, t_len)?;
        let loss = gpu.loss(tgt_slice)?;
        gpu.backward()?;
        let lr_t = learning_rate * (1.0 - step as f32 / cfg.num_steps as f32);
        gpu.adam(step, lr_t)?;

        print!("step {:4} / {:4} | loss {:.4}\r", step + 1, cfg.num_steps, loss);
        std::io::stdout().flush()?;
    }
    let elapsed = t_start.elapsed();
    println!(
        "\ntrain time: {:.2}s ({:.1} steps/s)",
        elapsed.as_secs_f64(),
        cfg.num_steps as f64 / elapsed.as_secs_f64()
    );

    // Inference: re-forward the whole prefix per character (T <= 16), sample on the host
    let temperature = 0.5f64;
    println!("--- inference (new, hallucinated names) ---");
    for sample_idx in 0..20 {
        let mut ctx = vec![bos as i32];
        let mut sample = String::new();
        for _ in 0..cfg.block_size {
            gpu.forward(&ctx, ctx.len())?;
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

        launch2d!(k, "matmul_nt", m, n; &mut dy, &da, &db, &mi, &ni, &ki, &zero).unwrap();
        let y = s.clone_dtoh(&dy).unwrap();
        for r in 0..m {
            for c in 0..n {
                let want: f32 = (0..kk).map(|p| a[r * kk + p] * b[c * kk + p]).sum();
                assert!((y[r * n + c] - want).abs() < 1e-4, "matmul_nt [{r},{c}]");
            }
        }

        launch2d!(k, "matmul_nn", m, n; &mut dy, &da, &db, &mi, &ni, &ki, &zero).unwrap();
        let y = s.clone_dtoh(&dy).unwrap();
        for r in 0..m {
            for c in 0..n {
                let want: f32 = (0..kk).map(|p| a[r * kk + p] * b[p * n + c]).sum();
                assert!((y[r * n + c] - want).abs() < 1e-4, "matmul_nn [{r},{c}]");
            }
        }

        let (m2, n2, k2) = (4usize, 6usize, 9usize);
        let a2 = rand_vec(&mut rng, k2 * m2);
        let b2 = rand_vec(&mut rng, k2 * n2);
        let da2 = s.clone_htod(&a2).unwrap();
        let db2 = s.clone_htod(&b2).unwrap();
        let mut dw = s.alloc_zeros::<f32>(m2 * n2).unwrap();
        let (mi2, ni2, ki2) = (m2 as i32, n2 as i32, k2 as i32);
        launch2d!(k, "matmul_tn", m2, n2; &mut dw, &da2, &db2, &mi2, &ni2, &ki2).unwrap();
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
        // batch of 2 exercises the batched indexing in every attention kernel
        let (b_n, h_n, t, d) = (2usize, 2usize, 5usize, 3usize);
        let c = h_n * d;
        let rows = b_n * t;
        let q = rand_vec(&mut rng, rows * c);
        let kv = rand_vec(&mut rng, rows * c);
        let v = rand_vec(&mut rng, rows * c);
        let w = rand_vec(&mut rng, rows * c); // loss = dot(w, attention(q,k,v))

        let fwd = |q: &[f32], kk: &[f32], v: &[f32]| -> f32 {
            let (dq, dk, dv) =
                (s.clone_htod(q).unwrap(), s.clone_htod(kk).unwrap(), s.clone_htod(v).unwrap());
            let mut att = s.alloc_zeros::<f32>(b_n * h_n * t * t).unwrap();
            let mut y = s.alloc_zeros::<f32>(rows * c).unwrap();
            let (bi, hi, ti, di) = (b_n as i32, h_n as i32, t as i32, d as i32);
            launch!(k, "attn_softmax_fwd", b_n * h_n * t; &mut att, &dq, &dk, &bi, &hi, &ti, &di)
                .unwrap();
            launch!(k, "attn_mix_fwd", rows * c; &mut y, &att, &dv, &bi, &hi, &ti, &di).unwrap();
            let y = s.clone_dtoh(&y).unwrap();
            y.iter().zip(&w).map(|(a, b)| a * b).sum()
        };

        // analytic: chain the four backward kernels
        let (bi, hi, ti, di) = (b_n as i32, h_n as i32, t as i32, d as i32);
        let (dq_in, dk_in, dv_in) =
            (s.clone_htod(&q).unwrap(), s.clone_htod(&kv).unwrap(), s.clone_htod(&v).unwrap());
        let mut att = s.alloc_zeros::<f32>(b_n * h_n * t * t).unwrap();
        launch!(k, "attn_softmax_fwd", b_n * h_n * t; &mut att, &dq_in, &dk_in, &bi, &hi, &ti, &di)
            .unwrap();
        let dy = s.clone_htod(&w).unwrap();
        let mut d_att = s.alloc_zeros::<f32>(b_n * h_n * t * t).unwrap();
        let mut d_scores = s.alloc_zeros::<f32>(b_n * h_n * t * t).unwrap();
        let mut gq = s.alloc_zeros::<f32>(rows * c).unwrap();
        let mut gk = s.alloc_zeros::<f32>(rows * c).unwrap();
        let mut gv = s.alloc_zeros::<f32>(rows * c).unwrap();
        launch!(k, "attn_mix_bwd_dp", b_n * h_n * t * t; &mut d_att, &dy, &dv_in, &bi, &hi, &ti, &di)
            .unwrap();
        launch!(k, "attn_mix_bwd_dv", rows * c; &mut gv, &att, &dy, &bi, &hi, &ti, &di).unwrap();
        launch!(k, "attn_softmax_bwd", b_n * h_n * t; &mut d_scores, &att, &d_att, &bi, &hi, &ti)
            .unwrap();
        launch!(k, "attn_scores_bwd_dq", rows * c; &mut gq, &d_scores, &dk_in, &bi, &hi, &ti, &di)
            .unwrap();
        launch!(k, "attn_scores_bwd_dk", rows * c; &mut gk, &d_scores, &dq_in, &bi, &hi, &ti, &di)
            .unwrap();
        let (gq, gk, gv) = (
            s.clone_dtoh(&gq).unwrap(),
            s.clone_dtoh(&gk).unwrap(),
            s.clone_dtoh(&gv).unwrap(),
        );

        let h = 1e-2f32;
        for i in 0..rows * c {
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
        let (t, v) = (4usize, 7usize);
        let logits = rand_vec(&mut rng, t * v);
        let targets: Vec<i32> = vec![2, 0, -1, 5]; // one padded (masked) row
        let valid = targets.iter().filter(|&&x| x >= 0).count();
        let dtgt = s.clone_htod(&targets).unwrap();

        let fwd = |logits: &[f32]| -> f32 {
            let dl = s.clone_htod(logits).unwrap();
            let mut dloss = s.alloc_zeros::<f32>(t).unwrap();
            let mut dprobs = s.alloc_zeros::<f32>(t * v).unwrap();
            let (ti, vi) = (t as i32, v as i32);
            launch!(k, "ce_fwd", t; &mut dloss, &mut dprobs, &dl, &dtgt, &ti, &vi).unwrap();
            let l = s.clone_dtoh(&dloss).unwrap();
            l.iter().sum::<f32>() / valid as f32
        };

        let dl = s.clone_htod(&logits).unwrap();
        let mut dloss = s.alloc_zeros::<f32>(t).unwrap();
        let mut dprobs = s.alloc_zeros::<f32>(t * v).unwrap();
        let mut dgrad = s.alloc_zeros::<f32>(t * v).unwrap();
        let (ti, vi, validi) = (t as i32, v as i32, valid as i32);
        launch!(k, "ce_fwd", t; &mut dloss, &mut dprobs, &dl, &dtgt, &ti, &vi).unwrap();
        launch!(k, "ce_bwd", t * v; &mut dgrad, &dprobs, &dtgt, &ti, &vi, &validi).unwrap();
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
        let host_params = init_params(&mut rng, &PARITY, vocab);
        let mut gpu = Gpu::new(PARITY, vocab, &host_params).unwrap();
        let l = gpu.l.clone();
        let lo = l.layers[0];

        let tokens: Vec<i32> = vec![26, 4, 12, 12, 0]; // BOS e m m a
        let targets: Vec<i32> = vec![4, 12, 12, 0, 26];

        let loss0 = {
            gpu.forward(&tokens, tokens.len()).unwrap();
            gpu.loss(&targets).unwrap()
        };
        // untrained loss should sit near the uniform floor ln(27) ~ 3.296
        let uniform = (vocab as f32).ln();
        assert!((loss0 - uniform).abs() < 0.5, "untrained loss {loss0} vs ln(V) {uniform}");

        gpu.backward().unwrap();
        let grads = gpu.kern.stream.clone_dtoh(&gpu.grads).unwrap();

        // probe a few parameters from every matrix in the model
        let probes: Vec<usize> = [
            l.wte + 26 * 16 + 3, // a token actually used (BOS row)
            l.wte + 4 * 16 + 7,  // 'e' row
            l.wpe + 5,
            l.wpe + 2 * 16 + 1,
            l.lm_head + 4 * 16 + 2,
            l.lm_head + 26 * 16 + 9,
            lo.wq, lo.wq + 100, lo.wk + 37, lo.wv + 200, lo.wo + 141,
            lo.fc1 + 11, lo.fc1 + 500, lo.fc2 + 300, lo.fc2 + 777,
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
                gpu.forward(&tokens, tokens.len()).unwrap();
                gpu.loss(&targets).unwrap()
            };
            let numeric = (eval(h) - eval(-h)) / (2.0 * h);
            assert_close(grads[i], numeric, &format!("end-to-end dparam[{i}]"));
        }
    }

    #[test]
    fn multilayer_batched_gradients_match_finite_differences() {
        // A tiny scale-shaped config: multiple layers, a batch of two documents of
        // different lengths, so the layer loop, batched attention, padding, and the
        // loss mask are all on the gradient path.
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
        let mut gpu = Gpu::new(tiny, vocab, &host_params).unwrap();
        let l = gpu.l.clone();

        // doc 0 fills the block; doc 1 is short, padded with masked (-1) targets
        let tokens: Vec<i32> = vec![
            10, 3, 1, 4, 1, 5, 9, 2, // doc 0: 8 positions
            10, 2, 7, 1, 10, 10, 10, 10, // doc 1: 4 valid + BOS padding
        ];
        let targets: Vec<i32> = vec![
            3, 1, 4, 1, 5, 9, 2, 6, //
            2, 7, 1, 10, -1, -1, -1, -1,
        ];

        gpu.forward(&tokens, tiny.block_size).unwrap();
        gpu.loss(&targets).unwrap();
        gpu.backward().unwrap();
        let grads = gpu.kern.stream.clone_dtoh(&gpu.grads).unwrap();

        // probe parameters from both layers and the embeddings/head
        let probes: Vec<usize> = vec![
            l.wte + 10 * 8 + 2, // BOS row (also the pad token: its grad must survive FD too)
            l.wte + 3 * 8 + 5,
            l.wpe + 6 * 8 + 1, // a position only doc 0 reaches
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
                let mut p = host_params.clone();
                p[i] += delta;
                let mut view = gpu.params.slice_mut(..);
                gpu.kern.stream.memcpy_htod(&p, &mut view).unwrap();
                drop(view);
                gpu.forward(&tokens, 8).unwrap();
                gpu.loss(&targets).unwrap()
            };
            let numeric = (eval(h) - eval(-h)) / (2.0 * h);
            assert_close(grads[i], numeric, &format!("multilayer dparam[{i}]"));
        }
    }

    #[test]
    fn training_is_bit_deterministic() {
        let run = || -> (f32, Vec<f32>) {
            let mut rng = Rng::new(42);
            let vocab = 27;
            let host_params = init_params(&mut rng, &PARITY, vocab);
            let mut gpu = Gpu::new(PARITY, vocab, &host_params).unwrap();
            let tokens: Vec<i32> = vec![26, 4, 12, 12, 0];
            let targets: Vec<i32> = vec![4, 12, 12, 0, 26];
            let mut last = 0.0;
            for step in 0..20 {
                gpu.forward(&tokens, tokens.len()).unwrap();
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
