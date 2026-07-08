# microgpt-mlx — Follow-on Plan (Rust + MLX, Apple Silicon GPU)

Status: **M0 spike done (2026-07-07)** — MLX itself validated on this
machine's GPU via Python (mlx 0.29.3): array math, `value_and_grad`
(checked against hand-derived gradients), and `categorical` sampling all
work. **The Rust path (`mlx-rs` 0.25.3) is blocked on the Metal shader
compiler**: `mlx-sys` builds MLX C++ from source and its kernel-compile
step needs the `metal` CLI, which ships with full Xcode, not the Command
Line Tools (cmake is now installed; the build gets as far as the
`.metallib` step). Unblock: install Xcode from the App Store, run
`sudo xcode-select -s /Applications/Xcode.app` and, if prompted,
`xcodebuild -downloadComponent MetalToolchain`. Fallback: implement in
Python MLX (validated working today).

Spike also confirmed the honest-performance prediction below: a 16×16
matmul costs ~650 µs on GPU (pure dispatch overhead) — the same as a
1024×1024 matmul (~500 µs). Parity-mode GPU *will* lose to the 0.6 s CPU
demo; scale mode is where the GPU wins.

## What changes conceptually

The CPU demo's thesis is "this is the complete algorithm; everything else is
just efficiency." The MLX demo is the *everything else*: the identical
algorithm re-expressed as tensor operations so a GPU can execute it. Three
consequences, all pedagogically interesting:

1. **We stop writing our own autograd.** MLX brings function-transformation
   autograd (`value_and_grad`), like JAX. The hand-rolled tape from
   `microgpt-rs` is replaced by "define the loss as a function of the
   parameter arrays; ask MLX for its gradient." The demo's teaching moment
   shifts from *how backprop works* to *how frameworks package backprop*.
2. **Scalars become tensors.** Per-position loops over `Value` become one
   `(T, C)` matmul per weight matrix; the training-time KV cache disappears
   in favor of causal-masked attention over the whole sequence at once
   (the cache returns for inference).
3. **Unified memory.** On Apple Silicon there is no host↔device copy story
   at all — arrays live in shared memory and the Metal GPU reads them in
   place. This is the demo's main contrast with the later CUDA version.

## Technology choice

**`mlx-rs` crate** (community Rust bindings over the MLX C API; Apache-2.0,
actively maintained). Requires macOS on Apple Silicon; Metal backend is the
default device.

- Verify at kickoff: `mlx-rs` version pin, availability of
  `transforms::value_and_grad`, `ops::{matmul, softmax, rsqrt, where/tril}`,
  and `random::{normal, categorical}` — all present as of late 2025; pin
  whatever version builds and note it in the README.
- **Fallback if the bindings block us** (missing op, broken transform):
  same plan in Python + `mlx` (~100 lines). The document stands either way;
  only the milestone estimates change. Decide at the end of M1, not later.

The zero-dependency rule is explicitly relinquished for this demo: `mlx-rs`
is the point. Everything else (RNG for shuffling/sampling on the host, data
loading, tokenizer) is reused verbatim from `microgpt-rs`.

## Architecture deltas vs microgpt-rs

| Aspect | microgpt-rs (CPU) | microgpt-mlx (GPU) |
|---|---|---|
| Autograd | hand-rolled tape | `value_and_grad(loss_fn)` |
| Data unit | scalar `Value` | `Array` (f32 — MLX GPU has no f64) |
| Attention (train) | per-position, KV cache | full-sequence, causal mask `tril` |
| Attention (infer) | per-position, KV cache | same as CPU version (cache of Arrays) |
| Params | tape slots 0..P | `Vec<Array>` (flat list, like Python's `params`) |
| Adam | loop over slots | elementwise Array ops over the param list |
| Batch | 1 doc/step | 1 doc/step for parity mode (see below) |
| Precision | f64 | f32 (expect small loss-curve differences) |

Model code shape (single `main.rs` again, ~250 lines):

```rust
fn gpt(params: &StateDict, tokens: &Array /* (T,) i32 */) -> Array /* (T, V) logits */
fn loss_fn(params: &[Array], tokens: &Array, targets: &Array) -> Array // scalar
// per step: let (loss, grads) = value_and_grad(loss_fn)(params, ...);
//           adam update: params[i] = params[i] - lr_t * m_hat / (sqrt(v_hat) + eps)
//           mx::eval(params) — MLX is lazy; force the step to actually run
```

## The honest-performance section (important)

At this model size (4,192 params, seq ≤ 16), **the GPU will lose**. Per-step
work is a few hundred FLOPs of matmul; kernel dispatch overhead dominates
and the CPU tape demo (~0.6 ms/step) will beat MLX-on-GPU. The demo must not
pretend otherwise — that would undercut the whole series. So it runs in two
modes:

- **parity mode** (default): exact micro config, 1000 steps, prints the same
  loss band and sampled names. Expected: GPU *slower* than microgpt-rs, and
  the README says so and explains why (dispatch latency vs. arithmetic).
- **scale mode** (`--scale`): `n_layer=4, n_embd=256, n_head=8,
  block_size=32, batch=64` (~3M params, docs padded & loss-masked into
  batches). Here the GPU pulls decisively ahead of any scalar CPU loop —
  this is the crossover the series is built to show. Report steps/sec for
  both modes next to microgpt-rs numbers, measured on the same machine.

Batching (scale mode only) is the one real semantic extension: pad each doc
to the block, mask padded positions out of the mean loss. Parity mode keeps
one-doc-per-step semantics so loss curves are directly comparable.

## Milestones

- **M0 — Spike (½ day, go/no-go):** `cargo add mlx-rs`; multiply two arrays
  on GPU; take a `value_and_grad` of a toy scalar function; sample from
  `categorical`. If any of these fail → invoke the Python-MLX fallback.
- **M1 — Model forward (½ day):** tokenizer/data reuse; `gpt()` with causal
  mask; untrained loss ≈ ln 27 ≈ 3.30 (same sanity gate as the CPU port).
- **M2 — Training (1 day):** `value_and_grad`, hand-written Adam over the
  param list (no `mlx-rs` optimizer module — Adam-by-hand is part of the
  demo's identity), lazy-eval discipline (`eval` once per step). Exit: loss
  lands in the 2.0–2.2 band over 1000 steps.
  - Cross-check gradients once against microgpt-rs: load the same tiny
    weight set into both (add a weight dump/load helper on each side),
    forward one doc, compare a handful of gradient values to ~1e-3 (f32).
- **M3 — Inference + parity report (½ day):** KV-cache sampling, 20 names,
  README with the two-mode performance table.
- **M4 — Scale mode (½ day):** batched/masked training, `--scale` flag,
  measure and record the CPU-vs-GPU crossover.

## Exit criteria for "it works" (gate for the CUDA follow-on)

1. Parity mode reproduces the loss band and name quality of microgpt-rs.
2. Scale mode shows GPU ≥ 10× over a comparable CPU tensor loop (or a
   documented honest number, whatever it is).
3. Single-file, builds with `cargo run --release` on Apple Silicon, no
   Python at runtime.

# microgpt-cuda — Sketch (Rust + CUDA on Arch Linux, contingent on MLX demo)

Status: **landed (2026-07-07)** — implemented as sketched below
(`cudarc` 0.19 + NVRTC, toolkit 13.2, RTX 5060 Ti). All per-kernel and
end-to-end gradient checks pass; results in the README. **Scale mode
landed (2026-07-08)**: `--scale` (~800K params, 4 layers, 128-dim, batch
32 — sized so the CPU control run stays under 3 minutes) on the same code
path, matmuls upgraded naive → shared-memory tiled, plus a
`microgpt-scale/` CPU crate as the control group. **M4 (MLX scale mode)
landed (2026-07-08)**: same `--scale` flag and config on `microgpt-mlx`,
batched masked training, host-RNG sampling shared with the siblings.
M1 Max results: 6.0 s train (165.9 steps/s) vs 77.1 s for the CPU control
on the same machine (12.8×); loss 2.128 vs 2.129 (CPU control) vs 2.132
(CUDA); samples open with the same names (`adari`, `annila`, `yuriana`).
Finding worth recording: MLX on Metal is *not* bit-deterministic run to
run (ulp-level reduction-order noise every few hundred steps, compounding
thereafter) — unlike the CPU/CUDA crates, which fix every summation
order. With that, the demo series is complete. The original sketch, for
the record:

- **Thesis:** MLX showed "let a framework differentiate tensor ops." The
  CUDA demo goes one level *down* instead: hand-written forward **and
  backward** kernels — a micro-sized, Rust-flavored `llm.c`. No autograd
  anywhere: every layer implements `forward()` and `backward()` explicitly,
  which is the natural sequel to the tape demo (same chain rule, now derived
  per-layer on paper instead of per-scalar by machine).
- **Technology:** `cudarc` crate (driver API + NVRTC): kernels written as
  CUDA C strings compiled at startup, launched from safe Rust. Explicit
  H2D/D2H copies — the deliberate contrast with unified memory on the Mac.
  Alternatives rejected: `burn`/`candle` (framework hides everything the
  demo exists to show), Rust-GPU kernel authoring (immature).
- **Environment:** Arch Linux, `pacman -S cuda`, NVIDIA driver; pin the
  toolkit version in the README; `CUDA_PATH` handling for NVRTC.
- **Kernels (~8):** embedding-gather, rmsnorm fwd/bwd, matmul (naive then
  tiled), causal-softmax fwd/bwd, relu fwd/bwd, cross-entropy fwd/bwd,
  fused Adam step. Batched scale-mode config from the MLX demo as default;
  parity micro-mode for correctness.
- **Correctness gate:** every kernel's backward is finite-difference-checked
  against its forward on small tensors (same discipline as microgpt-rs
  tests, now per-kernel), plus end-to-end loss-band parity.
- **Risk to flag now:** matmul backward and causal softmax backward are the
  two places hand-derived gradients typically go wrong; both get dedicated
  unit tests before the training loop exists.
