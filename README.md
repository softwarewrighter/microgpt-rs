# microgpt-rs

Ports of [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
— "the most atomic way to train and run inference for a GPT" — exploring the
same complete algorithm at different levels of the hardware/software stack,
in Rust. This project was inspired by the write-up
["Andrej Karpathy Just Built an Entire GPT in 243 Lines of Python"](https://www.towardsdeeplearning.com/andrej-karpathy-just-built-an-entire-gpt-in-243-lines-of-python-7d66cfdfa301)
(Towards Deep Learning, on Medium).

The original is ~200 lines of dependency-free Python: a character tokenizer,
a scalar autograd engine, a GPT-2-style transformer (1 layer, 16-dim, 4
heads, 4,192 parameters), the Adam optimizer, a training loop over ~32k baby
names, and temperature sampling that hallucinates new names. Its point is
pedagogical: *this file is the complete algorithm; everything else is just
efficiency.* This repo is the "everything else," one rung at a time:

| crate | target | autograd | status |
|---|---|---|---|
| `.` (**microgpt-rs**) | CPU, zero dependencies | hand-rolled scalar tape | done |
| [`microgpt-mlx/`](microgpt-mlx/) | Apple Silicon GPU ([mlx-rs]) | MLX `value_and_grad` | done |
| [`microgpt-cuda/`](microgpt-cuda/) | NVIDIA GPU ([cudarc], CUDA C kernels) | hand-written per-layer backward | done |
| [`microgpt-scale/`](microgpt-scale/) | CPU at the scale config, zero dependencies | hand-written per-layer backward | done |

Two model configurations run through those crates: **parity** (Karpathy's
original: 1 layer, 16-dim, 4,192 params, one document per step) and
**scale** (4 layers, 128-dim, 8 heads, ~800K params, 32 documents per
batched step) — parity to prove correctness against the original, scale to
show what the efficiency machinery is actually *for*. MLX at scale is the
remaining leg, planned next on the Mac.

[mlx-rs]: https://crates.io/crates/mlx-rs
[cudarc]: https://crates.io/crates/cudarc

## Run

```sh
cargo run --release                 # CPU demo (this crate)
cargo run --release -p microgpt-mlx # Apple Silicon GPU demo (needs Xcode's Metal toolchain)
cargo test                          # gradient checks vs finite differences + sanity checks

cargo run --release -p microgpt-scale           # CPU at the ~800K-param scale config
cargo run --release -p microgpt-scale -- --parity # same crate at the original 4K config
cargo test --release -p microgpt-scale

cd microgpt-cuda                    # NVIDIA GPU demo -- standalone, not a workspace member
cargo run --release                 # (needs the CUDA toolkit on a Linux/NVIDIA box)
cargo run --release -- --scale      # same binary at the ~800K-param scale config
cargo test --release                # per-kernel + end-to-end gradient checks, on the GPU
```

`input.txt` (the makemore names dataset) is downloaded via `curl` on first
run if missing. Expected output:

```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 1.9146
--- inference (new, hallucinated names) ---
sample  1: amanion
sample  2: alik
sample  3: zarani
...
```

Output is deterministic (seed 42): run it twice, get identical bits. That
holds for the CUDA version too (its kernels use no atomics) — and in fact
the CPU and CUDA versions print the identical 20 sample names.

## Results and comparison

Same dataset, same 1000 training steps everywhere. "Loss band" is where the
per-document training loss settles (it starts at ln 27 ≈ 3.30, the
uniform-guessing floor). Two machines, so two tables — comparisons are only
made within a table.

**Apple M1 Max:**

| implementation | total wall time | loss band | sample quality |
|---|---|---|---|
| `microgpt.py` (CPython 3.9) | 98.3 s | ~2.0–2.2 | `karai`, `keylen`, `anton` |
| **microgpt-rs** (CPU, scalar tape) | **1.0 s** | ~2.0–2.2 | `amani`, `kayli`, `delana` |
| **microgpt-mlx** (Metal GPU, tensors) | 2.2 s | ~2.0–2.2 | `anala`, `celin`, `kayan` |

**Linux box (Intel Xeon W-2135, RTX 5060 Ti, CUDA 13.2):**

| implementation | total wall time | training only | loss @ step 1000 | sample quality |
|---|---|---|---|---|
| **microgpt-rs** (Xeon CPU, scalar tape) | **0.75 s** | ~0.7 s | 1.9146 | `amanion`, `alik`, `zarani` |
| **microgpt-cuda** (RTX 5060 Ti, hand kernels) | 1.0 s | **0.10 s** | 1.9146 | `amanion`, `alik`, `zarani` |

(CUDA total wall time is dominated by one-time startup: CUDA context init
plus NVRTC-compiling the kernels is ~0.9 s. Training itself runs at ~10,000
steps/s once the GPU is at clock; from idle clocks the first run ramps
through ~0.5 s. Both numbers from warm repeat runs.)

Four observations worth more than the tables:

1. **All four produce equivalent models.** Same loss band, same name
   quality. Python's exact numbers differ only because its RNG differs.
2. **The Rust versions are near-twins by construction.** They share the
   same host RNG for initialization and the same document order, so their
   per-step losses track each other (final step: 1.9146 CPU-f64 vs 1.9160
   MLX-f32). The only divergence is float precision and reduction order.
   This is the cheapest possible cross-implementation correctness proof.
3. **The CUDA port passes that test to the last decimal.** Its f32 kernels
   accumulate in the same order as the CPU's scalar loops, so against the
   f64 CPU run, 998 of the 1000 printed step losses match to all 4 decimals
   (the other two differ by one in the last digit) and all 20 sampled names
   come out *character-for-character identical* — autograd tape and
   hand-derived per-layer backward agreeing end to end.
4. **Whether a GPU loses at this scale depends on the cost of a dispatch.**
   A Metal kernel dispatch costs ~0.5 ms whether the matmul is 16×16 or
   1024×1024, so on the Mac a 4,192-parameter model is pure dispatch
   overhead and the GPU loses to the CPU by 2×. A CUDA kernel launch costs
   ~2 µs — ~40 launches per step still leaves the RTX training 7× faster
   than the Xeon, even at a batch size of one document. (CPython, for
   scale, pays ~160× interpreter tax on the same math.) The GPU's real win
   needs real work per launch — which is what the scale config below is
   for.

## Results at scale

The scale config (`--scale` on the CUDA crate; the default for
`microgpt-scale`) is 4 layers, 128-dim, 8 heads, 795,392 parameters, 32
documents per batched step — ~2.4 GFLOP per training step instead of ~1
MFLOP, over the same 1000 steps. Same Linux box as above. Init std drops
to 0.02 and Adam's learning rate to 0.001; the parity values (0.08, 0.01)
diverge at this depth. The size is chosen so the whole demo stays fast:
seconds on the GPU, under three minutes on the CPU.

| implementation | train time | steps/s | loss @ step 1000 | sample quality |
|---|---|---|---|---|
| microgpt-rs (scalar tape) | — | — | — | *cannot run: one tape node per scalar op is billions of nodes (~100 GB) per step* |
| **microgpt-scale** (Xeon CPU, 12 threads) | 168.5 s | 5.9 | 2.133 | `adari`, `annila`, `yuriana` |
| **microgpt-cuda --scale** (RTX 5060 Ti) | **13.1 s** | **76.2** | 2.132 | `adari`, `annina`, `yuriana` |
| microgpt-mlx (M1 Max) | *planned — next up, on the Mac* | | | |

We also ran a 4× bigger config once (4 layers, 256-dim, batch 64: 3.16M
params, ~19 GFLOP/step) before sizing the demo down: GPU 26.0 s vs CPU
558 s, loss 2.01 on both. The GPU's lead *grows* with model size (13× →
21×) — that curve is the whole point of GPUs — but a 9-minute CPU run is
a bad demo, so the shipped config trades a slice of the gap for wall-clock
sanity.

What the scale run shows:

1. **This is the regime GPUs exist for.** At 4K params the RTX won on
   cheap launches; at 800K params it wins on arithmetic: 13× faster than
   the same algorithm on all 12 Xeon threads (21× at 3.16M params), with
   the same number of kernel launches per step as parity mode.
2. **The scalar tape's absence is the lesson, not an omission.** One tape
   node per scalar op × ~2.4 GFLOP per step doesn't fit in memory. That is
   *why* every real framework differentiates tensors (or writes per-layer
   backward like `llm.c` and these two crates) — it's a memory argument
   before it's a speed argument.
3. **The two scale implementations validate each other.** Same init bits,
   same document batches, mirror-image summation order: their printed
   losses are identical for the first tens of steps and drift only in the
   third decimal by step 1000 (2.1326 vs 2.1315 — the GPU's `expf`/`rsqrtf`
   are faithfully-rounded approximations; the drift is float noise, not
   disagreement). Both are bit-deterministic run to run.
4. **The bigger model is visibly better at the job.** Loss lands at ~2.13
   *per batch of 32* (a much less noisy estimate than parity's per-doc
   ~2.0–2.2 band), and the samples sound more like names: `yuriana`,
   `anayah`, `anysha` — and the 3.16M run more still (`weston`, `arielle`,
   `emmalina`, `cambrie`).
5. **Efficiency work happens on both sides, and it's the same lesson
   twice.** The CUDA matmuls went naive → shared-memory tiled (2.2× on the
   3.16M config): stage each 16×16 patch once, coalesced, instead of
   letting every warp scatter reads across 32 cache lines. The CPU
   backward matmuls got the mirror fix — a loop interchange so weights
   stream unit-stride and vectorize — worth 2.8× (2.1 → 5.9 steps/s).
   Neither change touches any element's summation order, so both crates'
   outputs stayed bit-identical through their optimizations — "everything
   else is just efficiency," verifiably.

## Design notes

**CPU (this crate).** Python's autograd is objects pointing at objects,
cleaned up by the garbage collector. Rust has no GC, and imitating one with
`Rc<RefCell<...>>` teaches the wrong lesson. Instead the computation graph
is a *tape* (arena): one `Vec<Node>`, and a `Value` is a copyable index into
it. Two things fall out:

- Nodes are appended in forward order, which is already a topological order,
  so `backward()` is a single reverse sweep — the Python's DFS + visited set
  disappears.
- Parameters are the first 4,192 tape slots and persist across steps;
  `tape.truncate(num_params)` frees an entire step's graph in O(1).

Zero crates — including the RNG (splitmix64 + Box–Muller, ~40 lines).

**MLX (`microgpt-mlx/`).** The same model re-expressed as tensor ops: the
hand-rolled tape is replaced by MLX's `value_and_grad`, per-position loops
become `(T, C)` matmuls, and the training-time KV cache becomes causal-mask
attention over the whole sequence. Parameters live in unified memory, read
by the GPU in place. One dependency: `mlx-rs`.

**CUDA (`microgpt-cuda/`).** The opposite move: down instead of up. No
autograd at all — every layer implements forward *and* backward as a
hand-written CUDA kernel (~20 kernels: embedding, rmsnorm, matmul, causal
softmax, attention mix, ReLU, cross-entropy, and a fused Adam step), the
chain rule derived per-layer on paper instead of per-scalar by machine. A
micro-sized, Rust-flavored `llm.c`. The kernels are CUDA C strings compiled
at startup with NVRTC and launched from safe Rust via `cudarc` (pinned to
0.19 / toolkit 13.2); parameters live in one flat device buffer, and every
number the host sees — the loss, the sampled logits — crosses the PCIe bus
in an explicit device↔host copy, the deliberate contrast with Apple's
unified memory. No atomics anywhere (embedding backward *gathers* per vocab
row instead of scattering with `atomicAdd`), so training is bit-
deterministic, same as the CPU crate. Each backward kernel is finite-
difference-checked against its own forward, plus an end-to-end gradient
check through the whole model (`cargo test`, needs the GPU). Kept out of
the cargo workspace so macOS builds never touch it; it builds standalone on
a Linux/NVIDIA box. One code path serves both configs — parity is just
`batch=1, layers=1` — so the parity output stayed bit-identical through the
scale work, including the switch from naive to shared-memory-tiled matmuls
(tiling changes *where* operands are staged, not any element's summation
order).

**Scale on CPU (`microgpt-scale/`).** The control group for the scale
experiment, and the reason it's a separate crate: the tape crate *cannot*
run this config. The tape materializes one node per scalar op, and at ~2.4
GFLOP per training step that's billions of nodes — over a hundred
gigabytes — per step. So this crate does on the CPU exactly what `microgpt-cuda` does
on the GPU: the same per-layer hand-derived backward over the same flat
`f32` parameter buffer, function-for-kernel identical down to summation
order. Still zero dependencies — parallelism is `std::thread::scope`
splitting each output buffer into disjoint row chunks, which preserves
bit-determinism because no element's arithmetic changes. Being pure Rust it
is a workspace member and builds anywhere, including the Mac (where it will
be the CPU baseline for the MLX scale run). `--parity` runs the 4K config
as a cross-check: it reproduces the tape crate's samples exactly.

Full architecture/design/milestone docs: [`docs/`](docs/) —
[ARCHITECTURE.md](docs/ARCHITECTURE.md), [DESIGN.md](docs/DESIGN.md),
[PLAN.md](docs/PLAN.md), [PLAN-MLX.md](docs/PLAN-MLX.md).

## License

MIT — Copyright (c) 2026 Michael A. Wright. See [LICENSE](LICENSE).
