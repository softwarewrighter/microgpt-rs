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

[mlx-rs]: https://crates.io/crates/mlx-rs
[cudarc]: https://crates.io/crates/cudarc

## Run

```sh
cargo run --release                 # CPU demo (this crate)
cargo run --release -p microgpt-mlx # Apple Silicon GPU demo (needs Xcode's Metal toolchain)
cargo test                          # gradient checks vs finite differences + sanity checks

cd microgpt-cuda                    # NVIDIA GPU demo -- standalone, not a workspace member
cargo run --release                 # (needs the CUDA toolkit on a Linux/NVIDIA box)
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
   scale, pays ~160× interpreter tax on the same math.) The GPUs' real win
   would need a `--scale` mode: ~3M params, batched documents.

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
a Linux/NVIDIA box.

Full architecture/design/milestone docs: [`docs/`](docs/) —
[ARCHITECTURE.md](docs/ARCHITECTURE.md), [DESIGN.md](docs/DESIGN.md),
[PLAN.md](docs/PLAN.md), [PLAN-MLX.md](docs/PLAN-MLX.md).

## License

MIT — Copyright (c) 2026 Michael A. Wright. See [LICENSE](LICENSE).
