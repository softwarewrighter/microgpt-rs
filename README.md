# microgpt-rs

Ports of [Andrej Karpathy's microgpt.py](https://github.com/karpathy/microgpt)
— "the most atomic way to train and run inference for a GPT" — exploring the
same complete algorithm at different levels of the hardware/software stack,
in Rust.

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
| `microgpt-cuda/` | NVIDIA GPU (cudarc, Arch Linux) | hand-written per-layer backward | planned |

[mlx-rs]: https://crates.io/crates/mlx-rs

## Run

```sh
cargo run --release                 # CPU demo (this crate)
cargo run --release -p microgpt-mlx # Apple Silicon GPU demo (needs Xcode's Metal toolchain)
cargo test                          # gradient checks vs finite differences + sanity checks
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

Output is deterministic (seed 42): run it twice, get identical bits.

## Results and comparison

All measured on the same M-series Mac, same dataset, same 1000 training
steps. "Loss band" is where the per-document training loss settles (it
starts at ln 27 ≈ 3.30, the uniform-guessing floor).

| implementation | total wall time | loss band | sample quality |
|---|---|---|---|
| `microgpt.py` (CPython 3.9) | 98.3 s | ~2.0–2.2 | `karai`, `keylen`, `anton` |
| **microgpt-rs** (CPU, scalar tape) | **1.0 s** | ~2.0–2.2 | `amani`, `kayli`, `delana` |
| **microgpt-mlx** (Metal GPU, tensors) | 2.2 s | ~2.0–2.2 | `anala`, `celin`, `kayan` |

Three observations worth more than the table:

1. **All three produce equivalent models.** Same loss band, same name
   quality. Python's exact numbers differ only because its RNG differs.
2. **The two Rust versions are near-twins by construction.** They share the
   same host RNG for initialization and the same document order, so their
   per-step losses track to ~3 decimal places (final step: 1.9146 CPU-f64
   vs 1.9160 GPU-f32). The only divergence is float precision. This is the
   cheapest possible cross-implementation correctness proof.
3. **The GPU loses at this scale — honestly and predictably.** A Metal
   kernel dispatch costs roughly the same (~0.5 ms) for a 16×16 matmul as
   for a 1024×1024 one; a 4,192-parameter model is pure dispatch overhead.
   CPython, meanwhile, pays ~160× interpreter tax on the same scalar math.
   The GPU's win appears when the model grows (a planned `--scale` mode:
   ~3M params, batched documents).

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

**CUDA (planned).** The opposite move: down instead of up. Hand-written
forward *and* backward kernels via `cudarc` — no autograd at all, a
micro-sized Rust `llm.c` — with explicit host↔device copies as the contrast
to Apple's unified memory. Kept out of the cargo workspace so macOS builds
never touch it; it builds standalone on a Linux/NVIDIA box.

Full architecture/design/milestone docs: [`docs/`](docs/) —
[ARCHITECTURE.md](docs/ARCHITECTURE.md), [DESIGN.md](docs/DESIGN.md),
[PLAN.md](docs/PLAN.md), [PLAN-MLX.md](docs/PLAN-MLX.md).

## License

MIT — Copyright (c) 2026 Michael A. Wright. See [LICENSE](LICENSE).
