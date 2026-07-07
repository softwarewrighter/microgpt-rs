# microgpt-rs — Architecture

## What we are porting

`microgpt.py` (Karpathy, ~200 lines) is a complete GPT — tokenizer, scalar
autograd engine, transformer forward pass, Adam optimizer, training loop, and
sampling — in pure, dependency-free Python. It trains a character-level model
on ~32k baby names and hallucinates new ones. Its value is pedagogical: every
number the model computes is visible in one file, with no framework hiding the
math.

The Rust port must preserve that property. **The goal is not a fast GPT in
Rust (candle/burn exist); the goal is the same "complete algorithm, one
readable file" demo, expressed in idiomatic Rust.** Speed is a free side
effect: scalar autograd in Rust runs ~50–100× faster than CPython, which turns
the 1000-step training run from minutes into seconds.

## Component map

The Python file has six components in sequence. The Rust port keeps the same
six, in the same order, in one `main.rs`:

| # | Component | Python mechanism | Rust mechanism |
|---|-----------|------------------|----------------|
| 1 | Dataset loader | `urllib` download + `open()` | `std::fs`; shell out to `curl` if `input.txt` missing |
| 2 | Tokenizer | `sorted(set(chars))`, index lookup | `BTreeSet<char>` → `Vec<char>` + `HashMap<char, usize>` |
| 3 | Autograd | `Value` objects forming a GC'd DAG | **Tape (arena) of nodes; `Value` is a `Copy` handle (index + tape ref)** |
| 4 | Model (GPT) | functions over `list[Value]` | functions over `Vec<Value>` |
| 5 | Optimizer (Adam) | parallel `m`/`v` float lists | parallel `Vec<f64>` buffers |
| 6 | Training + sampling loop | top-level script | `fn main` |

Also needed (Python got them from the stdlib, Rust's std has no RNG):

| Component | Rust mechanism |
|-----------|----------------|
| RNG | small deterministic PCG/xoshiro, ~15 lines |
| Gaussian sampling | Box–Muller on top of the RNG |
| Weighted choice | cumulative-sum inverse sampling |

Zero crate dependencies. `Cargo.toml` has an empty `[dependencies]` section —
that constraint is the point of the demo.

## The central architectural decision: tape-based autograd

Python's `Value` builds an implicit DAG through object references and relies
on the garbage collector; `backward()` does an explicit topological sort.
A literal Rust translation (`Rc<RefCell<Node>>`) works but is the wrong
lesson — it fights the borrow checker and buries the algorithm in `.borrow()`
noise.

Instead we use a **tape** (a.k.a. Wengert list / arena):

- A `Tape` owns a `Vec<Node>`. Each `Node` stores `data`, `grad`, and up to
  two `(parent_index, local_grad)` pairs.
- A `Value` is `Copy`: `{ tape: &'t Tape, idx: u32 }`. Operator overloading
  (`impl Add`, `Mul`, …) pushes a node and returns a new handle.
- **Backward pass needs no topological sort**: nodes are appended in forward
  order, which *is* a topological order. `backward()` seeds `grad = 1` at the
  loss node and walks the tape in reverse, accumulating into parents.

This is simpler than the Python version in one respect (no DFS/visited-set)
and is exactly how production reverse-mode AD engines work — a genuinely
better pedagogical story, not just a workaround for the borrow checker.

### Parameter persistence across steps

Parameters must survive between training steps while the per-step computation
graph is discarded. The tape makes this a one-liner:

1. At startup, push all P parameters as the **first P nodes** of the tape.
2. Each training step appends forward-pass nodes P.., runs `backward()`, and
   the optimizer reads gradients from slots `0..P` and writes updated `data`
   back into them.
3. End of step: `tape.truncate(P)` — the whole computation graph is freed in
   O(1) with no allocator churn (the `Vec`'s capacity is reused next step).

This replaces Python's "let the GC collect the graph" with an explicit,
visible lifetime for the graph — again, arguably clearer.

## Data flow (identical to Python)

```
input.txt ──▶ docs: Vec<String> ──▶ shuffle
                    │
                    ▼
        tokenizer: chars ↔ ids, BOS = vocab_size-1
                    │
     per step: one doc → [BOS, c1..cn, BOS]
                    │
                    ▼
  for each position t: gpt(token, pos, &mut kv_cache) → logits
        └ embeddings → 1×{rmsnorm → attn(4 heads, KV cache) → +residual
                          → rmsnorm → MLP(4×, ReLU) → +residual} → lm_head
                    │
                    ▼
   softmax → -log p(target) → mean loss ──▶ loss.backward()
                    │
                    ▼
   Adam (lr 0.01 linearly decayed, β=0.85/0.99) updates slots 0..P
                    │ tape.truncate(P)
                    ▼
   after 1000 steps: sample 20 names at temperature 0.5
```

Hyperparameters are copied verbatim: `n_layer=1`, `n_embd=16`,
`block_size=16`, `n_head=4`, init `std=0.08`, 1000 steps — 4,192 parameters.

## Parity contract

Bitwise parity with CPython is a non-goal (different RNG, different float
summation order is fine — we keep the same left-to-right summation anyway).
The contract is **behavioral parity**:

- identical parameter count printed (`num params: 4192`),
- loss decreasing from ~3.3 to roughly the same band Python reaches (~2.0–2.2),
- sampled output that looks like plausible names,
- fixed seed ⇒ bit-identical output across runs of the Rust binary itself.

## Non-goals

- No SIMD, no threads, no GPU, no tensor type. Scalars only — that is the demo.
- No CLI flags, no config files, no checkpointing.
- No crates, including `rand` — the RNG is part of the "everything from
  scratch" story.
