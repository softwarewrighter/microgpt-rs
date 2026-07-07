# microgpt-rs — Design

Concrete signatures and the reasoning behind each choice. Line references like
`microgpt.py:59` point at the Python original.

## 1. Autograd: `Tape` + `Value`

### Types

```rust
#[derive(Clone, Copy)]
struct Parent {
    idx: u32,        // index of the parent node on the tape
    local_grad: f64, // d(this node) / d(parent), captured at forward time
}

struct Node {
    data: f64,
    grad: f64,
    parents: [Option<Parent>; 2], // every primitive op has ≤ 2 inputs
}

struct Tape {
    nodes: RefCell<Vec<Node>>,
}

#[derive(Clone, Copy)]
struct Value<'t> {
    tape: &'t Tape,
    idx: u32,
}
```

Rationale:

- **`RefCell` on the whole `Vec`, not per node.** Ops need `&Tape` (shared)
  so that `Value` can be `Copy` and operator overloading works on plain
  values, exactly like the Python code reads (`q_h[j] * k_h[t][j]`). Borrows
  are always short (push one node, read one field), so `RefCell` never panics
  and its cost is negligible.
- **Fixed `[Option<Parent>; 2]` instead of `Vec<Parent>`.** Matches the
  Python primitives (add/mul are binary; pow/log/exp/relu are unary) and
  keeps `Node` inline with zero heap allocations per node. This mirrors
  `__slots__` at `microgpt.py:31` — same optimization, Rust flavor.
- **Local grads captured at forward time** — a direct translation of
  `_local_grads` (`microgpt.py:37`). E.g. `mul` stores
  `(other.data, self.data)`.

### Operations

Same primitive set as Python (`microgpt.py:39-57`):

```rust
impl Tape {
    fn value(&self, data: f64) -> Value;      // leaf (parameter / constant)
    fn push(&self, data: f64, p: [Option<Parent>; 2]) -> Value; // internal
    fn truncate(&self, len: usize);           // discard step's graph
    fn backward(&self, loss: Value);
}

impl<'t> Value<'t> {
    fn data(self) -> f64;
    fn grad(self) -> f64;
    fn powf(self, k: f64) -> Value<'t>;   // __pow__ (exponent is a plain float)
    fn log(self) -> Value<'t>;
    fn exp(self) -> Value<'t>;
    fn relu(self) -> Value<'t>;
}

// std::ops impls: Add, Sub, Mul, Div, Neg for Value⊕Value and Value⊕f64.
```

Python's reflected operators (`__radd__` etc.) become `impl Add<Value> for
f64`. Division translates literally as `self * other.powf(-1.0)` to keep the
derivation identical, or directly with local grads `(1/b, -a/b²)` — we choose
the direct form and note the equivalence in a comment.

### `backward()` — simpler than the Python

```rust
fn backward(&self, loss: Value) {
    let mut nodes = self.nodes.borrow_mut();
    nodes[loss.idx as usize].grad = 1.0;
    for i in (0..=loss.idx as usize).rev() {
        let g = nodes[i].grad;
        for p in nodes[i].parents.into_iter().flatten() {
            nodes[p.idx as usize].grad += p.local_grad * g;
        }
    }
}
```

The Python DFS + visited set (`microgpt.py:59-72`) exists only to produce a
topological order; the tape already has one by construction. A comment in the
code makes this point explicitly — it is the port's best teaching moment.

Gradients are zeroed by the optimizer for slots `0..P` (as in
`microgpt.py:182`) and discarded with the truncated graph for the rest.

## 2. RNG (replaces `random` / `random.gauss` / `random.choices`)

~25 lines, one struct:

```rust
struct Rng(u64); // splitmix64 / xorshift* core

impl Rng {
    fn new(seed: u64) -> Self;               // seed 42, as in the original
    fn uniform(&mut self) -> f64;            // [0, 1)
    fn gauss(&mut self, mu: f64, sigma: f64) -> f64; // Box–Muller
    fn shuffle<T>(&mut self, xs: &mut [T]);  // Fisher–Yates
    fn choices(&mut self, weights: &[f64]) -> usize; // cumsum + uniform
}
```

Sequence differs from CPython's Mersenne Twister, so trained weights differ;
behavior (loss curve, name quality) matches. Determinism within Rust runs is
exact.

## 3. Tokenizer

```rust
let uchars: Vec<char> = /* sorted unique chars across docs (BTreeSet) */;
let stoi: HashMap<char, usize> = /* reverse map */;
let bos: usize = uchars.len();
let vocab_size = uchars.len() + 1;
```

The Python uses `uchars.index(ch)` (O(n) scan, `microgpt.py:157`); we add the
`HashMap` because it is the idiomatic Rust spelling, not for speed. Dataset
is ASCII lowercase names, but we use `char` so the code is honest about
Unicode.

## 4. Model

Direct translation of `microgpt.py:94-144`, with `Vec<Value>` for activation
vectors and `Vec<Vec<Value>>` for weight matrices.

```rust
struct StateDict { /* wte, wpe, lm_head, and per-layer wq/wk/wv/wo/fc1/fc2 */ }

fn linear<'t>(x: &[Value<'t>], w: &[Vec<Value<'t>>]) -> Vec<Value<'t>>;
fn softmax<'t>(logits: &[Value<'t>]) -> Vec<Value<'t>>;
fn rmsnorm<'t>(x: &[Value<'t>]) -> Vec<Value<'t>>;

type KvCache<'t> = Vec<Vec<Vec<Value<'t>>>>; // [layer][position][n_embd]

fn gpt<'t>(
    sd: &StateDict<'t>, token_id: usize, pos_id: usize,
    keys: &mut KvCache<'t>, values: &mut KvCache<'t>,
) -> Vec<Value<'t>>;
```

Choices:

- **`StateDict` as a struct, not a `HashMap<String, _>`.** Python's string
  keys (`f'layer{i}.attn_wq'`) are dynamic-language idiom; a struct with a
  `Vec<Layer>` is the Rust idiom and gives field-name documentation for free.
  Layer count stays a `const N_LAYER: usize = 1` loop to preserve the "stack
  more layers" teaching point.
- **KV cache passed as `&mut`** — same explicit-cache design as the Python
  (`microgpt.py:121`), which is itself a nice demo of how inference caching
  works.
- `softmax` subtracts the max (`microgpt.py:98`) — keep, and keep the comment
  about numerical stability.
- All the math stays as loops over scalars. No `ndarray`. Resist the urge.

Hyperparameters as `const`s, verbatim: `N_LAYER=1, N_EMBD=16, BLOCK_SIZE=16,
N_HEAD=4, HEAD_DIM=4`, init `σ=0.08`.

## 5. Optimizer

Verbatim Adam with linear LR decay (`microgpt.py:146-182`):

```rust
let mut m = vec![0.0f64; num_params];
let mut v = vec![0.0f64; num_params];
// per step: read grad of tape slot i, update m/v with bias correction,
// write data back, zero grad. lr_t = 0.01 * (1 - step/1000).
```

Parameters are tape slots `0..P`, so "iterate over params" is
`for i in 0..num_params`. Same constants: β₁=0.85, β₂=0.99, ε=1e-8.

## 6. main(): training + inference

Same shape as the Python script:

1. Ensure `input.txt` (else `Command::new("curl")` the makemore names URL;
   on failure, print the URL and exit with instructions — no HTTP stack in
   the binary).
2. Load docs, shuffle, build tokenizer, init params on tape (print counts —
   must print `num params: 4192`).
3. 1000 steps: tokenize `[BOS] doc [BOS]`, forward ≤16 positions
   accumulating per-position losses, mean → `backward` → Adam →
   `tape.truncate(P)`. Progress line with `\r` like `microgpt.py:184`.
4. Sample 20 names at temperature 0.5: fresh KV cache per sample, feed BOS,
   sample from `softmax(logits / T)` via `rng.choices`, stop on BOS.

## Error handling & style rules

- `main` returns nothing; use `expect("message")` at the handful of I/O and
  parse points. A pedagogical demo should not thread `Result` through math.
- No `unsafe`, no clippy suppressions, `cargo fmt` clean.
- Comment style copies the Python's narrative register ("Let there be
  Autograd…") — the comments are part of what's being ported.
- Target ≤ ~400 lines including comments; single `src/main.rs`.

## Failure modes considered

- **Tape borrow panics:** avoided by never holding a `RefCell` borrow across
  an op that pushes (all ops copy out `data` first, then push).
- **u32 index overflow:** a 16-token doc builds ~700k nodes worst case —
  far under `u32::MAX`; assert on push in debug builds.
- **NaN loss:** same guards as Python (softmax max-subtraction, rmsnorm
  ε=1e-5, Adam ε=1e-8). Fixed seed makes any regression reproducible.
- **Memory:** peak tape ≈ 700k nodes × 48 B ≈ 34 MB, reused across steps.
