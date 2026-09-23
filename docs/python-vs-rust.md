# microgpt: Python vs. Rust — Program Flow and Lambdas

This is a companion to [`python-to-rust.org`](python-to-rust.org). That
document maps Karpathy's [`microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
to `src/main.rs` one section at a time. This one covers two other topics:

1. how control and data move through the Rust program when it runs, and
2. how the Python code uses `lambda` (and code that works like a lambda),
   and what the Rust port does instead.

Line numbers refer to `src/main.rs`.

---

## Part 1: How the Rust program runs

The whole program is `fn main` (`main.rs:369`) calling a handful of free
functions. Here it is in execution order.

### 1. Setup (`main.rs:370–404`)

```
Rng::new(42)                        seed the hand-written RNG
  → download input.txt if missing   (curl subprocess)
  → read_to_string → lines → trim   docs: Vec<&str>, borrowed from `text`
  → rng.shuffle(&mut docs)
  → BTreeSet<char> → uchars         sorted, unique chars (token ids 0..25)
  → stoi: HashMap<char, usize>      the reverse lookup; bos = 26, vocab = 27
  → Tape::new()                     empty arena
  → StateDict::init(&tape, &mut rng, 27)
  → num_params = tape.len()         4192
```

`StateDict::init` calls `matrix(...)` for each weight. Every call pushes
leaf nodes onto the empty tape. After init, the tape looks like this:

```
tape.nodes: [ wte | wpe | lm_head | wq | wk | wv | wo | fc1 | fc2 ]
             0 ────────────────────────────────────────────── 4191
```

`sd` holds no numbers itself. It holds `Value` handles, which are
`(&tape, index)` pairs pointing into those slots. The `'t` lifetime on
`StateDict<'t>` lets the compiler check that `tape` outlives `sd`.

### 2. Training loop, one iteration per document (`main.rs:413–457`)

**a. Tokenize.** Build `[bos] + stoi[c]... + [bos]`, and set
`n = min(16, len - 1)` prediction positions.

**b. Forward pass.** For each `pos_id` in `0..n`:

```
gpt(&sd, token, pos, &mut keys, &mut values)            → logits: Vec<Value> (27)
  x = wte[token] + wpe[pos]; rmsnorm
  per layer:
    rmsnorm → q, k, v = linear(x, W)
    push k, v into the KV cache (the cache grows by one position per call)
    per head: dot q with every cached k → softmax → weighted sum of cached v
    linear(attn_wo) + residual
    rmsnorm → fc1 → relu → fc2 + residual
  linear(lm_head)
softmax(logits) → probs
losses.push(-probs[target].log())
```

Each `+`, `*`, `exp`, `log` and so on goes through an operator impl, which
computes the value and its local gradient right away. It then calls
`tape.push`, appending one node after the parameters. One document adds
tens of thousands of nodes. None of them are freed during the forward
pass.

**c. Loss.** `(1/n) * reduce(+)` over `losses` gives the last node on the
tape. `loss.data()` is saved as a plain `f64` for printing later.

**d. Backward.** `tape.backward(loss)` sets `grad = 1` on the loss node,
then walks the indices downward. For each node it adds
`local_grad * grad` into each parent. Every parent has a lower index than
its child, so one pass is enough. When it finishes, slots `0..4192` hold
the parameter gradients.

**e. Adam.** Take one `borrow_mut()` of the node vector. For
`i in 0..num_params`, update `m[i]` and `v[i]`, adjust `nodes[i].data`,
and zero `nodes[i].grad`. This is plain `f64` math, so nothing new goes
on the tape.

**f. Free the graph.** `drop((keys, values))` releases the cache's
handles, then `tape.truncate(num_params)` cuts the tape back to just the
parameters. The `Vec` keeps its capacity, so the next step reuses the
same memory.

**g. Print.** Write the `step … | loss …\r` line and flush stdout.

The tape's size over one iteration:

```
[params]  →  [params | graph of one document ... loss]  →  backward  →  adam  →  [params]
```

### 3. Inference (`main.rs:459–482`)

Inference runs the same `gpt` 20 times, starting from `bos`:

1. Scale the logits by `1 / temperature`.
2. Apply `softmax`.
3. Pull the `.data()` values out into a `Vec<f64>`.
4. Pick the next token with `rng.choices`.
5. Stop at `bos`, or else push the character onto the name.

Each sample still builds graph nodes, because there's no "no-grad" mode,
and `tape.truncate(num_params)` discards them after the sample.

### Where the borrows are

There is one `Tape`. Every `Value` and `StateDict` borrows it immutably
(`&'t Tape`), and the node vector is changed through `RefCell`:

- Each operator takes a brief `borrow()` to read data and a brief
  `borrow_mut()` to push.
- `backward` and the Adam loop each take one long `borrow_mut()`.

These borrows never overlap, so the runtime `RefCell` check never panics.

---

## Part 2: `lambda` in the Python, and the Rust equivalents

### The one explicit `lambda`

```python
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
```

This is an anonymous function bound to a name, used as a small factory.
It relies on four things:

- a **default argument** (`std=0.08`);
- **implicitly captured globals**: the `random` module and the `Value`
  class;
- two **nested comprehensions**, which are themselves anonymous function
  scopes;
- **no graph handle**, because a Python `Value` is a free-standing object.

The Rust version is a named function, not a closure (`main.rs:265`):

```rust
fn matrix<'t>(tape: &'t Tape, rng: &mut Rng, nout: usize, nin: usize) -> Mat<'t> {
    let std = 0.08;
    (0..nout).map(|_| (0..nin).map(|_| tape.value(rng.gauss(0.0, std))).collect()).collect()
}
```

Each Python feature maps across like this:

| Python lambda feature | Rust | Why |
|---|---|---|
| `lambda nout, nin, ...` bound to a name | `fn matrix<'t>(...)` | A closure can't declare its own generic lifetime `'t`, and the return type `Mat<'t>` needs one. A `fn` can. |
| `std=0.08` default | `let std = 0.08;` in the body | Rust has no default arguments. Every call site used the default, so it became a local constant. |
| captures the `random` global | `rng: &mut Rng` parameter | There's no global RNG. Mutable state is passed in explicitly. |
| captures the `Value` class | `tape: &'t Tape` parameter | Creating a node needs the tape, and `tape.value(x)` replaces `Value(x)`. |
| outer comprehension `[... for _ in range(nout)]` | `(0..nout).map(\|_\| ...).collect()` | a closure that ignores its argument, the same as Python's `_` |
| inner comprehension `[Value(...) for _ in range(nin)]` | `(0..nin).map(\|_\| tape.value(rng.gauss(0.0, std))).collect()` | a closure that captures `tape` (shared borrow), `rng` (mutable borrow) and `std` |

That innermost closure is the real counterpart of the lambda's body.
Because it mutates `rng` through a captured `&mut`, it's an `FnMut`
closure. Python never makes that distinction; Rust's type system tracks
whether a closure reads, mutates, or consumes what it captures.

A closure version would compile inside `main`:

```rust
let mut matrix = |nout, nin| (0..nout).map(|_| (0..nin).map(|_| tape.value(rng.gauss(0.0, 0.08))).collect::<Vec<_>>()).collect::<Vec<_>>();
```

However, it would hold the `&mut rng` borrow for as long as `matrix` is
alive, which blocks `rng.shuffle` and `rng.choices` until its last use.
It also couldn't be called from `StateDict::init` without being passed in
as a generic `impl FnMut` parameter. A plain `fn` with explicit parameters
is simpler.

### Other Python constructs that act like lambdas

**`def build_topo(v)` inside `backward`.** This nested function captures
`visited` and `topo` from the enclosing scope and calls itself
recursively. The Rust port has no equivalent, because the tape makes the
depth-first search unnecessary. If Rust did need it, a closure couldn't
call itself directly, so it would be written as a nested `fn` taking
`&mut visited` and `&mut topo` as arguments.

**Comprehensions and generator expressions**
(`sum(wi * xi for wi, xi in zip(wo, x))`, `[t + p for t, p in zip(...)]`)
are anonymous per-element functions. They make up almost all of the
"lambda-like" code in the model, and each one becomes an iterator chain
with a closure:

| Python | Rust closure |
|---|---|
| `wi * xi for wi, xi in zip(wo, x)` | `.zip(x).map(\|(wi, xi)\| *wi * *xi)`, where the tuple pattern destructures like Python's `for wi, xi` |
| `sum(...)` | `.reduce(\|a, b\| a + b)`, a two-argument closure (Python's `sum` hides it) |
| `max(val.data for val in logits)` | `.map(\|v\| v.data()).fold(f64::NEG_INFINITY, f64::max)`, where `f64::max` is a function path passed where a closure could go |
| `[xi.relu() for xi in x]` | `.map(\|xi\| xi.relu())` |
| `attn_logits = [sum(...) / head_dim**0.5 for t in range(...)]` | `(0..t_len).map(\|t\| { let k_h = &keys[li][t][..]; ...; dot / ... })`, a multi-statement closure body that borrows `keys` and `q_h` from `gpt`'s scope |

Rust closures differ from Python lambdas in a few ways:

- **Capture mode is inferred and checked.** A closure borrows (`&`),
  mutably borrows (`&mut`) or moves each captured variable. The attention
  closure only reads `keys`, so it takes a shared borrow; that borrow ends
  before `gpt` pushes to the cache on the next call. A Python lambda just
  holds references to everything.
- **Closures can contain statements.** Python's `lambda` is limited to a
  single expression, which is why Python uses comprehensions for anything
  longer. Rust closures take full `{ ... }` blocks, so the Rust code puts
  multi-line logic in them.
- **No runtime cost.** Each closure has its own type and gets inlined into
  the iterator loop, so `linear` compiles to a plain nested loop, with no
  function-object call per element as in CPython.

### Closures in the tests

The tests (`main.rs:489–580`) use closures in the same style as the
Python lambda:

- `let eval = |delta: f64| { ... f(&vs).data() }` captures `xs`, `i` and
  the function `f`, a small helper named with `let` just like
  `matrix = lambda ...`.
- `let run = || { ... }` captures `tape`, `a`, `b` and `num_params`, and
  is called twice to compare gradients.

The test expressions (`expr_add_mul` and others) are named `fn`s, not
closures. That's because `grad_check` takes
`for<'t> fn(&[Value<'t>]) -> Value<'t>`, a function pointer that must work
for any tape lifetime. A closure can't be generic over a lifetime, which
is the same reason `matrix` is a `fn`.
