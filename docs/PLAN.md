# microgpt-rs — Implementation Plan

Milestones are ordered so every one ends with something runnable and
checkable. Total scope is a few hours of work; the risk is concentrated in
M2 (autograd), so it comes first after scaffolding and is tested in
isolation before any transformer code exists.

## M0 — Scaffold (15 min)

- `cargo new microgpt-rs`; empty `[dependencies]`; commit `rustfmt` defaults.
- `docs/` gets these three documents.
- **Done when:** `cargo run` prints hello, `cargo clippy` clean.

## M1 — RNG + dataset + tokenizer (30 min)

- `Rng` (splitmix64 core, `uniform`, `gauss` via Box–Muller, `shuffle`,
  `choices`).
- `input.txt` bootstrap (curl fallback), doc loading, shuffle, tokenizer.
- **Done when:** prints `num docs: 32033` and `vocab size: 27` against the
  real dataset; a 10k-sample histogram of `gauss(0, 1)` has mean ≈ 0,
  std ≈ 1 (throwaway assertion, then delete).

## M2 — Autograd tape (1–1.5 h) ← the risk milestone

- `Tape`, `Node`, `Value`, all `std::ops` impls, `powf/log/exp/relu`,
  `backward`, `truncate`.
- **Verification: gradient checking.** For ~20 random small expressions
  (nested add/mul/div/exp/log/relu), compare `backward()` grads against
  central finite differences `(f(x+h)-f(x-h))/2h`, tolerance 1e-6. Write
  this as `#[cfg(test)]` unit tests — they stay in the file.
- Also test: parameter slots survive `truncate`; second forward pass after
  truncate produces correct grads (catches stale-grad bugs).
- **Exit criterion:** all grad checks pass. Do not start M3 until they do —
  every downstream bug would otherwise be indistinguishable from an autograd
  bug.

## M3 — Model forward pass (1 h)

- `StateDict` init (4,192 params — assert the count), `linear`, `softmax`,
  `rmsnorm`, `gpt` with KV cache.
- **Done when:** with random weights, one forward pass over a short doc
  yields loss ≈ ln(27) ≈ 3.30 (the untrained-uniform sanity check), and
  `softmax` outputs sum to 1 ± 1e-9.

## M4 — Training loop + Adam (45 min)

- Per-doc loss, `backward`, Adam with bias correction and linear LR decay,
  `truncate(P)`, progress line.
- **Done when:** 1000 steps run in seconds; loss falls from ~3.3 into the
  ~2.0–2.2 band (Python parity); loss is bit-identical across two runs
  (determinism check).
- **Debug tactic if loss diverges:** shrink to 1 doc / 10 steps and diff the
  loss trajectory against the Python file patched to use fixed weights
  (dump/load a tiny weight file for both) — this isolates math errors from
  RNG differences.

## M5 — Sampling + polish (30 min)

- 20 samples at T=0.5, matching the Python output format.
- Pass over comments: port the narrative voice, add the two Rust-specific
  teaching notes (tape order = topological order; truncate = graph free).
- README: what it is, how to run, expected output, pointer to Karpathy's
  original, line-by-line correspondence note.
- **Done when:** `/verify`-style end-to-end run — fresh clone, `cargo run`,
  names look like names (`kaylee`, `jaxon`-ish, not `qqzzx`).

## M6 (optional, only if demo wants a wow moment)

- `--release` vs Python timing table in the README (expect ~50–100×).
- A `perf` note: where the time goes (it's all in `Vec<Node>` push and the
  backward sweep) — reinforces "everything else is just efficiency."

## Order of risk retirement

1. **Autograd correctness** — retired in M2 by finite-difference tests.
2. **Model math transcription errors** — retired in M3 by the ln(27) check
   and in M4 by the loss-band check.
3. **RNG quality** (bad gauss → bad init → training stalls) — retired in M1
   by the moment check; splitmix64 is well within what this needs.
4. **Ergonomics of `Value<'t>` lifetimes infecting signatures** — accepted:
   every function is generic over one lifetime `'t`; if it gets noisy, a
   type alias per milestone review, but do not switch to `unsafe` or
   thread-locals to hide it.

## Explicitly out of scope

Tensors, SIMD, rayon, GPU, CLI args, checkpointing, other datasets, any
crate. Each of these would make the demo better software and worse pedagogy.

## Deliverables checklist

- [ ] `microgpt-rs/src/main.rs` (≤ ~400 lines, zero deps)
- [ ] grad-check unit tests inline (`cargo test`)
- [ ] README with expected output and Python↔Rust correspondence table
- [ ] these three docs moved into the repo under `docs/`
