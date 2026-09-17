# The `--infer` route matrix

Every way `verith` can be handed a certificate, put to every benchmark and
property in the tree, as one static HTML page.

```bash
cd python

uv run python tests/bench_matrix/run_matrix.py     # measure  (hours, serial)
uv run python tests/bench_matrix/coldstart.py      # measure the cold start
uv run python tests/bench_matrix/render.py -o matrix.html
```

The page is a single local file — no server, no network, no build step.

---

## What is measured

One **cell** is one `(benchmark, property, route)`: one `uv run verith`
followed by one `lake build`, both timed. Those two numbers are the whole
point, so everything below exists to keep them meaningful.

`--artifacts reset` is passed on every run. Without it a route reads what a
previous run left in the project's `artifacts/`, and two identical command
lines quietly stop meaning the same thing — which is exactly the failure a
matrix cannot survive.

### The routes

The `--infer` routes of
[`zrth/lean/infer_route.py`](../../zrth/lean/infer_route.py), one column each
-- except `houdini`, which is two: the same proposals decided by cvc5
(`houdini`) or refuted by Vampire (`houdini-vampire`), so a cell where one
succeeds and the other times out says which half of the route the difference
is in. `vampire` is a third Vampire column and a different question again --
the certificate *derived* by Vampire from a template rather than proposed by
anything, which reaches one-component Int safety and refuses past that. The
two Vampire columns appear only when a binary was found.

Some routes need something from outside this repo, and the defaults in
`run_matrix.py` point at where they were found on the machine that ran it:

| | needs | env override |
|---|---|---|
| `ai`, `ai-cegis` | an Anthropic API key | read from `../../CLAUDE_KEY.txt`, or `ANTHROPIC_API_KEY` |
| `houdini-vampire` | a Vampire binary (the release zip, unpacked) | `VERITH_VAMPIRE` |
| `fbk-proveit` | a `lean-ltl-certifying` checkout | `VERITH_PROVEIT_DIR` |
| `fbk-proveit` | an `ic3ia` binary (or its build dir) | `VERITH_IC3IA` |
| `fbk-proveit` | the MathSAT Python bindings | `VERITH_MATHSAT_PY` |

The LLM routes' timings include network latency and are **not reproducible**:
a second pass may find a different certificate, or none. `VERITH_MODEL` pins
the model; unset, `verith`'s own default is used and recorded in the page.

For `fbk-proveit` the checkout needs its own `.lake/packages` symlinked to
the shared one first — [`../lean/fbk/README.md`](../lean/fbk/README.md) has
that setup. `run_matrix.py` adds the checkout to each generated project's
`lake-manifest.json` itself (it is a *path* require, so it needs no fetch;
a `lake update` would re-resolve the other thirteen packages).

### The suites

| suite | rows | source | property |
|---|---|---|---|
| `limits` | 30 | [`../limits/cases.py`](../limits/cases.py) | `--buchi`, the matrix's own reachability target |
| `fbk` | 39 | [`../lean/fbk/probes.py`](../lean/fbk/probes.py) | `--safety` |
| `tests` | 6 | [`../fixtures/`](../fixtures/) | `--buchi`, from each fixture's docstring |
| `svcomp` | 80 | [`../../benchmarks/svcomp/dsl/`](../../benchmarks/svcomp/dsl/) | both, derived — see below |

The limit matrix's 77 cases vary the *supplied* `--invariant` and
`--ranking`, which no `--infer` route reads, so they collapse onto their
(module, property, precondition) key exactly as `run_limits.py` collapses
them: 30 distinct questions, and the case names that fell together are kept
on the row.

**svcomp is the one suite whose properties are derived rather than read.**
The corpus is 57 SV-COMP termination benchmarks written for the Farkas
pipeline: `Bench.build()` returns a tuple, the package imports itself
relatively, and termination is stated as a claim over *wires*. `verith`
wants a file with a no-argument `module()` and a one-state predicate, so:

- [`svcomp_mod.py`](svcomp_mod.py) is that file, for all 57 at once — which
  one is named by `$SVCOMP_BENCH`, so a single checked-in adapter serves the
  corpus and the command a cell reports is still one that can be pasted.
- `terminates` (`--buchi`) is `G F ¬guard`. `_termination.terminates()`'s
  domain is "some column moves", and `resolve_domain` simplifies it to
  exactly the loop guard, because the encoding writes every update as
  `ite(guard, body, self)`. Leaving that domain is reaching a fixed point,
  which a run never leaves, so `G F ¬guard` *is* termination.
- `houdini-inv` (`--safety`) is the conjunction of what
  `_invariants.infer_invariants` finds — inductive by construction, so like
  the fbk suite's `inv-` probes it measures the plumbing rather than the
  search. Kept **unpruned**: it is the invariant the corpus's own inference
  produced, not a tidied one, and several rows are redundant as a result.
  Only 23 benchmarks have one. For the other 34 Houdini finds nothing, the
  conjunction is `true`, and a row for it would be verified by every route
  and say nothing — so there is none.

A column's z3 symbol is its name and `verith` indexes state by ctrl
declaration order, which `Bench.state` already is, so the rename onto
`s0..sN-1` is positional. All 57 check out.

---

## Verdicts

| | |
|---|---|
| `VERIFIED` | Lean discharged every obligation. |
| `REFUTED` | The route *disproved* the property, with a counterexample. The suites carry deliberately false controls to produce exactly this. |
| `NO-CERT` | The route searched its shape and returned nothing. For `sygus` and `smt-linear` a bounded shape that comes back empty is a **proof** that it is empty, not a search that ran out of time. |
| `GEN-FAIL` | No project was emitted at all — a codegen gap, or a shape refused up front. |
| `PROOF-FAIL` | A certificate was produced and Lean rejected it. |
| `BUILD-FAIL` | The project failed outside the certificate: one of the five encodings did not compile. |
| `SORRY` / `SORRY+FAIL` | An obligation left as `sorry`, the build otherwise clean / not. |
| `UNSUPPORTED` | The route does not take this property's kind (`sygus` and `fbk-proveit` are safety-only, `ai` Büchi-only); `verith` refused it up front. |

`NO-CERT`, `REFUTED` and `GEN-FAIL` are kept apart because they are
different measurements, and telling them apart is not a prefix match: `main`
wraps most refusals under `error: --infer:`, but a route whose row sets
`errors_self_named` names its own flag instead.

---

## Three ways this harness will lie to you

The first two are inherited from [`../limits/README.md`](../limits/README.md), which
paid for them.

**One shared build dir takes one writer, so the pass is strictly serial.**
Every generated project symlinks its `.lake` to one shared build dir, so
Mathlib, Core, LeanAI and ZerothHammer are compiled once and only
`System/*` and `Certificate/*` recompile per cell. Module names are
identical across projects, so two concurrent runs overwrite each other's
oleans — and it does not fail cleanly: it reads as heartbeat timeouts and
`unknown constant 'hrank'`, which looks exactly like a codegen bug. One
session lost an hour to five "regressions" that were another process
building into the same directory. Use `VERITH_BENCH_WORK` for anything
running alongside.

**Wall clock is only as quiet as the machine.** Another Lean build anywhere
is enough to inflate a timing several-fold and unevenly; one earlier pass
reported a case at 1996 s that re-timed at 75 s. Spotlight is the worst
offender, since a pass writes tens of thousands of `.olean` files. Hence
`.noindex` on the work directory, which is the documented way to make
Spotlight skip one — **keep the suffix** on anything passed through
`VERITH_BENCH_WORK`. Verdicts were never affected by any of this; only
timings lie. Re-time anything surprising before believing it.

**A runaway process is invisible in the numbers.** The failure above is
loud; this one is not. On 2026-09-16 a pass ran for three hours beside two
processes nobody was watching — a pytest orphaned 30 h earlier spinning in
z3, and a background waiter of a previous session busy-looping on a
condition that could never become true — plus, for part of it, a
`--infer ai-cegis` cell that the harness *believed* it had killed at the
timeout. `subprocess.run(timeout=)` signals only the process it started, so
capping `uv run verith` left the `verith` underneath it reparented to init
and running. Nothing in the results looked wrong, because a busy machine is
slower *uniformly*. Two defences now exist: `run_capped` puts each child in
its own process group and kills the group, and several passes can be
averaged (below), which is what makes such a pass detectable at all.

Before a long pass, check the machine is actually idle — orphans do not
announce themselves, and grepping for the process names you expect is how
that pytest survived a day and a half:

```bash
ps -axo pid,ppid,etime,time,%cpu,command -r | head        # who is busy
ps -axo pid,ppid,etime,%cpu,command | awk '$2==1'         # orphans
```

---

## Several passes, averaged

One pass is one sample: one machine, one afternoon, and — for `ai` and
`ai-cegis` — one set of answers from a model that need not give the same
ones twice. A repeat measurement is therefore a **new file**, not an
amendment to the old one:

```bash
uv run python tests/bench_matrix/run_matrix.py --results $VERITH_BENCH_WORK/run-2.json
uv run python tests/bench_matrix/render.py -r $VERITH_BENCH_WORK/run-{1,2}.json -o matrix.html
```

`render.py` takes any number of `-r` files and averages each cell across
them, showing the mean with its spread (`28.3 ±0.6`) and the individual
readings on hover. What is *not* averaged is as important:

- only runs that **agree on the verdict** contribute seconds, so a
  `TIMEOUT`'s 300.0 — the budget, not a measurement — never lands in a mean
  beside a completed run;
- a cell whose passes disagreed shows the modal verdict with a `!`, and its
  panel names the split. No mean can hide a non-deterministic route.

`merge.py` also raises warnings, printed to stderr and rendered onto the
page under *What these numbers do not support*. High severity means the
passes arguably should not be averaged at all:

| warning | severity | what it means |
| --- | --- | --- |
| `machine-mismatch` | high | different machines; the mean describes neither |
| `model-mismatch` | high | different model behind `ai`/`ai-cegis` |
| `gen-timeout-mismatch`, `build-timeout-mismatch` | high | different budgets, so different censoring |
| `overlapping-passes` | high | two passes ran concurrently, so both measured a loaded machine |
| `verdict-disagreement` | high | the same pair answered differently on different days |
| `high-spread` | low | agreeing runs differ by >30 % of their mean |
| `near-budget` | low | within 20 % of a timeout; another pass may record `TIMEOUT` |
| `uneven-coverage` | low | measured in fewer passes than were given |
| `unstamped-runs` | low | no `when`, so the overlap check cannot see them |
| `single-sample` | low | one pass; nothing is averaged |

`--strict` exits non-zero if any high-severity warning was raised, for a
pass whose page is generated unattended. The page is still written.

---

## Re-running part of it

`results.json` is written after every cell and a pair already in it is
skipped, so a pass is resumable and one route can be re-measured without
disturbing its neighbours:

```bash
uv run python tests/bench_matrix/run_matrix.py --suites limits tests
uv run python tests/bench_matrix/run_matrix.py --routes nuterm sygus
uv run python tests/bench_matrix/run_matrix.py --only Countdown --redo
uv run python tests/bench_matrix/run_matrix.py --no-build        # generation only
uv run python tests/bench_matrix/suites.py                      # just count the rows
uv run python tests/bench_matrix/run_matrix.py --reverdict       # reclassify, no re-measuring
uv run python tests/bench_matrix/run_matrix.py --prune           # forget rows suites.py dropped
uv run python tests/bench_matrix/run_matrix.py --pairs FILE      # exactly these, e.g. after a fix
```

`--reverdict` recomputes every recorded verdict from the output already
stored, so correcting a classification rule costs a second rather than a
re-measurement. `--results PATH` picks which file is written and resumed
(and `--label` what the page calls that sample); it must not run while
another pass is writing the same file.

Generated projects are deleted as they are measured — the olean cache they
matter for is the shared one, which is kept. Everything lives under
`$VERITH_BENCH_WORK` (default `/tmp/verith-bench.noindex`), never in the
tree; the rendered page is the only output that belongs in it.

## Layout

```
suites.py      every (benchmark, property) row, and the route column
svcomp_mod.py  one adapter so verith can load any SV-COMP bench ($SVCOMP_BENCH)
run_matrix.py  generate + build every cell, record verdicts and timings
coldstart.py   what the first project in a fresh build dir costs
merge.py       fold several passes into one averaged dataset, and flag it
render.py      averaged results -> one static local HTML page
```

The page's prose is generated too: the method cards are each route's own
`summary` from `infer_route.py`, so the page cannot drift from what
`--infer` accepts, and the machine block and every timing come from
`results.json`.
