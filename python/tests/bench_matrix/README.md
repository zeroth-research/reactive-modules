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
| `houdini-vampire`, `vampire` | a Vampire binary (the release zip, unpacked) | `VERITH_VAMPIRE` |
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
| `hybrid` | 26 | [`hybrid/cases.py`](hybrid/cases.py) | both, over 8 sampled hybrid systems |
| `petri` | 35 | [`petri/cases.py`](petri/cases.py) | both, over 13 nets and 2 extensions |

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

#### hybrid — continuous state, discrete time

Eight textbook hybrid systems under a first-order Euler step, which is what
makes each one a reactive module: **Real** state, discrete time,
piecewise-affine updates. A thermostat with hysteresis, the ARCH-COMP leaking
tank (and the same tank against an adversarial consumption disturbance), the
Fehnker–Ivančić two-tank and room-heating benchmarks, a bouncing ball, an
adaptive cruise controller, and Alur's reactor rod control. Each states where
it comes from in its own docstring.

Three conventions are worth knowing before adding to [`hybrid/`](hybrid/) —
the second is about the benchmarks, the other two about what compiles:

- **Every Real constant is a dyadic rational.** A wire's value is a float32
  tensor, so `0.9` reaches the certificate as `0.8999999761581421`, and the
  property is then about a constant nobody wrote. Eighths and quarters are
  exact; tenths are not.
- **The controllers sample.** A mode is read off the *current* state, so every
  one of these overshoots its own setpoints by a tick — the tank's controller
  switches at 6 and 12 and the level covers [5.5, 14.5]. That gap is not a
  modelling slip, it is what the benchmarks are about, and it is why the safe
  band in a row is wider than the guards in the module.
- **Every scaling multiplies a state variable.** `System/`'s
  scalar-to-matrix equivalence proof closes for a `Linear` term that reads a
  ctrl component and does not for one that reads a derived wire, so
  `-(v - 1)/2` is written `-v/2 + 1/2` and a saturation is distributed into
  its `ite` branches rather than applied and then scaled. A module that
  scales an intermediate value is `BUILD-FAIL` on every route and measures
  none of them — `m_bounce` and `m_cruise` both were, before being rewritten.

#### petri — nets, and four extensions

Thirteen place/transition nets live as data in
[`petri/nets.py`](petri/nets.py) and are built by
[`petri_mod.py`](petri_mod.py), one net per `$PETRI_NET` in the same shape as
`svcomp_mod.py`; a time Petri net and a hybrid Petri net, which carry clocks
and a fluid level rather than only a marking, are written out in
[`petri/`](petri/).

One tick is one attempted firing, and **which** transition fires is an
external input — so a `--safety` property is a claim about every firing
sequence rather than about one schedule, and a selection whose transition is
dead leaves the marking alone (that adds no reachable marking, and it makes
the transition relation total, which the obligation needs). Recurrence is
asked of the `-rr` variants instead, where a round-robin counter replaces the
input as a last state component: under a free input `G F p` is false for
nearly any `p`, because the input may select a dead transition for ever.

The four extensions — inhibitor, reset, transfer and continuous (real-valued,
fractionally-fired markings) — are each one field on `Trans` or one flag on
`Net`, and each carries a property that turns on it: two that hold *because*
of the arc (`inhibit/queue-bound`, `transfer/conserved`) and two classical
invariants it breaks (`reset/slots-eq`, `sem-cont/integral`). The `reset`
and `transfer` nets are the same shape with one arc different, so the pair
says what each arc costs rather than only that it exists.

**A place's next marking is a *sum* of guarded deltas, not a chain of
overrides.** Exactly one transition fires per tick, so the two are equivalent
— and only the first compiles. `System/Circ.lean` is laid out by walking back
from the block's outputs without merging a shared subterm, and each layer is
then bubble-sorted, so what that file costs is the number of distinct
output-to-input *paths*, squared. A chain reads its own running value twice
per transition and doubles that count each time: the seven-place
mutual-exclusion net reached a 131-wide layer and 800 KB of `Circ.lean`.
Summing `ite(g, d, 0)` puts constants in both branches, which is where the
walk stops, so the count is merely additive — 131 wide became 57.

Finding the wall this way is also what turned up a missing budget: the two
`*_circ_eq` theorems ran at Lean's default `maxHeartbeats` while `Scalar`'s
equivalents have long carried `2000000`. Every net above four places timed
out inside `simp` — `BUILD-FAIL` on every route, for modules nothing else was
wrong with. [`../../zrth/lean/translate/circ.py`](../../zrth/lean/translate/circ.py)
now raises it, and the same net builds `System.Circ` in 13 s.

**What is left of the wall, measured.** With the sum encoding and the raised
budget, a block up to about 70 paths wide compiles and one at 99 does not:

| net | widest layer | `System.Circ` |
|---|---|---|
| `reset`, `transfer`, `sem` | 25–34 | builds |
| `readwrite`, `inhibit`, `sem-cont` | 46–47 | builds |
| `prodcons`, `mutex` | 54–57 | builds |
| `philo` (forks only, 6 places) | 69 | builds |
| `philo` with a `thinking` place each | 99 | **fails** |
| `mutex` over the reals (7 places) | 104 | **fails** |

That is why two nets here are folded rather than written the textbook way:
the philosophers drop their redundant `thinking` places, and the continuous
net is built over the three-place `sem` rather than over `mutex`. Both folds
are behaviour-preserving on the places that remain, and both are explained in
`nets.py` where they are made. A net much larger than these does not belong
in this suite until `_circ_translate_body` merges shared subterms.

#### What `truth` rests on in these two

`Row.truth` decides whether a route's answer is *correct* — only a row whose
property fails may honestly be `REFUTED` — and a wrong one turns every honest
refutation into what looks like a route bug. The other four suites transcribe
it from something upstream that already paid for it. These two have no
upstream, so it is measured: [`test_rows.py`](test_rows.py) steps every one of
the 61 rows and compares the claim with the states the run reaches, under
`just py-test`.

That is one-sided on purpose, and [`sim.py`](sim.py) says so at length: a
`fails` row is *established* by its counterexample, a `holds` row is only
**not refuted**. Certifying a property is the matrix's own job.

Every module carries at least one property that fails — a test asserts it.
A suite of only-true properties measures half a route: nothing in it can show
that a `REFUTED` was right, so a route that answered `VERIFIED` to everything
would score perfectly on it.

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

## What the routes actually found

A `VERIFIED` says a route certified a row. It does not say **what** it
certified, and two routes that both certify one row need not have found the
same thing — which is the question the `houdini`/`houdini-vampire` column
pair exists to ask, and the one the table cannot answer.

```bash
uv run python tests/bench_matrix/compare_certs.py
uv run python tests/bench_matrix/compare_certs.py --routes houdini houdini-vampire
```

The comparison is **logical, not textual**, because the textual one is
wrong on the first row it meets: on `fbk/m_countdown/InvBase` all seven
routes derive `0 <= s0 <= 100` and they write it four ways, so a string
diff sorts them into four answers where there is one. Each invariant is
parsed back through the module's own cvc5 encoding — the same front end the
routes state their obligations in — and compared as a formula. Four
answers come out, and the ordering is the interesting part, since an
invariant that implies another is the *stronger* claim and the route that
reached it looked harder:

| | |
| --- | --- |
| `same` | one formula, however it is spelled |
| `stronger` | every pair is ordered: a chain from the tightest down |
| `mixed` | some pairs ordered, some not |
| `incomparable` | no pair is ordered: different arguments throughout |

With two answers the last three collapse to two; with three or more they do
not, and 19 of the recorded rows have three or more.

Measured over the recorded passes: of **148 rows where more than one route
found a certificate, 79 are one invariant, 64 are a chain and 5 are mixed —
and not one is wholly incomparable.** So the routes mostly differ in how
much they prove rather than in what they prove, and where they do diverge
it is a strand off a chain rather than two unrelated arguments.

On the `houdini` pair it separates 5 of 16 rows, and the implication goes
**both** ways — cvc5 stronger on two, Vampire on three — which is exactly
the "which half of the route the difference is in" that the pair exists to
show and that its verdicts, identical on all but one row, cannot.

Ranking functions are reported beside the invariants and compared only for
equality: `s0` and `2*s0` are both valid and neither is stronger, so no
implication is asked of them.

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
petri_mod.py   one adapter so verith can load any net in petri/ ($PETRI_NET)
hybrid/        8 sampled hybrid systems, a file each, + cases.py
petri/         nets.py (12 nets as data), 2 written-out extensions, + cases.py
sim.py         step a module and see whether the runs refute a row's property
test_rows.py   that check over every hybrid/petri row, under `just py-test`
run_matrix.py  generate + build every cell, record verdicts and timings
coldstart.py   what the first project in a fresh build dir costs
merge.py       fold several passes into one averaged dataset, and flag it
render.py      averaged results -> one static local HTML page
compare_certs.py  what the routes found, compared as formulas not as text
```

The page's prose is generated too: the method cards are each route's own
`summary` from `infer_route.py`, so the page cannot drift from what
`--infer` accepts, and the machine block and every timing come from
`results.json`.
