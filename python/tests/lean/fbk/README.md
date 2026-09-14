# `--fbk-proveit` experiments — cold start

`verith --fbk-proveit` certifies a module through the `proveit.py` driver of
[`lean-ltl-certifying`][ltl] instead of verith's own invariant machinery:

```
module.py ──verith──▶ <Proj>/ProveIt/<Proj>NA.lean     (NA encoding)
                            │
                            ▼   proveit.py
                  lean2vmt ─▶ ic3ia ─▶ vmt2lean
                            │
                            ▼
                  <Proj>/Certificate/ProveItCert.lean
```

This directory reproduces the sweep that measured it.  Everything here is
measurement only — no fixture in this repo depends on it, and nothing runs
in CI, because it needs three things that are not vendored: MathSAT, ic3ia
and a `lean-ltl-certifying` checkout.

| file | role |
|---|---|
| `probes.py` | the probe table: `(module, safety property, expected verdict)`, plus the modules the encoding refuses and why |
| `run_fbk.py` | the runner — strictly sequential, one probe at a time |
| `lean-smt-bug.md` | a reproducer for the one upstream bug the sweep found in `lean-smt`'s proof reconstruction |

---

## What you need

### 1. The fixture modules (`--mods`)

The probes run on the limit-probe fixture set — the `m_*.py` modules of
[`tests/limits/`](../../limits/README.md), which is in the repo at
`tests/limits/mods`.  Any directory of `m_*.py` files exposing `module()`
will do; the probe table names 11 of them.

### 2. MathSAT + its Python bindings

`vmt2lean.py` starts with `from mathsat import *`, and the bindings are a
compiled extension, so they must be built against the *same interpreter*
that will run `verith` — a `PYTHONPATH` pointing at a differently-versioned
build will not load.

```sh
# https://mathsat.fbk.eu/download.html  -> unpack to $MSAT
cd $MSAT/python
uv run --with setuptools python setup.py build_ext --inplace \
    -I$(brew --prefix gmp)/include -L$(brew --prefix gmp)/lib
export PYTHONPATH=$MSAT/python
uv run python -c 'import mathsat; print("ok")'
```

`--inplace` drops `_mathsat.<abi>.so` next to the `mathsat.py` that ships
with MathSAT, so that one directory is the whole `PYTHONPATH` and it
survives a reboot.  Building to a scratch directory works just as well
until the directory is cleaned up, and then the failure looks like a
MathSAT that was never built.  The linker prints a wall of
`was built for newer 'macOS' version` warnings; they are harmless.

The `.so` is tied to the interpreter that built it, so build it with the
same `uv run` that will run `verith` -- a `PYTHONPATH` pointing at a
build for another Python version will not load, and `verith` will report
exactly that.

`verith` checks this before it does anything else and aborts with the fix if
the interpreter cannot import `mathsat`.

### 3. ic3ia

```sh
# https://es-static.fbk.eu/people/griggio/ic3ia/
cd ic3ia && mkdir build && cd build
cmake .. -DMATHSAT_DIR=$MSAT -DCMAKE_BUILD_TYPE=Release && make
export IC3IA=$PWD/ic3ia
```

### 4. `lean-ltl-certifying`, on **v4.28.0**

Clone [`proof-prototyping`][ltl] and use `lean-ltl-certifying/`.  It has to
be on the same toolchain as the generated projects; if the checkout still
says `v4.27.0`, port it the way commit *"lean-ltl-certifying: port to Lean
v4.28.0"* did — `lean-toolchain` to `v4.28.0`, and the requires to `cslib
v4.28.0` + `smt f58d19d…`, with Mathlib inherited through cslib rather than
required directly.

That last point is what makes the cold start cheap: the package set then
matches this project's exactly, so **one already-built `.lake/packages`
serves both** and Mathlib is never rebuilt.

```sh
cd <lean-ltl-certifying>
mv .lake .lake.old 2>/dev/null; mkdir -p .lake
ln -s <repo>/python/tests/lean/.lake/packages .lake/packages
cp <repo>/python/tests/lean/lake-manifest.json .
lake build LTLCertifying.Safety.Lemmas lean2vmt      # ~80 s cold, ~2 s after
```

The symlinked `packages` is read-only in practice — the sweep only ever
writes to `lean-ltl-certifying`'s own `.lake/build`.  Without the symlink,
`lake exe cache get` fetches several GB instead.

---

## Running it

```sh
cd python

# no Lean, no ic3ia, no LTL checkout: which modules does the encoding accept?
uv run python tests/lean/fbk/run_fbk.py --mods <mods> --screen

# the full sweep, 39 probes, one at a time
PYTHONPATH=$MSAT/python uv run python tests/lean/fbk/run_fbk.py \
    --mods <mods> --ltl <lean-ltl-certifying> --ic3ia $IC3IA \
    --json /tmp/fbk.json    # results table, for diffing against a later run

# one group, route only (no Lean check) — about a third of the wall clock
PYTHONPATH=$MSAT/python uv run python tests/lean/fbk/run_fbk.py \
    --mods <mods> --ltl <ltl> --only '^Ni' --no-check

# the *limit matrix's* own invariants through this route, as a measurement
# rather than a pass/fail table (see below)
PYTHONPATH=$MSAT/python uv run python tests/lean/fbk/run_fbk_limits.py \
    --ltl <lean-ltl-certifying> --ic3ia $IC3IA --json /tmp/fbk-limits.json
```

**Do not run probes in parallel.** They share one `lean-ltl-certifying`
build directory and lake takes one writer; concurrent runners serve each
other's oleans, and the symptom is a type error inside a generated module
rather than a clean failure.

Roughly 18 s per certified probe: ~5 s route (of which ~1.6 s is a no-op
`lake build` of the model's imports) and ~12 s for Lean to check the
certificate.  `--no-check` drops the second.  Exit status is non-zero if any
probe disagrees with its `expect`.

---

## Reading the results

A verdict that differs from `expect` is the point of the table.  As of the
last sweep, 39 probes: 32 `certified`, 4 `unsafe`, 1 `unknown`, 1 `abort`,
1 `lean-fail`.

### On the choice of properties

The `P` column of the `tests/limits` case matrix is **not** usable here.
That `P` is a reachability target for verith's own certificate, which proves
`inv` + `ranking` ⇒ `P` is reached; this route proves `□ PROPERTY`.  They are
different properties.  Countdown starts at 100, so `□(x = 0)` is false at
step 0 and ic3ia is right to reject it — a `P`-driven sweep reports UNSAFE
for essentially every case and measures nothing.

What transfers is each case's **invariant**, the thing meant to hold always.
Those are inductive by construction, though, so they exercise the plumbing
and not the model checker.  `probes.py::HAND_WRITTEN` therefore adds
properties written for this route: true-but-not-inductive (ic3ia has to
synthesise the bound), relational over two state variables, nonlinear,
`ite`-in-the-property, and false-but-only-refutable-after-100-steps.
`probes.py::HARDER` goes one step further — a coupling between two state
components, a parity argument asked as an implication, a linear combination
of two counters, three-way band splits — and is what
[`lean-smt-bug.md`](lean-smt-bug.md)'s narrowing came out of: `Step2Odd`
and `NiS2Odd3` ask the same parity question of the same module and land on
opposite sides of the `sum_ub` bug.

### `run_fbk_limits.py`: the whole limit matrix, as a measurement

`run_fbk.py` is a pass/fail table — every probe carries the verdict it
should get.  `run_fbk_limits.py` is the other thing: it takes all 77 cases
of `tests/limits/cases.py`, feeds each case's `--invariant` to `--safety`,
and reports what happens, with no expectation attached.  Distinct
`(module, invariant)` pairs only, since 23 countdown cases share one
invariant.

The number it exists to produce is how many cases never reach ic3ia.  Last
run: 45 pairs covering 75 cases — 30 pairs (**53 cases**) certified, 12
pairs (18 cases) refused by `check_na_supported` before the model checker,
2 `lean-fail` (3 cases), and 1 `unsafe`: `BadInv`, whose invariant is
deliberately not inductive — the case matrix's own negative control,
refuted here with a counterexample in 5 s.

Of the 18 refusals, 11 are `Real` state, 4 external input wires, 2 an op
`smt_encode` has no term for (`Transpose`, `Uninterpreted`), and 1 a
property outside the Bool fragment (`mod`).

The previous run of the same sweep certified 40 cases and refused 30.  What
moved is the multi-element ctrl wire restriction and the op allowlist that
went with it: `m_mixed`, `m_relu_vec`, `m_relu_net`, `m_relu_net8`,
`m_relu_net16`, `m_max`, `m_min` and `m_argmax` all certify now, and the
two `m_vec32` cases changed from `abort` to `lean-fail` — they encode, and
ic3ia proves them, but checking the certificate for a 32-slot model runs
out of Lean heartbeats (200000) in `whnf`.  The `m_relu` cases changed from
`lean-fail` to `certified`, because the transition no longer spells a ReLU
`Max.max`.

### What the sweep found

Two of the four obstacles it turned up have since been fixed; these are
the standing ones.

1. **`mod` stops at the witness, not the model** (`InvMod`, `abort`).
   `lean2vmt` does translate `%`, and ic3ia proves such a property — but
   MathSAT eliminates the mod from the *witness*, returning
   `x + (-2) * to_int ((1/2) * to_real x) = 0`, and `vmt2lean.py` renders
   neither `to_real`/`to_int` nor a Real inside a `Bool`-valued `INVAR`.
   `smt_to_lean_bool` therefore refuses `INTS_MODULUS` at the front of the
   pipeline, where the message can say why rather than dying inside
   `vmt2lean`.

2. **`lean-smt` miscompiles a `sum_ub` proof** (`NiS2Odd3`, `lean-fail`).
   The step returns an equality where a `≤` is wanted, and the kernel
   rejects the term.  Standalone reproducer in
   [`lean-smt-bug.md`](lean-smt-bug.md), narrowed there to the exact
   shape: an equality *disjunct* in the hypothesis, a *disequality* goal,
   and that disequality on the variable the linear equation defines.  The
   VMT and the certificate are correct; only the tactic fails.  `Step2Odd`
   asks the same parity question of the same module without a disequality
   and certifies, which is both the evidence for the narrowing and the
   workaround.

   The file's *second* symptom — `Max` tripping universe-level
   bookkeeping — no longer fires on this route: the NA encoding takes its
   transition from `smt_encode`, where a ReLU is an `ite`, so no `Max`
   reaches Lean.  `ReluTrans` certifies.  The upstream bug is unfixed.

3. **Nonlinear ⇒ `unknown`** (`NonlinLexMul`).  Expected: ic3ia is IC3 with
   implicit predicate abstraction over *linear* arithmetic.  The route
   aborts cleanly.

Fixed since: a trivially safe property produced an uncompilable
certificate (two `vmt2lean` template bugs plus a tactic that ran past a
closed goal), and `Ne`/`ReLU`/`Max`/`Min` were emitted as unapplied
leaves by `lean2vmt`.  `InvTrue` and `MixedBoolInt` cover the first,
`ReluTrans` the second.

[ltl]: https://github.com/zeroth/proof-prototyping
