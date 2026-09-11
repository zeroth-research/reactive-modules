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

The probes run on the limit-probe fixture set — the `m_*.py` modules behind
[`VERITH_LIMITS.md`](../../../zrth/lean/VERITH_LIMITS.md).  They are not in
the repo; they live in the harness scratch directory that produced that
file.  Any directory of `m_*.py` files exposing `module()` will do; the
probe table names 11 of them.

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

# the full sweep, ~30 probes, one at a time
PYTHONPATH=$MSAT/python uv run python tests/lean/fbk/run_fbk.py \
    --mods <mods> --ltl <lean-ltl-certifying> --ic3ia $IC3IA \
    --json /tmp/fbk.json    # results table, for diffing against a later run

# one group, route only (no Lean check) — about a third of the wall clock
PYTHONPATH=$MSAT/python uv run python tests/lean/fbk/run_fbk.py \
    --mods <mods> --ltl <ltl> --only '^Ni' --no-check
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
last sweep, 29 probes: 23 `certified`, 3 `unsafe`, 1 `unknown`, 1 `abort`,
2 `lean-fail`.

### On the choice of properties

The `P` column of the `VERITH_LIMITS.md` case matrix is **not** usable here.
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

2. **`lean-smt` miscompiles two kinds of proof** (`NiS2Odd3`,
   `ReluTrans`, both `lean-fail`).  A `sum_ub` step returns an equality
   where a `≤` is wanted, and `Max` trips universe-level bookkeeping.  Two
   standalone reproducers in [`lean-smt-bug.md`](lean-smt-bug.md).  In both
   cases the VMT and the certificate are correct and only the tactic
   fails — neither is verith's or `lean-ltl-certifying`'s.

3. **Nonlinear ⇒ `unknown`** (`NonlinLexMul`).  Expected: ic3ia is IC3 with
   implicit predicate abstraction over *linear* arithmetic.  The route
   aborts cleanly.

Fixed since: a trivially safe property produced an uncompilable
certificate (two `vmt2lean` template bugs plus a tactic that ran past a
closed goal), and `Ne`/`ReLU`/`Max`/`Min` were emitted as unapplied
leaves by `lean2vmt`.  `InvTrue` and `MixedBoolInt` cover the first,
`ReluTrans` the second.

[ltl]: https://github.com/zeroth/proof-prototyping
