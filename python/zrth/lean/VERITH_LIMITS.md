# What `uv run verith` can verify

Measured, not guessed: 76 modules were put through the whole pipeline — Python
module → generated Lean project → `lake build` — and the outcome of each was
recorded. This file says which classes of module, invariant and ranking
function come out proved, which come out broken, and for the broken ones what
gives way.

Companion documents: [`GENERATED.md`](GENERATED.md) describes *what* the
generator emits, [`README.md`](README.md) *how* the translation works, and
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) is the running defect catalogue. This one
is about the capability envelope.

---

## Method

Each case is a full project (`verith … -o <dir> -p Rea`), because that is what
users get and because the standalone `--cert-file` path skips four of the six
encodings.

Two build targets answer two independent questions, and they must be read
separately:

| target | question | imports |
|---|---|---|
| `System` | do all six encodings compile? | functional, Circ, Rel, Scalar, ScalarRel, FBK |
| `Certificate` | do `init_inv`, `step_inv` and `hrank` discharge? | only `System.System` + `System.Data` |

A broken Scalar or FBK encoding therefore does **not** block the certificate,
so the two must be read apart: until open issue 1 was fixed, five cases had a
verifying certificate inside a project whose `lake build` failed. No case is in
that state now — every remaining failure is a certificate failure.

There is now a third question, and it is answered in milliseconds rather than
seconds: `--pre-check cvc5` says whether the obligation is *true* at all. Run
it first on any case that fails — a failing `lake build` cannot tell a wrong
certificate from weak tactics, and more than one entry below was the former.
See [SMT_ASSIST.md](SMT_ASSIST.md).

The matrix itself, the runners and this file's baseline are in
[`tests/limits/`](../../tests/limits/README.md).

### Running it in seconds per case

A fresh project would refetch and rebuild Mathlib. Instead, point every
project's `.lake` at one shared build directory whose `packages` is the
already-built package set of `tests/lean`:

```sh
mkdir -p /tmp/shared/.lake/build
ln -s <repo>/python/tests/lean/.lake/packages /tmp/shared/.lake/packages

uv run verith mymodule.py --buchi … --invariant … --ranking … -o /tmp/case1 -p Rea
ln -s /tmp/shared/.lake /tmp/case1/Rea/.lake
cp <repo>/python/tests/lean/lake-manifest.json /tmp/case1/Rea/
cd /tmp/case1/Rea && lake build
```

Give every project the same package name (`-p Rea`) and `Core`, `LeanAI` and
`ZerothHammer` are byte-identical across them, so they build once and hit the
cache afterwards; only `System/*` and `Certificate/*` recompile. Measured:
**~9 s per project**, 20 MB of shared build output, Mathlib never rebuilt.

**One shared build directory takes one writer.** `System.System`, `System.Data`
and the rest have the same module names in every project, so the olean for
`System.System` in the shared `build/` belongs to whichever project compiled
last. Two sweeps running at once silently serve each other's modules, and the
symptom is not a clean failure: it is `unknown constant 'c0'` and type
mismatches inside `System/Rel.lean` or `System/FBK.lean`, which read exactly
like a codegen bug. Give each concurrent runner its own `build/` (the
`packages` symlink can still be shared — Mathlib is read-only).

---

## Results

75 of 76 cases generate; 75 compile all six encodings; 64 certificates discharge. Of the 11 that
do not, **8 are negative controls that are supposed to fail** — a missing
property, a non-inductive invariant, a constant or increasing ranking, a
dropped precondition. That leaves 3 real limits (`OpArgmax`, `OpMax`, `OpMin`),
all in the next section.

Against the pipeline as it stood before this work, measured on the 36 cases
that existed then (the other 40 — eight neural-network cases, three built to
break a fixed tactic chain, one negative control, and a 28-case batch built
to find where neural certificates stop working — were added along the way):

| | before | after |
|---|---|---|
| generates | 33 / 36 | **35 / 36** |
| `System` compiles | 20 / 36 | **35 / 36** |
| `Certificate` discharges | 15 / 36 | **24 / 36** |
| whole project green | 8 / 36 | **24 / 36** |

Four of those cases were also *corrected* along the way, and the gain is
theirs as much as the tool's: the three Real ones asked for an interval
invariant, which is not inductive over the reals (x = 0.5 escapes
`0 ≤ x ≤ 5`), and `InvImplies` was given a ranking that could never
decrease. A fifth, `RealConjDisj`, was wrong in the same family and is
dissected in open issue 3. A wrong test failing is not a tool limit.

Over the full 76-case matrix: 75 generate, all 75 compile all six encodings,
64 certificates discharge, and the same 64 projects are green end to end —
`System` no longer fails anywhere, and every remaining failure is a
certificate failure. Of the twelve, eight are negative controls, one is the
`Uninterpreted` codegen gap, and **the only three real limits left are
`OpMax`, `OpMin` and `OpArgmax`, which are all the same defect**: a matrix
fold that never reduces.

That diagnosis has since been confirmed from the outside. The
`--fbk-proveit` route now encodes all three modules -- it takes its
transition from `smt_encode` rather than from the scalar Lean printer, so
`Linear` and `Argmax` arrive as arithmetic -- and ic3ia certifies a safety
property for each. The modules are fine, the SMT encoding of them is fine,
and what is left is the Lean-side fold: exactly what `--pre-check cvc5`
already said when it discharged all three obligations in 4.6, 4.2 and
3.2 ms.

No case that verified at any earlier point in this work stopped verifying.
Each change was measured against the whole matrix in an isolated build
directory: the fixed-chain-to-generated-plan switch reproduced all 44 shared
verdicts exactly, the FBK fix moved five cases, the state-enumeration step
moved `BVState`, the `min`/`max` folding moved eleven, and the Real budget
rule moved two — each time with every other verdict unchanged.

| case | what it probes | gen | System | Certificate |
|---|---|---|---|---|
| `Countdown` | LIA 1x1, conjunctive bound, Ite ranking — the known-good control | ok | ok | ok |
| `TwoVars` | two 1x1 wires, relational invariant, difference ranking | ok | ok | ok |
| `NoCert` | no property at all: what does a bare `verith` project contain? | ok | ok | **fail** |
| `PropOnly` | `--buchi` but no --invariant/--ranking: inv defaults to True, ranking to sorry | ok | ok | **fail** |
| `InvTrue` | trivially inductive invariant; hrank has no bound to work with | ok | ok | **fail** |
| `InvDisj` | 6-way disjunctive invariant — needs the casesm* _ v _ branch | ok | ok | ok |
| `InvMod` | parity invariant: INTS_MODULUS through smt_to_lean, then omega | ok | ok | ok |
| `InvNe` | DISTINCT in an invariant | ok | ok | ok |
| `InvIte` | Ite in Prop position over a Bool state component | ok | ok | ok |
| `InvImplies` | Implies in a Prop position — same module and ranking as InvIte, so the only difference from that case is the connective | ok | ok | ok |
| `InvMixed` | 1x1 + 3x1 state: wire index and flat slot disagree; x is unbounded | ok | ok | ok |
| `RankConst` | constant ranking — hrank needs 0 < 0 | ok | ok | **fail** |
| `RankLinear` | bare linear ranking, no Ite guard | ok | ok | ok |
| `RankLex` | nested loops; lexicographic (y,x) folded into one Nat | ok | ok | ok |
| `RankQuadratic` | nonlinear but correct ranking — omega/linarith territory | ok | ok | ok |
| `RankToInt` | non-integral Real state (steps of 0.5): to_int has to scale | ok | ok | ok |
| `Relu` | ReLU in the transition: x' = relu(x-1) becomes Max.max 0 (x-1) | ok | ok | ok |
| `ReluRankRelu` | ReLU-shaped ranking (no max kind in smt_to_lean, so Ite) | ok | ok | ok |
| `ReluInvRelu` | ReLU-shaped invariant: `x = relu(x)` written as an Ite | ok | ok | ok |
| `ReluVec` | element-wise ReLU on a 3-vector state | ok | ok | ok |
| `ReluLRA` | ReLU over Real — noncomputable RM, linarith instead of omega | ok | ok | ok |
| `ReluNet` | Linear -> ReLU -> Linear, the shape a small Q-network compiles to | ok | ok | ok |
| `ReluInput` | ReLU + external input, sound under the precondition | ok | ok | ok |
| `ReluInputNoPre` | same module without --pre: e is unconstrained, ranking cannot decrease | ok | ok | **fail** |
| `BoolState` | Bool state (2-bit counter), Bool->Int ranking via Ite | ok | ok | ok |
| `BVState` | BitVec state: omega does not model BitVec, and a branch that is contradictory only because a 1-bit vector has two values needs the state enumerated | ok | ok | ok |
| `LRALinear` | Real state, no ReLU — an interval invariant is not inductive over the reals, so the invariant pins exact values | ok | ok | ok |
| `OpMax` | Max as a unary reduction: x' = max(x-1, 0), i.e. ReLU spelled Max | ok | ok | **fail** |
| `OpMin` | Min as a unary reduction: x' = min(x+1, 5) | ok | ok | **fail** |
| `OpArgmax` | Argmax in the transition — does argmax_1d reduce under the hammer? | ok | ok | **fail** |
| `OpTranspose` | Transpose in the transition (MatTranspose / Box.transpose) | ok | ok | ok |
| `OpUninterp` | an uninterpreted symbol — no Lean counterpart exists | **fail** | — | — |
| `Vec32` | 32-wide state, all six encodings — the scaling ceiling | ok | ok | ok |
| `Deep64` | 64-deep straight-line transition body | ok | ok | ok |
| `NNRank` | ranking is a 2-unit ReLU net: 2·relu(x-1) + relu(x) | ok | ok | ok |
| `NNRankWide` | ranking is a 3-unit ReLU net with distinct thresholds | ok | ok | ok |
| `NNRankDeep` | ranking is a two-hidden-layer ReLU net | ok | ok | ok |
| `NNInv` | invariant is a ReLU net: relu(x) + relu(9-x) = 9, i.e. 0 ≤ x ≤ 9 | ok | ok | ok |
| `NNInvIneq` | net invariant in inequality form — the shape a learned barrier/Lyapunov function takes | ok | ok | ok |
| `NNBoth` | invariant and ranking both nets, over the same state | ok | ok | ok |
| `NNTwoInput` | two-input net invariant: relu(y-x) + relu(x) = y | ok | ok | ok |
| `NNNetModule` | the module is Linear->ReLU->Linear *and* both certificate predicates are nets — the fully neural case | ok | ok | ok |
| `RealConjDisj` | Real, invariant is a conjunction of two 4-way disjunctions: omega does not apply and `constructor <;> linarith` cannot prove a disjunct, so the plan has to case on both | ok | ok | ok |
| `RealConjDisjUnsat` | two Real components that both *cycle*: the product of their ranges admits (1,2)->(2,5)->(3,2)->(0,5)->(1,2), which never reaches P, so hrank is false for every ranking. Must be rejected | ok | ok | **fail** |
| `RealNonlin` | Real *and* nonlinear: needs nlinarith over an ordered field, not omega | ok | ok | ok |
| `RealNet` | a ReLU net over Real as the ranking: ite branches, real literals and a floor, all at once | ok | ok | ok |
| `BadInv` | not inductive (init is 100) — init_inv must fail | ok | ok | **fail** |
| `BadRankDir` | ranking increases along the transition — hrank must fail | ok | ok | **fail** |
| `NN2Width4` | 4-unit ranking net, distinct thresholds — 8 ite in hrank | ok | ok | ok |
| `NN2Width6` | 6-unit ranking net — 12 ite in hrank | ok | ok | ok |
| `NN2Width8` | 8-unit ranking net — 16 ite in hrank | ok | ok | ok |
| `NN2Width10` | 10-unit ranking net — 20 ite in hrank | ok | ok | ok |
| `NN2Width12` | 12-unit ranking net — 24 ite in hrank | ok | ok | ok |
| `NN2WidthDup8` | 8 units but one distinct condition — control separating the cost of net *size* from the cost of net *branching* | ok | ok | ok |
| `NN2Deep3` | 3 dense hidden layers — 6 ite per copy, text doubles per layer | ok | ok | ok |
| `NN2Deep4` | 4 dense hidden layers — 8 ite per copy, 16 in hrank | ok | ok | ok |
| `NN2Deep5` | 5 dense hidden layers — 10 ite per copy, 20 in hrank | ok | ok | ok |
| `NN2Narrow8` | 8 layers of one unit each: same 16 ite as NN2Width8 but linear text — does depth or does size cost? | ok | ok | ok |
| `NN2InvWide4` | invariant is a 4-unit net equality that *is* the box 0..100 | ok | ok | ok |
| `NN2InvWide8` | the same, 8 units wide — step_inv now carries 16 ite | ok | ok | ok |
| `NN2RealWide4` | 4-unit net over Real — no omega, linarith on 2^8 branches | ok | ok | ok |
| `NN2RealFrac` | weights 1.5 / 0.25 / -0.5 and a negative output weight — the weights a trained net actually has | ok | ok | ok |
| `NN2RealNetInv` | a net *invariant* over Real: the exact-value disjunction keeps it inductive, the net box is the learned part | ok | ok | ok |
| `NN2VecNet` | 3-input net invariant over a 3-vector state: Σ relu(vᵢ) = Σ vᵢ is componentwise non-negativity, and Σ relu(vᵢ) ≤ 6 bounds it | ok | ok | ok |
| `NN2NetMod8` | the module's own net is 8 wide and both predicates are nets — module width against certificate width | ok | ok | ok |
| `NN2NetMod16` | the same with a 16-wide module net | ok | ok | ok |
| `NN2WideInput` | a net that reads 8 state slots but has only 2 units — input width without branch width, over the 32-wide state | ok | ok | ok |
| `NN2MixedSign` | mixed-sign weights, thresholds at 0 and 10 so two branches are live at once, output bias keeping the value non-negative | ok | ok | ok |
| `NN2BigWeights` | weights 10007 / 3001 / 499 — coefficient size, not branch count | ok | ok | ok |
| `NN2ToNatClamp` | net(x) = 3x - 2 is negative at x = 0, so Int.toNat clamps; the obligation is still true because the clamp only bites under P | ok | ok | ok |
| `NN2CmpBoth` | a net on each side of ≤ — relu(x)+relu(y-x) ≤ relu(y) is 0≤x≤y | ok | ok | ok |
| `NN2Lyapunov` | |x-5| as a 2-unit net, used as *both* invariant and ranking over a plant that converges from either side: a genuine piecewise-linear Lyapunov function whose decrease needs the ReLU split | ok | ok | ok |
| `NN2Lyap2D` | the same in two dimensions: a 4-unit Lyapunov net over a plant with two independent piecewise-linear legs | ok | ok | ok |
| `NN2Width3` | 3-unit ranking net — the last width that closes | ok | ok | ok |
| `NN2Width5` | 5-unit ranking net — bisects the knee | ok | ok | ok |
| `NN2Width6Big` | the 6-unit net of NN2Width6 over the 32-slot module: identical certificate and identical transition shape, but the plan calls the module slow and raises the heartbeat budget 5x | ok | ok | ok |

---

## What verifies

**State.** Integer (LIA) state of any shape tried: a single `1×1` wire,
several `1×1` wires, a 3-, 6- or 32-element vector, and a mixed state whose
wire indices and flat element slots disagree (`1×1` + `3×1`). Components that
grow without bound are fine as long as the invariant does not mention them.
Bool state works. External inputs work, with `--pre` constraining them.

**Invariants.** Conjunctive interval bounds; relational (`x ≤ y`); six-way
disjunctions; parity via `%`; `≠`; `Ite` and `Implies` in a `Prop` position
over a Bool component; per-component predicates over a vector.

**Ranking functions.** A bare linear `x`; the guarded `ite(P, 0, x)`; a
difference `y − x`; a lexicographic order on two counters folded into one
`Nat` as `4y + x`.

**ReLU — in every position tried.** In the transition at `1×1`
(`x' = relu(x−1)`, emitted as `Max.max 0 (x−1)`) and element-wise on a
3-vector; as a ReLU-shaped *ranking* (`ite(x>0, x, 0)` — the SMT→Lean
translator has no `max`, so it must be written as an `Ite`); as a ReLU-shaped
*invariant* (`x = ite(x≥0, x, 0)`); in a `Linear → ReLU → Linear` net, the
shape a small Q-network compiles to; and combined with an external input under
`--pre "(>= e0 1)"`. It works because `Core/Mat.lean` gives `ReLu` a `@[simp]
relu_apply` and `omega` understands `max` over `Int` — which is exactly what
`Min`/`Max` lack (open issue 2).

**Scale.** A 32-wide state verifies. A straight-line transition body verifies
to at least 64 chained operations now that `maxRecDepth` is raised; the
default limit gave out between 44 and 48.

---

## The tactics are generated, not fixed

The three obligations used to be closed by one hardcoded chain, textually
identical in every certificate. That is wasteful in both directions: an
integer module with a linear ranking paid for `decide` and `bv_decide` on
every goal, and a shape nobody had anticipated had no way to ask for the step
it needed.

`zrth/lean/tactics.py` reads the module and its predicates and emits two
macros, `cert_prep` (canonicalise) and `cert_close` (decide), which the three
proofs share. Each certificate carries a comment saying what was detected.

| detected | consequence |
|---|---|
| Int / Bool in the state | `omega` is available |
| Real | `omega` is dropped; `linarith`, `norm_num` and `norm_cast; linarith` take over |
| Real *and* equalities in the predicates | `simp_all` moves into **prep**, so the closers see numerals, not an opaque `⌊4 - x⌋` |
| a product of two state-dependent terms | `nlinarith`, `positivity` |
| finite state (Bool/BitVec only) | `decide`, last, because it is the most expensive step |
| an `Ite`/`ReLU`/`Min`/`Max` anywhere | `split_ifs` in prep |
| `∧` / `∨` in the predicates | `casesm*` and `simp only [not_and_or]` in prep |
| an `=` in the predicates | the `ne_iff_lt_or_gt` split, *after* `norm_num at *` |
| term count | `maxRecDepth` scales with it |
| branch points in the predicates | scales the heartbeat budget — this, not module size, is what a net costs |
| nothing known to be slow | heartbeat budget stays at 400 000 for economy (it does *not* bound failures — see "Nets over ℝ") |

Two decisions are worth recording because measurement contradicted the
obvious guess.

**`linarith` belongs in every plan.** Gating it on Real looked right and cost
two integer certificates that were closing on it — an integer goal that omega
cannot phrase is still ordinary linear arithmetic.

**`bv_decide` belongs in none.** Open issue 4 adds a second reason — on the
generated goals it abstracts `Mat` projections into opaque variables and
returns a spurious counterexample — but the measured one was enough. It only
ever reaches goals every cheaper
prover has already failed on, and on those it bit-blasts: including it took a
BitVec certificate from 10 s to 977 s while closing nothing `decide` had not
already closed. The obligation it might have helped with — `hrank`, mixing
BitVec with `Int.toNat` — it answers with "a potentially spurious
counterexample".

Because the plan depends on the predicates, `Certificate.lean` is no longer
stable across a change of invariant: `verith --infer` now rewrites it
alongside `Data.lean`.

`tests/test_lean_tactics.py` pins these decisions directly, so a plan
regression shows up in the fast suite rather than only in a slow Lean build.

---

## Neural-network invariants and ranking functions

A ReLU net can serve as the invariant, as the ranking, or as both, over a
module that is itself a net. Every case is generated from explicit weight
matrices (`net(...)` in the harness), never a hand-written formula, so none of
them can be quietly tuned to what the prover happens to close.

### What the ceiling turned out to be

The first eight cases all passed, so a second batch was built to find where
nets stop working: width 3 to 12 units, three to five dense hidden layers,
net invariants, net Lyapunov functions, nets over ℝ, nets over a vector
state, and a module that is itself `Linear → ReLU → Linear` with both
predicates nets.

Thirteen of them failed — **and every failure was the heartbeat budget, not
the proof.** Each one closed when `maxHeartbeats` was raised, some at 4M,
some only at 40M, taking 27 s to 491 s. That is the wrong kind of fix: it
turns "fails in 16 s" into "passes in 491 s".

The cause was in the translator, not the tactics. There is no `max` kind to
translate from, so every ReLU unit arrived as `(ite (>= e 0) e 0)`. Left as
an `ite` each unit costs a `split_ifs` branch, and `hrank` mentions the
ranking twice, so a k-unit net fans one goal out into 2^(2k) — each of them
paying for the whole prep chain. `omega` reasons about `min`/`max` over `Int`
natively, with **no split at all**.

Folding that shape back into `max` (`smt_to_lean.py`, Int only) collapses it:

| case | before | after |
|---|---|---|
| `NN2Width6` — 6-unit ranking net | 491 s (needed 40M heartbeats) | **9.6 s** at the base budget |
| `NN2Deep3` — three dense hidden layers | 164 s (40M) | **11.8 s** |
| `NN2Lyap2D` — 2-D piecewise-linear Lyapunov net | 220 s (40M) | **12.0 s** |
| `NN2Width12` — 12-unit layer | did not close | **11.6 s** |

Eleven cases went green on that one change. What is left of the cost tracks
the number of branch points in the *predicate*, which the budget rule now
reads directly — the module stays a one-wire countdown whether its ranking
has 3 branch points or 62, so module size was never the right proxy.

### Where it stands

Every integer net in the matrix verifies: up to a 12-unit layer, five dense
hidden layers (62 branch points, 79 s), nets as invariant, as ranking, as
both, over a vector state, with mixed-sign and large weights, with a net
module underneath, and as a piecewise-linear Lyapunov function in one and two
dimensions.

An affine layer is *linear* — `2 * x` is not a product of two
state-dependent terms — so nets do not drag `nlinarith` into the plan, which
is what keeps a small net at ~10 s.

### Nets over ℝ

A Real net cannot use the folding above — `linarith`, which is what a Real
goal gets, has no `min`/`max` support, so there the `ite` and its `split_ifs`
branch are still the way through. Those cases verify, but they cost a bigger
heartbeat budget and about 70 s where the integer equivalent takes 10 s.

Raising that budget is what made them verify at all, and the measurement
behind it inverts an assumption this report used to state. A low cap does
**not** make a failing proof fail fast:

| | base budget | raised |
|---|---|---|
| `NN2RealWide4` — 4-unit net over ℝ | **fails** in 106 s | **succeeds** in 71 s |
| `NN2RealFrac` — fractional weights | **fails** in 23 s | **succeeds** in 2.4 s |

Both are *faster* with more budget. When a tactic hits the cap it throws,
`first` catches that like any other failure and moves on to a more expensive
alternative, which burns up to the cap again; a low cap multiplies the wasted
work rather than cutting it short. The base budget is still low, but for
economy on ordinary goals, not to bound failures.

---

## Open issues

### 1. ~~`FBK` only compiles when it shares a file with `ScalarRel`~~ — fixed

**Symptom.** `System/FBK.lean` failed with `unsolved goals ⊢ X = X` — the two
sides printed identically — for any module with a multi-element ctrl wire.
It cost `InvMixed`, `ReluNet`, `OpTranspose`, `Vec32` and `NNNetModule`: every
remaining `System` failure. In all five the *certificate* discharged; only the
encoding failed to compile.

**Mechanism.** `pp.explicit` shows what the pretty printer hides. The goal was

```
matVecAffine … (fun i j => FBK.effect_0.match_1 …) 0 0
  = matVecAffine … (fun i j => ScalarRel.effect_0.match_1 …) 0 0
```

A multi-element wire is rebuilt with `fun i j => match i, j with …`, and every
`match` in a definition elaborates to an auxiliary matcher named after its
enclosing declaration — likewise `_proof_1` for the `Fin` literal bounds. The
two matchers have identical bodies and are definitionally equal, but they are
*different constants*, and `simp` closes a goal only up to syntactic equality
after rewriting. So `simp [effect_0, ScalarRel.effect_0]` reduced both sides to
the same printed term and then could not finish.

Lean reuses a matcher within a module and generates a fresh one across a module
boundary, which is why the single-file form compiled and the split form did not
— the module boundary was a symptom, not the cause. The earlier guess that
`let`-heavy equation lemmas were failing to export was wrong.

**Fix.** `translate/fbk.py` emits `first | rfl | simp […]` for `effect_i_eq` and
`init_i_eq`. `rfl` checks definitional equality at default transparency, which
unfolds both matchers; the `simp` fallback stays for anything `rfl` cannot do.
Modules whose ctrl wires are all `1×1` emit no `match` at all, which is why only
the multi-element ones were ever affected.

**Regression cover.** `Certs/RelEnc*` in the slow suite now generates four
separate modules — functional, Scalar, ScalarRel, FBK — exactly as
`create_project` splits them, instead of concatenating them into one file. The
concatenated form is a strictly weaker test: it shares matchers that a real
project does not. `tests/test_lean_diagram.py` additionally pins the `rfl`-first
tactic without needing `lake`.

### 2. `matMin` / `matMax` never reduce, so `Min`/`Max` modules cannot be proved

**Symptom.** `OpMax`, `OpMin`. The certificate goal keeps an opaque
`matMax (matVecAffine …) 0 0` and every closer fails on it.

**Mechanism.** `ReLu` has a `@[simp] relu_apply`; `matMin`/`matMax` have no
apply lemma and are not in `simp_mat`'s list. Adding the bare definitions to
that list is not enough — tried and reverted: they are folds over
`(List.finRange m).flatMap …`, and `simp_mat` carries no lemmas that compute
`List.finRange`, so the fold stays stuck.

**Resolution.** Either add computed `@[simp]` lemmas for `matMin`/`matMax`, or
redefine them over `List.ofFn`, whose `List.ofFn_succ` / `List.ofFn_zero` are
already in `simp_mat`. The second is a semantics-preserving redefinition of
`Core/Mat.lean` and needs the slow suite to confirm nothing that reasons about
these two regresses.

**Confirmed provable.** The SMT encoder could not build these modules at all
until this pass — `Min`/`Max` were wired to the binary `_elementwise` path
although they are unary reductions, so any cvc5 query about them raised
`TypeError`. With that fixed, `--pre-check cvc5` says all three obligations
hold for `OpMax` and `OpMin`, in 4.6 ms and 4.2 ms. So there is nothing wrong
with the certificates and nothing to find in the tactic plan: the Lean-side
fold is the whole of it.

### 3. ~~Real (LRA): a conjunction of disjunctions still defeats the plan~~ — not a tool limit

Real modules verify: `LRALinear`, `RankToInt` (a non-integral state in steps
of 0.5), `ReluLRA`, `RealNonlin`, `RealNet` and now `RealConjDisj` — an
invariant that is a conjunction of two 4-way disjunctions over two Real
components — all discharge.

`RealConjDisj` was previously listed here as the one shape the generated plan
could not serve. That was wrong: **the obligation it stated was false.** Its
module (`m_lra_two`) has two components that both *cycle*, x over
`{0,1,2,3}` and y over `{2,5}`, and the invariant was the product of their two
ranges. That product loses the phase relation between them and admits

```
(1,2) → (2,5) → (3,2) → (0,5) → (1,2)
```

— four states that all satisfy the invariant, none of which is `P = (0,2)`.
`hrank` asks for `ranking s' < ranking s` on every invariant state off `P`,
which no ranking function can satisfy around a cycle. Checked in Lean, not by
hand: each of the four states satisfies `inv`, none satisfies `P`, and each
`update` step is provable by `norm_num`. So the certificate was unprovable by
construction, and no tactic could ever have closed it.

Restated over two components that *converge* instead of cycling
(`m_lra_conv`: `x' = if x > 0 then x - 1 else 0`, `y' = if y > 2 then y - 1
else 2`), keeping the same conjunction-of-disjunctions shape, it verifies in
13 s with no change to the plan. The step that carries it is the
`casesm* _ ∧ _, _ ∨ _` the plan already emits for a disjunctive predicate:
deleting just that step from the generated `cert_prep` turns 0 errors into 9.

The original module is kept as `RealConjDisjUnsat`, a negative control — the
tool must go on rejecting it.

**Lesson for the harness.** Two independent cyclic components cannot be
covered by a conjunction of per-component ranges; the invariant has to relate
their phases. Three earlier cases in this matrix were wrong the same way (an
interval invariant is not inductive over the reals), and a wrong test failing
is not a tool limit.

### 4. ~~BitVec: the invariant obligations pass, the ranking one does not~~ — fixed

**Symptom.** `BVState`. `init_inv` and `step_inv` closed; `hrank` failed with
eight goals that were, literally, `⊢ False`.

**Mechanism.** After `split_ifs` the surviving branches are contradictory
only because a `BitVec 1` has exactly two inhabitants — `s.1 0 0 ≠ 0#1`
together with `s.1 0 0 ≠ 1#1`. Nothing in the plan could see that:
`contradiction` needs a literal `h`/`¬h` pair, `simp_all` has no lemma for
the cardinality of `BitVec 1`, and `omega` does not model BitVec at all.
`decide` is in the plan for exactly this case and yet can never fire — it
evaluates *closed* propositions, and every goal here is under a free state
variable, so all it can report is that `False` is false.

**Fix.** A generated prep step that enumerates the state. When every state
element has two values and there are at most `MAX_ENUMERABLE_SLOTS` of them,
the plan emits

```lean
macro "cert_states" v:ident : tactic =>
  `(tactic| (rcases BitVec.eq_zero_or_eq_one (($v).1 0 0) with h0 | h0 <;>
     rcases BitVec.eq_zero_or_eq_one (($v).2 0 0) with h1 | h1 <;>
     try simp only [h0, h1] at *))
```

and `step_inv` and `hrank` call it on their own state binder before the
closers run. Once the elements are concrete every later goal is closed, which
is what the rest of the plan was always built for. `BVState` verifies in 11 s;
`BoolState`, the same counter over `Bool`, went from 15 s to 10 s. Modules
whose state is not finite emit `skip`, so nothing else changes.

The bound matters: each element doubles the fan-out, so a wide finite state
opts out rather than producing `2^n` branches. `BitVec.eq_zero_or_eq_one` is
width-1 only, so a wider BitVec also opts out and would still hit the original
wall — untested, since nothing in the matrix has one.

**A note on two dead ends.** `bv_omega` closes seven of the eight goals and is
a reasonable prover for a BitVec state (the plan has no integer reasoner for
one, though `ranking` is always `Nat`-valued); it was not needed once the
state was enumerated, so it is not in the plan. `bv_decide` closes the eighth
in isolation but not in context: it reports *"a potentially spurious
counterexample — abstracted the following unsupported expressions as opaque
variables: [s.2 0 0, s.1 0 0, s.1 0 0]"*. Under `pp.explicit` the reason is
the same class of defect as open issue 1 — one occurrence carries
`@Fin.instOfNat 1 init_pre._proof_1 0` and another `@Fin.instOfNat 1
update._proof_1 0`, two auto-generated per-declaration instance proofs, so
`s.1 0 0` is two syntactically different terms that print identically.

### 5. Nonlinear ranking functions — fixed, with one caveat

`RankQuadratic` (`ranking = x * x`) and `RealNonlin` (the same over ℝ) both
verify. Three things were needed and all are now generated: `nlinarith` when
a product of two state-dependent terms is detected, the same prover inside
the `constructor` branch (an `Int.toNat` comparison unfolds to a
*conjunction*, `a < b ∧ 0 < b`, which no arithmetic prover proves directly),
and the `ne_iff_lt_or_gt` split, because `¬(x = 0)` yields the strict bound
`1 ≤ x` only through integrality — which linarith and nlinarith do not do.

The caveat is detection: `is_nonlinear` reads the predicate text, and errs
toward reporting nonlinear, since a superfluous `nlinarith` only costs time
on goals everything cheaper has already failed, whereas a missed one loses
the proof.

### 6. `Uninterpreted` has no Lean form

The only remaining generation failure (`No Lean expression mapping for:
Uninterpreted`). Unlike `Transpose` — which had a Lean counterpart all along
and is fixed below — an uninterpreted symbol has no obvious translation. It
needs a decision: emit an `opaque` declaration (and accept that no proof can
say anything about it), or reject it at the front end with a clear message.
`KNOWN_ISSUES.md` #23 lists this alongside the now-dead `TensorGet` /
`ToUnsigned` keys.

### 7. `zeroth_hammer` and the generated plan have diverged further

`Certificate.lean` still never calls `zeroth_hammer`; it now uses the
generated `cert_prep`/`cert_close` instead of the old fixed chain. Keeping
them apart is defensible — `zeroth_hammer`'s last phase is `sorry`, which
would turn a failed certificate into a build that "succeeds" with a warning —
but there are now two proof strategies in the tree, and only one of them is
shape-aware.

**Resolution.** Either give `zeroth_hammer` a `sorry`-free variant that
consumes the same plan, or drop it from the generated project and keep it as
the interactive/experimental tactic it is used as in `ManualTests`.

### 8. Nothing in CI builds a generated project

Every Lean test compiles a standalone `Certs/*.lean`. No test runs
`lake build` on the output of `verith -o`. That is why issue 1 survived, and
why 13 of 33 projects failed `System` before this pass while the suite was
green.

Partly addressed: `Certs/RelEnc*` now mirrors the *module layout* of a
generated project — four separate modules rather than one concatenated file —
which is what makes it able to see issue 1 at all. It still builds fixtures in
the warm project rather than a `verith -o` output.

**Resolution.** One slow test that generates a project, symlinks the shared
`.lake` (the recipe above; `test_generated_executable_builds_and_runs` already
does exactly this for the `-x` path) and builds `System` and `Certificate`.

The precondition is now met: the whole matrix -- 77 cases, 37 module
fixtures, the runners and the baseline -- lives in
[`tests/limits/`](../../tests/limits/README.md) instead of a scratch
directory, so every number in this file is reproducible with one command.
What remains is wiring a handful of its cases into a slow-marked test.

### 9. Argmax certificates

`OpArgmax` compiles all six encodings now but the certificate fails:
`argmax_1d` does not reduce under the tactic chain, the same shape of problem
as issue 2. The generated `argmax1d_scalar_n_eq` theorems tie the unrolled
scalar form to `Core.Mat.argmax_1d`, but nothing brings either into a goal.

`--pre-check cvc5` reports all three obligations holding in 3.2 ms, so this is
the same story as issue 2: a true certificate that Lean cannot reduce its way
to. `Argmax` was always encoded correctly for SMT (`_argmax_flat`), which is
why it could be checked before `Min`/`Max` could.

### 10. Pre-existing entries still open

`KNOWN_ISSUES.md` #24 (`_can_scalarize()` gates nothing) and #25 (`Real` has
no Lean IO, so `verith -x` refuses a Real wire) are unchanged by this pass.

---

## Fixed in this pass

| # | Defect | Fix |
|---|---|---|
| 1 | `convert_method(…, theory=LIA)` crashed on any Python `bool` literal — `const_bool` → `const` → the nonexistent `LIA.Const` | `builder.py`: emit `LIA.Bool` |
| 2 | Scalar encoding: a `1×1` `ReLU` took the matrix fallback and was ascribed `Int` while producing `Mat` | `_SCALAR_OP["ReLU"] = max 0 x`. Lifting the operand and projecting the result is *not* a fix — `ReLu`'s dimensions are then unconstrained |
| 3 | Same fallback never projected its result for any other matrix-form op | project at `0 0` when the write wire is scalar, as the `Argmax`/`Linear` branches already did |
| 4 | `flat_slots` were unparenthesised, so `argmax1d_scalar_int_3 x0 0 0 x0 1 0 x0 2 0` passed 9 arguments to a 3-argument axiom | parenthesise each slot |
| 5 | Lifted scalar operands had no type ascription, leaving `matVecAffine`'s batch dimension unsolved | ascribe `Mat <elem> 1 1` |
| 6 | `Implies` in any predicate hit `SMT→Lean: unsupported kind Kind.IMPLIES` | add the `IMPLIES` case (`→`) |
| 7 | `Transpose` had no entry in either op table, though `Core/Mat.lean` has had `MatTranspose` all along | map it, and add `Box.transpose` for the Circ encoding |
| 8 | A Real ranking emitted `⌊x⌋` with neither the notation nor the `FloorRing ℝ` instance in scope | import `Mathlib.Algebra.Order.Floor.Defs` and `Mathlib.Data.Real.Archimedean` in `Core/Mat.lean` |
| 9 | `def ranking` was not `noncomputable` for a Real state, though the `DecidablePred` instance beside it was | prefix it with the same `noncomp` |
| 10 | `simp` hit `maximum recursion depth` between 44 and 48 chained operations | `set_option maxRecDepth 100000` in the certificate and on the scalar equivalence theorems |
| 11 | `decide` on a finite state ran past the heartbeat budget before concluding | `set_option maxHeartbeats 2000000` in the certificate |
| 12 | The closer list ended at `tauto`, with no `decide` for finite state | append `decide`, `bv_decide`, `(simp_all; done)`, `(simp_all; omega)`, `(simp_all; linarith)` — appending is monotone, an alternative only runs when every earlier one has already failed |
| 13 | `zeroth_hammer` phase 5: `contradiction`/`decide`/`native_decide` act on the main goal but were followed by an unconditional `return`, so a sibling goal from an earlier split was left unproved *and* the `sorry` fallback was skipped | apply them with `all_goals` and return only when no goals remain |
| 14 | `test_scalar_encoding_lifts_scalar_operand_for_relu` asserted only that the *operand* was lifted, so it passed on ill-typed output | assert the emitted ReLU is scalar on both sides |
| 15 | `¬(x = 0)` gave linarith/nlinarith no usable bound — the strict `1 ≤ x` follows only by integrality | split it with `ne_iff_lt_or_gt`, after `norm_num at *` (which renormalises `≠` back) so the existing `casesm*` consumes it |
| 16 | `simp … at *` read the transition equation left-to-right and rewrote `s 0 0 - 1` in the goal *back* into `s' 0 0`, undoing the `rw [← heq]` above it | `clear heq` once the goal no longer mentions the successor state |
| 17 | An `Int.toNat` comparison unfolds to a conjunction, which no arithmetic prover proves directly | give the `constructor` branch every prover in the plan |
| 18 | Rational literals were emitted with `str(term)`, i.e. SMT-LIB: cvc5 prints 0.5 as `(/ 1 2)` | build the literal from `getRealValue()` |
| 19 | One fixed tactic chain for every module: `bv_decide` on integer goals, no `nlinarith` on nonlinear ones, no way to add a prep step for a new shape | generate `cert_prep`/`cert_close` from the module and predicates (`zrth/lean/tactics.py`), pinned by `tests/test_lean_tactics.py` |
| 20 | `Certificate.lean` was treated as stable across a change of predicates, but its tactics are now generated from them | `--infer` rewrites it alongside `Data.lean` |
| 24 | A branchy Real predicate cannot fold to `min`/`max`, keeps a `split_ifs` branch per ReLU unit, and exhausted the base budget | raise it for `Real` + branch points. Measured, this is also *faster*: the case failed in 106 s at the base budget and succeeds in 71 s at the raised one — a cap that is hit makes `first` fall through to more expensive alternatives, so a low cap multiplies wasted work instead of bounding it |
| 23 | Every ReLU unit reached Lean as `(ite (>= e 0) e 0)`, so a k-unit net fanned one goal into 2^(2k) under `split_ifs` and neural certificates died on the heartbeat budget rather than on the proof | fold that shape to `max` in `smt_to_lean.py` (Int only — `linarith` has no min/max), and scale the budget by the predicate's branch count rather than the module's size. Eleven cases went green; `NN2Width6` went from 491 s to 9.6 s |
| 22 | A finite state's obligations could produce a bare `False` goal, contradictory only because `BitVec 1` has two inhabitants. `decide` is in the plan for finite states but evaluates closed propositions, so under a free state variable it can never fire | generate a `cert_states` prep step that enumerates each two-valued state element (`BitVec.eq_zero_or_eq_one` / `Bool.eq_false_or_eq_true`), bounded by `MAX_ENUMERABLE_SLOTS` so the fan-out stays small |
| 21 | `FBK.effect_i_eq` closed with `simp`, which cannot equate two auto-generated matchers — `FBK.effect_i.match_1` vs `ScalarRel.effect_i.match_1` — so `System/FBK.lean` failed for every multi-element ctrl wire | `first \| rfl \| simp […]`: `rfl` unfolds both at default transparency. The slow suite now builds the four encodings as separate modules, as a real project does |
