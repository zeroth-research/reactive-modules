# What `uv run verith` can verify

Measured, not guessed: 48 modules were put through the whole pipeline — Python
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

### Running it in seconds per case

A fresh project would refetch and rebuild Mathlib. Instead, point every
project's `.lake` at one shared build directory whose `packages` is the
already-built package set of `tests/lean`:

```sh
mkdir -p /tmp/shared/.lake/build
ln -s <repo>/python/tests/lean/.lake/packages /tmp/shared/.lake/packages

uv run verith mymodule.py -P … --invariant … --ranking … -o /tmp/case1 -p Rea
ln -s /tmp/shared/.lake /tmp/case1/Rea/.lake
cp <repo>/python/tests/lean/lake-manifest.json /tmp/case1/Rea/
cd /tmp/case1/Rea && lake build
```

Give every project the same package name (`-p Rea`) and `Core`, `LeanAI` and
`ZerothHammer` are byte-identical across them, so they build once and hit the
cache afterwards; only `System/*` and `Certificate/*` recompile. Measured:
**~9 s per project**, 20 MB of shared build output, Mathlib never rebuilt.

---

## Results

47 of 48 cases generate; 47 compile all six encodings; 35 certificates discharge. Of the 12 that
do not, **8 are negative controls that are supposed to fail** — a missing
property, a non-inductive invariant, a constant or increasing ranking, a
dropped precondition. That leaves 4 real limits (`BVState`, `OpArgmax`, `OpMax`, `OpMin`),
all in the next section.

Against the pipeline as it stood before this work, measured on the 36 cases
that existed then (the other 12 — eight neural-network cases, three built to
break a fixed tactic chain, and one negative control — were added along the
way):

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

Over the full 48-case matrix: 47 generate, all 47 compile all six encodings,
35 certificates discharge, and the same 35 projects are green end to end —
`System` no longer fails anywhere. No case that verified at any earlier point
in this work stopped verifying: the switch from a fixed tactic chain to a
generated plan reproduced all 44 shared verdicts exactly, and the FBK fix moved
five cases from broken to green while leaving every other verdict untouched.

| case | what it probes | gen | System | Certificate |
|---|---|---|---|---|
| `Countdown` | LIA 1x1, conjunctive bound, Ite ranking — the known-good control | ok | ok | ok |
| `TwoVars` | two 1x1 wires, relational invariant, difference ranking | ok | ok | ok |
| `NoCert` | no -P at all: what does a bare `verith` project contain? | ok | ok | **fail** |
| `PropOnly` | -P but no --invariant/--ranking: inv defaults to True, ranking to sorry | ok | ok | **fail** |
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
| `BVState` | BitVec state — omega does not reason about BitVec | ok | ok | **fail** |
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
| nothing known to be slow | heartbeat budget stays at 400 000, so failures fail fast |

Two decisions are worth recording because measurement contradicted the
obvious guess.

**`linarith` belongs in every plan.** Gating it on Real looked right and cost
two integer certificates that were closing on it — an integer goal that omega
cannot phrase is still ordinary linear arithmetic.

**`bv_decide` belongs in none.** It only ever reaches goals every cheaper
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
module that is itself a net. There is no `max` in the SMT→Lean translator, so
a unit is an `ite` and a `k`-unit layer costs `k` branches to `split_ifs`.

All eight cases below discharge their certificate:

| case | net |
|---|---|
| `NNRank` | ranking `2·relu(x−1) + relu(x)` |
| `NNRankWide` | ranking, three units with distinct thresholds |
| `NNRankDeep` | ranking, two hidden layers |
| `NNInv` | invariant `relu(x) + relu(9−x) = 9`, i.e. exactly `0 ≤ x ≤ 9` |
| `NNInvIneq` | the same in inequality form — the shape a learned barrier takes |
| `NNBoth` | invariant and ranking both nets |
| `NNTwoInput` | two-input net invariant `relu(y−x) + relu(x) = y` |
| `NNNetModule` | module is `Linear → ReLU → Linear` *and* both predicates are nets |

The cases are generated from explicit weight matrices (`net(...)` in the
harness), not hand-written formulas, so they cannot be quietly tuned to what
the prover happens to close. All eight are green, `NNNetModule` included — its
project used to fail to build on open issue 1, which was never about the net.

Note that an affine layer is *linear* — `2 * x` is not a product of two
state-dependent terms — so nets do not drag `nlinarith` into the plan. That
is what keeps a 3-unit net at ~10 s.

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

### 4. BitVec: the invariant obligations pass, the ranking one does not

**Symptom.** `BVState`. `decide` closes `init_inv` and `step_inv`; `hrank`
fails, in 10 s.

**Mechanism.** After `split_ifs` the surviving branches are bare `False`
goals whose hypotheses are contradictory only because a `BitVec 1` has
exactly two values — `¬(b = 1#1)` together with `¬(b = 0#1)`. Closing that
needs case analysis over the bit, which `decide` cannot do (it ignores
hypotheses) and `contradiction` cannot see. `bv_decide` can, and is
deliberately excluded: see the note in the generated-tactics section.

**Resolution.** Either a prep step that reverts BitVec hypotheses into the
goal so `decide` can settle the implication, or emitting `Bool` rather than
`BitVec 1` for single-bit wires, which would put the whole thing in
`decide`'s reach.

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

### 9. Argmax certificates

`OpArgmax` compiles all six encodings now but the certificate fails:
`argmax_1d` does not reduce under the tactic chain, the same shape of problem
as issue 2. The generated `argmax1d_scalar_n_eq` theorems tie the unrolled
scalar form to `Core.Mat.argmax_1d`, but nothing brings either into a goal.

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
| 21 | `FBK.effect_i_eq` closed with `simp`, which cannot equate two auto-generated matchers — `FBK.effect_i.match_1` vs `ScalarRel.effect_i.match_1` — so `System/FBK.lean` failed for every multi-element ctrl wire | `first \| rfl \| simp […]`: `rfl` unfolds both at default transparency. The slow suite now builds the four encodings as separate modules, as a real project does |
