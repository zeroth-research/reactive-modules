# Using cvc5 inside `verith`

`verith` has always had a complete symbolic encoding of every module
(`smt_module.ModuleSMT`), but outside the `--infer ai-cegar` loop it never
asked the solver anything: cvc5 parsed the SMT-LIB predicates, handed them to
the Lean printer, and went home. This is what it is used for now, what that
bought, and what is left.

Every solving phase is bounded (`--smt-timeout` per query, `--smt-budget` per
phase, both enforced by cvc5's own `tlimit`) and every failure -- timeout,
`unknown`, an operator with no SMT form, a cvc5 exception -- degrades to the
behaviour `verith` had before. Nothing here may take information away.

---

## What runs, and when

| | flag | solving | default |
|---|---|---|---|
| Obligation pre-check | `--pre-check=cvc5` | yes | off |
| Sharing in the printer | — | no | **on** |
| Plan features from the terms | — | no | **on** |
| Settled branch conditions | `--smt-tactics=cvc5` | yes | off |
| Product hints for `nlinarith` | `--smt-tactics=cvc5` | yes | off |
| Unused-precondition `clear` | `--smt-tactics=cvc5` | yes | off |

The two always-on items do no solving at all -- they read the parsed terms --
so they need no budget and cannot hang.

Both flags act on the predicates *as supplied*, before `--infer` runs. They
have nothing to say about an inferred certificate: `magic` hands back Lean
text rather than SMT-LIB, so there is no term left to ask cvc5 about. Closing
that would mean `magic` returning the term alongside the rendering --
worthwhile, since `--infer ai` does no verification of its own, and cheap.

---

## 1. `--pre-check=cvc5`

A failing `lake build` cannot distinguish "the obligation is false" from "the
tactics are too weak", and it takes 9-79 s per case to not distinguish it.
cvc5 answers all three obligations in milliseconds, with a counterexample:

| case | `lake build` | cvc5, all three | verdict |
|---|---|---|---|
| Countdown | 8.7 s | 9.2 ms | all hold |
| NN2Deep5 | 78.8 s | 10.1 ms | all hold |
| NN2RealWide4 | 69.5 s | 6.5 ms | all hold |
| BadInv | 8.8 s fail | 4.5 ms | `init_inv` false at `s0=100`, `step_inv` at `s0=0` |
| BadRankDir | 6.0 s fail | 3.8 ms | `hrank` false at `s0=1` |
| RealConjDisjUnsat | 9.2 s fail | 5.8 ms | `hrank` false at `s0=3.0, s1=2.0` |

More than one "tool limit" in the earlier passes turned out to be a broken
test case instead — `RealConjDisjUnsat` in the table is the one kept as a
control, and establishing that by hand in Lean took the better part of an
afternoon. This is the check that answers it in 5.8 ms.

**Prerequisite.** `Min` and `Max` had to be fixed in the SMT encoder first.
They are unary reductions over a matrix, like `Argmax`, but were wired to the
binary `_elementwise` path, so building the term for any module using one
raised `TypeError: <lambda>() missing 1 required positional argument` and took
down every cvc5 query about that module — `--infer ai-cegar` included.

```
$ verith m.py -P '(= s0 0)' --invariant '(and (>= s0 0) (<= s0 100))' \
      --ranking '(- 100 s0)' --pre-check cvc5 -o out -p Rea
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         1 ms
   step_inv  holds         1 ms
   hrank     REFUTED       1 ms   counterexample: s0 = 1
   1 obligation(s) refuted (hrank): the certificate is wrong, not merely
   hard -- Lean cannot close it
```

Generation continues after a refutation: the project is still worth having to
look at. The exit status is unchanged.

---

## 2. Sharing (always on)

cvc5 hash-conses, so a predicate arrives as a DAG; the printer walked it as a
tree. A dense net whose hidden layer feeds the next shares every unit, so that
expansion is exponential in depth. Measured on `k` layers of two units:

| layers | units | chars, tree | chars, shared | `max`, tree | `max`, shared |
|---|---|---|---|---|---|
| 2 | 4 | 176 | 176 | 6 | 6 |
| 4 | 8 | 704 | 412 | 30 | 8 |
| 6 | 12 | 2816 | 620 | 126 | 12 |
| 8 | 16 | 11264 | 828 | 510 | 16 |
| 10 | 20 | 45056 | 1036 | 2046 | 20 |

Each subterm reached by two or more parent edges is `let`-bound. Arithmetic
only: binding a Bool subterm would hide the structure `split_ifs`, `casesm*`
and `not_and_or` match on. And only when it shrinks the output, so a shallow
net is emitted byte-identically to before.

**It does not make the proofs faster.** `simp` is zeta-reducing, so the goal
the closers see is the expanded one either way -- measured, NN2Deep5 runs
76.2 s against 78.8 s and NN2Deep4 22.4 s against 22.9 s. What it fixes is
generation: `Data.lean` no longer grows exponentially in net depth.

---

## 3. Plan features from the terms (always on)

`tactics.features_for` read the shape of the predicates off the *printed
Lean* with a regex, which meant the plan moved whenever the printer did:
sharing cut a five-layer net from 62 `max` to 10 and silently dropped it two
heartbeat tiers, failing it at 400k. Three features now come from the terms:

* **`nonlinear`** — a `MULT` with two state-dependent children, in place of
  ~60 lines of hand-rolled operand scanning around each `*`. On the 76-case
  matrix the two agree everywhere, so this is robustness, not a fix.
* **`n_branch`** — branch points in the *unfolded goal*. It follows the
  min/max fold, which genuinely emits a subterm once where the `ite` emitted
  it twice; it ignores `let`-sharing, which does not survive to the goal.
  The existing budget calibration is preserved exactly.
* **`n_conditions`** — distinct branch conditions, which is what `split_ifs`
  fans out over. Counting occurrences overestimates it.

`facts` is `None` for predicates compiled from Python IR, and every one of
these falls back to the text scan.

---

## 4 & 5. `--smt-tactics=cvc5`

Three additions, all strictly additive.

**Settled branch conditions.** `split_ifs` fans a goal out into 2^k over k
conditions and each branch pays for the whole prep chain. For each condition
the printer will *keep* as an `if` (a folded `min`/`max` never splits), cvc5
is asked whether `inv ∧ c` or `inv ∧ ¬c` is unsatisfiable. Either answer means
the proof can discharge the condition once instead of splitting on it:

```lean
-- NN2RealWide4, generated verbatim (wrapped here to fit)
macro "cert_facts" v:ident : tactic =>
  `(tactic| (try (have hf0 : (((1 : Real) * (($v) 0 0)) ≥ (0 : Real)) := by
                    (first | omega | linarith | norm_num);
                  try simp only [if_pos hf0])))
```

and in the obligations:

```lean
    try simp_defs      -- unfolds `inv`, so the `have` is provable
    try cert_facts s
    try simp_mat       -- from here the arithmetic moves and `if_pos` would miss
```

Every step is `try`: cvc5 is the stronger prover, so a `have` the Lean
tactics cannot reproduce has to cost nothing and leave the `split_ifs` route
intact.

**It fires, and it does not pay.** Rebuilt with the `try` removed, so that a
failing `have` would be an error rather than a no-op, NN2RealWide4's
`hf0 : 1 * s 0 0 ≥ 0` is proved and the certificate closes. But that is one
of its four conditions, so `NN2RealAllPos4` was built to give it every one:
a Real net whose every ReLU is non-negative under the invariant, all four
conditions settled. Alternating runs with the olean deleted each time, so
only the `Certificate.Certificate` job is being timed:

| | off | on |
|---|---|---|
| first pair | 77 s | 75 s |
| second pair | 76 s | 71 s |

3-7%, which is at the edge of the noise on this machine even quiet.

**Why: sharing and `cert_facts` are pulling against each other**, and both
are additions from this pass. `ranking` is emitted with its repeated
subterms `let`-bound, so its branch conditions read

```lean
let u3 := (u0 + (3 : Real))
… (if (u3 ≥ (0 : Real)) then u3 else (0 : Real)) …
```

while `cert_facts` states the *expanded* condition, because a `have` outside
the definition cannot refer to `u3`:

```lean
have hf0 : ((((1 : Real) * (($v) 0 0)) + (3 : Real)) ≥ (0 : Real)) := …
```

The `have` is proved either way -- it is a true statement about the state --
but `simp only [if_pos hf0]` has nothing to match unless the `let`s have
already been zeta-reduced into the goal, and only the rewrite removes the
split. That is the whole of the missing benefit.

So the predicate is now rendered **unshared whenever cvc5 settled a
condition in it** (`solver_hints` runs before `smt_predicates_to_lean`, and
passes `share=False`). That costs almost nothing: settled conditions only
survive as `if`s in Real predicates, where sharing saves little
(NN2RealWide4 is 556 chars against 422), while the deep Int nets where
sharing matters fold to `max` and offer no conditions to settle at all.
The two shapes now match on the nose.

**And it still does not pay.** Same measurement after the fix:

| | off | on |
|---|---|---|
| first pair | 68 s | 68 s |
| second pair | 69 s | 70 s |

Identical -- which also says the 3-7% before the fix was noise, not a
partial win.

And the rewrite is firing. Rebuilt with the inner `try` stripped, so that a
`simp only` making no progress would be an error, it builds clean: all four
branches really are collapsed, and the certificate still takes 70 s. So
`split_ifs` over four conditions was never the cost. The Real goal carries
`⌊·⌋` and `Int.toNat` and is closed by `linarith`; that is where to look
next. The flag stays off.

The premise was wrong, not the implementation, and that is worth more than
a speed-up: `n_conditions` is a bad cost model for a Real certificate, and
the next attempt should profile where the time actually goes before
optimising anything.

**Product hints for `nlinarith`.** cvc5's proofs are not translatable — this
build offers alethe, cpc, dot and lfsc, and Mathlib reads none of them — but
the proof does not need translating to be useful. Walking it for the
nonlinear rules (`ARITH_MULT_POS`, `ARITH_MULT_SIGN`, `ARITH_MULT_TANGENT`,
…) gives the products cvc5 had to reason about, and those become
`nlinarith [mul_self_nonneg (a - b), mul_self_nonneg (a + b)]`, which is the
difference between `nlinarith` checking a certificate and searching for one.
The hinted call is inserted *before* the bare one; nothing is removed.

**Unsat cores.** Each obligation is proved with its hypotheses asserted
separately, and the core says which were needed. Two uses: a precondition no
refutation needed is `clear`ed before anything else runs (`simp_all` is
superlinear in the hypothesis count and `nlinarith` multiplies pairs of
them), and the core is recorded in the generated comment, where it says which
part of the invariant is load-bearing when a proof does fail.

### What is deliberately *not* done

It is tempting to read "cvc5's refutation was linear" as licence to drop
`nlinarith` and `positivity` from the closer list. That is a bad trade twice
over. `first | a | b` costs nothing for `b` when `a` closes the goal, so
removing a closer only speeds up the runs that were going to fail anyway —
and cvc5 reasoning linearly about a goal is no promise that Mathlib's
`linarith` can.

### A generated tactic is not tested until Lean has parsed it

The unit tests check the *string* `facts_tactic` produces. That is not the
same as checking Lean accepts it, and the gap bit immediately: steps joined
with `;` across lines are rejected inside a `` `(tactic| …) `` quotation with

    unexpected token 'try'; expected ')'

which appears only from **two** settled conditions on. Every case in the
matrix with settled conditions had exactly one, so the whole 12-case subset
built clean and the bug surfaced only on a case constructed on purpose to
have four. Join on one line; and when adding anything to the generated
tactics, build a case that exercises more than one of it.

### Measured cost of getting this wrong

The first cut also derived the invariant at the successor state in `hrank`
(`have hinv_next : inv (RM.update s l) := step_inv s l ⟨hpre, hi⟩`), on the
reasoning that the closers were reasoning about `RM.update s l` knowing
nothing about it. It is sound, it is one line, and it is a catastrophe:
NN2Deep4 went from 22.4 s to **1077.5 s**, NN2Deep5 from 76.2 s to 981.5 s,
and NN2RealWide4 from 68 s to a 2027 s timeout. Unfolding a net-shaped
invariant at a net-shaped successor doubles everything downstream. Removed.

---

## Measuring any of this

Every project's `.lake` symlinks to one shared build directory, and the module
names (`System.Data`, `Certificate.Certificate`, …) are identical across
projects, so **two processes building different projects into one build
directory overwrite each other's oleans**. That does not surface as a clean
failure: it surfaces as heartbeat timeouts and `unknown constant 'hrank'`,
which reads exactly like a codegen bug.

Even with the harness isolated (`shared_lake_regress`, `projects_regress`),
an unrelated heavy process on the machine distorts wall-clock badly enough to
invent findings. One run of the 12-case subset reported Countdown at 801.8 s,
NN2Width12 at 978.9 s and NN2Deep5 at 1996.2 s — for a Countdown certificate
whose generated file differs from the baseline's by one comment line. Re-run
quiet: 9.4 s, 11.3 s, 74.9 s. Verdicts were unaffected throughout; only the
timings were nonsense. Re-time anything surprising before believing it.

---

## 6. Cold start: abduction to repair an invariant

**What it does.** When `step_inv` fails, `Solver.getAbduct(goal)` returns a
formula `B` with `A ∧ B` consistent and `A ∧ B ⊨ goal` — the conjunct missing
from a non-inductive invariant. Measured against the fixtures:

```
m_countdown  inv=(>= s0 1)   -> step_inv needs additionally: (= s0 100)
m_countdown  inv=(<= s0 99)  -> step_inv needs additionally: (= s0 1)
```

**The catch.** Both of those are *sufficient*, not weakest — `s0 = 100` is a
far stronger repair than `(>= s0 1)` needs. An abduct that is too strong is
useless as a certificate even though it type-checks as a repair.

**Steps.**

1. Extend `smt_query` with `abduce(q, budget) -> list[str]`, called only when
   `check_step_inv` came back `REFUTED`, and sharing the phase budget.
2. Enumerate with `getAbductNext` up to a small cap (4) — the first is
   frequently the strongest. Keep the *weakest* by pairwise entailment
   checks, which is one extra `checkSat` per pair and still milliseconds.
3. Filter for usability: reject an abduct that mentions no state variable
   (vacuous), or that is refuted at the initial state (`init_inv` would then
   fail instead — check it before offering it).
4. Surface it in the `--pre-check` report rather than applying it:
   `step_inv REFUTED … try adding: (>= s0 2) ∨ (= s0 0)`. Applying it silently
   would change a user-supplied certificate into a different one.
5. Optional second phase: re-run the pre-check with `inv ∧ abduct` and report
   whether all three obligations then hold, so the suggestion comes with
   evidence.

**Where it should not go.** Do not feed abducts into `--infer` as a fixpoint
loop without an iteration cap and a check that the strengthened invariant is
still true at init — abduction will happily converge on `False`.

**Effort.** Half a day. The API is one call; the work is steps 2 and 3.

---

## 7. Cold start: SyGuS to synthesise the ranking function

**What it does.** `Solver.synthFun` + `addSygusConstraint` + `checkSynth`
finds a ranking function outright, with no LLM, no API key, and a
deterministic answer. Measured against the fixtures:

```
m_countdown  (lambda ((x0 Int)) x0)                              5 ms
m_toward5    (lambda ((x0 Int)) (ite (<= x0 5) (- 6 x0) x0))    3.7 s
m_twovars    (lambda ((x0 Int) (x1 Int)) ...)                   10.9 s
```

The constraints are the obligations `magic_cegar` already builds:
`inv s → rank s ≥ 0` and `inv s ∧ ¬P s → rank (update s) < rank s`.

**Steps.**

1. `--infer sygus` alongside `ai` and `ai-cegar`, in a new `magic_sygus.py`
   that subclasses `TA2Magic`. It needs no `_make_client`, so it runs in CI.
2. Reuse `ModuleSMT.update_state` against `declareSygusVar` state variables —
   the encoding is identical to `magic_cegar._check_ranking_decrease`, only
   with `rank` as a `synthFun` instead of a parsed term.
3. Give it a grammar rather than the default. Unrestricted `LIA` took 10.9 s
   on a two-variable module and will not scale; a grammar of linear
   combinations of the state with small integer coefficients, plus one level
   of `ite` over comparisons, covers everything in the matrix and prunes hard.
4. Bound it, and check *how* first. Every other phase here rides on cvc5's
   `tlimit`; whether that actually interrupts `checkSynth` is untested and
   has to be established before it is relied on, because a synthesis that
   ignores its limit hangs generation. If it does not hold, this phase needs
   a subprocess with a wall-clock kill instead — and it would then be the
   one part of the integration not on cvc5's own leash.
5. Translate the result with the existing `smt_to_lean_nat`. A synthesised
   `ite` folds through the min/max peephole for free.
6. Fall back to `--infer ai-cegar` when synthesis returns no solution, and to
   nothing when there is no LLM configured.

**Where the value is.** Not in replacing the LLM for hard cases — SyGuS will
lose there — but in making the easy 80% reproducible and free, and in giving
the CEGAR loop a *starting* candidate instead of a blank prompt.

**Also worth trying.** The same machinery synthesises the *missing invariant
conjunct*: fix the shape as `inv ∧ ?B` and synthesise `?B` under the same
constraints. That is option 6 done properly, with a grammar bounding how
strong the repair may be.

**Effort.** A day and a half, most of it steps 3 and 4.
