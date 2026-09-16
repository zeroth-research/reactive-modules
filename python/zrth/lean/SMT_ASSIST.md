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

`--smt-tactics` acts on the predicates *as supplied*, and then again on the
inferred ones. Running only the first way was the same mistake `--pre-check`
used to make: every question `solver_hints` asks is asked *under* the
invariant, so with `--infer` there was no invariant to ask under and the run
printed "nothing cvc5 can add to the plan" before the route that would give
it something to add had run. `cert_facts` and `cert_thin` were therefore
`skip` on every inferred certificate. `main._replan` restates both the hints
and `predicate_facts` once the route has answered, from the `inv_smt` /
`ranking_smt` it kept -- which matters twice over, because `features_for`
takes `nonlinear` from the facts *over* the text scan rather than merging
them, so stale facts did not merely omit `nlinarith` but removed one the
printed Lean had already justified.

`--pre-check` no longer does either: run there, the invariant it would check does
not exist yet, so it reported "nothing to check" and the certificate that
reached Lean went unchecked. With `--infer` it runs **after** inference, on
what `magic` returned. `magic` hands back Lean text, which nothing parses
back, so `ai-cegar` keeps the SMT-LIB it gave the solver in
`CertificateData.inv_smt` / `.ranking_smt` and the check restates the
obligations from that.

`--infer ai` is still out of reach, and it is the route that needs it most:
it asks an LLM for Lean and verifies with a second LLM call, so there is no
SMT-LIB anywhere on that path and no Lean parser to make one. The pre-check
says exactly that rather than reporting nothing. Closing it means prompting
that route in SMT-LIB too, as `ai-cegar` does.

---

## 1. `--pre-check=cvc5`

A failing `lake build` cannot distinguish "the obligation is false" from "the
tactics are too weak", and it takes 9-79 s per case to not distinguish it.
cvc5 answers all three obligations in milliseconds, with a counterexample.
Which three depends on the property: `init_inv`, `step_inv` and `hrank` under
`--buchi`; `init_inv`, `step_inv` and `inv_imp_P` under `--safety`, where the
invariant has to imply the property and there is no ranking to rank.

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
$ verith m.py --buchi '(= s0 0)' --invariant '(and (>= s0 0) (<= s0 100))' \
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
condition in it** (`solver_hints` runs before `smt_predicates_to_lean` and
passes `share=False`; after inference `_replan` reprints the same way, and
reprints *every* predicate rather than the two that were inferred, since a
settled condition can come from the property). That costs almost nothing: settled conditions only
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
branches really are collapsed, and the certificate still takes 70 s.

### Where the 70 s actually goes

`set_option profiler true`, elaborating the file directly:

| | |
|---|---|
| **typeclass inference** | **17.7 s** |
| simp | 12.2 s |
| norm_num | 6.2 s |
| ring | 2.6 s |
| everything else | < 1 s each |

and, replacing `hrank`'s proof with `sorry`, the whole rest of the file --
`init_inv`, `step_inv`, `hinv'`, `hinv`, `buchi`, every definition -- comes
to **5.2 s against 75.2 s**. So `hrank` is 93% of the certificate, and
inside it typeclass inference is the largest single item, bigger than simp
and about seven times anything `split_ifs` could have saved.

That is `Real`'s algebraic hierarchy. Every `≥`, `+`, `*`, `⌊·⌋` and
`Int.toNat` in the goal triggers an instance search through Mathlib's
ordered-field tower, and `norm_num`, `ring` and `linarith` each re-resolve
them on every branch. Pruning branches divides that cost; it does not touch
the per-branch constant, and the per-branch constant is the bill.

**So the cost model is wrong in kind, not in calibration.** `n_branch` and
`n_conditions` count branches; what is being paid for is the prep chain
walking a large goal, over and over, synthesising instances each time.

The same profile on an *integer* net says it even more plainly.
`NN2Deep5`, whose branches the min/max fold had already removed entirely:

| | |
|---|---|
| **`norm_num` tactic execution** | **15.3 s + 32 s = 47.3 s** |
| typeclass inference | 12.5 s |
| simp | 1.0 s |
| **`omega`, which closes the goal** | **0.79 s** |

48.7 s of tactic execution, 47.3 s of it in a canonicalisation step that
`omega` does not need -- it normalises numerals itself and reads
`min`/`max`/`Int.toNat` natively. Dropping `norm_num at *` from prep for a
non-Real state takes that case from **75.9 s to 7.0 s** with the same
proof and no errors, and the full matrix unchanged at 0 regressions.
Restricting it to the goal instead saves nothing (77.9 s).

Over the reals it has to stay: goal-only leaves two of `NN2RealAllPos4`'s
obligations unproved, because `linarith` needs the hypotheses normalised.

### Writing the instances explicitly cannot help

A reasonable-looking idea, and measurably wrong: emit `Real.add x y` or
`@HAdd.hAdd ℝ ℝ ℝ instHAdd x y` in place of `x + y`, to spare the
elaborator the instance search.

`Real.add` does not exist -- `x + y` over ℝ elaborates to
`@HAdd.hAdd ℝ ℝ ℝ (@instHAdd ℝ Real.instAdd) x y`, so there is no such
constant to bypass to. And the explicit form saves nothing, because the
search is not happening where it looks:

| | typeclass inference | elaboration |
|---|---|---|
| `Certificate/Data.lean` -- every `+`, `*`, `≥`, `⌊·⌋` and numeral, in notation | **6.17 ms** | 8.01 ms |
| `Certificate/Certificate.lean` -- the proofs | **17,700 ms** | 322 ms |

Source-level resolution is 0.03% of the bill. By the time the tactics run,
`ranking`'s instances are already resolved and baked into the term; the
17.7 s is the tactics synthesising instances for terms *they* build while
rewriting, which no annotation in the source reaches.

Per class, of the searches over 1 ms:

| class | total | calls |
|---|---|---|
| **`CanonicallyOrderedAdd`** | **2641 ms** | **862** |
| `CharZero` | 322 ms | 298 |
| `NeZero`, `NatCast`, `AddRightMono`, … | < 30 ms each | |

ℝ is not canonically ordered, so all 862 of those are *failures* -- simp
and norm_num retrying lemmas guarded on a class ℝ can never satisfy, each
one walking the instance tree before giving up. They come to 3.0 s; the
remaining ~14.7 s is in searches under the 1 ms reporting threshold, which
is to say thousands of cheap ones. Death by a thousand cuts, plus one
class retried 862 times.

The lever is therefore the same one that worked for Int: fewer passes over
the goal, not a different way of spelling the arithmetic.

There is no second `norm_num`-shaped win here, though. Real prep runs two
full-context passes -- `norm_num at *` and `simp_all` -- and dropping the
second gives 69.9 -> 66.5 s, 62.4 -> 60.2 s, 67.4 -> 65.8 s on the three
Real cases, with no errors. Three to five percent, inside the noise, and
`simp_all` is load-bearing for the invariants that pin exact values
(`RealConjDisj`, `LRALinear`), so it stays. The Real cost is genuinely
diffuse: no one step to remove.

Two incidentals from the same profile, both cosmetic but in every generated
file: `init_inv`'s `all_goals (cert_prep <;> cert_close)` is flagged
"tactic does nothing" -- `simp_mat` and `simp_defs` have already closed it
-- and `cert_states`'s macro binder `v` is an unused-variable warning
whenever the state is not finite and the body is `skip`.

The flag stays off.

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
names (`Certificate.Data`, `Certificate.Certificate`, …) are identical across
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

## 7. Synthesis: `--infer sygus` and `--infer smt-linear` — **built**

Written as a plan for a SyGuS *ranking function*; built as an invariant
synthesiser and a template query, because that is what the measurements
said. What follows is the plan's own claims against what running them
showed (cvc5 1.3.4, `tests/limits` fixtures), and then what shipped.

**What the plan had right.** `Solver.synthFun` + `addSygusConstraint` +
`checkSynth` does find ranking functions with no LLM and no API key, and
`ModuleSMT.update_state` against `declareSygusVar` state variables is the
encoding — step 2 was exactly right and is what both routes do.

**Step 3 — "give it a grammar" — was half wrong.** A grammar is not
automatically a pruning:

```
                     default LIA    loose grammar    template grammar
m_countdown              10 ms           7 ms             7 ms
m_toward5               4.5 s           8.6 s            2.0 s
m_twovars              14.4 s          12.8 s           14.6 s
m_lex                      --              --             124 s
```

The "linear combinations plus one `ite`" grammar the plan describes is
*slower than no grammar at all* on `m_toward5`. What prunes is fixing the
affine form and enumerating only its coefficients — and even that does not
rescue two variables. `m_lex` is the case that settles it: 124 s for a rank
`--infer nuterm` certifies in seconds, and the answer that comes back is
`ite (-2 + 2*x1 <= 12 - 2*x0 - x1) (100 + 2*x0 + 100*x1) (1 + 10*x0 + 3*x1)`
where `4*x1 + x0` would do.

**Step 4 — "check how it is bounded first" — was the right worry and has an
answer.** `tlimit` does **not** stop `checkSynth`: a 1000 ms limit ran 13.9 s
and returned a solution. It does not stop a quantified `checkSat` either — a
20 s limit ran past 400 s. **`tlimit-per` stops both** (1001 ms and 5007 ms,
`unknown`). No subprocess is needed; `smt_query._solver` already sets the
pair, and `smt_synth.bounded_solver` is that pair for the searches.

**Step 5 — "a synthesised `ite` folds through the min/max peephole for free"
— is wrong.** `smt_to_lean.min_max_of` fires only on `ite (a ⋈ b) a b`, where
the branches *are* the comparison's operands. No solution cvc5 returned here
has that shape, so each one costs a `split_ifs` branch, doubled because
`hrank` mentions the ranking twice.

**Step 6 — "fall back to `--infer ai-cegar`" — was not built, deliberately.**
A route is a row and routes do not chain: a route that silently becomes
another one makes two identical command lines mean different things, which
is the argument `_resume_inv` already makes about inheriting a predicate.
The handoff is through `artifacts/` instead, which is explicit, inspectable
and survives the process.

**Where the value actually was: the invariant, and the `no`.**

```
inv + rank together, four hand-rolled constraints (m_step2)   nothing in 300 s
the invariant alone, hand-rolled                                        10 ms
the invariant alone, addSygusInvConstraint                              11 ms
```

`m_step2` is the one Buchi case `--infer nuterm` misses on the certificate
rather than on reach, and the cause is the invariant: `x` is even, and no
candidate Houdini has says so. A grammar with `(= (mod lin 2) 0)` in it says
it in 11 ms — through the *dedicated* SyGuS-IF invariant track, which is not
a convenience but the difference between 11 ms and not returning.

The other half is the linear template asked as a plain quantified query,
which needs no SyGuS at all:

```
m_countdown  rank        sat    6 ms    (rank = s0)
m_toward5    rank      unsat    2 ms
m_twovars    rank      unsat    2 ms
m_lex        rank      unsat   30 ms
m_countdown  inv k=1     sat   11 ms    (200 - 2*s0 >= 0)
m_countdown  inv k=2   unknown 30 s     -- the width is the cost, not the module
m_step2      inv k=2   unsat    2 ms    -- needs the congruence, as above
```

An `unsat` there is a **proof that the shape is empty**. That is the one
thing neither an LLM nor a learner produces, and it is what the routes were
built around: it goes to `artifacts/` as a `no_solution` note and into the
`--infer ai-cegar` prompt on the next run.

**What shipped.**

* `--infer sygus` (`--safety`) — `magic_sygus.py`. `addSygusInvConstraint`
  over a grammar of affine comparisons plus congruences, with the constants
  seeded from the program's own literals. The conjunction is bounded
  (`--sygus-conjuncts`, default 3) and that is load-bearing: `B -> (and B B)`
  is an infinite space, where "nothing works" can only come back as
  `unknown`, and a bounded one is finite, where cvc5 answers `hasNoSolution`
  — measured at 10 / 25 / 166 ms for 1 / 2 / 3 atoms on `m_step2` with the
  congruences dropped. External inputs are existentially quantified inside
  `pre`/`trans`, so `--pre` works and a module with inputs is in reach
  (`m_relu_input`: invariant found, and all three obligations hold). Writes
  the invariant as a resumable `inv`.
* `--infer smt-linear` (`--safety` and `--buchi`) — `magic_linear.py`. One
  quantified query per shape; widths tried smallest first; a supplied or
  resumed invariant is *strengthened* rather than replaced. Every scalar
  component is a column whatever its sort — `Int` as itself, `Bool` as
  `(ite s0 1 0)`, a bitvector as `(ubv_to_int s0)` — so a Bool-state module,
  which `--infer nuterm` refuses, is in reach and so is a BitVec one.
* **A bitvector column needs a second engine.** Measured: with one in the
  formula cvc5 answers the `exists c. forall s.` query
  `unknown (INCOMPLETE)` in *one millisecond*, at every width — a refusal to
  state it, not a timeout. So when the direct query comes back unknown the
  same question runs as a counterexample-guided loop whose halves are both
  quantifier-free: propose coefficients against the states seen so far
  (linear, whatever the module's sorts), verify against the whole
  transition, add the counterexample. An 8-bit counter's invariant takes 55
  samples and 538 ms. The loop needs a *bounded* candidate space to
  terminate, so its `unsat` is a proof about that box rather than about
  every integer — and the note says so, because the difference is exactly
  what an LLM route would otherwise be told wrongly. The bound is read off
  the program's own constants, like the grammar route's: measured, a tight
  bound is what makes it converge (55 samples inside one, no answer in 200
  outside).
* **`bv_omega` joined the closer list** (`tactics.py`), for BitVec states
  only. A bitvector read as `BitVec.toNat` is arithmetic no other closer can
  phrase, and `decide` — which cannot fire while the state is a free
  variable — *errors* there rather than failing, which escapes the `first`
  chain and fails the build. With it, an 8-bit counter's certificate builds
  in 3.4 s. `BVState` and `BoolState` in the limit matrix still verify.
* `artifacts.STATUSES` gained `no_solution`, told apart from `unknown`
  because the difference is a proof versus a timeout, and a consumer that
  confused them would put a falsehood in a prompt.
* `--infer ai-cegar` resumes `note` artifacts as well as `inv` ones, and
  states them in the prompt as established facts.

**What is still open.** Both searches are integer-only and refuse a Real,
Bool, bitvector or matrix-shaped state by name; the `--safety` template
stops being decidable in practice at two rows, which a CEGIS loop (sample
states, solve for coefficients over the samples, verify, repeat) would
push much further — it turns one hard `NIA` query into a loop of easy `LIA`
ones, and an unsat sample system is still a proof of absence.
