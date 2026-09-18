- houdini: store results to artifacts
- Try using facts from Vampire to generate tactics; dump to artifacts/
- Allow --pre for more configurations
- Prune selectively -- if Vampire proves inv but not ranking, try preserving that inv
- [later] Cooperative infer through artifacts: smt + vampire in parallel for 4 seconds; then one after another (with known artifacts) and nuterm; then ai-cegis; then ai-fix
- [later] Try vampire instead of lean-smt in hammer

## From the matrix

One pass, 2026-09-18, 1158 measured cells over 216 rows
(`/tmp/verith-bench.noindex/results.json`, rendered as
`python/tests/bench_matrix/matrix.html`). Numbered by priority; every count is
cells from that pass. No soundness defect: nothing verified a row known to
fail, and every `REFUTED` landed on a `truth=fails` row.

1. **`houdini-vampire` renders nowhere -- 155 cells, 112 `VERIFIED`, dropped on
   the floor.** Two causes: `run_matrix.py` records those runs with
   `"route": "vampire"`, and `render.py:1018` builds the column list from that
   field (`any(x["route"] == r ...)`) while every cell lookup uses the `::` key
   suffix -- and `route_order` comes from `ROUTES`, which has 8 names and no
   `houdini-vampire`. The page shows a 36-cell `vampire` column instead of the
   155-cell one that worked. Fix both halves; it changes what the page says,
   not just how it looks.

2. ~~**A `Real` component has no ranking function.**~~ **Done** (`bf01f67`).
   A Real column is `(to_int s0)`, scaled first by `denominator_scale`.
   The reading alone was not enough and would have turned 12 honest
   refusals into build failures: `linarith` reads `⌊x⌋` as an opaque `Int`
   atom and cannot reach a hypothesis about the `Real` under it. `has_floor`
   was detected and gated nothing; it now gates `cert_floors`, which states
   each floored quantity's sign in `Int`. `m_lra_lin` and `m_lra_half` both
   build end to end through a route that refused them outright.

3. **`matMin`/`matMax` never reduce -- 8 cells** (KNOWN_ISSUES #26). `OpMax`
   and `OpMin` are `PROOF-FAIL` with `linarith failed` on all four columns that
   produce a certificate. `--pre-check cvc5` says the obligations hold in
   ~4 ms, so the fold is the whole problem. Redefine over `List.ofFn`
   (`ofFn_succ`/`ofFn_zero` are already in `simp_mat`). Also unblocks the
   `fbk_bridge` equivalence for these modules (#31).

4. **sygus gives up at the 5 s default -- 30 of its 34 `NO-CERT`s are budget
   exhaustion**, all stopping at 5.6-5.9 s (`DEFAULT_CALL_MS = 5000`,
   `smt_query.py:41`) inside a 300 s cell budget; its mean gen over all 155
   cells is 1.7 s. Exactly 1 of the 34 is a proof of emptiness -- so the
   README's "a bounded shape that comes back empty is a proof" is true of
   smt-linear (87 of 103) and backwards for sygus. Cheapest column improvement
   available: it costs a config default.

5. **`argmax_1d` does not reduce -- 4 cells** (KNOWN_ISSUES #30). `OpArgmax`,
   same four columns, same failure and same shape of fix as 3.

6. **`lean2vmt` models only `state`/`statenext`, so external inputs have no VMT
   counterpart -- 13 cells.** All 13 are svcomp `houdini-inv` rows with
   `truth=holds` whose invariant is inductive by construction -- the easiest
   certificates on the page, and the only thing between `fbk-proveit` and the
   whole svcomp safety column. VMT-LIB can carry an input as an unconstrained
   variable absent from the next-state relation.

7. ~~**Tuple- and matrix-shaped components have no route.**~~ **Partly done**
   (`c2264db`). `smt-linear` reads a matrix component as one column per
   element through the tuple selectors `smt_to_lean` already rendered, and
   `m_relu_vec` builds. Finding the invariant unaided does not follow: cvc5
   will not state the quantified form once a tuple is in it, and the
   counterexample loop did not finish in 200 s at width 0, so these reach
   `NO-CERT` with a note unless the invariant is supplied or resumed -- see
   4. **Still open: houdini (7 cells) and vampire (12).** Their candidates
   and templates are stated over components rather than over readings, so
   the shared `readings` layer does not reach them; each needs its own
   candidate machinery taught about elements.

8. **`nuterm` reads only scalar `Int` -- 15 cells**: 7 `Int([n,m])`, 5 `Real`,
   3 `Bool`. **Not one fix, and bigger than it looks.** The refusal is in
   `benchmarks/svcomp/_farkas.read_system`, but the scalar assumption is the
   procedure's data model, not that check: `_nodes._eval` wraps every read
   into a 1x1 array (`np.array([[r[0]]])`) and says so in its docstring, and
   `node_view` gives each non-affine term one symbol (`Node(kind, sym,
   tuple(r[0] for r in reads))`). A matrix component means one column per
   element through `_eval`, `values`/`opaque`, the region machinery and the
   Lean emitter's `Vector n Int` state. The current refusal is at least
   honest -- it names the wire and its sort -- so this is worth doing
   deliberately or not at all.

9. ~~**smt-linear searches the ranking under the invariant `true`.**~~
   **Done** (`946e6ec`). The rank and `k` invariant rows are one query,
   widening `k` from 0. Searching the invariant first does not work and is
   why it is joint: `true` is inductive, so an invariant asked for on its
   own comes back vacuous and the rank is no better off. `m_toward5` and
   `m_lex` were both reported as proofs that no linear rank exists; both
   have one, and `m_lex`'s is a lexicographic argument as a single linear
   rank. `m_twovars` is still empty at every width.

10. **`Uninterpreted` crashes instead of refusing -- 7 cells, the matrix's only
    `GEN-FAIL`** (KNOWN_ISSUES #27), all `m_uninterp/OpUninterp`, one per
    column. `fbk-proveit` already refuses it cleanly in `check_module`; every
    other route dies in a traceback out of `native.py`. Hoisting that refusal
    into the shared front end is separable from the open `opaque`-vs-reject
    decision and turns a crash into an honest verdict today.

11. **`NO-CERT` conflates three different measurements.** Of 312: 88 are a
    proof the space is empty, 34 are budget exhaustion, ~190 are the route
    declining the shape. The underlying messages already distinguish them
    cleanly; the verdict does not. Split it.

12. **`Certificate.lean`'s heartbeat floor is 400000** (`tactics.py:642`,
    unless `slow`), while every other generated file carries 2000000
    (`scalar.py:94,270`, `circ.py:221,251`, `fbk_bridge.py:141`) -- and raising
    exactly this budget is what unblocked the petri nets. 6 cells hit the
    limit; 2 of the 6 already ran at 2000000, so those want the tactic plan
    looked at rather than the budget.

13. **`(kernel) unknown constant 'hrank'` is always a cascade** -- all 6
    occurrences follow a `whnf` heartbeat timeout. Suppress the follow-on error
    so the log names its own cause instead of a kernel symbol.

14. **The `ai` route writes the model's string straight into `Data.lean`.** 2
    of its 15 `PROOF-FAIL`s are unparseable rather than wrong:
    `fun s => s 0 0.toNat` (Lean reads `0.toNat` as a decimal literal) and
    `Real.toNat`, which does not exist. `ai-cegis` catches these by
    re-querying; `ai` has no such loop. An elaboration pre-check turns them
    into either a repair or an honest `NO-CERT`.

15. **`--pre` is refused by ai / nuterm / fbk-proveit** -- 38 `UNSUPPORTED`
    cells now, but 50 rows carry a `--pre` and 29 of those are petri, so this
    becomes ~150 cells once the missing suites run: the largest single category
    on the page. Sizes "Allow --pre for more configurations" above.

16. `fbk/m_step2/NiS2Odd3::fbk-proveit` -- `(kernel) application type mismatch`
    in `Certificate.lean`. A certificate ic3ia produced that `vmt2lean`
    rendered ill-typed. The only one of its kind, so likely a single
    translation rule.

17. `svcomp/genady/houdini-inv::fbk-proveit` -- cvc5 "Failed to reconstruct
    term" over `@purify_3` / `to_real`. Same family as the `mod` note on
    `fbk/m_step2/InvMod`: the solver rewrites the witness into a vocabulary
    `vmt2lean` cannot render back.

18. **The pass is ~67% of what the page claims.** hybrid (26 rows) and petri
    (35 rows) have zero runs on every column -- the two newest suites, and the
    two whose `truth` is measured by `test_rows.py` rather than transcribed.
    With houdini at 37/216 and vampire at 36/216, 570 of 1728 cells are blank.

19. **houdini and houdini-vampire agree on every verdict on the 37 rows both
    ran** (0 of 37 differ), so the split has so far produced none of the
    "which half of the route the difference is in" it exists for. Either run
    both everywhere so the comparison means something, or drop one.

20. **`truth` is `None` for all 30 limits/buchi and all 57 svcomp/buchi rows**
    -- 40% of the page, where the matrix can only say a route answered, not
    that it answered correctly.
