- houdini: store results to artifacts
- Try using facts from Vampire to generate tactics; dump to artifacts/
- ~~Allow --pre for more configurations~~ **Done** -- item 15 below: every route that has somewhere to put one now takes it.
- Prune selectively -- if Vampire proves inv but not ranking, try preserving that inv
- [later] Cooperative infer through artifacts: smt + vampire in parallel for 4 seconds; then one after another (with known artifacts) and nuterm; then ai-cegis; then ai-fix
- [later] Try vampire instead of lean-smt in hammer

## From the matrix

One pass, 2026-09-18, 1158 measured cells over 216 rows
(`/tmp/verith-bench.noindex/results.json`, rendered as
`python/tests/bench_matrix/matrix.html`). Numbered by priority; every count is
cells from that pass. No soundness defect: nothing verified a row known to
fail, and every `REFUTED` landed on a `truth=fails` row.

1. ~~**`houdini-vampire` renders nowhere -- 155 cells, 112 `VERIFIED`, dropped
   on the floor.**~~ **Done** (`73b6d9e`). Both halves were one mistake: a
   column is not a route, and the page asked `infer_route.ROUTES` what the
   columns were. `houdini-vampire` is `--infer houdini` over its second
   solver, so it can never be in that list; and the presence test read the
   `route` field while every cell lookup used the `::` key suffix. The
   harness that measured the columns now lists them (`suites.routes`,
   `column_of`), and a suffix it no longer declares is carried at the end
   rather than dropped. The 155 cells' recorded commands say `--infer
   vampire` -- which meant this search before `25bfb58` split the flag and
   means a different route now -- so each says so in its panel; the verdict
   is what was measured, the command is what no longer reproduces it.
   Re-measuring that column would make the commands true again, which is a
   pass, not a fix.

2. ~~**A `Real` component has no ranking function.**~~ **Done** (`bf01f67`).
   A Real column is `(to_int s0)`, scaled first by `denominator_scale`.
   The reading alone was not enough and would have turned 12 honest
   refusals into build failures: `linarith` reads `⌊x⌋` as an opaque `Int`
   atom and cannot reach a hypothesis about the `Real` under it. `has_floor`
   was detected and gated nothing; it now gates `cert_floors`, which states
   each floored quantity's sign in `Int`. `m_lra_lin` and `m_lra_half` both
   build end to end through a route that refused them outright.

3. ~~**`matMin`/`matMax` never reduce -- 8 cells**~~ **Done** (`be70a43`),
   together with 5. The three reductions enumerate with `List.ofFn` and are
   in `simp_mat`; adding them to `simp_mat` alone had been tried and is not
   enough, because opening the definition only exposes the fold that will
   not compute. Measured: `OpMax`, `OpMin` and `OpArgmax` go `PROOF-FAIL` ->
   `VERIFIED`, the slow Lean suite passes, and 20 `VERIFIED` cells sampled
   over four routes and five suites are unchanged. Order, seeds and
   tie-breaking are untouched, which is what `ManualTests/Argmax.lean` pins
   by `decide`. Still to check: the `fbk_bridge` equivalence for these
   modules (#31) -- the bridge's simp set got the same four defs, but no
   cell of the matrix measures it.

4. ~~**sygus gives up at the 5 s default -- 30 of its 34 `NO-CERT`s are budget
   exhaustion**~~ **Done** (`5fdc214`), and the premise was wrong twice.
   The measurements are worth keeping in the order they landed, because
   each one killed the fix the one before it suggested.

   The reading was right: all 30 stop at 5.6-5.9 s (`DEFAULT_CALL_MS =
   5000`, `smt_query.py:41`) inside a 300 s cell budget, and exactly 1 of
   the 34 is a proof of emptiness -- so the README's "a bounded shape that
   comes back empty is a proof" was true of smt-linear (87 of 103) and
   backwards for sygus.

   **"Cheapest column improvement available: it costs a config default" was
   wrong, and measuring it is what says so.** Six of those cells re-run at
   12-24x the budget: `m_twovars/TvRelational` at 60 s, and `CdBands`,
   `InvLex`, `InvDisj`, `T5Exact`, `InvTwoVars` at 83-121 s against a 120 s
   per-call budget. Every one still did not finish. The grammar is what is
   missing, not the seconds. What came of it is a message that no longer
   sends the reader to spend two minutes arriving back at it (`47dce43`).

   **"A smaller default grammar would trade reach for proofs of emptiness"
   was wrong too, and there is no trade.** The sweep it asked for is all 59
   cells that reach the grammar, at 1 / 2 / 3 atoms separately, and what it
   shows is that **the search is not monotone in the width**. The widths
   nest -- the start rule at width `w` is `A | A /\ A | ...` up to `w` -- so
   an answer found at 2 lies inside the space searched at 3, and cvc5 still
   does not find it there: `wise` finds a two-atom invariant at 3 and times
   out at 2, `T5Exact` finds one at 2 and times out at 3, and `NiLexNe4`
   and `RelTvImplies` find *one-atom* invariants in under a second and time
   out at 3. The ceiling is the worst single width to ask at and it was the
   only one asked. So the fix is not a smaller grammar but the ladder
   `--infer smt-linear` has run all along: widths `1, 2, ... N`, first to
   answer wins.

   Asked only at the ceiling: 28 found, 1 space decided empty, 30 cells
   that learned nothing. A width at a time: **31 found, 22 decided, 6 that
   learned nothing** -- predicted from the per-width sweep and then
   reproduced exactly by the route, no disagreement on any of the 59. The
   column is 28 -> 31 `VERIFIED` with nothing lost, and 22 of its 31
   remaining `NO-CERT`s carry a proof of absence rather than a shrug. The
   cost is the widths that find nothing: 201 s -> 352 s of search over the
   59, bounded by `SmtBudget.phase_ms`, which is what a phase already
   means.

   What is left is a different knob, and the 6 cells that still learn
   nothing are all of one kind: their *one-atom* space does not finish in
   5 s either. A single atom is `|consts|^(columns+1)` linear forms times
   the atom shapes, so those six are 10^5 to 10^6 atoms wide at the
   narrowest rung -- `easy1` at 3 columns and 16 constants, `NoriSharma
   Fig7`/`Fig8` at 7 columns and 5. **That** is where a smaller default
   grammar would be the question this item first guessed it was, and the
   knob is `_CONSTANT_COUNT` (`smt_synth.py`), not `--sygus-conjuncts`.
   Worth its own item if the 6 are worth chasing; the seeding comment in
   `program_constants` argues the constants are the half that must not
   shrink.

5. ~~**`argmax_1d` does not reduce -- 4 cells**~~ **Done** (`be70a43`), in
   the same commit as 3 and for the same reason; 2-D `argmax` went with them
   rather than be left the odd one out.

6. ~~**`lean2vmt` models only `state`/`statenext`, so external inputs have
   no VMT counterpart.**~~ **Done** (`263214f`), and not the way this item
   proposed. "VMT-LIB can carry an input as an unconstrained variable" is
   true and is not the obstacle: `lean2vmt` annotates `:next` only on slots
   written as `var_k statenext`, so an unwritten slot is *already* an
   unconstrained variable. Lean is what refuses -- `TS.transfer` carries
   the certificate back along a **function** from module states to model
   states, and a module state does not say which input produced it. A
   simulation relation would express it; a function cannot.
   What works instead costs no variable at all: measured, all 21 modules
   with inputs read one in exactly one place -- `init`, as the initial
   value of one slot -- and none reads one while stepping. So such a slot
   gets **no `Init_k`**: `x := nondeterministic` is "`x` starts anywhere",
   which is the module's own initial set rather than a superset of it. That
   distinction is the whole point, because this route reports REFUTED and a
   model admitting more than the module would report counterexamples the
   module has not got -- so everything else that touches an input is
   refused in its own words rather than approximated. 8 cells `NO-CERT` ->
   `VERIFIED`; the fbk suite's 39 are unchanged. Of the 15 svcomp rows
   still not verified, 8 are `--pre` (15) and 5 are those precise refusals.

7. ~~**Tuple- and matrix-shaped components have no route.**~~ **Done**
   (`c2264db`, then `30fe154`, `e129ba5`, `68ac231`, `5101824`, `4943def`).
   `smt-linear` reads a matrix component as one column per element through
   the tuple selectors `smt_to_lean` already rendered, and `m_relu_vec`
   builds. Finding the invariant unaided does not follow there: cvc5 will
   not state the quantified form once a tuple is in it, and the
   counterexample loop did not finish in 200 s at width 0, so those reach
   `NO-CERT` with a note unless the invariant is supplied or resumed.

   **houdini now verifies six of the seven, end to end.** `m_relu_vec`,
   `m_relu_net`, `m_mixed`, `m_relu_net8`, `m_relu_net16` and `m_vec32`, on
   both provers -- so twelve cells across the `houdini` and
   `houdini-vampire` columns, every one of them from refused. The
   certificate is `0 <= v[0]` with rank `v[0]` in all six: the shape the
   modules always had, and unsayable while a fact had to be about a whole
   tuple. The seventh, `m_transpose`, is refused for its `Transpose`, which
   has no cvc5 term at all.

   **"vampire (12)" was two counts added together.** Seven of those cells
   are matrix-shaped; the other five (`m_lra_half`, `m_relu_lra`,
   `m_lra_lin`, `m_lra_conv`, `m_lra_two`) are `Real`, which is a different
   limitation -- a ranking function has to land in `Nat` -- and is not what
   this item is about. So it was 7 and 7, not 7 and 12.

   **The vampire half derives nothing, and that is the honest result.**
   `--infer vampire` can now *state* these: its rows and rank coefficients
   are columns, and the test instantiates the certificate `m_relu_vec`
   actually has into the template and has cvc5 prove it. Vampire does not
   pin the coefficients down -- 120 s each on three modules, 60 s on a
   safety property as easy as this shape gets. Three ReLUs split the round
   into eight branches against six holes, where scalar `m_relu` is two and
   two and is one of the 114 cells the route does verify. No verdict moves;
   what moves is the note, from "cannot read this shape" to "searched it
   and derived nothing", which is what 11 wants to tell apart.

   Two things found on the way, both recorded rather than folded in:
   `c2264db` removed the sort gate that happened to catch `m_transpose`, so
   every SMT route on it had been dying with a `ValueError` traceback since
   -- unmeasured, because the `smt-linear` column has not been re-run since
   that commit (`e129ba5`). And under the vampire prover `m_relu_net16`
   comes back with a 50-fact invariant where cvc5 cuts it to one: correct,
   verbose, and a budget-bound minimisation rather than anything the
   columns did.

8. **`nuterm` reads only scalar `Int`.** **Not one fix, and it is three
   different ones** -- measured, 17 cells rather than 15, and the split is
   what decides whether any of it is worth doing:

   * **7 matrix `Int`** (`m_relu_vec`, `m_relu_net`, `net8`, `net16`,
     `m_vec32`, `m_mixed`, `m_transpose`). The one this item described, and
     the only one that keeps the machinery's arithmetic: a matrix of
     integers is just more integer columns.
   * **5 scalar `Real`** (`m_lra_*`, `m_relu_lra`). A different arithmetic,
     not a shape -- the rows are integer-affine and the rank lands in `Nat`.
   * **4 scalar `Bool` + 1 `BitVec`** (`m_boolint` x3, `twobit_lia`,
     `twobit`). **Not a matrix problem at all, and not a cheap widening**:
     the columns are integer-*affine* (`affine_coeffs` fits by 0/1
     substitution and then verifies the fit), so a Bool column is a
     *branch*, not a wider element type. Reading it as `If(b,1,0)` does not
     help, because the transition is affine in `b`'s branches rather than
     in that reading.

   For the matrix slice the sites are known and a probe gets most of the
   way: making `_nodes._eval` shape-aware -- `np.array(r).reshape(shape(w))`
   instead of `np.array([[r[0]]])` -- already walks `m_relu_vec` and
   `m_vec32` to correct per-element transitions (`If(w0_0 - 1 > 0, w0_0 - 1,
   0)`). What is left is `node_view` giving a non-affine term one symbol per
   *element* rather than one per wire (`Node(kind, sym, tuple(r[0] for r in
   reads))`, `opaque.update(zip(t_.write, [[sym]]))`), `read_system` making
   a column per element, and `System.sp_syms` / `index` / `entry`, which
   read `[0]` off each var and so need the wire-vs-column split that
   `magic.houdini.Column` is on the other side of the house.

   **What should decide it**: the payoff is now *second-route coverage*, not
   reach. `--infer ai` verifies all 7 of those modules and `--infer houdini`
   verifies 6 of them since `68ac231`, so this buys a second opinion on
   modules that are already certified, at the cost of a wire/column split
   through ~1550 lines (`_farkas.py`, `_nodes.py`), `_termination.py`,
   `magic/learn.py` and five test files. The current refusal is honest --
   it names the wire and its sort.

9. ~~**smt-linear searches the ranking under the invariant `true`.**~~
   **Done** (`946e6ec`). The rank and `k` invariant rows are one query,
   widening `k` from 0. Searching the invariant first does not work and is
   why it is joint: `true` is inductive, so an invariant asked for on its
   own comes back vacuous and the rank is no better off. `m_toward5` and
   `m_lex` were both reported as proofs that no linear rank exists; both
   have one, and `m_lex`'s is a lexicographic argument as a single linear
   rank. `m_twovars` is still empty at every width.

10. ~~**`Uninterpreted` crashes instead of refusing -- 7 cells, the matrix's
    only `GEN-FAIL`**~~ **Done** (`fd775b5`), with the premise corrected:
    nothing crashed. It was a clean `Refused`, but raised from the middle of
    codegen and phrased as a missing cell of `ops.py` (`No Lean expression
    mapping for: Uninterpreted (matrix form)`) for a user who asked about a
    module. The op table now answers as a value (`lean_gap`) and
    `native.lean_gaps` asks it of a whole module, before any of the project
    exists and after the route's own precheck; the message names the block,
    the wire and the operator. It covers all 24 unsupported ops, not
    `Uninterpreted` alone, and cannot refuse what generates: no op with an
    unsupported matrix form emits in another Lean column, and the walk
    prunes terms nothing reads. The verdict stays `GEN-FAIL`, which is what
    the page already promises for a module shape refused up front -- the
    open `opaque`-vs-reject decision (KNOWN_ISSUES #27) is untouched.

11. ~~**`NO-CERT` conflates three different measurements.**~~ **Done**
    (`8a8dea4`, `926b641`). Of 312: 88 a proof the space is empty, 34 budget
    exhaustion, ~190 the route declining the shape -- and classifying by
    regex over the prose left ~160 unmatched across a dozen wordings per
    route, with 7 matching two buckets at once.

    Every route records which it was now, and the mechanism is one flag
    rather than one edit per route: `Refused` carries `searched`, defaulting
    to `False` because most refusals are *gates* -- a sort the route cannot
    weigh, an operator with no encoding, a property kind it does not take --
    all reached before a solver starts. `main._infer` writes the note
    centrally, so a route added later cannot forget, and only when the route
    wrote none of its own (`smt-linear`, `sygus`, `houdini` and `vampire`
    each leave a richer one).

    `declined` is the new status, told from `unknown` in the direction
    `no_solution` is: `unknown` means a search ran and did not finish, which
    is a fact about how hard the module is, while `declined` means none ran
    and there is no such fact. `ai` and `ai-cegis` were raising
    `RuntimeError` on running out of attempts -- a traceback, and nothing
    recorded; they refuse like every other route now.

    The harness reads the status and the panel shows it. **The verdict is
    unchanged, deliberately**: `NO-CERT` is load-bearing in the truth
    comparison, the colouring and the counts, and splitting the vocabulary
    would churn all three to say what the panel now says directly. Also
    relieved by `74393b0`: a failed cell's one-line summary is what was
    raised rather than the frames above it.

    Still to do, and it is 18's pass rather than this item's code: the
    counts above are from the old classification, so what each bucket really
    holds is not known until a pass recorded the statuses.

12. ~~**`Certificate.lean`'s heartbeat floor is 400000**~~ **Done**
    (`932ab14`). It is 2000000 now, what every other generated file carries.
    The floor had six rungs climbing back to exactly 2M -- finite state, a
    bitvector, more than 8 slots, more than 32 terms, a branchy Real
    predicate, more than 16 branch points -- and the list still growing,
    which is the tell. **A low floor never bounded a failure**: a tactic
    that hits the cap *throws*, `first` catches that like any other failure
    and moves to a dearer alternative, so it capped the cheap attempts and
    let the expensive ones run anyway.

    Measured, one project apiece with only that number changed:
    `svcomp_collatz_bounded` fails in 29 s at 400k and **builds in 66 s** at
    2M -- so the floor was the whole difference, and it cost 37 s to find
    that out rather than saving any.
    `ChenFlurMukhopadhyay-SAS2012-Ex1.01` still fails, but with `linarith
    failed to find a contradiction` instead of a heartbeat timeout, which is
    the truth about it: cvc5 refutes that certificate's `hrank` outright at
    `s0 = 1`. **A budget error had been standing in front of a wrong
    certificate.** So 4 of the 6 cells move, 2 of them to a proof. The other
    2 (`m_lra_two`) already ran at 2M and want the tactic plan, as this item
    said. Only `n_branch > 32 -> 8M` survives, being the one rung that went
    anywhere above the standard budget.

13. ~~**`(kernel) unknown constant 'hrank'` is always a cascade**~~ **Done**
    (`c8294f3`). A declaration whose elaboration fails is never added to the
    environment, so the next one that mentions it fails again in the kernel.
    `without_cascades` sits next to the build reporting in `project.py`, so
    verith's own `--build-cert` output and the bench harness apply one rule
    rather than two that have to agree; checked against the real failing
    build rather than a fixture. It only ever drops a *later* kernel
    `unknown constant` -- one that is the first thing to go wrong is kept,
    since then it is nobody's shadow and something really is missing.

14. ~~**The `ai` route writes the model's string straight into
    `Data.lean`.**~~ **Done** (`3657574`), and the two cells wanted opposite
    treatment rather than one pre-check.

    `fun s => s 0 0.toNat` is a slip, and Lean's own error is the repair --
    "consider parenthesizing the number". Exactly one parenthesisation
    elaborates, because the projection is of the state *element*; both
    halves checked against Lean rather than argued, `(s 0 0).toNat`
    compiles and `s 0 0.toNat` is the recorded type mismatch. The model
    writes the parenthesised form on nine other cells, so it is rewritten in
    place, as `_unquote` already rewrites the backticks a model wraps an
    answer in.

    `Real.toNat` is **not** rewritten: it could be a floor, a ceiling or a
    truncation, and picking one would be inventing the certificate rather
    than reading it. It goes back as feedback naming the missing constant,
    through the retry loop the route already has, and survives to an honest
    failure if the model keeps it. Such a candidate never reaches `_verify`
    -- there is nothing for an auditor to be right or wrong about in an
    expression Lean will not read.

15. ~~**`--pre` is refused by ai / nuterm / fbk-proveit**~~ **Done**
    (`da78190`, `89b6ad7` fbk-proveit; `8aaaf28` ai; `ffad4ab` nuterm) -- 38
    `UNSUPPORTED` cells, and 50 rows carry a `--pre` with 29 of those petri,
    so it was ~150 cells once the missing suites run: the largest single
    category on the page. Sized "Allow --pre for more configurations" above.

    All 8 of the svcomp safety column's `fbk-proveit` cells were this. **6
    are now `VERIFIED`** and 2 are refused in the encoding's own words (two
    slots starting at one input, which the NA has nowhere to say). The
    39-cell fbk suite is unchanged.

    `nuterm`'s 19: **10 `VERIFIED`**, 6 `NO-CERT`, 3 `TIMEOUT`, from 19
    `UNSUPPORTED`. All eight svcomp `houdini-inv` rows verify, and the
    control says what they needed -- without `--pre` the same cells come
    back "no inductive invariant implying the property was found". Nothing
    that did not verify failed for a `--pre` reason: three are module
    shapes the procedure refuses by name and had been masked by the seed
    refusal firing first -- a `Bool` wire and a `Bv1` wire, which are item
    8, and a step reading a nondeterministic input, which is not on this
    list -- three are the convex-rank limit (a rank that is a non-negative
    sum of ReLUs is convex, and these runs wrap around), three the 300 s
    budget.

    `ai`'s 19 are unmeasured: 8 are `--safety` and stay `UNSUPPORTED` for
    the *kind*, and the other 11 need an API key this environment has not
    got. Its refusal was simply stale -- both prompts already stated the
    obligations with `init_pre`/`update_pre` in them and already wrote the
    predicates into each message, so the row was declining a flag the
    module beside it was using.

    A precondition is a predicate over inputs and the NA model has no
    inputs -- but `init` writes each input it reads to one slot, so reading
    that slot back *is* reading the input, and `--pre` becomes one more
    conjunct of `INIT`. Exactly the module's initial set, not a superset,
    which this route needs because it reports `REFUTED`. Two refusals keep
    it that way: a *latched* input, which `init` never read, and a next
    input that starts no slot -- both would need the NA to say "some value
    satisfying it exists", and it has no quantifier.

    `INIT` and not `TRANS`, though a precondition *is* an auxiliary
    invariant and in an encoding with real input variables would belong in
    both. Here the transition reads no input at all, so `update_pre`
    constrains a label `update` never looks at; the conjunct one could
    write, `PRE statenext`, is a claim about the *initial* inputs asserted
    of an evolved state. Lean catches that rather than leaving it to the
    argument: `TS.transfer` takes a simulation, so `step_maps` has to hold
    for every pair of states and not only the reachable ones. And with
    `PRE` in `INIT` the model is already exactly the module, so there is no
    imprecision for an auxiliary invariant to remove.

    For `ai` and `nuterm` the "both halves" reading is the right one and is
    already what `smt_query` states -- `init_pre e -> inv (init e)` and
    `update_pre e /\ inv s -> inv (update s e)`. `ai` needed nothing for
    it: `Certificate/Data.lean` carries both whatever route ran, so `lake
    build` was always going to check under them.

    `nuterm` takes the init half as `System.assuming`, the entry assumption
    the svcomp harness already uses for a benchmark's own `if (P)` gate.
    That assumption is a predicate over *columns* and `--pre` is one over
    inputs; they meet where they met in the other direction when the
    harness derived the flag (`suites.entry_pre`), at a column whose init
    value is a bare input. The update half needs nothing there either, and
    for a sharper reason than fbk-proveit's: `_farkas.check_supported`
    already refuses every module whose next value reads an input, so the
    step reads none and `update_pre` constrains nothing it looks at.
    Dropping it is exact rather than an over-approximation -- which is the
    thing to keep an eye on, since a route that certifies rather than
    searches must not quietly widen what it assumed.

16. ~~`fbk/m_step2/NiS2Odd3::fbk-proveit` -- `(kernel) application type
    mismatch`~~ **Done** (`0737c49`), together with 17, and the premise here
    was wrong. It is not "a certificate ic3ia produced that `vmt2lean`
    rendered ill-typed" and so not a translation rule: the term the kernel
    rejects is `Smt.Reconstruct.Prop.eqResolve` over a chain of
    `Smt.Reconstruct.Int.sum_ub`, which is **lean-smt reconstructing a cvc5
    proof**. Nothing `vmt2lean` wrote. That makes 16 and 17 one bug and not
    two, which is why they close together.

17. ~~`svcomp/genady/houdini-inv::fbk-proveit` -- cvc5 "Failed to
    reconstruct term"~~ **Done** (`0737c49`), and the "same family" guess was
    right for a better reason than the one given: not a witness `vmt2lean`
    cannot render back, but lean-smt failing to turn a cvc5 proof into a
    kernel-valid term. 17 fails loudly at reconstruction; 16 hands the kernel
    a term it will not take. One cause.

    The repair is neither solver's: a validity check is linear integer
    arithmetic, which `omega` decides, and `omega` builds its own proof
    instead of translating cvc5's -- so where it applies there is no
    reconstruction to get wrong. It is offered before the `smt` that closes
    a check, costing a millisecond failure where it does not apply.

    Measured end to end from ic3ia to `lake build`: both cells go
    `PROOF-FAIL` -> build, and five that already built (`InvBase`,
    `InvDisj`, `InvLex`, `InvTwoVars`, `CdBands`) still do.

    A third member of this family is closed (`89b6ad7`), and it was not
    about a witness: `vmt2lean`'s proof of the first validity check
    generalises each state slot out of the initial condition, and
    `generalize` cannot abstract a slot that a `Decidable` instance still
    mentions -- `with_reducible reduce` rewrites `var_k (trace 0)` to
    `trace 0 k` in the proposition under a `decide` and not in the instance
    beside it. Invisible while `INIT` was all `var_k state == c`; `--pre`
    is the first thing to put a comparison there. The installed certificate
    now leaves `Bool` before generalising. Worth knowing for 16 and 17: the
    route's "processing" step is a real place to repair a rendering, and
    one that reports how many proofs it touched.

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
