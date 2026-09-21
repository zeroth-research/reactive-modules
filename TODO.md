- houdini: store results to artifacts
- Try using facts from Vampire to generate tactics; dump to artifacts/
- Prune selectively -- if Vampire proves inv but not ranking, try preserving that inv
- [later] Cooperative infer through artifacts: smt + vampire in parallel for 4 seconds; then one after another (with known artifacts) and nuterm; then ai-cegis; then ai-fix
- [later] Try vampire instead of lean-smt in hammer
- In Sygus, Houdini, ... (all that uses templates), allow generating the templates from the system -- simple way is to look at the definition of the system and guess the right template from that (linear, quadratic, using max/min, ...). More complex would be to unroll the transition relation (as SMT formula) several times and see what expressions we get and classify then what kind of template we might need.

## From the matrix

One pass, 2026-09-18, 1158 measured cells over 216 rows
(`/tmp/verith-bench.noindex/results.json`, rendered as
`python/tests/bench_matrix/matrix.html`). Numbered by priority; every count is
cells from that pass. No soundness defect: nothing verified a row known to
fail, and every `REFUTED` landed on a `truth=fails` row.

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

18. **The pass is ~67% of what the page claims.** hybrid (26 rows) and petri
    (35 rows) have zero runs on every column -- the two newest suites, and the
    two whose `truth` is measured by `test_rows.py` rather than transcribed.
    With houdini at 37/216 and vampire at 36/216, 570 of 1728 cells are blank.

    **Sized, so it is not started blind.** hybrid + petri is 61 rows over 9
    routes = 549 cells, and the API key is present so every route runs.
    Measured throughput on this machine: **50 s per cell of wall clock**, so
    **about 8 hours**, and that is with `smt-linear` -- `ai` and `ai-cegis`
    cost 4 to 7 times as much per cell by recorded time, so the real figure
    is higher. An overnight job, not an afternoon one.

    Note the gap that makes it so: the *recorded* `gen`+`build` seconds
    average 14 s, and the wall is 50 s. The other 36 s is per-cell process
    overhead -- a fresh `uv run verith`, the shared `.lake` symlink, the
    manifest, the teardown. Anyone estimating from `results.json`'s own
    timings will be out by a factor of three.

    **Done for these two suites** -- 549 cells in 4h29m, results in
    `/tmp/verith-bench.noindex/results-hybrid-petri.json`. And the answer is
    not "blank" but two different facts:

        hybrid (234)  178 NO-CERT  37 UNSUPPORTED  12 REFUTED   7 PROOF-FAIL   0 VERIFIED
        petri  (315)  189 NO-CERT  43 UNSUPPORTED  20 REFUTED  14 PROOF-FAIL  45 VERIFIED
                                                    3 TIMEOUT   1 GEN-FAIL

    **hybrid is out of reach for every route** -- 26 rows, 9 routes, not one
    certificate -- and petri verifying 45 is what says why: the shutout is
    hybrid's **Real dynamics**, not the suites being new. Every route's Real
    limitation bites at once. The page should say that rather than show
    emptiness.

    Per route over the 61 rows: `houdini` 14, `ai-cegis` 13,
    `houdini-vampire` 13, `smt-linear` 3, `sygus` 2 (4 since 21), and **0
    for `ai`, `nuterm`, `vampire` and `fbk-proveit`**.

    This once said that `ai` certifying none while `ai-cegis` certifies 13
    was worth its own look. It is not: **42 of `ai`'s 61 cells are the CLI
    gate** -- it infers a ranking function `rule_globally` cannot take, so
    it refuses every safety row -- and it was only ever eligible on the 19
    buchi ones, where it produced a certificate 9 times and Lean rejected
    all 9. The two routes were never running the same cells.

    Soundness holds on all 549: **no `REFUTED` off a row declared `fails`,
    and no `VERIFIED` on one**. Worth more than usual here, because these
    are the two suites whose `truth` is *measured* by `test_rows.py` rather
    than transcribed -- nine independent routes agreeing with the simulator
    is evidence in both directions, in the one family where 20's refutation
    rule is calibrated.

19. **houdini and houdini-vampire agree on every verdict on the 37 rows both
    ran** (0 of 37 differ), so the split has so far produced none of the
    "which half of the route the difference is in" it exists for. Either run
    both everywhere so the comparison means something, or drop one.

    **Answered: keep it.** Over the 61 hybrid and petri rows both ran, they
    differ on two -- and the two go in *opposite* directions, so each prover
    reaches somewhere the other does not:

        hybrid/m_tank_dist/refills   houdini NO-CERT    houdini-vampire PROOF-FAIL
        petri/mutex/semiflow         houdini VERIFIED   houdini-vampire PROOF-FAIL

    **One cause underneath both: the vampire prover minimises worse.** On
    `semiflow` it keeps a *superset* of cvc5's invariant -- the same four
    conjuncts plus three more -- and Lean then times out at 2000000
    heartbeats on the bigger certificate, where cvc5's four close. On
    `m_relu_net16` it is the same story at 50 facts against 1. And on
    `refills` that same reluctance to cut is what lets it reach a
    certificate cvc5 never finds. So the fix is one thing, not two: minimise
    the vampire path as well as the cvc5 one, and `semiflow` should verify
    while `refills` keeps its reach.

    `houdini-vampire` is **the only route in the matrix that derives a
    certificate for `refills`**:

        inv   (=> (not s1) (<= 5.0 s0))
        rank  (ite s1 (to_int (+ 49.0 (* (- 4.0) s0)))
                      (to_int (+ 13.0 (* 4.0 s0))))

    cvc5-backed `houdini` gives up on it ("at most 18 ranking functions
    dropped on every reached and sampled round"), and every other route is
    `NO-CERT` or `UNSUPPORTED`. Lean then fails with unsolved goals on the
    floored piecewise Real rank, so the cell is not a proof -- but the
    *search* reached somewhere cvc5 did not, which is exactly the "which
    half of the route the difference is in" this column exists for. Thirty
    seven rows of agreement said nothing because they were rows neither half
    found hard.

    A second, qualitative difference from the same work: since `5101824`
    both solvers read a matrix-shaped module, and on `m_relu_net16` the
    vampire prover returns a 50-fact invariant where cvc5 cuts it to one.
    Same verdict, very different certificate -- which a verdict-only
    comparison also cannot show, and which suggests the column should be
    compared on *certificates*, not just verdicts.

    **The minimisation is fixed** (`15d2e5a`), and "the vampire prover
    minimises worse" was the symptom rather than the cause. `_minimise` is
    solver-independent and was doing its job; what differed was the step
    before it. `_shrink` asks "which facts do the preservation proofs use?"
    *once*, of everything Houdini kept, and an unsat core is what the
    refutation happened to touch rather than the least it could have -- so
    the wider the pool, the more it touches. cvc5 cut 87 facts to 9;
    Vampire cut the same 87 to 35.

    So the cut is now taken **again over its own result** until a round cuts
    nothing, which costs almost nothing: a round's first query is one the
    last round's core already proved, so it is a proof either solver finds
    fast, and a round that cannot answer leaves the round before it
    standing. Vampire goes 35 -> 13, and then `_minimise` reaches 4 where it
    reached 7:

        cvc5     cut 9 of 87  -> minimised 4     (unchanged)
        vampire  cut 35 of 87 -> minimised 7     before
        vampire  cut 13 of 87 -> minimised 4     after

    The two solvers now return the **same certificate**, which is what this
    column pair exists to compare, and the vampire half is *faster* for it
    -- 8.9 s over 45 calls against 2.8 s over 39, because `_minimise` is one
    call per fact and had 22 fewer to ask about. `lake build` closes it, so
    `semiflow` goes `PROOF-FAIL` -> `VERIFIED` as predicted; `refills`
    derives the same invariant and rank as before, so the reach is kept.

    **The certificate comparison is done too** (`compare_certs.py`), and it
    had to be logical rather than textual -- which the first row it meets
    proves: on `fbk/m_countdown/InvBase` all seven routes derive `0 <= s0 <=
    100` and write it four ways, so a string diff sorts them into four
    answers where there is one. Each invariant is parsed back through the
    module's own cvc5 encoding and compared as a formula, and the groups are
    then ordered by *implication*, which is the relation worth having: an
    invariant that implies another is the stronger claim.

    What it says about the column pair, over the re-measured pass: **11 of
    16 rows are one invariant, 5 are ordered, and the implication goes both
    ways** -- cvc5 stronger on `prodcons-rr/empties` and `reset/slots-le`,
    Vampire on `mutex/mutex`, `sem-cont/semiflow` and `sem/integral`. So the
    pair does earn its place on certificates where its verdicts, now
    identical on all but one row, say nothing.

    Over all routes and both passes, of 148 rows where more than one route
    found a certificate: **79 one invariant, 64 a chain, 5 mixed, 0 wholly
    incomparable.** The routes differ in how much they prove rather than in
    what they prove.

    One correction worth recording, because the first count was wrong and
    the code was what made it wrong: "0 incomparable" was an artefact of
    reporting a row as `stronger` when *any* pair of answers was ordered. 19
    rows have three or more answers, and on 5 of them some pair is ordered
    and some pair is not -- `fbk/m_lex/LexLinComb`, where `smt-linear` keeps
    `s0 >= -2` and nothing else says it, is the shape. The verdict counts
    pairs now, and `mixed` is its own answer.

20. **`truth` is `None` for all 30 limits/buchi and all 57 svcomp/buchi rows**
    -- 40% of the page, where the matrix can only say a route answered, not
    that it answered correctly.

    **Do not fill these in with `sim.observe`.** Measured over all 87: it
    reports `fails` on five, and four of those five -- `m_countdown/
    Countdown`, `m_lex/RankLex`, `m_deep/Deep64`, `svcomp/genady/terminates`
    -- are `VERIFIED` by four or five routes each, with Lean proofs. The
    rule is why: a buchi row is refuted when the property recurs fewer than
    `TAIL` = 4 times in the last quarter of a `STEPS` = 200 tick run, so it
    assumes a recurrence period of about twelve ticks or less. That is the
    hybrid and petri family it was written for; `m_countdown` counts 100
    down to 0 and resets, period 101, and hits its property at most once in
    a 50-tick tail. **On this family it is a false-refutation machine**, and
    `test_rows.py`'s own warning applies to it -- a wrong `truth` turns every
    honest `REFUTED` into what looks like a route bug.

    A further 12 of the 87 error rather than answer, and they are 8's cells
    plus three more: the simulator carries the same scalar assumption
    (`sim: Int([3,1]) is not a 1x1 component`, `Bv1([1,1])`), `m_max`,
    `m_min` and `m_argmax` raise a matmul shape error, and `m_uninterp`
    cannot be evaluated at all.

    So this needs either a period-aware refutation rule -- which would have
    to be re-checked against the 4 hybrid/petri buchi rows currently
    declared `fails`, since it can only loosen them -- or truth transcribed
    from somewhere upstream, which for the svcomp rows means the task
    definitions the benchmarks were converted from. The one row that looks
    genuinely `fails` is `limits/m_relu_input/ReluInputNoPre`: `NO-CERT` on
    all seven routes, and it is the variant with the precondition removed.

    **The rule is fixed** (`fb5170f`, `9683f98`), and it wanted to be
    period-*free* rather than period-aware. `G F p` cannot be refuted by any
    finite prefix, so the only thing a run can prove is a **cycle**: a
    return to a configuration it has already been in, with `p` false all the
    way round. The inputs along that loop are the ones the run drew, so
    replaying them from the start repeats it for ever. A configuration is
    `(state, held input)` and not the state alone -- the update reads the
    latched input too.

    Measured: over the 19 hybrid and petri recurrence rows it reproduces
    every declared `truth`, all four `fails` among them, each now a concrete
    loop (periods 60, 17, 6, 3) rather than a count. Over the 75 runnable
    rows here it agrees with the tail count on 71 and differs on **exactly
    the four the tail count had wrong**. So the false-refutation machine is
    gone and nothing else moved.

    What is *not* fixed is this item's headline, and the rule says why. It
    proves `fails` and cannot prove `holds`, so of the 87 it fills in one --
    `ReluInputNoPre`, the row this item already singled out. The other 74
    runnable ones close a cycle their property survives, which is evidence
    and not proof, and filling `holds` in from it would assert exactly what
    `sim.py` says a simulation cannot. The upstream half is also not
    available here: the svcomp benchmarks are checked in as C and DSL
    (`benchmarks/svcomp/c`, `.../dsl`) with no `.yml` task definition, so
    the expected verdicts are not in this tree.

    One thing the fix turned up that was invisible before: 18 of the 19
    recurrence rows close a cycle at all, and `hybrid/m_reactor/cools` does
    not -- a Real plant whose core temperature is a fresh float every tick,
    so no configuration is ever revisited exactly. Its `holds` is "never
    contradicted" where its neighbours' is "no cycle avoids it". That is the
    shape of what the remaining 74 would be worth, if they were filled in
    from runs: `--pre`-sampled Real dynamics mostly cannot close a cycle.

21. ~~**The Real-state gate refuses 114 cells before any search.**~~
    **Half done** (`--infer sygus`); the rest is sized below.

    Counted over the 549-cell hybrid+petri pass, the non-`VERIFIED` cells
    are not what their verdicts say. 80 `UNSUPPORTED` are the CLI declining
    a property *kind* (`ai` refuses all 42 safety rows, `sygus` and
    `fbk-proveit` all 19 buchi ones) and 32 `REFUTED` are correct
    refutations on `fails` rows. Of the 392 real failures, **114 are a
    sorts gate that runs no solver at all** -- and that is what keeps
    hybrid at 0/234, because all 26 hybrid rows carry a Real component
    where only 9 of 35 petri rows do.

    * **`sygus` (22 cells) -- done.** The route was the one that never
      adopted the `Reading` layer: it stated its `synthFun` over the state
      *components*, assumed each was an `Int`, and refused everything else
      at `SynthContext.build`. It now takes the components in their own
      sorts -- `pre` and `trans` are the module's own transition, which is
      where the sorts have to match -- and makes the grammar affine over
      their **columns**, which also gets it Bool, bitvector and
      matrix-shaped state, not only Real.

      The part worth keeping: a Real column here is **not** read through
      `--infer smt-linear`'s floor. That reading exists because a *ranking
      function* has to land in `Nat`; `sygus` only ever states an
      invariant, and flooring a predicate is not conservative in either
      direction -- `to_int s0 + to_int s1 <= 1` holds at `s0 = s1 = 0.6`
      where `s0 + s1 <= 1.0` fails. Measured both ways: the floored version
      finds nothing on `petri/sem-cont`, the Real one proves it. That is
      now `component_readings(floor_reals=)`, with the counterexample in
      the docstring so the next route does not rediscover it.

      Measured over all 61 rows, `--redo`: **2 `VERIFIED` (`sem-cont/mutex`,
      `sem-cont/semiflow`), 0 regressions**, and 17 of the remaining 20
      formerly-gated cells now return *a proof that the space is empty*
      rather than a refusal -- which is the honest answer the route exists
      to give. Soundness clean: no `VERIFIED` on a `fails` row.

    * **`vampire` -- done for Real, and only 9 of the 35 were Real.**
      The count was the surprise: lifting the sorts gate admitted 9 cells,
      and **26 are refused on a Bool column**, which is a different problem
      (below). The 9 now search honestly -- 120 s, 4 Vampire calls -- and
      **not one of them produced a certificate**. Unlike `sygus`, this gate
      lift bought no verdicts: the intervals-and-differences templates
      state single columns and pairwise differences, and `sem-cont`'s
      invariant is a *sum*. The column now reads 23 searched, 26 refused on
      Bool/bitvector, 12 refused at the `ite` branch cap.

      Two things were worth the trip. A Real column gets an interval with
      *integer* endpoints, because Vampire reports an answer as a literal
      and this route reads an integer one -- a restriction on reach and not
      on soundness, since `_checks_out` re-asks cvc5. And the rank floors
      where the invariant does not (`hrank` lands in `Nat`), so one route
      wants **both** readings of one column, which is what earns
      `floor_reals` its existence rather than a per-route rule.

      The bug underneath: **`script_for` never passed its term through
      `decimals`**. A rational prints as `(/ 1 2)`, whose arguments are
      Int, and Vampire answers "invalid sort $int for interpretation /".
      `houdini_solver` has documented that since it began routing its own
      scripts through the function; this route did not, and nothing noticed
      because the sorts gate refused every module with a rational in it.
      Six of the nine failed to parse until it was fixed.

