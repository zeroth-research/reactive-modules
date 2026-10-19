- Try using facts from Vampire to generate tactics; dump to artifacts/
- Prune selectively -- if Vampire proves inv but not ranking, try preserving that inv
- [later] Cooperative infer through artifacts: smt + vampire in parallel for 4 seconds; then one after another (with known artifacts) and nuterm; then ai-cegis; then ai-fix
- [later] Try vampire instead of lean-smt in hammer
- fbk-proveit: `_REVERT` / `_SMT_CLOSER` (`python/zrth/lean/fbk_proveit.py`) regex-patch the Lean proveit.py generates. A run warns when a patch finds nothing to change while the tactic it hooks onto is still there (2026-10-19). Revisit: fix it upstream in lean-ltl-certifying's template
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

20. **`truth` is `None` on 86 of the 87 limits/buchi and svcomp/buchi rows**,
    so on 40% of the page the matrix can only say a route answered, not that
    it answered correctly.

    The simulator cannot fill them in. Its refutation rule (`fb5170f`,
    `9683f98`) proves `fails` by finding a cycle on which the property stays
    false, and it cannot prove `holds`: of the 87 rows it fills in one,
    `limits/m_relu_input/ReluInputNoPre`. The other 74 runnable rows close a
    cycle that their property survives, which is evidence and not proof. 12
    more error in the simulator rather than answer: it assumes scalar
    components as `nuterm` does (item 8), `m_max`, `m_min` and `m_argmax`
    raise a matmul shape error, and `m_uninterp` cannot be evaluated.

    What would fill them is truth from upstream. For the svcomp rows that is
    the task definitions the benchmarks were converted from, which are not
    in this tree: `benchmarks/svcomp/c` and `.../dsl` carry no `.yml`.

## Pipeline shape

Paths are relative to `python/zrth/lean/`.

- **An artifact does not record the precondition it was found under.**
  `ArtifactStore._stale` drops an artifact written for another module,
  property or proof rule, but not one found under another `--pre`. An
  invariant houdini or nuterm proved under `--pre` holds only from the
  states that admits, so a later `--infer ai-cegis --safety` without it
  resumes it as the whole certificate, and fails `init_inv` instead of
  asking the model.
- **Not every SMT user is on `smt/`.** `sygus._pre`/`_trans` and their
  existential relations are not on the obligation builder
  `smt/obligations.py`; smt-linear uses `_bound_vars`/`_fresh_consts`
  instead of `ModuleEncoding.fresh`; houdini and vampire parse with
  `env.parse_expr` (their candidates may be Real) where the rest use
  `ModuleEncoding.parse`; nuterm's z3 encoding and
  `_farkas.Inductive`/`LexDecrease` wait for the nuterm move (next item).
- **nuterm needs the editable install.** `magic/learn.py` imports
  `benchmarks.svcomp`, which is not packaged. Left for the branch that moves
  the engine, to be rebased onto `mch/verith-1`.
- **Check the Lean encoding against the SMT one.** `pre_check` asks cvc5
  about the certificate (`init_inv`, `step_inv`, `hrank`); nothing asks
  whether `native.py`'s Lean and `smt_encode`'s cvc5 terms agree about the
  module, which they do only by construction. The equivalence theorems do
  not cover this: they can stay provable while both sides are wrong, as
  `argmax1d_scalar_n_eq` did while both sides had argmax's tie-breaking
  wrong. Re-parsing the emitted Lean would be circular. Instead, give
  `native.py`'s op table a cvc5 *evaluator* sibling in the same registry
  (`ops.py`), and check the two agree on random concrete inputs, in
  milliseconds per module.

## Lean generation

- **Nothing in CI builds a `verith -o` project.** The Lean tests compile
  standalone `Certs/*.lean`, except the slow ones in
  `tests/test_lean_svcomp.py`, which build `create_project` output with
  `Data.lean` rewritten afterwards rather than a run of `main`. A matcher
  defect in the relational encodings went unnoticed that way until the
  limits pass, while 13 of 33 projects failed `System`. Add one slow test
  that runs a handful of `tests/limits/` cases through `verith -o`, links
  the shared `.lake` (the recipe is in `tests/limits/README.md`;
  `test_generated_executable_builds_and_runs` does it for `-x`), and builds
  `System` and `Certificate`.
- **`Uninterpreted` has no Lean form** (needs a decision). It is the limit
  matrix's only generation failure (`OpUninterp`). Every route refuses such
  a module before writing the project (`native.lean_gaps`, naming the block
  and the wire). Keep that refusal as the answer, or emit an `opaque`
  declaration and accept that no proof can say anything about the symbol.
- **`TensorGet` / `ToUnsigned` are dead op keys** (needs a decision). No
  theory defines them and there is no variant to rename them to: drop the
  rows, or implement the ops. `tests/test_lean_ops.py` allowlists them
  meanwhile, so its guard still catches new dead keys.
- **`zeroth_hammer` and the generated plan have diverged** (needs a
  decision). `Certificate.lean` closes its goals with the generated
  `cert_prep`/`cert_close` and never calls `zeroth_hammer`, whose last
  phase is `sorry` and so would turn a failed certificate into a build that
  "succeeds" with a warning. Either give `zeroth_hammer` a `sorry`-free
  variant that consumes the same plan, or drop it from the generated
  project and keep it as the interactive tactic `ManualTests` uses.
- **`Real` has no Lean IO** (needs a decision). `generate_main_lean`
  refuses a `Real` wire, since Lean's `Real` is noncomputable. A float
  approximation at the IO boundary would work, if `verith -x` is wanted for
  LRA modules.

## Proof automation and cvc5

- **`--smt-tactics cvc5` does not pay, so it stays off.** Its best case is
  `NN2RealAllPos4`: a Real net whose every ReLU is non-negative under the
  invariant, so cvc5 settles all four branch conditions, and the rewrite
  collapses all four branches. The certificate still takes the same time
  (68/69 s off against 68/70 s on). Profiled, `hrank` is 70 of the 75 s,
  and inside it typeclass inference is 17.7 s, simp 12.2 s, norm_num 6.2 s
  and ring 2.6 s: Real's algebraic hierarchy, searched again on every
  branch by `norm_num`, `ring` and `linarith`. Pruning branches divides
  that cost but leaves the per-branch constant, which is most of it, so
  `n_branch`/`n_conditions` are the wrong cost model in kind. Start the
  next attempt from a profile, aiming at fewer passes over the goal.
- **cvc5 abduction to repair an invariant.** When `step_inv` fails,
  `Solver.getAbduct(goal)` gives a conjunct `B` with `inv ∧ B` consistent
  and `inv ∧ B ⊨ goal` (`tests/limits/probes/abduction.py`). The catch:
  the answers are sufficient, not weakest. On `m_countdown`,
  `inv=(>= s0 1)` gets `(= s0 100)`. Plan, about half a day:
  1. `smt_query.abduce(q, budget)`, called only when `check_step_inv` came
     back `REFUTED`, sharing the phase budget;
  2. enumerate with `getAbductNext` up to 4, and keep the weakest by
     pairwise entailment checks;
  3. reject an abduct that mentions no state variable, or that is refuted
     at the initial state;
  4. report it in `--pre-check` (`step_inv REFUTED … try adding: …`)
     rather than applying it, since applying it would silently change a
     user's certificate;
  5. optionally re-run the pre-check with `inv ∧ abduct`, so the
     suggestion comes with evidence.

  Never loop it into `--infer` without an iteration cap and an init check:
  abduction will happily converge on `False`.
- **`--infer ai` is out of reach of `--pre-check` and `--smt-tactics`.**
  It answers in Lean and checks with a second LLM call, so there is no
  SMT-LIB on that path. Closing it means prompting that route in SMT-LIB, as
  `ai-cegis` does.

## fbk-proveit

- **Rebuild the bridge for `m_max`, `m_min` and `m_argmax`.** The bridge
  (`translate/fbk_bridge.py`, which writes `Certificate/Equivalence.lean`)
  closed for 14 of the 17 module shapes the route accepts. These three
  failed because `matMin`/`matMax`/`argmax_1d` did not reduce. `be70a43`
  made them reduce and put them in `MAT_SIMP`, which the bridge's cascade
  uses, but nothing has rebuilt those three bridges since.
- **The last link of `module_safety`.** Its hypothesis is the model-side
  statement that the installed certificate proves, in
  `lean-ltl-certifying`'s LTL vocabulary rather than `Core.LTL`'s.
  Translating between the two is fixed work, but both libraries declare a
  top-level `LTLFormula`, so a module importing the certificate and the
  bridge together is rejected before it is elaborated. That is why
  `--build-cert` reaches the bridge through the `Certificate` lean_lib's
  glob. Discharging the hypothesis means moving one of the two
  `LTLFormula`s.
- **An UNSAFE verdict is not carried back.** The bridge's `hinit`/`hstep`
  are implications (every module run is a model run), which is what `G P`
  needs. Concluding that the *module* is unsafe needs the converse. The
  obligations as measured are equalities, so the converse is there to be
  stated; ic3ia's counterexample also has to be brought into Lean.
- **Upstream: `mod` stops at the witness** (probe `InvMod`). `lean2vmt`
  translates `%` and ic3ia proves the property, but MathSAT eliminates the
  mod from the witness (`x + (-2) * to_int ((1/2) * to_real x) = 0`), and
  `vmt2lean.py` renders neither `to_real`/`to_int` nor a Real inside a
  Bool-valued `INVAR`. Meanwhile `smt_to_lean_bool` refuses
  `INTS_MODULUS` up front.
- **Upstream: `vmt2lean.py` maps only Int and Bool**, so the route refuses
  Real and BitVec state. If that changes, the bridge's cascade needs a
  `linarith` arm: `omega` stops applying, and the bridge cannot use `smt`
  (lean-smt's `Auto.instBEqInt_auto` outranks the `BEq Int` the model is
  built over).
- **Upstream: lean-smt rejects a reconstructed `ARITH_SUM_UB` proof**
  (probe `NiS2Odd3`; ufmg-smite/lean-smt; Lean v4.28.0, smt `f58d19d5`,
  cvc5 `ef0efbf4`). `smt` closes the goal, but the kernel rejects the
  term: a `sum_ub` step concludes `=` where `≤` is wanted.

  ```lean
  import Smt

  example (x x' : Int) (h1 : 2 <= x ∨ x = 0) (htr : x' = x + 2) : ¬ (x' = 3) := by
    revert h1 htr
    smt
  ```

  It needs all three of these: an equality disjunct in the hypothesis, a
  disequality goal, and that disequality on the variable the linear
  equation defines (`x'`, not `x`). The workaround is to state the property
  so that its invariant needs no disequality: `Step2Odd` asks the same
  parity question and certifies.
- **Upstream: lean-smt fails on `Max.max`** with "incorrect number of
  universe levels Max":

  ```lean
  example (x x' : Int) (h : x' = Max.max 0 (x - 1)) (hx : 0 <= x) : 0 <= x' := by
    revert h hx
    smt
  ```

  The route no longer reaches it, because a ReLU now comes from
  `smt_encode` as an `ite`, but any producer of `Max.max` does.
