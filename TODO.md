- Try using facts from Vampire to generate tactics; dump to artifacts/
- Prune selectively -- if Vampire proves inv but not ranking, try preserving that inv
- [later] Cooperative infer through artifacts: smt + vampire in parallel for 4 seconds; then one after another (with known artifacts) and nuterm; then ai-cegis; then ai-fix
- [later] Try vampire instead of lean-smt in hammer
- fbk-proveit: `_REVERT` / `_SMT_CLOSER` (`python/zrth/lean/fbk_proveit.py`) regex-patch the Lean proveit.py generates, and do nothing, silently, when its output changes. Revisit: fix it upstream in lean-ltl-certifying's template, or at least warn when a patch matches nothing (kept as is in phase 6, 2026-10-19)
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
