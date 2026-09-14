# Known issues in the Lean generation pipeline

Findings from a review of the pipeline (`python/zrth/lean/`, its templates,
and the parts of `zrth` they read). Recorded so the open ones are not
rediscovered; each says how it fails, not just that it does.

`CONFIRMED` was reproduced by running the code; `PLAUSIBLE` comes from
reading it and has not been executed.

For what the pipeline can and cannot *prove* — as opposed to what is
outright broken — see
[`tests/limits/README.md`](../../tests/limits/README.md), which measures 77
cases end to end; the proof-automation limits it turns up are #26-#30 below.

None of this came from the `origin/main` rebase. The dead-name family
predates it (`IType`-era names that never existed at the merge base) and the
rest is older still. The suite passed because none of these paths were
reached by it: `!=`, `%`, `Argmax`, Bool matrix constants, the scalar and
relational encodings, BV codegen and the whole `verith -x` path were all
uncovered.

---

## Fixed

| # | Issue | Commit |
|---|---|---|
| 1 | `Ne` was keyed as `"Neq"`, so every `!=` failed codegen with "No Lean expression mapping" | `c50beab` |
| 2 | `argmax_1d` seeded `(0, default)` under a non-strict `≤`: ties went to the last index and every all-negative row reported 0 | `a3a042d` |
| 3 | 2-D `argmax` returned an `[i, j]` pair where torch returns one row-major flat index, and was mis-seeded the same way | `bca45dc` |
| 4 | The theory admitted an Argmax output no backend could fill (any vector, under a `FIXME`); now `[1, 1]` | `3769d3f` |
| 5 | `_is_scalar_tensor` ignored shape for Bool, so a Bool matrix constant hit `.item()` | `9b34259` |
| 6 | The scalar encoding flattened the signature but not the body — 4 type errors per multi-element wire | `5b32d2f` |
| 7 | `Main.lean` called `update` with one merged tuple against three curried params, dropping `extl_l` | `7aa62c2` |
| 8 | `generate_main_lean` reversed tuple order in both directions, unpaired | `5c2d5f5` |
| 9 | `parseExtl`/`showCtrl` pinned every element to `Int`; `Main.lean` imported a file nothing writes | `80b074c` |
| 10 | BV modules got Boolean Lean operators (`!`, `&&`, bare `if c then`) on `BitVec 1` values | `2485e73` |
| 11 | An `unknown` solver result fell into the model-reading path and crashed CEGAR | `ec2419a` |
| 12 | `BitVec.ofNat` cannot take a negative literal (latent: nothing feeds one today) | `34edcfc` |
| 13 | Scalar argmax variants were named by width alone, colliding across element types | `db777c0` |
| 14 | `IndexError` in the FBK encoding for a module with no ctrl wires | `00f5f8f` |
| 15 | Theory and evaluators disagreed about `Min`/`Max`; resolved as unary reductions | `6511e9f` |
| 16 | `_linear` floored float weights against an int input — `[[0.5,0.5]]·[[3],[3]]` gave 0, not 3.0 | `e2935ec` |
| 17 | `README.md` still described the pre-`Var` IR | `53fa3de` |
| 18 | Two computed-but-unused simp lists in `rel.py` | `6122591` |
| 19 | `Mod` assumed an integer modulo no theory has; `MatAdd` duplicated `Add` | `862d247` |
| 20 | FBK's state was per wire while `ScalarRel.effect_i` takes it per element | `d49ebdb` |
| 21 | `_prepend_recon` gave every body every reconstruction, so call sites under-applied | `fa7bfa6` |
| 22 | `ScalarRel`/`FBK` projected component `i` of a per-element tuple for wire `i`; now an offset-and-length slice against a flat `effect_i` | *this change* |

Two of these carried the same lesson: **the equivalence theorems can stay
provable while both sides are wrong.** `argmax1d_scalar_n_eq` proved the
unrolled scalar form equal to `argmax_1d` by unfolding both, so it went on
passing while each had the tie-breaking and seeding wrong. A fix to one side
alone would have broken it.

The corollary is which theorem to point a new certificate at. #22's slice is
computed the same way on both sides of `effect_i_eq`, so that theorem alone
would hold at a wrong offset; `TransRel_scalar_eq` into `TransRel_func_eq`
is what does not, because it carries the relation back to the functional
`update` through `pack`/`unpack`. Mutating `_flat_layout` to return offset 0
for every wire turns `Certs/RelEncMixed` from 0 errors into 6.

### From the limits pass

The 77-case matrix in [`tests/limits/`](../../tests/limits/README.md) was
built to find where the pipeline gives way, and these are what it found on
the way there. Numbered independently of the table above: these are its own,
and each says what the fix *was*, because in several the obvious fix is the
wrong one.

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
| 32 | `visit_If` merged the two branch scopes through a `set` of variable-*name* strings, so two runs of the generator over one unchanged module emitted different `System.lean`/`Scalar.lean` — the `let`s for independent `Ite`s swapped and the `x{n}` numbering shifted with them. Same program either way, but a regenerated project was byte-different, so `lake` rebuilt everything downstream and byte-diffing generated Lean could not show that a refactor changed nothing | merge in the order the two scopes were built (`zrth/analyzer.py`); `tests/test_lean_determinism.py` pins the six modules under two `PYTHONHASHSEED`s |

---

## Open

### 23. `TensorGet` / `ToUnsigned` are dead keys with no modern equivalent · CONFIRMED

Unlike `Mod` and `MatAdd`, these have no variant to rename to. **Needs a
decision:** drop the entries, or implement the ops. Allowlisted in
`tests/test_lean_ops.py` meanwhile, so the guard still catches new ones.

`Transpose` and `Uninterpreted` were the reverse case — real variants absent
from the tables. `Transpose` is now mapped (`MatTranspose` in `_LEAN_OP`,
`Box.transpose` in `_LEAN_OP_BOX`); it had a Lean counterpart in
`Core/Mat.lean` all along. `Uninterpreted` still has none and needs its own
decision: emit an `opaque` declaration, or reject it at the front end.

### 24. `_can_scalarize()` gates nothing · CONFIRMED

`len(dtype_shape(w.dtype)) <= 2` is vacuous: every shape here is 2-D. It no
longer produces ill-typed output now that the flattening is finished, so
this is tidiness — but the docstring and the callers' "not available"
message both claim it rejects matrix wires, and it does not.

### 25. `Real` has no Lean IO · by design, but worth revisiting

`generate_main_lean` refuses a `Real` wire: Lean's `Real` is noncomputable,
so the generated executable can neither parse nor print it. A float
approximation at the IO boundary would work if `verith -x` is wanted for LRA
modules; that is a decision about what the executable is for.

### 31. The NA model is not tied to the module · CONFIRMED

Every other encoding carries an equivalence theorem back to the functional
one; the `--fbk-proveit` route's model carries none, because trust leaves
Lean through ic3ia and returns as `vmt2lean.py`'s certificate. A
mistranslation in `smt_encode`, `_scalar_element`, cvc5's rewriter or
`smt_to_lean_bool` therefore yields a model that parses, model-checks and
carries a machine-checked certificate about a *different* system.

`tests/test_lean_fbk.py::test_the_emitted_transition_agrees_with_the_module`
evaluates the emitted slot bodies against the module's own execution, which
catches a flattening or indexing error but is not a proof and does not run in
the generated project.

**Built**, for 14 of the 17 module shapes the route accepts:
`translate/fbk_bridge.py` emits `Certificate/Equivalence.lean`, whose
`module_safety` says safety of the model is safety of the module, over the
module's own `init`/`update` and its own `P`. See
[`FBK_EQUIVALENCE.md`](FBK_EQUIVALENCE.md).

What is left is the three modules of #26 and #30 -- `matMin`/`matMax`/
`argmax_1d` do not reduce, so the bridge cannot compute the side it has to
compare against, and fixing the fold fixes both -- and the last link of
`module_safety`, whose hypothesis is the model-side statement the installed
certificate proves in `lean-ltl-certifying`'s LTL vocabulary rather than
`Core.LTL`'s. Translating between the two is fixed work that does not vary
per module.

### 26. `matMin` / `matMax` never reduce, so `Min`/`Max` modules cannot be proved · CONFIRMED

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

**Confirmed provable, twice.** The SMT encoder could not build these modules
at all until `Min`/`Max` were moved off the binary `_elementwise` path they
were wired to — they are unary reductions, so any cvc5 query about such a
module raised `TypeError`. With that fixed, `--pre-check cvc5` says all three
obligations hold for `OpMax` and `OpMin`, in 4.6 ms and 4.2 ms. And the
`--fbk-proveit` route, which takes its transition from `smt_encode` rather
than from the scalar Lean printer, certifies a safety property for both.
There is nothing wrong with the certificates and nothing to find in the
tactic plan: the Lean-side fold is the whole of it.

### 27. `Uninterpreted` has no Lean form · CONFIRMED

The decision #23 asks for, from the other side: this is the limit matrix's
*only* generation failure (`OpUninterp`, `No Lean expression mapping for:
Uninterpreted`, `native.py:291`). Unlike `Transpose`, which had a Lean
counterpart all along, an uninterpreted symbol has no obvious translation —
emit an `opaque` declaration and accept that no proof can say anything about
it, or reject it at the front end with a message.

The `--fbk-proveit` route already takes the second option: `check_module`
runs before anything is generated and reports "smt_encode has no term for
Uninterpreted". The ordinary route still dies in a traceback out of the
functional encoder.

### 28. `zeroth_hammer` and the generated plan have diverged further

`Certificate.lean` still never calls `zeroth_hammer`; it now uses the
generated `cert_prep`/`cert_close` instead of the old fixed chain. Keeping
them apart is defensible — `zeroth_hammer`'s last phase is `sorry`, which
would turn a failed certificate into a build that "succeeds" with a warning —
but there are now two proof strategies in the tree, and only one of them is
shape-aware.

**Resolution.** Either give `zeroth_hammer` a `sorry`-free variant that
consumes the same plan, or drop it from the generated project and keep it as
the interactive/experimental tactic it is used as in `ManualTests`.

### 29. Nothing in CI builds a generated project

Every Lean test compiles a standalone `Certs/*.lean`. No test runs
`lake build` on the output of `verith -o`. That is why the `FBK`/`ScalarRel`
matcher defect (#21) survived, and why 13 of 33 projects failed `System`
before that pass while the suite was green.

Partly addressed: `Certs/RelEnc*` now mirrors the *module layout* of a
generated project — four separate modules rather than one concatenated file —
which is what makes it able to see a defect like #21 at all. It still builds fixtures in
the warm project rather than a `verith -o` output.

**Resolution.** One slow test that generates a project, symlinks the shared
`.lake` (`tests/limits/README.md` has the recipe;
`test_generated_executable_builds_and_runs` already
does exactly this for the `-x` path) and builds `System` and `Certificate`.

The precondition is met: the whole matrix -- 77 cases, 37 module fixtures,
the runners and the baseline -- lives in
[`tests/limits/`](../../tests/limits/README.md) instead of a scratch
directory, so every verdict is reproducible with one command. What remains
is wiring a handful of its cases into a slow-marked test.

### 30. Argmax certificates · CONFIRMED

`OpArgmax` compiles every encoding now but the certificate fails:
`argmax_1d` does not reduce under the tactic chain, the same shape of problem
as #26. The generated `argmax1d_scalar_n_eq` theorems tie the unrolled
scalar form to `Core.Mat.argmax_1d`, but nothing brings either into a goal.

`--pre-check cvc5` reports all three obligations holding in 3.2 ms, so this is
the same story as #26: a true certificate that Lean cannot reduce its way
to. `Argmax` was always encoded correctly for SMT (`_argmax_flat`), which is
why it could be checked before `Min`/`Max` could.

---

## Not bugs (checked)

* Every simp-lemma name in `translate/circ.py::_LAYER_SIMP` resolves to a
  real declaration in `Core/{Box,Basic,Mat,LTL}.lean`.
* `linear_list_literals`' `out_m` matches `check_linear_affine`'s `a_rows`
  convention.
* `_circ_compute_swapping`/`_dups`/`_dels` terminate and stay consistent —
  sorting makes duplicates adjacent.
* `analyzer.py`'s `call_repr` read is safe: a dataclass field defaulting to
  `None`.
* `"Implies"`/`"ToInt"` in `smt_prompt.py` are predicate-DSL keys, not itype
  dispatch.

## Traps worth knowing

* `templates/project/System/Scalar.lean.j2` is **never rendered** — nothing
  passes it to `render()`. The live emitter is
  `translate/scalar.py::_argmax_scalar_def_lines`. Kept in step anyway so
  reviving it does not resurrect old semantics.
* `tests/lean/Core/` is **generated** by the `sync_core_templates` fixture
  and gitignored. Running `lake build` there without running pytest first
  compiles a stale copy — and a stale `Core` olean makes a new definition
  look like an unknown identifier, which `autoImplicit` then reports as
  "function expected ... has type ?m.1" rather than "unknown identifier".
  `lake build Core` cannot work (the lib has no root file); build
  `Core.Mat`/`Core.Box` directly.
* `templates/` used to hold a second, byte-identical set of the Core files.
  Only `static/` was ever read, so editing a top-level copy did nothing —
  which happened in `879d9f7`. Removed in `7f3f753`, guarded by
  `tests/test_lean_templates.py`.
* Issues 14, 20, 21 and 22 above are about `System/FBK.lean`, the
  Bool-valued relational encoding. It is gone: every project generated one
  and nothing ever read it, and the name now belongs to
  `translate/fbk.py`, the NA model the `--fbk-proveit` route feeds to
  `lean2vmt`. The entries stay because they say what went wrong and how it
  was found, which outlives the file.
* The standalone certificates in `Certs/` carry no Scalar or ScalarRel
  section, which is why those encodings went uncompiled for so long.
  `Certs/ScalarEnc*` and `Certs/RelEnc*` now cover them. `RelEncMixed` is
  the only one whose state has both several wires and a multi-element one —
  every other spec is either all-1×1 (slot `i` *is* wire `i`) or a single
  wire (its slice is the whole tuple), so neither catches a per-wire
  projection of a per-element tuple.
* `templates/project/System/ScalarRel.lean.j2` is not rendered either (same
  as `Scalar.lean.j2`), and is kept in step for the same reason.
