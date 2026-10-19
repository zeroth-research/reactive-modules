# Known issues in the Lean generation pipeline

Open defects in the pipeline (`python/zrth/lean/`, its templates, and the
parts of `zrth` they read), recorded so they are not rediscovered; each says
how it fails, not just that it does. A fixed issue is removed, and the fix
is in its commit message. Numbers are not reused, so a gap is a fixed issue.

`CONFIRMED` was reproduced by running the code; `PLAUSIBLE` comes from
reading it and has not been executed.

For what the pipeline can and cannot *prove* — as opposed to what is
outright broken — see
[`tests/limits/README.md`](../../tests/limits/README.md), which measures 77
cases end to end; the proof-automation limits it turns up are #27-#29 below.

---

## Open

### 23. `TensorGet` / `ToUnsigned` are dead keys with no modern equivalent · CONFIRMED

Unlike `Mod` and `MatAdd`, these have no variant to rename to. **Needs a
decision:** drop the entries, or implement the ops. Allowlisted in
`tests/test_lean_ops.py` meanwhile, so the guard still catches new ones.

`Uninterpreted` is the reverse case, a real variant absent from the tables:
#27.

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

### 27. `Uninterpreted` has no Lean form · CONFIRMED

The limit matrix's only generation failure (`OpUninterp`), and the decision
#23 asks for, from the other side. Every route refuses such a module before
a line of the project exists, and says which block writes which wire with
what (`native.lean_gaps`, asked by `main` after the route's own precheck;
`fd775b5`). Still open is whether that refusal is the final answer, or
whether to emit an `opaque` declaration and accept that no proof can say
anything about the symbol.

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

### 29. Nothing in CI builds a `verith -o` project

The Lean tests compile standalone `Certs/*.lean`, except the slow ones in
`tests/test_lean_svcomp.py`, which build `create_project` output with
`Data.lean` rewritten afterwards rather than a run of `main`. That is how a
matcher defect in the relational encodings went unnoticed until the limits
pass, while 13 of 33 projects failed `System` and the suite was green.
`Certs/RelEnc*` mirror a generated project's module layout (separate
modules rather than one concatenated file), which is what lets them see that
class of defect, but they build fixtures in the warm project.

**Resolution.** One slow test that runs a handful of
[`tests/limits/`](../../tests/limits/README.md) cases through `verith -o`,
symlinks the shared `.lake` (`tests/limits/README.md` has the recipe;
`test_generated_executable_builds_and_runs` already does this for the `-x`
path), and builds `System` and `Certificate`.

### 31. The NA model's tie to the module is not complete · CONFIRMED

`translate/fbk_bridge.py` emits `Certificate/Equivalence.lean`, whose
`module_safety` says safety of the `--fbk-proveit` route's NA model is
safety of the module ([`FBK_EQUIVALENCE.md`](FBK_EQUIVALENCE.md)). Two
things are left:

- **Three module shapes are unmeasured.** The bridge closed for 14 of the
  17 shapes the route accepts. `m_max`, `m_min` and `m_argmax` failed
  because `matMin`/`matMax`/`argmax_1d` did not reduce. `be70a43` made them
  reduce and put them in `MAT_SIMP`, which the bridge's cascade uses, but
  nothing has rebuilt those three bridges since.
- **The last link of `module_safety`.** Its hypothesis is the model-side
  statement the installed certificate proves, in `lean-ltl-certifying`'s
  LTL vocabulary rather than `Core.LTL`'s. Translating between the two is
  fixed work that does not vary per module, but it is not only a
  translation: both libraries declare a top-level `LTLFormula`, so a module
  importing the certificate and the bridge together is rejected before it
  is elaborated. That is why `--build-cert` reaches the bridge through the
  `Certificate` lean_lib's glob and not through the root `Certificate.lean`'s
  import list. The day the hypothesis is discharged, one of the two
  `LTLFormula`s has to move.

---

## Not bugs (checked)

* Every simp-lemma name in `translate/circ.py::_LAYER_SIMP` resolves to a
  real declaration in `Core/{Box,Basic,Mat,LTL}.lean`.
* `_circ_compute_swapping`/`_dups`/`_dels` terminate and stay consistent —
  sorting makes duplicates adjacent.
* `analyzer.py`'s `call_repr` read is safe: a dataclass field defaulting to
  `None`.
* `"Implies"`/`"ToInt"` in `smt_prompt.py` are predicate-DSL keys, not itype
  dispatch.

## Traps worth knowing

* **The equivalence theorems can stay provable while both sides are
  wrong.** `argmax1d_scalar_n_eq` proved the unrolled scalar form equal to
  `argmax_1d` by unfolding both, so it went on passing while each had the
  tie-breaking and seeding wrong; a fix to one side alone would have broken
  it. The corollary is which theorem to point a new certificate at. The
  per-element slice `ScalarRel` takes is computed the same way on both
  sides of `effect_i_eq`, so that theorem alone would hold at a wrong
  offset; `TransRel_scalar_eq` into `TransRel_func_eq` is what does not,
  because it carries the relation back to the functional `update` through
  `pack`/`unpack`. Mutating `_flat_layout` to return offset 0 for every
  wire turns `Certs/RelEncMixed` from 0 errors into 6.
* `RelEncMixed` is the only standalone certificate whose state has both
  several wires and a multi-element one. Every other one is all-1×1 (slot
  `i` *is* wire `i`) or a single wire (its slice is the whole tuple), so
  none of them catches a per-wire projection of a per-element tuple.
* `System/*.lean` has no templates: `translate/` prints every encoding
  (`ENCODINGS` in `project.py` names each producer).
* `tests/lean/Core/` is **generated** by the `sync_core_templates` fixture
  and gitignored. Running `lake build` there without running pytest first
  compiles a stale copy — and a stale `Core` olean makes a new definition
  look like an unknown identifier, which `autoImplicit` then reports as
  "function expected ... has type ?m.1" rather than "unknown identifier".
  `lake build Core` cannot work (the lib has no root file); build
  `Core.Mat`/`Core.Box` directly.
