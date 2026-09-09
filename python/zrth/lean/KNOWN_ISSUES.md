# Known issues in the Lean generation pipeline

Findings from a review of the pipeline (`python/zrth/lean/`, its templates,
and the parts of `zrth` they read). Recorded so the open ones are not
rediscovered; each says how it fails, not just that it does.

`CONFIRMED` was reproduced by running the code; `PLAUSIBLE` comes from
reading it and has not been executed.

For what the pipeline can and cannot *prove* — as opposed to what is
outright broken — see [`VERITH_LIMITS.md`](VERITH_LIMITS.md), which measures
36 modules end to end and carries the open proof-automation issues.

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
* The standalone certificates in `Certs/` carry no Scalar, ScalarRel or FBK
  section, which is why those encodings went uncompiled for so long.
  `Certs/ScalarEnc*` and `Certs/RelEnc*` now cover them. `RelEncMixed` is
  the only one whose state has both several wires and a multi-element one —
  every other spec is either all-1×1 (slot `i` *is* wire `i`) or a single
  wire (its slice is the whole tuple), so neither catches a per-wire
  projection of a per-element tuple.
* `templates/project/System/ScalarRel.lean.j2` is not rendered either (same
  as `Scalar.lean.j2`), and is kept in step for the same reason.
