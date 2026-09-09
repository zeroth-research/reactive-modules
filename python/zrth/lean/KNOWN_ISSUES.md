# Known issues in the Lean generation pipeline

Findings from a review of the pipeline (`python/zrth/lean/`, its templates,
and the parts of `zrth` they read). Recorded so the open ones are not
rediscovered; each says how it fails, not just that it does.

`CONFIRMED` was reproduced by running the code; `PLAUSIBLE` comes from
reading it and has not been executed.

None of this came from the `origin/main` rebase. The dead-name family
predates it (`IType`-era names that never existed at the merge base) and the
rest is older still. The suite passed because none of these paths were
reached by it: `!=`, `%`, `Argmax`, Bool matrix constants, the scalar
encoding, BV codegen and the whole `verith -x` path were all uncovered.

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

Two of these carried the same lesson: **the equivalence theorems can stay
provable while both sides are wrong.** `argmax1d_scalar_n_eq` proved the
unrolled scalar form equal to `argmax_1d` by unfolding both, so it went on
passing while each had the tie-breaking and seeding wrong. A fix to one side
alone would have broken it.

---

## Open

### 19. Dead dispatch keys: `Mod`, `TensorGet`, `ToUnsigned`, `MatAdd` · CONFIRMED

Same root cause as (1): emission dispatches on the name `itype_name()`
returns, so a key matching no variant can never fire.

| Key | Where | Should be |
|---|---|---|
| `Mod` | `native.py` (both tables), `smt_encode.py` | BV has `SMod`/`UMod`; LIA/LRA have no modulo |
| `MatAdd` | `circ.py` | `Add` |
| `TensorGet` | `native.py` ×2, `smt_encode.py` | no modern equivalent |
| `ToUnsigned` | `native.py` ×2, `smt_encode.py` | no modern equivalent |

`Mod` is the sharp one: `87bcec0 lean: support 'Mod' operation` is dead code
as written. `MatAdd` -> `Add` is mechanical. **`TensorGet`/`ToUnsigned` need
a decision** — remove them, or implement the ops.

`tests/test_lean_ops.py::test_op_table_has_no_new_dead_keys` allowlists
these four, so the guard catches *new* dead keys and the allowlist is the
todo list. `"Implies"`/`"ToInt"` in `smt_prompt.py` are **not** in this
family — they are predicate-DSL namespace keys.

Also absent from the tables though they are real variants: `Transpose` and
`Uninterpreted`. (`Argmax`, `Linear`, `Min`, `Max` have dedicated branches.)

### 20. FBK's state tuple is incompatible with `ScalarRel.effect_i` · PLAUSIBLE

`translate/fbk.py`. `_state_tuple` builds its `TypeMap` from
`dtype_to_lean_type(w, simple_types=True)`, so a multi-element ctrl wire
stays a matrix, while `ScalarRel`'s state type comes from
`_product_type_scalar`, which flattens it. For a ctrl component of
`Int([1,3])` the generated `effect_i_eq`/`R_i_iff` theorems are ill-typed —
for exactly the matrix modules `_can_scalarize()` admits.

Worth re-checking against `5b32d2f`: the Scalar encoding now flattens its
body, so this may be the same defect one layer up, and the same four pieces
may apply.

### 21. `_prepend_recon` makes call sites under-apply `effect_i` · PLAUSIBLE

`translate/fbk.py`. The reconstruction `let`s are emitted unconditionally
from the shared `update_recon`, so Lean's `variable` auto-binding pulls
`extl_l`/`extl_n` into `effect_i`/`init_i`, while `_effect_args` derives the
argument list from `_consumed()` and lists only the groups actually read.
For a module with a multi-element `extl_l` and an `effect_0` depending only
on state, the emitted `effect_0 state` is missing an argument.

### 22. `_can_scalarize()` gates nothing · CONFIRMED

`len(dtype_shape(w.dtype)) <= 2` is vacuous: every shape here is 2-D. It no
longer causes ill-typed output now that the flattening is finished
(`5b32d2f`), so this is tidiness rather than a bug — but the docstring and
the callers' "not available" message both claim it rejects matrix wires,
and it does not.

### 23. `Real` has no Lean IO · by design, but worth revisiting

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
* The standalone certificates in `Certs/` carry no Scalar section, which is
  why the scalar encoding went uncompiled for so long. `Certs/ScalarEnc*`
  now cover it.
