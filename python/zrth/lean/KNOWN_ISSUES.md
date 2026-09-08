# Known issues in the Lean generation pipeline

Findings from a review of the pipeline (`python/zrth/lean/`, its templates,
and the parts of `zrth` they read). Recorded so the open ones are not
rediscovered from scratch; each says how it fails, not just that it does.

**Status legend** — `FIXED` names the commit. `OPEN` items are untouched.
`CONFIRMED` was reproduced by running the code; `PLAUSIBLE` comes from
reading it and has not been executed.

Nothing here is a regression from the `origin/main` rebase: the dead-name
family predates it (they are `IType`-era names that never existed at the
merge base), and the rest is older still. The test suite passes because none
of these paths are reached by it — `!=`, `%`, `Argmax`, Bool matrix
constants, negative `BV.Const`, the scalar encoding's matrix paths and the
whole `verith -x` executable path were all uncovered.

---

## Fixed

### 1. `Ne` was undispatchable — every `!=` failed codegen · CONFIRMED
`FIXED` in `c50beab`.

The op tables keyed inequality as `"Neq"`, but emission dispatches on the
name `itype_name()` returns, which strips only the theory prefix. The real
variant (`LIA::Ne`) therefore arrived as `Ne` and matched nothing:
`ValueError: No Lean expression mapping for: Ne` in the functional and
scalar encodings, a bare `KeyError` in the circuit one. `Box.neq` already
existed in `Core/Box.lean`, so only the Python keys were wrong.

### 2. `argmax_1d` disagreed with `torch.argmax` · CONFIRMED
`FIXED` in `a3a042d`.

The fold seeded `(0, default)` and updated on a non-strict `≤`. So `0`
entered as a candidate — any all-negative row reported index 0, e.g.
`[-5, -3]` gave 0 where the maximum is at 1 — and ties resolved to the last
index rather than the first, e.g. `[3, 3]` gave 1.

The generated scalar variant mirrored the same fold, which is why this
survived: `argmax1d_scalar_n_eq` proves the two equal by unfolding both, so
it stayed provable while both sides were wrong. A fix to one side alone
would have broken that proof.

Dangerous direction: the SMT encoder already matched torch, so CEGAR could
prove an invariant UNSAT that is false of the system the certificate states.

### 3. 2-D `argmax` returned the wrong kind of answer · CONFIRMED
`FIXED` in `bca45dc`.

It reported the `[i, j]` position of the maximum packed as `Mat Nat 1 2`,
where torch (`eval.py` evaluates Argmax as `r[0].argmax()`) flattens
row-major and returns the single index `i * n + j`. It also carried the same
mis-seeding as (2), and the `1 2` result was ascribed to the write wire's
type by the emitters, so it did not elaborate even for a `[1, 1]` output.

The SMT encoder had no 2-D path at all — `_argmax_1d` rejected `m != 1` — so
a 2-D Argmax could be emitted to Lean but never verified. Generalised to
`_argmax_flat` over the row-major flattening.

---

## Open

### 4. Dead dispatch keys: `Mod`, `TensorGet`, `ToUnsigned`, `MatAdd` · CONFIRMED

Same root cause as (1) and the same consequence — the operator they were
meant to serve falls through to the "no mapping" path.

| Key | Where | Should be |
|---|---|---|
| `Mod` | `native.py` (`_LEAN_OP`, `_SCALAR_OP`), `smt_encode.py` | BV has `SMod`/`UMod`; LIA/LRA have no modulo |
| `TensorGet` | `native.py` ×2, `smt_encode.py` | no such variant |
| `ToUnsigned` | `native.py` ×2, `smt_encode.py` | no such variant |
| `MatAdd` | `circ.py` (`_LEAN_OP_BOX`) | `Add` |

`Mod` is the sharp one: commit `87bcec0 lean: support 'Mod' operation` is
dead code as written. `TensorGet`/`ToUnsigned` are `IType`-era names with no
modern equivalent, so they need a decision rather than a rename.

`tests/test_lean_ops.py::test_op_table_has_no_new_dead_keys` allowlists
these four, so the guard catches *new* dead keys while the allowlist stands
as the todo list. `"Implies"`/`"ToInt"` in `smt_prompt.py` are **not** in
this family — they are keys in the user-facing predicate DSL namespace.

Also missing from the op tables, though they are real variants: `Xor` (all
three theories) and `Transpose`/`Uninterpreted`. `Argmax` and `Linear` are
absent by design — they have dedicated branches.

### 5. `_is_scalar_tensor` ignores shape for Bool wires · CONFIRMED

`common.py`: `if isinstance(dt, Bool): return True` skips the
`_is_scalar_shape` check that `Int`/`BitVec` get, contradicting the
function's own docstring ("not a matrix"). Verified:
`_is_scalar_tensor(Wire(Bool([2,3])))` is `True` while `Wire(Int([2,3]))`
is `False`.

A Bool matrix constant is therefore never interned by
`ConstantRegistry.intern`, falls through to `_constant_expr` →
`_tensor_to_lean_inline`, and hits `tensor.item()`:
`RuntimeError: a Tensor with 6 elements cannot be converted to Scalar`.

### 6. Scalar encoding mixes bare scalars with `Mat`-expecting builders · CONFIRMED

`native.py`, the `_LEAN_OP` fallback in `_translate_terms_scalar`. The
scalar encoding binds 1×1 wires to bare `Bool`/`Int`, but `_LEAN_OP`
builders index operands with `0 0`. A matrix `Ite` with a scalar Bool
condition emits

```lean
let x1 : (Mat Int 1 3) := (if ctrl.2.2.2 0 0 then x0 else _m0)
```

where `unpack_ctrl` types `ctrl.2.2.2` as `Bool`. Applying a `Bool` to two
arguments does not elaborate, so `System/Scalar.lean` fails to build for any
module combining a matrix wire with a scalar-conditioned op.

### 7. `Linear` in the scalar encoding emits wrong operand and result types · CONFIRMED

`native.py`. Unlike the adjacent `Argmax` branch there is no `0 0`
extraction and no scalar path, so a 1×1 `LIA.Linear` emits

```lean
let x0 : Int := (matVecAffine 1 ([[1]] : List (List Int)) ([1] : List Int) ctrl)
```

with `ctrl : Int`. Both the argument (needs `Mat t n batch`) and the ascribed
result (`Mat t 1 batch`, not `Int`) are wrong.

### 8. `generate_main_lean` calls `update` with one tuple instead of three args · CONFIRMED

`project.py` emits `update (state.1, state.2, extl)` while
`atom_to_lean_functional` emits `def update (ctrl) (extl_l) (extl_n)` —
three curried parameters. `extl_l` is silently dropped. Any project
generated with `verith -x` fails to elaborate. Verified on
`tests/fixtures/twobit.py`.

### 9. `generate_main_lean` reverses tuple component order · CONFIRMED

`project.py`: `reversed(parse_vars)` and `reversed(destr_vars)`, but
`_product_type`/`_build_tuple` never reverse, so the reversal is unpaired in
both directions. Verified on the twobit fixture: emits `let (v1, v0) := v`
then `s!"{showMat 1 1 v0} {showMat 1 1 v1}"`, so the executable prints `b1`
before `b0`. With heterogeneous ctrl types it is a hard type error, applying
the wrong `showBool`/`showMat` to each slot.

### 10. `parseExtl`/`showCtrl` pin the element type to `Int` · CONFIRMED

`project.py`. The `Bool`/`Int` branches compare against bare `"Bool"`/`"Int"`,
but `dtype_to_lean_type(w)` is called with the default `simple_types=False`
and always returns `(Mat ty 1 1)` — so those branches are unreachable and
everything takes the matrix path, which hardcodes `Int`. Verified on the
BitVec fixture: `parseExtl` is declared `IO ((Mat (BitVec 1) 1 1))` while
its body builds `let e0 : Fin 1 → Fin 1 → Int`, and `showMat` only accepts
`Fin m → Fin n → Int`. Bool, BitVec and Real IO all mistype.

### 11. `unknown` solver result crashes the CEGAR driver · PLAUSIBLE

`magic_cegar.py`. Only `res.isUnsat()` short-circuits; an `unknown` result
falls into the counterexample path and calls `solver.getValue(v)` with no
model available, raising `CVC5ApiException`. A nonlinear ranking function or
an `--infer ai-cegar` query that times out crashes the whole run instead of
producing feedback.

### 12. `BitVec.ofNat` emitted for negative values · PLAUSIBLE

`common.py`: `f"(BitVec.ofNat {dtype._0} {int(item)})"` yields
`(BitVec.ofNat 8 -3)` for a negative entry, but `ofNat` takes a `Nat`.
Reachable from `_tensor_to_lean_def` and `linear_list_literals`.
`BitVec.ofInt` is the right constructor.

### 13. FBK's state tuple is incompatible with `ScalarRel.effect_i` · PLAUSIBLE

`translate/fbk.py`. `_state_tuple` builds its `TypeMap` from
`dtype_to_lean_type(w, simple_types=True)`, so a multi-element ctrl wire
stays a matrix, while `ScalarRel`'s state type comes from
`_product_type_scalar`, which flattens it. For a ctrl component of
`Int([1,3])` the generated `effect_i_eq`/`R_i_iff` theorems are ill-typed —
for exactly the matrix modules `_can_scalarize()` admits.

### 14. argmax scalar variants collide on name · PLAUSIBLE

`translate/scalar.py`. `_collect_argmax_variants` dedupes on `(ety, n)`
while `_argmax_scalar_name(n)` ignores the element type. A module with an
Argmax over `Mat Int 1 4` and another over `Mat Real 1 4` emits two
`def argmax1d_scalar_4` declarations and two `_eq` theorems — duplicate
declaration error, plus the same name twice in the simp set.

### 15. `_prepend_recon` makes call sites under-apply `effect_i` · PLAUSIBLE

`translate/fbk.py`. The reconstruction `let`s are emitted unconditionally
from the shared `update_recon`, so Lean's `variable` auto-binding pulls
`extl_l`/`extl_n` into `effect_i`/`init_i`, while `_effect_args` derives the
argument list from `_consumed()` and lists only the groups actually read.
For a module with a multi-element `extl_l` and an `effect_0` depending only
on state, the emitted `effect_0 state` is missing an argument.

### 16. `IndexError` when a module has no ctrl wires · PLAUSIBLE

`translate/fbk.py`. `ctrl_types[0]`, and `ctrl_types[-1]` in the `else`
branch (`all_same` is `False` for the empty list), both index an empty list.
A purely combinational atom crashes `to_lean_bool_rel()` instead of emitting
the "not available" comment that `to_lean_scalar`/`to_lean_rel` produce.

### 17. Theory and evaluator disagree about `Min`/`Max` · CONFIRMED

Noticed while tightening the `Argmax` shape rule. `check_mat_ops` groups
`Min`/`Max` with `Argmax` and requires **one** read plus a vector output —
i.e. treats them as reductions. `eval.py` evaluates them as **binary**
elementwise ops (`torch.minimum(r[0], r[1])`). Both cannot be right. The
same `FIXME: we should fix which dimension is 1..` sits on their output rule
in `lia.rs` and `lra.rs`.

Left alone deliberately when `Argmax` was split out and tightened, because
which of the two readings is intended is a design question.

### 18. `README.md` described the pre-`Var` IR · CONFIRMED
`FIXED` alongside this file.

The "Reactive Module IR" section still said `ctrl`/`extl` were lists of
`(latched, next)` wire pairs, and referred to `DType`/`Float`/`IType`. Since
the `origin/main` rebase, `ctrl` yields `Sequence[Var]` (a `Var` stands for
its latched wire, `X(v)` is the next one), sorts are `Bool`/`Int`/`Real`/
`BitVec` under `Sort`, and operations come from the `LIA`/`LRA`/`BV`
namespaces. Missed when the code was migrated in `1c88c5b`.

### 19. Unused `List.foldl`-era duplication in `rel.py` · PLAUSIBLE

`translate/rel.py` computes `unpack_pack_simp`/`unpack_pack_simp_str` twice
and never uses either.

### 20. `eval.py::_linear` truncates float weights against an int input · PLAUSIBLE

`_linear` does `weight.to(x.dtype)` in both directions. Baked *float*
weights against an int-typed `x` would truncate to 0. The cast was added to
handle the opposite case (baked Int weights against float runtime tensors);
it is unguarded in the direction that loses information.

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
* `"Implies"`/`"ToInt"` in `smt_prompt.py` are predicate-DSL namespace keys,
  not itype dispatch.

## Traps worth knowing

* `templates/project/System/Scalar.lean.j2` is **never rendered** — nothing
  passes it to `render()`. The live emitter is
  `translate/scalar.py::_argmax_scalar_def_lines`. The template is kept in
  step anyway so reviving it does not resurrect old semantics.
* `tests/lean/Core/` is **generated** — the `sync_core_templates` fixture
  copies the templates in, and the directory is gitignored. Running
  `lake build` there without running pytest first compiles a stale copy.
