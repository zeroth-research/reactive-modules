"""Tests for the Module-to-Lean4 functional converter."""

import pytest
import torch
from zrth import Wire, Term, Module, Bool, Int, LIA, Var, X
from zrth.lean import ModuleToLean4
from zrth.lean.common import LeanContext, itype_name
from zrth.lean.cert import generate_certificate_lean
from zrth.lean.project import generate_main_lean


def test_itype_name_strips_prefix():
    assert itype_name(LIA.Add()) == "Add"
    assert itype_name(LIA.Int(torch.tensor([[0]]))) == "Int"
    assert itype_name(LIA.Ite()) == "Ite"


def _make_twobitcounter():
    """Bool-only module: two-bit counter with enable."""
    b0 = Var(Bool([1, 1]))
    b1 = Var(Bool([1, 1]))
    enable = Var(Bool([1, 1]))

    not_b0 = Wire(Bool([1, 1]))
    not_b1 = Wire(Bool([1, 1]))
    b0_and_enable = Wire(Bool([1, 1]))

    init = [
        Term(LIA.Bool(torch.tensor([[False]])), [X(b0)]),
        Term(LIA.Bool(torch.tensor([[False]])), [X(b1)]),
    ]
    update = [
        Term(LIA.Not(), [not_b0], [b0]),
        Term(LIA.Ite(), [X(b0)], [X(enable), not_b0, b0]),
        Term(LIA.And(), [b0_and_enable], [b0, X(enable)]),
        Term(LIA.Not(), [not_b1], [b1]),
        Term(LIA.Ite(), [X(b1)], [b0_and_enable, not_b1, b1]),
    ]
    return Module.sequential([b0, b1, enable], init, update)


def _make_matrix_module():
    """Matrix module using LIA.Linear (affine map Y = A·X + B).

    Both transitions are constant-matrix maps, so they are linear and expressed
    with `LIA.Linear` (A and B baked into the op) rather than a (BV-only) MatMul:
      init:   x' = A · u          with A = [[0,0],[1,0],[0,1]]  (no bias)
      update: x' = B · x + e1     with B = I₃, e1 = [1,0,0]ᵀ
    """
    x = Var(Int([3, 1]))
    u = Var(Int([2, 1]))

    A = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    init = [
        Term(LIA.Linear(A, torch.zeros((3, 1), dtype=torch.int64)), [X(x)], [X(u)]),
    ]

    B = torch.eye(3, dtype=torch.int64)
    e1 = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    update = [
        Term(LIA.Linear(B, e1), [X(x)], [x]),
    ]
    return Module.sequential([x, u], init, update)


# ── Two-bit counter ──────────────────────────────────────────────────


def test_twobitcounter_generates_lean():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()

    assert "def init" in lean
    assert "def update" in lean


def test_twobitcounter_has_inlined_scalars():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()

    # Scalar Bool constants are inlined as Mat-typed literals, not top-level defs
    assert "def c0" not in lean
    assert "let x0 : (Mat Bool 1 1) := (fun _ _ => false)" in lean


def test_twobitcounter_update_has_let_bindings():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()

    assert "let x0" in lean
    assert "let x1" in lean


def test_twobitcounter_update_has_expected_ops():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()

    assert "!" in lean  # not
    assert "if " in lean  # ite
    assert "&&" in lean  # and


def test_twobitcounter_init_signature():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()

    assert (
        "def init (extl_n: (Mat Bool 1 1)) : (Mat Bool 1 1) × (Mat Bool 1 1)" in lean
    )


def test_twobitcounter_update_signature():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()

    assert (
        "def update (ctrl: (Mat Bool 1 1) × (Mat Bool 1 1)) (extl_l: (Mat Bool 1 1)) (extl_n: (Mat Bool 1 1)) : (Mat Bool 1 1) × (Mat Bool 1 1)"
        in lean
    )


def test_twobitcounter_output_tuple():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()

    # Output should be a plain tuple like (x1, x4)
    assert "(x1, x4)" in lean


# ── Matrix module ────────────────────────────────────────────────────


def test_matrix_module_generates_lean():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    assert "def init" in lean
    assert "def update" in lean


def test_matrix_module_has_matrix_types():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    # The state (3×1) and external input (2×1) matrix types appear in the
    # signatures / let-bindings. A and B are baked into the Linear op and
    # rendered as inline literals, so their types are not annotated here.
    assert "Mat Int 3 1" in lean
    assert "Mat Int 2 1" in lean


def test_matrix_module_uses_matvecaffine():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    # LIA.Linear is emitted in the reflected form over list literals: matVecAffine
    # on the native path, Box.linear on the circuit path — neither uses the
    # (unreduced) affineLinear/MatMul the prover would have to expand.
    assert "affineLinear" not in lean
    assert "matVecAffine" in lean
    assert "Box.linear" in lean
    assert "[[0, 0], [1, 0], [0, 1]]" in lean  # A of init, as a list literal


def test_matrix_init_signature():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    assert "def init (extl_n: (Mat Int 2 1)) : (Mat Int 3 1)" in lean


def test_matrix_update_signature():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    assert (
        "def update (ctrl: (Mat Int 3 1)) (extl_l: (Mat Int 2 1)) (extl_n: (Mat Int 2 1)) : (Mat Int 3 1)"
        in lean
    )


# ── Main.lean generation ────────────────────────────────────────────────


def test_main_lean_bool_module():
    m = _make_twobitcounter()
    src = generate_main_lean("Rea", m, "ReactiveModule")

    assert "parseBool" in src
    assert "parseExtl" in src
    assert "showCtrl" in src
    assert "def main" in src


def test_main_lean_matrix_module():
    m = _make_matrix_module()
    src = generate_main_lean("Rea", m, "ReactiveModule")

    assert "parseIntOrFail" in src
    assert "showMat" in src
    assert "parseExtl" in src
    assert "showCtrl" in src
    assert "def main" in src


def test_main_lean_bool_signatures():
    m = _make_twobitcounter()
    src = generate_main_lean("Rea", m, "ReactiveModule")

    # parseExtl returns Bool, showCtrl takes Bool × Bool
    assert "Mat Bool 1 1" in src
    assert "(Mat Bool 1 1) × (Mat Bool 1 1)" in src


def test_main_lean_matrix_signatures():
    m = _make_matrix_module()
    src = generate_main_lean("Rea", m, "ReactiveModule")

    assert "Mat Int 2 1" in src
    assert "Mat Int 3 1" in src


# ── Certificate generation ───────────────────────────────────────────


def _cert_for(make_module, module_name="ReactiveModule"):
    m = make_module()
    ctx = LeanContext(m)
    return generate_certificate_lean(ctx)


def test_certificate_bool_has_rm():
    cert = _cert_for(_make_twobitcounter)
    assert "def RM : ReactiveModule" in cert
    assert "(Mat Bool 1 1)" in cert
    assert "(Mat Bool 1 1) × (Mat Bool 1 1)" in cert


def test_certificate_bool_rm_uses_plain_functions():
    cert = _cert_for(_make_twobitcounter)
    # Should use init directly, not init.fn
    assert "init := fun e => init e.2" in cert
    assert ".fn" not in cert


def test_certificate_bool_has_theorems():
    cert = _cert_for(_make_twobitcounter)
    assert "theorem init_inv" in cert
    assert "theorem step_inv" in cert
    assert "theorem hinv'" in cert
    assert "theorem hinv" in cert


def test_certificate_bool_has_simp_mod():
    cert = _cert_for(_make_twobitcounter)
    assert 'macro "simp_mat"' in cert
    assert "init, update, inv" in cert


def test_certificate_bool_has_no_scalar_constants_in_simp():
    cert = _cert_for(_make_twobitcounter)
    # Bool module has no matrix constants, so no c0/c1 in simp
    assert "c0, c1" not in cert


def test_data_lean_has_sorry_when_no_property():
    from zrth.lean.cert import generate_data_lean
    m = _make_twobitcounter()
    ctx = LeanContext(m)
    data = generate_data_lean(ctx)
    assert "sorry" in data


def test_certificate_matrix_has_rm():
    cert = _cert_for(_make_matrix_module)
    assert "def RM : ReactiveModule" in cert
    assert "Mat Int 2 1" in cert
    assert "Mat Int 3 1" in cert


def test_certificate_has_hrank():
    cert = _cert_for(_make_twobitcounter)
    assert "theorem hrank" in cert
    # Should not be commented out
    for line in cert.splitlines():
        if "theorem hrank" in line:
            assert not line.lstrip().startswith("--"), (
                "hrank should not be commented out"
            )
            break


def test_certificate_has_buchi():
    cert = _cert_for(_make_twobitcounter)
    assert "def buchi" in cert
    # Should not be commented out
    for line in cert.splitlines():
        if "def buchi" in line:
            assert not line.lstrip().startswith("--"), (
                "buchi should not be commented out"
            )
            break


def test_certificate_matrix_simp_reduces_matmul():
    cert = _cert_for(_make_matrix_module)
    # A/B are baked into the Linear op (no interned c0/c1 constants). affineLinear
    # unfolds to MatMul + b, so the proof simp set must carry MatMul_apply to
    # reduce the matrix obligations.
    assert "MatMul_apply" in cert


# ──────────────────────────────────────────────────────────────
# Degenerate modules
# ──────────────────────────────────────────────────────────────


def _no_ctrl_module():
    """A module with no controlled state: `module.ctrl` is empty."""
    w = Wire(Bool([1, 1]))
    return Module.combinatorial([], [Term(LIA.Bool(torch.tensor([[True]])), [w])])


@pytest.mark.parametrize(
    "encoding",
    ["to_lean_functional", "to_lean_scalar", "to_lean_rel"],
)
def test_no_ctrl_module_does_not_crash_any_encoding(encoding):
    """Every encoder must cope with an empty `ctrl`, not raise.

    An encoder that indexed `ctrl_types[0]` unguarded raised IndexError
    where its siblings returned a comment.
    """
    out = getattr(ModuleToLean4(_no_ctrl_module()), encoding)()
    assert isinstance(out, str)



# ──────────────────────────────────────────────────────────────
# Relational encoding: characterisation
# ──────────────────────────────────────────────────────────────


def test_rel_encoding_emits_its_equivalence_theorems():
    """Pins what the Rel encoding produces, so refactors there stay silent.

    Added when two computed-but-unused simp lists were removed from
    `rel.py`; the proofs inline their own `simp only` sets, so dropping them
    left the output byte-identical.
    """
    lean = ModuleToLean4(_make_twobitcounter()).to_lean_rel()
    for expected in (
        "namespace ScalarRel",
        "theorem InitCond_func_eq",
        "theorem TransRel_func_eq",
        "rw [TransRel_scalar_eq, update_scalar_eq]",
        "simp only [Scalar.pack, Scalar.unpack_ctrl, ← Mat_1_1_eq, Prod.eta]",
    ):
        assert expected in lean, f"missing from the Rel encoding: {expected!r}"


def test_rel_encoding_has_no_unused_simp_placeholders():
    """The removed lists were never interpolated; nothing should reference them."""
    lean = ModuleToLean4(_make_twobitcounter()).to_lean_rel()
    assert "unpack_pack_simp" not in lean


# ──────────────────────────────────────────────────────────────
# Scalar encoding: operand forms
# ──────────────────────────────────────────────────────────────


def test_scalar_encoding_emits_a_scalar_relu():
    """A 1x1 `ReLU` must be scalar on both sides, operand *and* result.

    It used to take the `_LEAN_OP` matrix fallback, which lifts the operand
    but never projects the result: `let x : Int := ReLu (fun _ _ => y)`
    ascribes `Mat ℤ ?m ?n` to `Int`. Asserting only that the operand was
    lifted (as this test once did) passes on exactly that output, so the
    check has to be that no `Mat`-valued form survives at all — lifting the
    operand and projecting the result is not a fix either, since `ReLu`'s
    dimensions are then unconstrained ("typeclass instance problem is
    stuck").
    """
    s = Var(Int([1, 1]))
    module = Module.sequential(
        [s],
        [Term(LIA.Int(torch.tensor([[0]])), [X(s)])],
        [Term(LIA.ReLU(), [X(s)], [s])],
    )
    lean = ModuleToLean4(module).to_lean_scalar()
    relu = [l for l in lean.splitlines() if "Max.max" in l or "ReLu" in l]
    assert relu, f"no ReLU emitted:\n{lean}"
    for line in relu:
        assert "ReLu" not in line, f"matrix ReLu in the scalar encoding: {line.strip()}"
        assert "Max.max 0" in line, f"not a scalar ReLU: {line.strip()}"


def test_scalar_encoding_extracts_a_scalar_from_linear():
    """`matVecAffine` yields a `Mat`, but a 1x1 write wire is ascribed `Int`."""
    s = Var(Int([1, 1]))
    module = Module.sequential(
        [s],
        [Term(LIA.Int(torch.tensor([[0]])), [X(s)])],
        [Term(LIA.Linear(torch.tensor([[2]]), torch.tensor([[1]])), [X(s)], [s])],
    )
    lean = ModuleToLean4(module).to_lean_scalar()
    line = next(l for l in lean.splitlines() if "matVecAffine" in l)
    assert line.strip().endswith("0 0)"), f"result not extracted: {line.strip()}"
    # Ascribed, not a bare `fun _ _ => ctrl`: `matVecAffine`'s batch
    # dimension is not determined by the lambda, and an un-ascribed lift
    # leaves `n` unsolved wherever the surrounding term does not pin it.
    assert "(fun _ _ => ctrl : Mat Int 1 1)" in line, (
        f"operand not lifted with its type: {line.strip()}"
    )
    assert ": Int :=" in line, "a 1x1 write wire should stay scalar-typed"


def test_scalar_encoding_lifts_a_scalar_ite_condition():
    """A matrix `Ite` with a scalar Bool condition emitted
    `if ctrl.2.2.2 0 0 then ...`, applying a bare `Bool` to two arguments.

    (The rest of this module's scalar encoding is still ill-typed — the
    signature flattens multi-element wires while the body keeps them as
    matrices — so this checks the operand form, not the whole file.)
    """
    m3 = Var(Int([1, 3]))
    c = Var(Bool([1, 1]))
    init = [
        Term(LIA.Int(torch.zeros(1, 3, dtype=torch.int64)), [X(m3)]),
        Term(LIA.Bool(torch.tensor([[True]])), [X(c)]),
    ]
    update = [
        Term(LIA.Id(), [X(c)], [c]),
        Term(LIA.Ite(), [X(m3)], [c, m3, m3]),
    ]
    lean = ModuleToLean4(Module.sequential([m3, c], init, update)).to_lean_scalar()
    ite = [l for l in lean.splitlines() if "if " in l]
    assert ite, "no Ite emitted"
    for line in ite:
        assert "(fun _ _ =>" in line, f"scalar operand not lifted: {line.strip()}"


# ──────────────────────────────────────────────────────────────
# Scalar flattening
# ──────────────────────────────────────────────────────────────


def test_scalar_translator_flattening_is_opt_in():
    """`rel.py` types `effect_i` as the wire's `Mat`, so it opts out.

    Making the flattening unconditional in `_translate_terms_scalar` broke
    it, and nothing compiled it to notice.
    """
    from zrth.lean.native import _translate_terms_scalar
    import inspect

    sig = inspect.signature(_translate_terms_scalar)
    assert sig.parameters["flatten_outputs"].default is False, (
        "flattening must be opt-in so the matrix-typed callers keep working"
    )


# ──────────────────────────────────────────────────────────────
# Reconstruction lets follow what the body uses
# ──────────────────────────────────────────────────────────────


def _split_effects(lean):
    """Map `effect_i` -> the reconstruction lets prepended to its body."""
    out, cur = {}, None
    for line in lean.splitlines():
        stripped = line.strip()
        if stripped.startswith("abbrev effect_") or stripped.startswith(
            "@[simp] def effect_"
        ):
            cur = stripped.split("effect_")[1].split()[0].rstrip(":")
            out[cur] = []
        elif cur is not None and stripped.startswith("let _"):
            out[cur].append(stripped)
        elif cur is not None and stripped.startswith(("abbrev ", "theorem ", "def ")):
            cur = None
    return out


def _mixed_dependency_module():
    """Two effects: one reads only state, the other a multi-element extl."""
    s1 = Var(Int([1, 1]))
    s2 = Var(Int([1, 1]))
    e = Var(Int([3, 1]))
    init = [
        Term(LIA.Int(torch.tensor([[0]])), [X(s1)]),
        Term(LIA.Int(torch.tensor([[0]])), [X(s2)]),
    ]
    update = [
        Term(LIA.Id(), [X(s1)], [s1]),
        Term(
            LIA.Linear(torch.tensor([[1, 0, 0]]), torch.tensor([[0]])),
            [X(s2)],
            [e],
        ),
    ]
    return Module.sequential([s1, s2, e], init, update)


def test_effect_gets_only_the_reconstructions_it_uses():
    """An unused let mentions its parameter, which auto-binding then adds.

    The call sites are built from what the body consumes (`_effect_args`),
    so an extra parameter means every call under-applies. `effect_0` here
    reads only `state`, but got the `extl_l` and `extl_n` reconstructions.
    """
    lean = ModuleToLean4(_mixed_dependency_module()).to_lean_rel()
    effects = _split_effects(lean)
    assert effects, f"no effects found in:\n{lean[:400]}"
    assert effects["0"] == [], (
        f"effect_0 depends only on state but carries: {effects['0']}"
    )
    assert any("extl_l" in l for l in effects["1"]), (
        f"effect_1 reads a multi-element extl_l but has: {effects['1']}"
    )
    assert not any("extl_n" in l for l in effects["1"]), (
        f"effect_1 does not read extl_n but has: {effects['1']}"
    )


def test_recon_filter_keeps_order_and_transitive_uses():
    """The filter is order-preserving and reaches a fixed point."""
    from zrth.lean.translate._shared import _needed_recon

    lets = [
        "  let _m0 : T := f extl_l",
        "  let _m1 : T := g extl_n",
        "  let _m2 : T := h _m0",
    ]
    assert _needed_recon(lets, "  x := _m1") == [lets[1]]
    # _m2 pulls in _m0 transitively, and order is preserved
    assert _needed_recon(lets, "  x := _m2") == [lets[0], lets[2]]
    assert _needed_recon(lets, "  x := 0") == []
    assert _needed_recon([], "  x := 0") == []
