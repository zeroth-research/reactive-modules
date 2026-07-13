"""Tests for the Module-to-Lean4 functional converter."""

import pytest
import torch
from zrth import Wire, Term, Module, Sort as dt, LIA
from zrth.lean import ModuleToLean4
from zrth.lean.common import LeanContext, itype_name
from zrth.lean.cert import generate_certificate_lean
from zrth.lean.project import generate_main_lean


def test_itype_name_strips_prefix():
    assert itype_name(LIA.Add()) == "Add"
    assert itype_name(LIA.ConstInt(torch.tensor([[0]]))) == "ConstInt"
    assert itype_name(LIA.Ite()) == "Ite"


def _make_twobitcounter():
    """Bool-only module: two-bit counter with enable."""
    b0 = (Wire(dt.Bool([1, 1])), Wire(dt.Bool([1, 1])))
    b1 = (Wire(dt.Bool([1, 1])), Wire(dt.Bool([1, 1])))
    enable = (Wire(dt.Bool([1, 1])), Wire(dt.Bool([1, 1])))

    not_b0 = Wire(dt.Bool([1, 1]))
    not_b1 = Wire(dt.Bool([1, 1]))
    b0_and_enable = Wire(dt.Bool([1, 1]))

    init = [
        Term(LIA.ConstBool(torch.tensor([[False]])), [b0[1]]),
        Term(LIA.ConstBool(torch.tensor([[False]])), [b1[1]]),
    ]
    update = [
        Term(LIA.Not(), [not_b0], [b0[0]]),
        Term(LIA.Ite(), [b0[1]], [enable[1], not_b0, b0[0]]),
        Term(LIA.And(), [b0_and_enable], [b0[0], enable[1]]),
        Term(LIA.Not(), [not_b1], [b1[0]]),
        Term(LIA.Ite(), [b1[1]], [b0_and_enable, not_b1, b1[0]]),
    ]
    return Module.sequential(init, update, obs=[b0, b1, enable])


def _make_matrix_module():
    """Matrix module using LIA.Linear (affine map Y = A·X + B).

    Both transitions are constant-matrix maps, so they are linear and expressed
    with `LIA.Linear` (A and B baked into the op) rather than a (BV-only) MatMul:
      init:   x' = A · u          with A = [[0,0],[1,0],[0,1]]  (no bias)
      update: x' = B · x + e1     with B = I₃, e1 = [1,0,0]ᵀ
    """
    x = (Wire(dt.Int([3, 1])), Wire(dt.Int([3, 1])))
    u = (Wire(dt.Int([2, 1])), Wire(dt.Int([2, 1])))

    A = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    init = [
        Term(LIA.Linear(A, torch.zeros((3, 1), dtype=torch.int64)), [x[1]], [u[1]]),
    ]

    B = torch.eye(3, dtype=torch.int64)
    e1 = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    update = [
        Term(LIA.Linear(B, e1), [x[1]], [x[0]]),
    ]
    return Module.sequential(init, update, obs=[x, u])


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
