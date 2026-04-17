"""Tests for the Module-to-Lean4 functional converter."""

import torch
from zrth import Wire, Term, Module, DType as dt, IType as it
from zrth.lean import ModuleToLean4
from zrth.lean.common import LeanContext, itype_name
from zrth.lean.project import generate_main_lean, generate_certificate_lean


def test_itype_name_strips_prefix():
    assert itype_name(it.Add()) == "Add"
    assert itype_name(it.Tensor(torch.tensor([0]))) == "Tensor"
    assert itype_name(it.Ite()) == "Ite"


def _make_twobitcounter():
    """Bool-only module: two-bit counter with enable."""
    b0 = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))
    b1 = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))
    enable = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))

    not_b0 = Wire(dt.Bool([1]))
    not_b1 = Wire(dt.Bool([1]))
    b0_and_enable = Wire(dt.Bool([1]))

    init = [
        Term(it.Tensor(torch.tensor([False])), [b0[1]]),
        Term(it.Tensor(torch.tensor([False])), [b1[1]]),
    ]
    update = [
        Term(it.Not(), [not_b0], [b0[0]]),
        Term(it.Ite(), [b0[1]], [enable[1], not_b0, b0[0]]),
        Term(it.And(), [b0_and_enable], [b0[0], enable[1]]),
        Term(it.Not(), [not_b1], [b1[0]]),
        Term(it.Ite(), [b1[1]], [b0_and_enable, not_b1, b1[0]]),
    ]
    return Module.sequential(init, update, obs=[b0, b1, enable])


def _make_matrix_module():
    """Matrix module similar to Counter."""
    x = (Wire(dt.Int([3, 1])), Wire(dt.Int([3, 1])))
    u = (Wire(dt.Int([2, 1])), Wire(dt.Int([2, 1])))

    A_wire = Wire(dt.Int([3, 2]))
    init = [
        Term(it.Tensor(torch.tensor([[0, 0], [1, 0], [0, 1]])), [A_wire]),
        Term(it.MatMul(), [x[1]], [A_wire, u[1]]),
    ]

    B_wire = Wire(dt.Int([3, 3]))
    e1_wire = Wire(dt.Int([3, 1]))
    Bx_wire = Wire(dt.Int([3, 1]))
    update = [
        Term(it.Tensor(torch.eye(3, dtype=torch.int64)), [B_wire]),
        Term(it.MatMul(), [Bx_wire], [B_wire, x[0]]),
        Term(it.Tensor(torch.tensor([[1], [0], [0]])), [e1_wire]),
        Term(it.Add(), [x[1]], [Bx_wire, e1_wire]),
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

    # Scalar Bool constants are inlined, not top-level defs
    assert "def c0" not in lean
    assert "let x0 := false" in lean


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


def test_matrix_module_has_matrix_constants():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    assert "Mat Int 3 2" in lean
    assert "Mat Int 3 3" in lean
    assert "Mat Int 3 1" in lean


def test_matrix_module_uses_matmul_and_add():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    assert "MatMul" in lean
    assert "(x1 + x2)" in lean


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
    return generate_certificate_lean("Rea", module_name, ctx)


def test_certificate_bool_has_rm():
    cert = _cert_for(_make_twobitcounter)
    assert "def RM : ReactiveModule" in cert
    assert "(Mat Bool 1 1)" in cert
    assert "(Mat Bool 1 1) × (Mat Bool 1 1)" in cert


def test_certificate_bool_rm_uses_plain_functions():
    cert = _cert_for(_make_twobitcounter)
    # Should use init directly, not init.fn
    assert "init := init" in cert
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


def test_certificate_bool_has_sorry():
    cert = _cert_for(_make_twobitcounter)
    assert "sorry" in cert


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


def test_certificate_matrix_has_constants_in_simp():
    cert = _cert_for(_make_matrix_module)
    assert "c0" in cert
