"""Tests for the Module-to-Lean4 Box wiring diagram converter."""

import torch
from zrth import Wire, Term, Module, DType as dt, IType as it
from zrth.lean.diagram import ModuleToLean4, dtype_to_lean_ty, itype_name
from zrth.lean.project import generate_main_lean, generate_certificate_lean


def test_dtype_bool_scalar():
    assert dtype_to_lean_ty(Wire(dt.Bool([1]))) == ".bool"


def test_dtype_int_scalar():
    assert dtype_to_lean_ty(Wire(dt.Int([1]))) == ".int"


def test_dtype_int_matrix():
    assert dtype_to_lean_ty(Wire(dt.Int([3, 2]))) == ".mat 3 2"


def test_dtype_int_vector():
    assert dtype_to_lean_ty(Wire(dt.Int([3]))) == ".mat 3 1"


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


def test_twobitcounter_generates_lean():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()

    assert "open Box" in lean
    assert "def init" in lean
    assert "def update" in lean


def test_twobitcounter_has_constants():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()

    # init uses two Tensor(False) constants
    assert "def c0 : Bool := false" in lean
    assert "def c1 : Bool := false" in lean


def test_twobitcounter_update_has_expected_ops():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()

    assert "not" in lean
    assert "ite" in lean
    assert "and" in lean


def test_twobitcounter_has_layer_defs():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()
    assert "def L1" in lean
    assert "def L2" in lean


def test_twobitcounter_has_layer_lemmas():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()
    assert "theorem L1_1" in lean
    assert "by rfl" in lean


def test_twobitcounter_update_uses_layers():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()
    assert "L1 ≫" in lean


def test_twobitcounter_update_signature():
    m = _make_twobitcounter()
    lean = ModuleToLean4(m).to_lean()

    # update takes (b0_latched, b1_latched, enable_next) and returns (b0_next, b1_next)
    assert "def update : Box [.bool, .bool, .bool] [.bool, .bool]" in lean


# ── Matrix module ────────────────────────────────────────────────────────


def test_matrix_module_has_layer_defs():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()
    assert "def L1" in lean
    assert "def L2" in lean


def test_matrix_module_generates_lean():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    assert "open Box" in lean
    assert "def init" in lean
    assert "def update" in lean


def test_matrix_module_has_matrix_constants():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    assert "Fin 3 → Fin 2 → Int" in lean
    assert "Fin 3 → Fin 3 → Int" in lean
    assert "Fin 3 → Fin 1 → Int" in lean


def test_matrix_module_uses_matmul_and_matadd():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    assert "matMul" in lean
    assert "matAdd" in lean


def test_matrix_init_signature():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    # init takes u (mat 2 1) and produces x (mat 3 1)
    assert "def init : Box [.mat 2 1] [.mat 3 1]" in lean


def test_matrix_update_signature():
    m = _make_matrix_module()
    lean = ModuleToLean4(m).to_lean()

    # update takes (x_latched, u_next) and produces x_next
    assert "def update : Box [.mat 3 1, .mat 2 1] [.mat 3 1]" in lean


# ── Main.lean generation ────────────────────────────────────────────────


def test_main_lean_bool_module():
    m = _make_twobitcounter()
    src = generate_main_lean("Rea", m, "ReactiveModule")

    assert "parseBool" in src
    assert "parseExtl" in src
    assert "showCtrl" in src
    assert "def main" in src
    assert "ValTuple.append" in src


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

    # parseExtl takes 1 bool (enable)
    assert "ValTuple [.bool]" in src
    # showCtrl takes 2 bools (b0, b1)
    assert "ValTuple [.bool, .bool]" in src


def test_main_lean_matrix_signatures():
    m = _make_matrix_module()
    src = generate_main_lean("Rea", m, "ReactiveModule")

    # parseExtl takes u (mat 2 1)
    assert "ValTuple [.mat 2 1]" in src
    # showCtrl takes x (mat 3 1)
    assert "ValTuple [.mat 3 1]" in src


# ── Certificate generation ───────────────────────────────────────────


def _cert_for(make_module, module_name="ReactiveModule"):
    m = make_module()
    m2l = ModuleToLean4(m)
    m2l.to_lean()
    return generate_certificate_lean("Rea", m, module_name, m2l.const_names, m2l.update_layer_count)


def test_certificate_bool_has_rm():
    cert = _cert_for(_make_twobitcounter)
    assert "def RM : ReactiveModule" in cert
    assert "ValTuple [.bool]" in cert
    assert "ValTuple [.bool, .bool]" in cert


def test_certificate_bool_has_theorems():
    cert = _cert_for(_make_twobitcounter)
    assert "theorem init_inv" in cert
    assert "theorem step_inv" in cert
    assert "theorem hinv'" in cert
    assert "theorem hinv" in cert


def test_certificate_bool_has_box_simp():
    cert = _cert_for(_make_twobitcounter)
    assert 'macro "box_simp"' in cert
    assert "init, update, inv" in cert


def test_certificate_bool_has_constants_in_simp():
    cert = _cert_for(_make_twobitcounter)
    assert "c0, c1" in cert


def test_certificate_bool_has_layers_in_simp():
    cert = _cert_for(_make_twobitcounter)
    assert "L1" in cert
    assert "L2" in cert


def test_certificate_bool_has_sorry():
    cert = _cert_for(_make_twobitcounter)
    assert "sorry" in cert


def test_certificate_matrix_has_rm():
    cert = _cert_for(_make_matrix_module)
    assert "def RM : ReactiveModule" in cert
    assert "ValTuple [.mat 2 1]" in cert
    assert "ValTuple [.mat 3 1]" in cert


def test_certificate_matrix_has_constants_in_simp():
    cert = _cert_for(_make_matrix_module)
    # matrix module has c0 (A), c1 (B), c2 (e1)
    assert "c0" in cert
