"""Unit tests for the reflected `Linear` (pre-contraction) codegen.

These are fast, Lean-free checks that the Python side emits the correct list
literals for a `LIA.Linear`/`LRA.Linear` op: right orientation (row-major
`A[i][l]`, not transposed), signs, bias (incl. empty → zeros), non-square
shapes, and element formatting (Int/Real). They complement the lake-verified
`Core.Mat.matVecAffine_eq`, which checks the *Lean* side reduces to the intended
affine map — here we guard the *generator* that feeds it.
"""
from fractions import Fraction

import pytest
import torch

from zrth import BitVec, Bool, Wire, Term, LIA, LRA, Int, Real
from zrth.lean.literals import linear_list_literals
from zrth.lean.native import _linear_expr


def _int_linear(A, B, out_rows):
    """A `LIA.Linear(A, B)` term with out=[out_rows,1], in=[A.cols,1]."""
    in_cols = A.shape[1]
    out_w = Wire(Int([out_rows, 1]))
    x_w = Wire(Int([in_cols, 1]))
    return Term(LIA.Linear(A, B), [out_w], [x_w]), x_w


def test_identity_with_bias():
    A = torch.eye(3, dtype=torch.int64)
    B = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    term, _ = _int_linear(A, B, 3)
    out_m, a_lit, b_lit, ty = linear_list_literals(term)
    assert out_m == 3
    assert a_lit == "([[(1 : Int), (0 : Int), (0 : Int)], [(0 : Int), (1 : Int), (0 : Int)], [(0 : Int), (0 : Int), (1 : Int)]] : List (List Int))"
    assert b_lit == "([(1 : Int), (0 : Int), (0 : Int)] : List Int)"
    assert ty == "Int"


def test_orientation_is_row_major():
    # Asymmetric, non-square: a transpose bug would change both shape and values.
    A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)  # out=2, in=3
    B = torch.zeros((2, 1), dtype=torch.int64)
    term, _ = _int_linear(A, B, 2)
    out_m, a_lit, b_lit, _ = linear_list_literals(term)
    assert out_m == 2
    # rows are A[i][:] in order — NOT the transpose [[1,4],[2,5],[3,6]]
    assert a_lit == "([[(1 : Int), (2 : Int), (3 : Int)], [(4 : Int), (5 : Int), (6 : Int)]] : List (List Int))"
    assert b_lit == "([(0 : Int), (0 : Int)] : List Int)"


def test_negative_coefficients():
    A = torch.tensor([[-1, 2], [0, -3]], dtype=torch.int64)
    B = torch.tensor([[-5], [7]], dtype=torch.int64)
    term, _ = _int_linear(A, B, 2)
    _, a_lit, b_lit, _ = linear_list_literals(term)
    assert a_lit == "([[(-1 : Int), (2 : Int)], [(0 : Int), (-3 : Int)]] : List (List Int))"
    assert b_lit == "([(-5 : Int), (7 : Int)] : List Int)"


def test_empty_bias_becomes_zeros():
    A = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)  # out=3, in=2
    empty = torch.zeros((0, 0), dtype=torch.int64)
    term, _ = _int_linear(A, empty, 3)
    out_m, a_lit, b_lit, _ = linear_list_literals(term)
    assert out_m == 3
    assert a_lit == "([[(0 : Int), (0 : Int)], [(1 : Int), (0 : Int)], [(0 : Int), (1 : Int)]] : List (List Int))"
    assert b_lit == "([(0 : Int), (0 : Int), (0 : Int)] : List Int)"  # one zero per output row


def test_real_lra_formatting():
    A = torch.tensor([[1.5, 0.0]], dtype=torch.float64)  # out=1, in=2
    B = torch.tensor([[2.0]], dtype=torch.float64)
    out_w = Wire(Real([1, 1]))
    x_w = Wire(Real([2, 1]))
    term = Term(LRA.Linear(A, B), [out_w], [x_w])
    out_m, a_lit, b_lit, ty = linear_list_literals(term)
    assert out_m == 1
    assert ty == "Real"
    assert a_lit == "([[((3 : Real) / 2), (0 : Real)]] : List (List Real))"
    assert b_lit == "([(2 : Real)] : List Real)"


def test_linear_expr_emits_matvecaffine():
    A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    B = torch.tensor([[5], [0]], dtype=torch.int64)
    term, x_w = _int_linear(A, B, 2)
    expr = _linear_expr(term, {x_w.id: "extl_n"})
    assert expr == (
        "(matVecAffine 2 ([[(1 : Int), (2 : Int), (3 : Int)], [(4 : Int), (5 : Int), (6 : Int)]] : List (List Int)) "
        "([(5 : Int), (0 : Int)] : List Int) extl_n)"
    )


def test_linear_expr_uses_read_wire_accessor():
    A = torch.eye(2, dtype=torch.int64)
    B = torch.zeros((2, 1), dtype=torch.int64)
    term, x_w = _int_linear(A, B, 2)
    # whatever accessor the caller bound for the read wire is threaded through as X
    assert "ctrl.2.1" in _linear_expr(term, {x_w.id: "ctrl.2.1"})


# ──────────────────────────────────────────────────────────────
# Bool matrix constants
# ──────────────────────────────────────────────────────────────


def test_is_scalar_tensor_respects_shape_for_bool():
    """Bool skipped the shape check that Int/BitVec get, so a Bool matrix
    was treated as a scalar and inlined via `tensor.item()`."""
    from zrth import Wire, Bool, Int, BitVec
    from zrth.lean.sorts import is_scalar_tensor

    assert is_scalar_tensor(Wire(Bool([1, 1]))) is True
    assert is_scalar_tensor(Wire(Bool([2, 3]))) is False
    # unchanged for the sorts that already checked
    assert is_scalar_tensor(Wire(Int([1, 1]))) is True
    assert is_scalar_tensor(Wire(Int([2, 3]))) is False
    assert is_scalar_tensor(Wire(BitVec(8, [2, 3]))) is False


def test_bool_matrix_constant_is_interned_not_inlined():
    """A Bool matrix constant must become a top-level def, not `.item()`."""
    import torch
    from zrth import Term, Module, Bool, LIA, Var, X
    from zrth.lean import ModuleToLean4

    flags = Var(Bool([2, 3]))
    data = torch.tensor([[True, False, True], [False, True, False]])
    module = Module.sequential(
        [flags],
        [Term(LIA.Bool(data), [X(flags)])],
        [Term(LIA.Id(), [X(flags)], [flags])],
    )
    lean = ModuleToLean4(module).to_lean_functional()
    assert "Mat Bool 2 3" in lean, "the Bool matrix constant was not emitted"


# ──────────────────────────────────────────────────────────────
# BitVec literals
# ──────────────────────────────────────────────────────────────


def test_bitvec_literal_uses_of_int():
    """`BitVec.ofNat w (-3)` has no `Neg ℕ` instance and does not elaborate.

    Latent rather than live: `BV.Const` rejects negative tensors ("Const:
    tensor values do not fit in N bits") and BV has no `Linear`, so nothing
    currently feeds a negative here. `ofInt` closes the trap and agrees with
    `ofNat` on non-negatives — both facts pinned in
    tests/lean/Playground.lean.
    """
    from zrth import Wire, BitVec
    from zrth.lean.literals import lean_literal

    dt = Wire(BitVec(8, [1, 1])).dtype
    assert lean_literal(dt, -3) == "(BitVec.ofInt 8 (-3))"
    assert lean_literal(dt, 3) == "(BitVec.ofInt 8 (3))"
    assert "ofNat" not in lean_literal(dt, -1)


def test_non_negative_bitvec_constant_still_emits():
    """The reachable case must be unchanged by the switch to `ofInt`."""
    import torch
    from zrth import Term, Module, BitVec, BV, Var, X
    from zrth.lean import ModuleToLean4

    v = Var(BitVec(8, [1, 1]))
    module = Module.sequential(
        [v],
        [Term(BV.Const(torch.tensor([[5]])), [X(v)])],
        [Term(BV.Id(), [X(v)], [v])],
    )
    lean = ModuleToLean4(module).to_lean_functional()
    assert "BitVec.ofInt 8 (5)" in lean
    assert "BitVec.ofNat" not in lean


@pytest.mark.parametrize("sort, value, lean", [
    (Int([1, 1]), 3, "(3 : Int)"),
    (Int([1, 1]), -3, "(-3 : Int)"),
    (Real([1, 1]), 2.0, "(2 : Real)"),
    (Real([1, 1]), -0.5, "((-1 : Real) / 2)"),
    (Real([1, 1]), 0.1, "((1 : Real) / 10)"),
    (Real([1, 1]), 1e-07, "((1 : Real) / 10000000)"),
    (Bool([1, 1]), True, "true"),
    (BitVec(8, [1, 1]), -3, "(BitVec.ofInt 8 (-3))"),
])
def test_one_printer_spells_every_literal(sort, value, lean):
    """Ints ascribed whatever their sign, Reals exact -- a float read by its
    decimal form, which is also what the SMT encoding states."""
    from zrth.lean.literals import lean_literal

    assert lean_literal(sort, value) == lean


def test_a_float_constant_in_scientific_notation_encodes():
    """`cvc5.mkReal(float)` reads `str(1e-07)` and refuses it; the encoder
    builds the constant from the same exact fraction the Lean literal
    states."""
    import cvc5
    from zrth.lean.smt.encode import translate_terms

    w = Wire(Real([1, 1]))
    term = Term(LRA.Real(torch.tensor([[1e-07]], dtype=torch.float64)), [w], [])
    tm = cvc5.TermManager()
    assert translate_terms(tm, [term], {})[w.id].getRealValue() == Fraction(1, 10**7)
