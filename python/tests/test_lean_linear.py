"""Unit tests for the reflected `Linear` (pre-contraction) codegen.

These are fast, Lean-free checks that the Python side emits the correct list
literals for a `LIA.Linear`/`LRA.Linear` op: right orientation (row-major
`A[i][l]`, not transposed), signs, bias (incl. empty → zeros), non-square
shapes, and element formatting (Int/Real). They complement the lake-verified
`Core.Mat.matVecAffine_eq`, which checks the *Lean* side reduces to the intended
affine map — here we guard the *generator* that feeds it.
"""
import torch

from zrth import Wire, Term, LIA, LRA, Int, Real
from zrth.lean.common import linear_list_literals
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
    assert a_lit == "([[1, 0, 0], [0, 1, 0], [0, 0, 1]] : List (List Int))"
    assert b_lit == "([1, 0, 0] : List Int)"
    assert ty == "Int"


def test_orientation_is_row_major():
    # Asymmetric, non-square: a transpose bug would change both shape and values.
    A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)  # out=2, in=3
    B = torch.zeros((2, 1), dtype=torch.int64)
    term, _ = _int_linear(A, B, 2)
    out_m, a_lit, b_lit, _ = linear_list_literals(term)
    assert out_m == 2
    # rows are A[i][:] in order — NOT the transpose [[1,4],[2,5],[3,6]]
    assert a_lit == "([[1, 2, 3], [4, 5, 6]] : List (List Int))"
    assert b_lit == "([0, 0] : List Int)"


def test_negative_coefficients():
    A = torch.tensor([[-1, 2], [0, -3]], dtype=torch.int64)
    B = torch.tensor([[-5], [7]], dtype=torch.int64)
    term, _ = _int_linear(A, B, 2)
    _, a_lit, b_lit, _ = linear_list_literals(term)
    assert a_lit == "([[-1, 2], [0, -3]] : List (List Int))"
    assert b_lit == "([-5, 7] : List Int)"


def test_empty_bias_becomes_zeros():
    A = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)  # out=3, in=2
    empty = torch.zeros((0, 0), dtype=torch.int64)
    term, _ = _int_linear(A, empty, 3)
    out_m, a_lit, b_lit, _ = linear_list_literals(term)
    assert out_m == 3
    assert a_lit == "([[0, 0], [1, 0], [0, 1]] : List (List Int))"
    assert b_lit == "([0, 0, 0] : List Int)"  # one zero per output row


def test_real_lra_formatting():
    A = torch.tensor([[1.5, 0.0]], dtype=torch.float64)  # out=1, in=2
    B = torch.tensor([[2.0]], dtype=torch.float64)
    out_w = Wire(Real([1, 1]))
    x_w = Wire(Real([2, 1]))
    term = Term(LRA.Linear(A, B), [out_w], [x_w])
    out_m, a_lit, b_lit, ty = linear_list_literals(term)
    assert out_m == 1
    assert ty == "Real"
    assert a_lit == "([[(1.5 : Real), (0 : Real)]] : List (List Real))"
    assert b_lit == "([(2 : Real)] : List Real)"


def test_linear_expr_emits_matvecaffine():
    A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    B = torch.tensor([[5], [0]], dtype=torch.int64)
    term, x_w = _int_linear(A, B, 2)
    expr = _linear_expr(term, {x_w.id: "extl_n"})
    assert expr == (
        "(matVecAffine 2 ([[1, 2, 3], [4, 5, 6]] : List (List Int)) "
        "([5, 0] : List Int) extl_n)"
    )


def test_linear_expr_uses_read_wire_accessor():
    A = torch.eye(2, dtype=torch.int64)
    B = torch.zeros((2, 1), dtype=torch.int64)
    term, x_w = _int_linear(A, B, 2)
    # whatever accessor the caller bound for the read wire is threaded through as X
    assert "ctrl.2.1" in _linear_expr(term, {x_w.id: "ctrl.2.1"})
