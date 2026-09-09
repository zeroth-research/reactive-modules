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


# ──────────────────────────────────────────────────────────────
# Bool matrix constants
# ──────────────────────────────────────────────────────────────


def test_is_scalar_tensor_respects_shape_for_bool():
    """Bool skipped the shape check that Int/BitVec get, so a Bool matrix
    was treated as a scalar and inlined via `tensor.item()`."""
    from zrth import Wire, Bool, Int, BitVec
    from zrth.lean.common import _is_scalar_tensor

    assert _is_scalar_tensor(Wire(Bool([1, 1]))) is True
    assert _is_scalar_tensor(Wire(Bool([2, 3]))) is False
    # unchanged for the sorts that already checked
    assert _is_scalar_tensor(Wire(Int([1, 1]))) is True
    assert _is_scalar_tensor(Wire(Int([2, 3]))) is False
    assert _is_scalar_tensor(Wire(BitVec(8, [2, 3]))) is False


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
    from zrth.lean.common import _get_dtype_item

    dt = Wire(BitVec(8, [1, 1])).dtype
    assert _get_dtype_item(dt, -3) == "(BitVec.ofInt 8 (-3))"
    assert _get_dtype_item(dt, 3) == "(BitVec.ofInt 8 (3))"
    assert "ofNat" not in _get_dtype_item(dt, -1)


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
