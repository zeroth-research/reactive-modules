"""Tests for the eager, theory-baked Expression (`zrth.expr`).

Covers `expr()`, the `AExpr`/`BExpr`/`WExpr` split, explicit-sort literals, signed/unsigned
word ops, no-implicit-promotion (+ `cast`), the `collecting()` term collector, and eval.
"""

import torch
import pytest

from zrth import LIA, LRA, BV, Wire, Bool, Int, Real, BitVec, Var, X as _X
from zrth.builder import NonLinearError
from zrth.eval import eval_itype
from zrth.expr import expr, cast, ite, collecting, AExpr, BExpr, WExpr, X

INT = Int([1, 1])
REAL = Real([1, 1])
BV32 = BitVec(32, [1, 1])


@pytest.fixture(autouse=True)
def _collector():
    # Building a Term requires an active collector; wrap every test in one. Tests that
    # need the captured terms open their own nested `with collecting() as terms:`.
    with collecting():
        yield


# --- helpers ----------------------------------------------------------------


def _var(sort):
    # return (Wire(sort), Wire(sort))
    return Var(sort)


def _varexpr(sort, theory=LIA, signed=False):
    return expr(_var(sort), theory=theory, signed=signed)


def _run(terms, state):
    for t in terms:
        read = [state[w] for w in t.read]
        out_sort = t.write[0].dtype if len(t.write) else None
        for w, val in zip(t.write, eval_itype(t.itype, read, out_sort)):
            state[w] = val
    return state


def _int(n):
    return torch.tensor([[n]], dtype=torch.int64)


# --- expr() factory & guards ------------------------------------------------


def test_factory_picks_subclass_from_sort():
    assert isinstance(expr(_var(INT), theory=LIA), AExpr)
    assert isinstance(expr(True, sort=Bool, theory=LRA), BExpr)
    assert isinstance(expr(_var(BV32), theory=BV), WExpr)


def test_expr_requires_theory():
    with pytest.raises(TypeError):
        expr(True)


def test_expr_is_not_idempotent():
    e = expr(True, theory=LRA)
    with pytest.raises(TypeError):
        expr(e, theory=LRA)


def test_numeric_literal_needs_explicit_sort():
    with pytest.raises(TypeError):
        expr(3, theory=LRA)
    assert isinstance(expr(3, theory=LRA, sort=Real), AExpr)
    assert isinstance(expr(True, theory=LRA), BExpr)  # bool is exempt


# expr is maximally permissive - anything that torch accepts we accept
# def test_float_value_rejected_for_an_integral_sort():
#     # would be silently truncated, so `3.14 + x` on an Int x must fail rather than add 3
#     x = _varexpr(INT)
#     with pytest.raises(TypeError, match="unsupported"):
#         expr(3.14, theory=LIA, sort=INT)
#
#     with pytest.raises(TypeError, match="unsupported"):
#         expr(3.14, theory=LIA, sort=INT)
#
#     with pytest.raises(TypeError, match="unsupported"):
#         expr(torch.tensor([[1.5]]), theory=LIA, sort=INT)
#
#     with pytest.raises(TypeError, match="unsupported"):
#         expr([0.1, 1.0], theory=LIA, sort=Int([1, 2]))
#
#     with pytest.raises(TypeError, match="unsupported"):
#         expr(3.14, theory=BV, sort=BV32)
#
#     with pytest.raises(TypeError, match="unsupported"):
#         3.14 + x
#
#     with pytest.raises(TypeError, match="unsupported"):
#         x + 3.14


# def test_non_bool_value_rejected_for_a_bool_sort():
#     # 5 -> True collapses just as lossily as 3.14 -> 3
#     for value in (5, -1, 3.14):
#         with pytest.raises(TypeError, match="unsupported"):
#             expr(value, theory=LIA, sort=Bool([1, 1]))


def tensor_of_term(term):
    match term.itype:
        case LRA.Real(t):
            return t
        case LRA.Bool(t):
            return t
        case LIA.Bool(t):
            return t
        case LIA.Int(t):
            return t
        case _:
            return None


def test_lossless_value_conversions_are_not_happening():
    with collecting() as terms:
        expr(3, theory=LRA, sort=REAL)  # int -> Real
        assert tensor_of_term(terms[0]).dtype == torch.float

    with collecting() as terms:
        expr(True, theory=LIA, sort=INT)  # bool -> Int
        assert tensor_of_term(terms[0]).dtype == torch.int

    with collecting() as terms:
        expr(True, theory=LRA, sort=REAL)  # int -> Real
        assert tensor_of_term(terms[0]).dtype == torch.float

    with collecting() as terms:
        expr(torch.tensor([[2]]), theory=LRA, sort=REAL)
        assert tensor_of_term(terms[0]).dtype == torch.long

    # assert expr(True, theory=LIA, sort=INT)._value.dtype is torch.int64  # bool -> Int
    # assert expr(True, theory=LRA, sort=REAL)._value.dtype is torch.float32  # bool -> Real
    # # an int tensor on a Real wire is converted, not left as int
    # assert expr(torch.tensor([[2]]), theory=LRA, sort=REAL)._value.dtype is torch.float32


def test_explicit_sort_wins_for_bool_value():
    # bool is a subtype of int: an explicit sort must be honored, not overridden to Bool
    x = expr(True, theory=LIA, sort=Int([1, 1]))
    assert isinstance(x, AExpr) and isinstance(x.dtype, Int)
    assert isinstance(expr(True, theory=LIA), BExpr)  # but no sort -> still Bool


# --- variables & nxt --------------------------------------------------------


def test_variable_reads_latched_and_nxt():
    pair = _var(INT)
    x = expr(pair, theory=LIA)
    assert x.wire is pair
    assert X(x).wire == _X(pair)


def test_nxt_requires_a_variable():
    x = _varexpr(INT)
    with pytest.raises(TypeError):
        X(x + 1)


# not valid anymore
# def test_variable_wire_pair_must_share_a_sort():
#     # otherwise x.dtype != nxt(x).dtype; raise (not assert, so -O keeps the check)
#     with pytest.raises(TypeError, match="same sort"):
#         expr(Var(INT), theory=LIA)


# --- result sorts -----------------------------------------------------------


def test_arith_and_compare_result_sorts():
    x, y = _varexpr(INT), _varexpr(INT)
    assert isinstance(x + y, AExpr) and (x + y).dtype == INT
    assert isinstance(x < y, BExpr) and (x < y).dtype == Bool([1, 1])
    assert isinstance(x == y, BExpr)


def test_theory_baked_real_and_bv():
    xr = _varexpr(REAL, theory=LRA)
    assert (xr + 1.0).dtype == REAL
    xb, yb = _varexpr(BV32, theory=BV), _varexpr(BV32, theory=BV)
    assert (xb + 1).dtype == BV32
    assert (xb < yb).dtype == BitVec(1, [1, 1])


# --- equality builds predicates; truth-testing is refused -------------------


def test_equality_operators_build_predicates():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        e = x == y
    assert isinstance(e, BExpr) and isinstance(terms[-1].itype, LIA.Eq)
    with collecting() as terms:
        e = x != y
    assert isinstance(e, BExpr) and isinstance(terms[-1].itype, LIA.Ne)


def test_equality_coerces_a_raw_operand():
    x = _varexpr(INT)
    with collecting() as terms:
        e = x == 3
    assert isinstance(e, BExpr) and isinstance(terms[-1].itype, LIA.Eq)


def test_truth_testing_raises():
    x, y = _varexpr(INT), _varexpr(INT)
    for thunk in (lambda: bool(x),
                  lambda: "t" if (x < y) else "f",
                  lambda: "t" if (x == y) else "f",
                  lambda: x and y,
                  lambda: x or y,
                  lambda: not x):
        with pytest.raises(TypeError, match="no truth value"):
            thunk()


def test_expr_is_not_hashable():
    x = _varexpr(INT)
    with pytest.raises(TypeError, match="unhashable"):
        {x: 1}


# --- no implicit promotion; cast is explicit --------------------------------


def test_mixed_sorts_raise():
    with pytest.raises(TypeError):
        _varexpr(INT, theory=LIA) + expr(True, theory=LIA)  # Int + Bool
    with pytest.raises(TypeError):
        ite(expr(True, theory=LIA), _varexpr(INT), _varexpr(REAL, theory=LRA))  # Int vs Real branches


def test_cast_identity_and_unsupported():
    x = _varexpr(INT)
    assert cast(x, Int) is x
    with pytest.raises(NotImplementedError):
        cast(x, Real)


# --- signed / unsigned word ops ---------------------------------------------


def test_bv_signedness_picks_op():
    xs, ys = _varexpr(BV32, theory=BV, signed=True), _varexpr(BV32, theory=BV, signed=True)
    with collecting() as terms:
        xs < ys
    assert isinstance(terms[-1].itype, BV.SLt)

    xu, yu = _varexpr(BV32, theory=BV), _varexpr(BV32, theory=BV)
    with collecting() as terms:
        xu < yu
    assert isinstance(terms[-1].itype, BV.ULt)


def test_coercion_inherits_signedness_regardless_of_operand_order():
    # a coerced literal takes the variable's signedness, so `5 + xs` and `xs + 5`
    # both produce a *signed* comparison for a signed bit-vector `xs`.
    xs = _varexpr(BV32, theory=BV, signed=True)
    with collecting() as terms:
        (xs + 5) < 3
    assert isinstance(terms[-1].itype, BV.SLt)
    with collecting() as terms:
        (5 + xs) < 3
    assert isinstance(terms[-1].itype, BV.SLt)


def test_mixing_signed_and_unsigned_bv_raises():
    xs = _varexpr(BV32, theory=BV, signed=True)
    yu = _varexpr(BV32, theory=BV, signed=False)
    with pytest.raises(TypeError, match="signed and an unsigned"):
        xs + yu


# --- collector --------------------------------------------------------------


def test_collecting_records_deps_first():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        ite(x < y, x + 1, y)
    optypes = [type(t.itype) for t in terms]
    assert optypes == [LIA.Lt, LIA.Int, LIA.Add, LIA.Ite]


def test_shared_subexpression_recorded_once():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        g = x < y
        ite(g, x + 1, x)
        ite(g, y, y)
    assert sum(isinstance(t.itype, LIA.Lt) for t in terms) == 1


# --- evaluation end-to-end --------------------------------------------------


def test_eval_arith():
    x = _varexpr(INT)
    with collecting() as terms:
        e = x + 1
    assert _run(terms, {x.wire: _int(5)})[e.wire].item() == 6


def test_eval_ite_both_branches():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        e = ite(x < y, x + 1, y)
    assert _run(terms, {x.wire: _int(1), y.wire: _int(5)})[e.wire].item() == 2
    assert _run(terms, {x.wire: _int(5), y.wire: _int(1)})[e.wire].item() == 1


def test_mul_by_constant_folds_to_linear():
    x = _varexpr(INT)
    with collecting() as terms:
        e = x * 2
    assert isinstance(terms[-1].itype, LIA.Linear)
    assert _run(terms, {x.wire: _int(3)})[e.wire].item() == 6


@pytest.mark.parametrize("n", [1, 2, 3])
def test_scaling_a_column_vector_by_a_constant(n):
    x = _varexpr(Int([n, 1]))
    with collecting() as terms:
        e = x * 3
    assert isinstance(terms[-1].itype, LIA.Linear)
    vals = torch.arange(1, n + 1, dtype=torch.int64).reshape(n, 1)
    got = _run(terms, {x.wire: vals})[e.wire]
    assert got.flatten().tolist() == [3 * v for v in range(1, n + 1)]


def test_mul_of_two_variables_is_nonlinear():
    x, y = _varexpr(INT), _varexpr(INT)
    with pytest.raises(NonLinearError):
        _ = x * y


# --- single entry point: const / var / pair / wire all go through expr() -----


def test_single_entry_point_spec():
    """The one `expr()` handles every leaf; there is no separate var/const/pair API,
    and no string distinguishes a variable. (Spec from the `marco/dsl` review.)"""
    BOOL = Bool([1, 1])

    # a wire pair with no theory -> not enough information to pick a class
    with pytest.raises(TypeError):
        expr((Wire(BOOL), Wire(BOOL)))

    # wire pair -> state variable; bool literal and bare wire -> all BExpr (Bool sort)
    x = expr(Var(BOOL), theory=LRA)
    y = expr(True, theory=LRA)
    w = expr(Wire(BOOL), theory=LRA)
    assert isinstance(x, BExpr) and isinstance(y, BExpr) and isinstance(w, BExpr)

    # boolean composition stays BExpr; a raw operand coerces through expr()
    assert isinstance(x & y & w, BExpr)
    assert isinstance(x & y & w & True, BExpr)

    # a numeric literal needs an explicit sort; with one it is an AExpr
    with pytest.raises(TypeError):
        expr(0.3, theory=LRA)
    b = expr(0.3, theory=LRA, sort=Real)
    assert isinstance(b, AExpr)
    assert isinstance(b + 1, AExpr)  # raw operand coerces

    # list / tensor constants; matmul with a raw tensor coerces the tensor
    t = expr([0.1, 1.0], theory=LRA, sort=Real)
    assert isinstance(t, AExpr)
    u = t @ torch.tensor([[0.2], [0.5]])
    assert isinstance(u, AExpr)

    # not idempotent: passing an Expr back into expr() is an error
    with pytest.raises(TypeError):
        expr(u)


def test_tag_is_a_label_with_no_logical_meaning():
    # `tag` is optional, purely a display label; it does not affect the class or ops.
    p = Var(Bool([1, 1]))
    x = expr(p, theory=LRA, tag="x")
    assert x.tag == "x"
    assert isinstance(x, BExpr)
    assert "x" in repr(x)
    # untagged leaves and derived expressions simply have no tag
    assert expr(True, theory=LRA).tag is None
    assert (x & True).tag is None
