"""Tests for the eager, theory-baked Expression (`zrth.expr`).

Covers the `expr()` entry point, the `AExpr`/`BExpr`/`WExpr` split, explicit-sort literals,
signed/unsigned word ops, no-implicit-promotion (+ `cast`), term collection, and eval.
"""

import torch
import pytest

from zrth import LIA, LRA, BV, Sort, Wire
from zrth.builder import NonLinearError
from zrth.eval import eval_itype
import zrth.expr as E
from zrth.expr import expr, cast, nxt, ite, eq, AExpr, BExpr, WExpr

INT = Sort.Int([1, 1])
REAL = Sort.Real([1, 1])
BV32 = Sort.BitVec(32, [1, 1])


# --- helpers ----------------------------------------------------------------


def _pair(sort):
    return (Wire(sort), Wire(sort))


def _var(sort, theory=LIA, signed=False):
    return expr(_pair(sort), theory=theory, signed=signed)


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
    assert isinstance(expr(_pair(INT), theory=LIA), AExpr)
    assert isinstance(expr(True, theory=LRA), BExpr)
    assert isinstance(expr(_pair(BV32), theory=BV), WExpr)


def test_expr_requires_theory():
    with pytest.raises(TypeError):
        expr(True)


def test_expr_is_not_idempotent():
    e = expr(True, theory=LRA)
    with pytest.raises(TypeError):
        expr(e, theory=LRA)


def test_numeric_literal_needs_explicit_sort():
    with pytest.raises(TypeError):
        expr(3, theory=LRA)                    # numeric, no sort
    assert isinstance(expr(3, theory=LRA, sort=Sort.Real), AExpr)
    assert isinstance(expr(True, theory=LRA), BExpr)   # bool is exempt


# --- variables & nxt --------------------------------------------------------


def test_variable_reads_latched_and_nxt():
    pair = _pair(INT)
    x = expr(pair, theory=LIA)
    assert x.wire is pair[0]
    assert nxt(x).wire is pair[1]


def test_nxt_requires_a_variable():
    x = _var(INT)
    with pytest.raises(ValueError):
        nxt(x + 1)                             # a computed expr is not a variable


# --- sorts of results -------------------------------------------------------


def test_arith_and_compare_result_sorts():
    x, y = _var(INT), _var(INT)
    assert isinstance(x + y, AExpr) and (x + y).dtype == INT
    assert isinstance(x < y, BExpr) and (x < y).dtype == Sort.Bool([1, 1])
    assert isinstance(eq(x, y), BExpr)


def test_theory_baked_real_and_bv():
    xr = _var(REAL, theory=LRA)
    assert (xr + 1.0).dtype == REAL            # literal coerced to Real by the operator
    xb, yb = _var(BV32, theory=BV), _var(BV32, theory=BV)
    assert (xb + 1).dtype == BV32
    assert (xb < yb).dtype == Sort.BitVec(1, [1, 1])


# --- no implicit promotion; cast is explicit --------------------------------


def test_mixed_sorts_raise():
    with pytest.raises(TypeError):
        _var(INT, theory=LIA) + expr(True, theory=LIA)      # Int + Bool

    with pytest.raises(TypeError):
        ite(expr(True, theory=LIA), _var(INT), _var(REAL, theory=LRA))  # Int vs Real branches


def test_cast_identity_and_unsupported():
    x = _var(INT)
    assert cast(x, Sort.Int) is x              # same sort -> no-op
    with pytest.raises(NotImplementedError):
        cast(x, Sort.Real)                     # real conversion op not in the IR yet


# --- signed / unsigned word ops ---------------------------------------------


def test_bv_signedness_picks_op():
    xs, ys = _var(BV32, theory=BV, signed=True), _var(BV32, theory=BV, signed=True)
    assert isinstance((xs < ys).term.itype, BV.SLt)
    xu, yu = _var(BV32, theory=BV), _var(BV32, theory=BV)   # unsigned default
    assert isinstance((xu < yu).term.itype, BV.ULt)


# --- term collection --------------------------------------------------------


def test_collect_terms_deps_first():
    x, y = _var(INT), _var(INT)
    e = ite(x < y, x + 1, y)
    itypes = [type(t.itype).__name__ for t in E.collect_terms(e)]
    assert itypes == ["LIA_Lt", "LIA_ConstInt", "LIA_Add", "LIA_Ite"]


# --- evaluation end-to-end --------------------------------------------------


def test_eval_arith():
    x = _var(INT)
    e = x + 1
    state = _run(E.collect_terms(e), {x.wire: _int(5)})
    assert state[e.wire].item() == 6


def test_eval_ite_both_branches():
    x, y = _var(INT), _var(INT)
    e = ite(x < y, x + 1, y)
    terms = E.collect_terms(e)
    assert _run(terms, {x.wire: _int(1), y.wire: _int(5)})[e.wire].item() == 2
    assert _run(terms, {x.wire: _int(5), y.wire: _int(1)})[e.wire].item() == 1


def test_mul_by_constant_folds_to_linear():
    x = _var(INT)
    e = x * 2
    assert isinstance(e.term.itype, LIA.Linear)
    assert _run(E.collect_terms(e), {x.wire: _int(3)})[e.wire].item() == 6


def test_mul_of_two_variables_is_nonlinear():
    x, y = _var(INT), _var(INT)
    with pytest.raises(NonLinearError):
        _ = x * y
