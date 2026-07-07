"""Tests for the eager, theory-baked Expression (`zrth.expr`).

Building an Expr eagerly produces Terms directly for its theory; these tests check
sort propagation, literal coercion, wire-pair leaves / `nxt`, term collection, and
end-to-end evaluation of the collected terms.
"""

import torch
import pytest

from zrth import LIA, LRA, BV, Sort, Wire
from zrth.builder import NonLinearError
from zrth.eval import eval_itype
import zrth.expr as E


# --- helpers ----------------------------------------------------------------


def _pair(sort):
    return (Wire(sort), Wire(sort))


def _var(name, sort, theory):
    return E.var(_pair(sort), theory, name)


def _run(terms, state):
    """Evaluate collected terms in order, threading wire values through `state`."""
    for t in terms:
        read = [state[w] for w in t.read]
        out_sort = t.write[0].dtype if len(t.write) else None
        for w, val in zip(t.write, eval_itype(t.itype, read, out_sort)):
            state[w] = val
    return state


def _int(n):
    return torch.tensor([[n]], dtype=torch.int64)


# --- leaves & nxt -----------------------------------------------------------


def test_var_reads_latched_and_nxt_reads_next():
    b = LIA
    pair = _pair(Sort.Int([1, 1]))
    x = E.var(pair, b, "x")
    assert x.wire is pair[0]
    assert x.term is None
    assert E.nxt(x).wire is pair[1]


def test_nxt_requires_a_variable():
    b = LIA
    x = _var("x", Sort.Int([1, 1]), b)
    with pytest.raises(ValueError):
        E.nxt(x + 1)  # a computed expr is not a wire-pair variable


# --- sort propagation -------------------------------------------------------


def test_arith_and_compare_sorts_lia():
    b = LIA
    x = _var("x", Sort.Int([1, 1]), b)
    y = _var("y", Sort.Int([1, 1]), b)
    assert (x + y).dtype == Sort.Int([1, 1])
    assert (x - 1).dtype == Sort.Int([1, 1])
    assert (x < y).dtype == Sort.Bool([1, 1])
    assert E.eq(x, y).dtype == Sort.Bool([1, 1])


def test_theory_is_baked_lra_and_bv():
    xr = _var("x", Sort.Real([1, 1]), LRA)
    assert (xr + 1.0).dtype == Sort.Real([1, 1])

    xb = _var("x", Sort.BitVec(32, [1, 1]), BV)
    yb = _var("y", Sort.BitVec(32, [1, 1]), BV)
    assert (xb + 1).dtype == Sort.BitVec(32, [1, 1])
    assert (xb < yb).dtype == Sort.BitVec(1, [1, 1])


# --- coercion ---------------------------------------------------------------


def test_literal_coercion_both_sides():
    b = LIA
    x = _var("x", Sort.Int([1, 1]), b)
    assert isinstance(x + 1, E.Expr)      # __add__ coerces the int
    assert isinstance(1 + x, E.Expr)      # __radd__ coerces the int
    assert isinstance(1 - x, E.Expr)      # __rsub__


# --- term collection --------------------------------------------------------


def test_collect_terms_is_deps_first_and_deduped():
    b = LIA
    x = _var("x", Sort.Int([1, 1]), b)
    y = _var("y", Sort.Int([1, 1]), b)
    e = E.ite(x < y, x + 1, y)
    terms = E.collect_terms(e)
    # Lt, Const(1), Add, Ite — each op-node's term appears exactly once.
    itypes = [type(t.itype).__name__ for t in terms]
    assert itypes == ["LIA_Lt", "LIA_ConstInt", "LIA_Add", "LIA_Ite"]
    # the root's term is last (everything it depends on comes before it)
    assert terms[-1] is e.term


# --- evaluation end-to-end --------------------------------------------------


def test_eval_arith():
    b = LIA
    x = _var("x", Sort.Int([1, 1]), b)
    e = x + 1
    state = _run(E.collect_terms(e), {x.wire: _int(5)})
    assert state[e.wire].item() == 6


def test_eval_ite_both_branches():
    b = LIA
    x = _var("x", Sort.Int([1, 1]), b)
    y = _var("y", Sort.Int([1, 1]), b)
    e = E.ite(x < y, x + 1, y)
    terms = E.collect_terms(e)

    s = _run(terms, {x.wire: _int(1), y.wire: _int(5)})
    assert s[e.wire].item() == 2          # 1 < 5 -> x + 1

    s = _run(terms, {x.wire: _int(5), y.wire: _int(1)})
    assert s[e.wire].item() == 1          # not(5 < 1) -> y


def test_eval_mul_by_constant_folds_to_linear():
    b = LIA
    x = _var("x", Sort.Int([1, 1]), b)
    e = x * 2
    assert isinstance(e.term.itype, LIA.Linear)
    state = _run(E.collect_terms(e), {x.wire: _int(3)})
    assert state[e.wire].item() == 6


def test_mul_of_two_variables_is_nonlinear():
    b = LIA
    x = _var("x", Sort.Int([1, 1]), b)
    y = _var("y", Sort.Int([1, 1]), b)
    with pytest.raises(NonLinearError):
        _ = x * y
