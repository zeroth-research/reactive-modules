"""Tests for the module -> cvc5 encoder (`zrth.lean.smt_encode`).

`Min` / `Max` are unary *reductions* in the theory: one read wire of any
shape, a `Mat t 1 1` written, matching `Core.Mat.matMin` / `matMax` and the
shape of `Argmax`. They had been wired to the binary `_elementwise` path, so
building the SMT term for any module using one raised `TypeError` and took
down every cvc5 query about that module.

One `TermManager` is shared by the whole file, and every `Solver` is kept
alive in `_KEEP`: the cvc5 Python bindings segfault at interpreter shutdown
when a manager is collected before terms minted from it.
"""

import cvc5
import pytest
import torch

from zrth import Module, Term, Wire, Var, X, LIA, Int
from zrth.lean.smt_module import ModuleSMT

_TM = cvc5.TermManager()
_KEEP: list = []


def _reduce_module(itype, rows: int) -> Module:
    """`x' = <reduce> (A·x + b)` with `b = (0, -1, -2, ...)`.

    So the reduction sees `x, x-1, ..., x-(rows-1)`: `Max` is the identity
    and `Min` subtracts `rows-1`, whatever `x` is. A reduction that only
    looked at one element would still pass for `Max`, which is why `Min` is
    tested alongside it.
    """
    x = Var(Int([1, 1]))
    v = Wire(Int([rows, 1]))
    out = Wire(Int([1, 1]))
    A = torch.tensor([[1]] * rows, dtype=torch.int64)
    b = torch.tensor([[-i] for i in range(rows)], dtype=torch.int64)
    return Module.sequential(
        [x],
        [Term(LIA.Int(torch.tensor([[5]])), [X(x)])],
        [
            Term(LIA.Linear(A, b), [v], [x]),
            Term(itype, [out], [v]),
            Term(LIA.Id(), [X(x)], [out]),
        ],
    )


def _step(module: Module, value: int) -> int:
    """Run one symbolic step from `x = value` and read the state back off."""
    m = ModuleSMT(tm=_TM, module=module)
    nxt = m.update_state([_TM.mkInteger(value)], [], [])
    solver = cvc5.Solver(_TM)
    _KEEP.append(solver)
    solver.setLogic("ALL")
    solver.setOption("produce-models", "true")
    solver.checkSat()
    return solver.getValue(nxt[0]).getIntegerValue()


@pytest.mark.parametrize("value", [-3, 0, 4, 17])
def test_max_reduces_over_the_whole_column(value):
    assert _step(_reduce_module(LIA.Max(), 3), value) == value


@pytest.mark.parametrize("value", [-3, 0, 4, 17])
def test_min_reduces_over_the_whole_column(value):
    assert _step(_reduce_module(LIA.Min(), 3), value) == value - 2


def test_reduction_over_a_single_element_is_the_identity():
    assert _step(_reduce_module(LIA.Max(), 1), 7) == 7


def test_max_module_encodes_at_all():
    """The regression itself: building the term used to raise TypeError."""
    m = ModuleSMT(tm=_TM, module=_reduce_module(LIA.Max(), 2))
    assert m.update_state(m.fresh_ctrl("s"), [], [])
