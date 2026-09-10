"""Two independent 1x1 plants, each converging to 5 from either side.

`m_toward5` in two dimensions, so that |x-5| + |y-5| is a four-unit ReLU
Lyapunov function over a two-wire state.
"""
import torch
from zrth import Module, Term, Wire, Int, Bool, LIA, Var, X


def _leg(v, init_val):
    five, one = Wire(Int([1, 1])), Wire(Int([1, 1]))
    lo, hi = Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    inc, dec, up = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))
    init = [Term(LIA.Int(torch.tensor([[init_val]])), [X(v)])]
    update = [
        Term(LIA.Int(torch.tensor([[5]])), [five]),
        Term(LIA.Int(torch.tensor([[1]])), [one]),
        Term(LIA.Lt(), [lo], [v, five]),
        Term(LIA.Gt(), [hi], [v, five]),
        Term(LIA.Add(), [inc], [v, one]),
        Term(LIA.Sub(), [dec], [v, one]),
        Term(LIA.Ite(), [up], [hi, dec, five]),
        Term(LIA.Ite(), [X(v)], [lo, inc, up]),
    ]
    return init, update


def module() -> Module:
    x, y = Var(Int([1, 1])), Var(Int([1, 1]))
    ix, ux = _leg(x, 10)
    iy, uy = _leg(y, 0)
    return Module.sequential([x, y], ix + iy, ux + uy)
