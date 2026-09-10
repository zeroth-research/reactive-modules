"""LIA 1x1 that converges to 5 from either side — a piecewise-linear plant.

x' = x+1 if x < 5, x-1 if x > 5, else 5; init 10.

The point of the fixture is that |x - 5| is a genuine Lyapunov function whose
decrease needs the ReLU case analysis on *both* sides of 5, while only the
descending side is ever reached from init. `hrank` quantifies over every state
satisfying the invariant, so the unreachable branch has to close too.
"""
import torch
from zrth import Module, Term, Wire, Int, Bool, LIA, Var, X


def module() -> Module:
    x = Var(Int([1, 1]))
    five, one = Wire(Int([1, 1])), Wire(Int([1, 1]))
    lo, hi = Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    inc, dec, up = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))
    init = [Term(LIA.Int(torch.tensor([[10]])), [X(x)])]
    update = [
        Term(LIA.Int(torch.tensor([[5]])), [five]),
        Term(LIA.Int(torch.tensor([[1]])), [one]),
        Term(LIA.Lt(), [lo], [x, five]),
        Term(LIA.Gt(), [hi], [x, five]),
        Term(LIA.Add(), [inc], [x, one]),
        Term(LIA.Sub(), [dec], [x, one]),
        Term(LIA.Ite(), [up], [hi, dec, five]),
        Term(LIA.Ite(), [X(x)], [lo, inc, up]),
    ]
    return Module.sequential([x], init, update)
