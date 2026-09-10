"""LIA 1x1 with ReLU in the transition: x' = relu(x - 1), x0 = 5.

Counts down to 0 and stays there, so `x = 0` holds infinitely often. The
Lean side becomes `Max.max 0 (x - 1)`.
"""
import torch
from zrth import Module, Term, Wire, Int, LIA, Var, X


def module() -> Module:
    x = Var(Int([1, 1]))
    one = Wire(Int([1, 1]))
    diff = Wire(Int([1, 1]))
    init = [Term(LIA.Int(torch.tensor([[5]])), [X(x)])]
    update = [
        Term(LIA.Int(torch.tensor([[1]])), [one]),
        Term(LIA.Sub(), [diff], [x, one]),
        Term(LIA.ReLU(), [X(x)], [diff]),
    ]
    return Module.sequential([x], init, update)
