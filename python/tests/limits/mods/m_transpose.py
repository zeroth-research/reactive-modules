"""LIA Transpose — an op with no entry in the Lean expression tables."""
import torch
from zrth import Module, Term, Wire, Int, LIA, Var, X


def module() -> Module:
    v = Var(Int([1, 3]))
    t = Wire(Int([3, 1]))
    tt = Wire(Int([1, 3]))
    init = [Term(LIA.Int(torch.tensor([[1, 2, 3]])), [X(v)])]
    update = [
        Term(LIA.Transpose(), [t], [v]),
        Term(LIA.Transpose(), [tt], [t]),
        Term(LIA.Id(), [X(v)], [tt]),
    ]
    return Module.sequential([v], init, update)
