"""ctrl = [x : 1x1, v : 3x1] — wire indices and flat element slots disagree.

x accumulates v[0]; v counts (1,2,3) down on its head and resets at 0.
"""
import torch
from zrth import Module, Term, Wire, Int, Bool, LIA, Var, X


def module() -> Module:
    x = Var(Int([1, 1]))
    v = Var(Int([3, 1]))
    z11 = torch.zeros((1, 1), dtype=torch.int64)
    v0 = torch.tensor([[1], [2], [3]], dtype=torch.int64)

    init = [
        Term(LIA.Int(torch.zeros((1, 1), dtype=torch.int64)), [X(x)]),
        Term(LIA.Int(v0), [X(v)]),
    ]

    head, zc = Wire(Int([1, 1])), Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    reset, dec = Wire(Int([3, 1])), Wire(Int([3, 1]))
    row0 = torch.tensor([[1, 0, 0]], dtype=torch.int64)
    zero33 = torch.zeros((3, 3), dtype=torch.int64)
    bneg = torch.tensor([[-1], [0], [0]], dtype=torch.int64)
    update = [
        Term(LIA.Linear(row0, z11), [head], [v]),
        Term(LIA.Add(), [X(x)], [x, head]),
        Term(LIA.Int(torch.zeros((1, 1), dtype=torch.int64)), [zc]),
        Term(LIA.Eq(), [cond], [head, zc]),
        Term(LIA.Linear(zero33, v0), [reset], [v]),
        Term(LIA.Linear(torch.eye(3, dtype=torch.int64), bneg), [dec], [v]),
        Term(LIA.Ite(), [X(v)], [cond, reset, dec]),
    ]
    return Module.sequential([x, v], init, update)
