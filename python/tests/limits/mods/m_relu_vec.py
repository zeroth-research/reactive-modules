"""LIA 3-vector with element-wise ReLU: v' = relu(v - 1), v0 = (3, 2, 1)."""
import torch
from zrth import Module, Term, Wire, Int, LIA, Var, X


def module() -> Module:
    v = Var(Int([3, 1]))
    ones = Wire(Int([3, 1]))
    diff = Wire(Int([3, 1]))
    init = [Term(LIA.Int(torch.tensor([[3], [2], [1]])), [X(v)])]
    update = [
        Term(LIA.Int(torch.ones((3, 1), dtype=torch.int64)), [ones]),
        Term(LIA.Sub(), [diff], [v, ones]),
        Term(LIA.ReLU(), [X(v)], [diff]),
    ]
    return Module.sequential([v], init, update)
