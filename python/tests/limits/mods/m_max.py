"""LIA Max as a unary reduction: v = (x-1, 0), x' = max v — ReLU spelled Max."""
import torch
from zrth import Module, Term, Wire, Int, LIA, Var, X


def module() -> Module:
    x = Var(Int([1, 1]))
    v = Wire(Int([2, 1]))
    hi = Wire(Int([1, 1]))

    init = [Term(LIA.Int(torch.tensor([[5]])), [X(x)])]
    A = torch.tensor([[1], [0]], dtype=torch.int64)
    b = torch.tensor([[-1], [0]], dtype=torch.int64)
    update = [
        Term(LIA.Linear(A, b), [v], [x]),
        Term(LIA.Max(), [hi], [v]),
        Term(LIA.Id(), [X(x)], [hi]),
    ]
    return Module.sequential([x], init, update)
