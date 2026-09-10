"""LIA Min as a unary reduction: v = (x+1, 5), x' = min v — saturates at 5."""
import torch
from zrth import Module, Term, Wire, Int, LIA, Var, X


def module() -> Module:
    x = Var(Int([1, 1]))
    v = Wire(Int([2, 1]))
    lo = Wire(Int([1, 1]))

    init = [Term(LIA.Int(torch.tensor([[0]])), [X(x)])]
    A = torch.tensor([[1], [0]], dtype=torch.int64)
    b = torch.tensor([[1], [5]], dtype=torch.int64)
    update = [
        Term(LIA.Linear(A, b), [v], [x]),
        Term(LIA.Min(), [lo], [v]),
        Term(LIA.Id(), [X(x)], [lo]),
    ]
    return Module.sequential([x], init, update)
