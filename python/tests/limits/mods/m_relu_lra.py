"""LRA (Real) 1x1 with ReLU: x' = relu(x - 1.0), x0 = 5.0."""
import torch
from zrth import Module, Term, Wire, Real, LRA, Var, X


def module() -> Module:
    x = Var(Real([1, 1]))
    one = Wire(Real([1, 1]))
    diff = Wire(Real([1, 1]))
    init = [Term(LRA.Real(torch.tensor([[5.0]])), [X(x)])]
    update = [
        Term(LRA.Real(torch.tensor([[1.0]])), [one]),
        Term(LRA.Sub(), [diff], [x, one]),
        Term(LRA.ReLU(), [X(x)], [diff]),
    ]
    return Module.sequential([x], init, update)
