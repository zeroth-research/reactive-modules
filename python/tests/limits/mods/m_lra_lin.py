"""LRA (Real) countdown without ReLU: x' = x - 1.0 while x > 0, else 5.0."""
import torch
from zrth import Module, Term, Wire, Real, Bool, LRA, Var, X


def module() -> Module:
    x = Var(Real([1, 1]))
    zero = Wire(Real([1, 1]))
    one = Wire(Real([1, 1]))
    five = Wire(Real([1, 1]))
    pos = Wire(Bool([1, 1]))
    dec = Wire(Real([1, 1]))
    init = [Term(LRA.Real(torch.tensor([[5.0]])), [X(x)])]
    update = [
        Term(LRA.Real(torch.tensor([[0.0]])), [zero]),
        Term(LRA.Real(torch.tensor([[1.0]])), [one]),
        Term(LRA.Real(torch.tensor([[5.0]])), [five]),
        Term(LRA.Gt(), [pos], [x, zero]),
        Term(LRA.Sub(), [dec], [x, one]),
        Term(LRA.Ite(), [X(x)], [pos, dec, five]),
    ]
    return Module.sequential([x], init, update)
