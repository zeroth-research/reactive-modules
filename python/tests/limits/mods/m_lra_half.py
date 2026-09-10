"""LRA: x steps down by 0.5 from 3.0, resetting at 0 — a non-integral Real state."""
import torch
from zrth import Module, Term, Wire, Real, Bool, LRA, Var, X


def module() -> Module:
    x = Var(Real([1, 1]))
    zero = Wire(Real([1, 1]))
    half = Wire(Real([1, 1]))
    three = Wire(Real([1, 1]))
    pos = Wire(Bool([1, 1]))
    dec = Wire(Real([1, 1]))
    init = [Term(LRA.Real(torch.tensor([[3.0]])), [X(x)])]
    update = [
        Term(LRA.Real(torch.tensor([[0.0]])), [zero]),
        Term(LRA.Real(torch.tensor([[0.5]])), [half]),
        Term(LRA.Real(torch.tensor([[3.0]])), [three]),
        Term(LRA.Gt(), [pos], [x, zero]),
        Term(LRA.Sub(), [dec], [x, half]),
        Term(LRA.Ite(), [X(x)], [pos, dec, three]),
    ]
    return Module.sequential([x], init, update)
