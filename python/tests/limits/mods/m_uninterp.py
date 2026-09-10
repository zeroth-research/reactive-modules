"""LIA Uninterpreted as a source (one write, no read) — the only shape the
theory check accepts: `Uninterpreted` is a lone read *or* a lone write."""
import torch
from zrth import Module, Term, Wire, Int, LIA, Var, X


def module() -> Module:
    x = Var(Int([1, 1]))
    y = Wire(Int([1, 1]))
    init = [Term(LIA.Int(torch.tensor([[0]])), [X(x)])]
    update = [
        Term(LIA.Uninterpreted("f"), [y]),
        Term(LIA.Id(), [X(x)], [y]),
    ]
    return Module.sequential([x], init, update)
