"""Two-bit counter reactive module fixture.

Boolean state: b0, b1 are bits of a 2-bit counter, enable is external input.
Property: b0 = False ∧ b1 = False holds infinitely often (counter visits 00).
"""
import torch
from zrth import Wire, Term, Module, Bool, LIA, Var, X


def module() -> Module:
    b0 = Var(Bool([1, 1]))
    b1 = Var(Bool([1, 1]))
    enable = Var(Bool([1, 1]))

    not_b0 = Wire(Bool([1, 1]))
    not_b1 = Wire(Bool([1, 1]))
    b0_and_enable = Wire(Bool([1, 1]))

    init = [
        Term(LIA.Bool(torch.tensor([[False]])), [X(b0)]),
        Term(LIA.Bool(torch.tensor([[False]])), [X(b1)]),
    ]
    update = [
        Term(LIA.Not(), [not_b0], [b0]),
        Term(LIA.Ite(), [X(b0)], [X(enable), not_b0, b0]),
        Term(LIA.And(), [b0_and_enable], [b0, X(enable)]),
        Term(LIA.Not(), [not_b1], [b1]),
        Term(LIA.Ite(), [X(b1)], [b0_and_enable, not_b1, b1]),
    ]
    return Module.sequential([b0, b1, enable], init, update)
