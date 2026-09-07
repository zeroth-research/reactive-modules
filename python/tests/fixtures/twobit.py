"""Two-bit counter reactive module fixture (BV encoding).

BV<1> state: b0, b1 are bits of a 2-bit counter, enable is external input.
Property: b0 = 0 ∧ b1 = 0 holds infinitely often (counter visits 00).
"""
import torch
from zrth import Wire, Term, Module, BitVec, BV, Var, X


def module() -> Module:
    b0 = Var(BitVec(1, [1, 1]))
    b1 = Var(BitVec(1, [1, 1]))
    enable = Var(BitVec(1, [1, 1]))

    not_b0 = Wire(BitVec(1, [1, 1]))
    not_b1 = Wire(BitVec(1, [1, 1]))
    b0_and_enable = Wire(BitVec(1, [1, 1]))

    init = [
        Term(BV.Const(torch.tensor([[0]])), [X(b0)]),
        Term(BV.Const(torch.tensor([[0]])), [X(b1)]),
    ]
    update = [
        Term(BV.Not(), [not_b0], [b0]),
        Term(BV.Ite(), [X(b0)], [X(enable), not_b0, b0]),
        Term(BV.And(), [b0_and_enable], [b0, X(enable)]),
        Term(BV.Not(), [not_b1], [b1]),
        Term(BV.Ite(), [X(b1)], [b0_and_enable, not_b1, b1]),
    ]
    return Module.sequential([b0, b1, enable], init, update)
