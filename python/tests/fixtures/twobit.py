"""Two-bit counter reactive module fixture (BV encoding).

BV<1> state: b0, b1 are bits of a 2-bit counter, enable is external input.
Property: b0 = 0 ∧ b1 = 0 holds infinitely often (counter visits 00).
"""
import torch
from zrth import Wire, Term, Module, Sort as dt, BV


def module() -> Module:
    b0 = (Wire(dt.BitVec(1, [1, 1])), Wire(dt.BitVec(1, [1, 1])))
    b1 = (Wire(dt.BitVec(1, [1, 1])), Wire(dt.BitVec(1, [1, 1])))
    enable = (Wire(dt.BitVec(1, [1, 1])), Wire(dt.BitVec(1, [1, 1])))

    not_b0 = Wire(dt.BitVec(1, [1, 1]))
    not_b1 = Wire(dt.BitVec(1, [1, 1]))
    b0_and_enable = Wire(dt.BitVec(1, [1, 1]))

    init = [
        Term(BV.Const(torch.tensor([[0]])), [b0[1]]),
        Term(BV.Const(torch.tensor([[0]])), [b1[1]]),
    ]
    update = [
        Term(BV.Not(), [not_b0], [b0[0]]),
        Term(BV.Ite(), [b0[1]], [enable[1], not_b0, b0[0]]),
        Term(BV.And(), [b0_and_enable], [b0[0], enable[1]]),
        Term(BV.Not(), [not_b1], [b1[0]]),
        Term(BV.Ite(), [b1[1]], [b0_and_enable, not_b1, b1[0]]),
    ]
    return Module.sequential(init, update, obs=[b0, b1, enable])
