"""Two-bit counter reactive module fixture.

Boolean state: b0, b1 are bits of a 2-bit counter, enable is external input.
Property: b0 = False ∧ b1 = False holds infinitely often (counter visits 00).
"""
import torch
from zrth import Wire, Term, Module, DType as dt, IType as it


def module() -> Module:
    b0 = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))
    b1 = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))
    enable = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))

    not_b0 = Wire(dt.Bool([1]))
    not_b1 = Wire(dt.Bool([1]))
    b0_and_enable = Wire(dt.Bool([1]))

    init = [
        Term(it.Tensor(torch.tensor([False])), [b0[1]]),
        Term(it.Tensor(torch.tensor([False])), [b1[1]]),
    ]
    update = [
        Term(it.Not(), [not_b0], [b0[0]]),
        Term(it.Ite(), [b0[1]], [enable[1], not_b0, b0[0]]),
        Term(it.And(), [b0_and_enable], [b0[0], enable[1]]),
        Term(it.Not(), [not_b1], [b1[0]]),
        Term(it.Ite(), [b1[1]], [b0_and_enable, not_b1, b1[0]]),
    ]
    return Module.sequential(init, update, obs=[b0, b1, enable])
