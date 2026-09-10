"""Mixed Bool+Int state, built with explicit Terms.

`convert_method(..., theory=LIA)` cannot build this: any bool literal reaches
`LIATermBuilder.const` which emits the nonexistent `LIA.Const` (builder.py:432).
These Terms are what the fixed builder would produce (`LIA.Bool`).

b toggles between "counting up" and "counting down"; x walks 0..5..0.
"""
import torch
from zrth import Module, Term, Wire, Int, Bool, LIA, Var, X


def module() -> Module:
    b = Var(Bool([1, 1]))
    x = Var(Int([1, 1]))

    init = [
        Term(LIA.Bool(torch.tensor([[True]])), [X(b)]),
        Term(LIA.Int(torch.tensor([[0]])), [X(x)]),
    ]

    five, zero, one = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))
    lt5, gt0 = Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    up, dn = Wire(Int([1, 1])), Wire(Int([1, 1]))
    notb = Wire(Bool([1, 1]))
    x_if_b, b_if_b = Wire(Int([1, 1])), Wire(Bool([1, 1]))
    x_else, b_else = Wire(Int([1, 1])), Wire(Bool([1, 1]))

    update = [
        Term(LIA.Int(torch.tensor([[5]])), [five]),
        Term(LIA.Int(torch.tensor([[0]])), [zero]),
        Term(LIA.Int(torch.tensor([[1]])), [one]),
        Term(LIA.Lt(), [lt5], [x, five]),
        Term(LIA.Gt(), [gt0], [x, zero]),
        Term(LIA.Add(), [up], [x, one]),
        Term(LIA.Sub(), [dn], [x, one]),
        Term(LIA.Not(), [notb], [b]),
        # b: x' = x+1 while x<5, else flip b
        Term(LIA.Ite(), [x_if_b], [lt5, up, x]),
        Term(LIA.Ite(), [b_if_b], [lt5, b, notb]),
        # not b: x' = x-1 while x>0, else flip b
        Term(LIA.Ite(), [x_else], [gt0, dn, x]),
        Term(LIA.Ite(), [b_else], [gt0, b, notb]),
        Term(LIA.Ite(), [X(x)], [b, x_if_b, x_else]),
        Term(LIA.Ite(), [X(b)], [b, b_if_b, b_else]),
    ]
    return Module.sequential([b, x], init, update)
