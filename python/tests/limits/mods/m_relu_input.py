"""ReLU plus an external input: x' = relu(x - e), x0 = 5, e supplied each step.

Sound only under a precondition `e >= 1`; without it the ranking does not
decrease. Used twice, with and without `--pre`.
"""
import torch
from zrth import Module, Term, Wire, Int, LIA, Var, X


def module() -> Module:
    x = Var(Int([1, 1]))
    e = Var(Int([1, 1]))
    diff = Wire(Int([1, 1]))
    init = [Term(LIA.Int(torch.tensor([[5]])), [X(x)])]
    update = [
        Term(LIA.Sub(), [diff], [x, X(e)]),
        Term(LIA.ReLU(), [X(x)], [diff]),
    ]
    return Module.sequential([x, e], init, update)
