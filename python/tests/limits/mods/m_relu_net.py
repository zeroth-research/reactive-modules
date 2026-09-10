"""Linear -> ReLU -> Linear over Int: the shape a small Q-network compiles to.

state is a 2-vector; the step is W2 @ relu(W1 @ s + b1) + b2 with a countdown
embedded in the first component.
"""
import torch
from zrth import Module, Term, Wire, Int, LIA, Var, X


def module() -> Module:
    s = Var(Int([2, 1]))
    h = Wire(Int([2, 1]))
    hr = Wire(Int([2, 1]))

    init = [Term(LIA.Int(torch.tensor([[4], [0]])), [X(s)])]

    # W1 = I, b1 = (-1, 0): h = (s0 - 1, s1)
    W1 = torch.eye(2, dtype=torch.int64)
    b1 = torch.tensor([[-1], [0]], dtype=torch.int64)
    # W2 = I, b2 = (0, 1): out = (relu(s0 - 1), relu(s1) + 1)
    W2 = torch.eye(2, dtype=torch.int64)
    b2 = torch.tensor([[0], [1]], dtype=torch.int64)

    update = [
        Term(LIA.Linear(W1, b1), [h], [s]),
        Term(LIA.ReLU(), [hr], [h]),
        Term(LIA.Linear(W2, b2), [X(s)], [hr]),
    ]
    return Module.sequential([s], init, update)
