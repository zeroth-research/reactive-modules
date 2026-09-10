"""Linear -> ReLU -> Linear over an 8-wide Int state.

`m_relu_net` scaled up: the module's own net is 8 units wide, while the
certificate only ever mentions slots 0 and 1. Isolates the cost of the
*module's* width from the cost of the certificate's.

s0' = relu(s0 - 1), s1' = relu(s1) + 1, si' = relu(si) elsewhere; init s0 = 6.
"""
import torch
from zrth import Module, Term, Wire, Int, LIA, Var, X

N = 8


def module() -> Module:
    s = Var(Int([N, 1]))
    h, hr = Wire(Int([N, 1])), Wire(Int([N, 1]))

    init_vec = torch.zeros((N, 1), dtype=torch.int64)
    init_vec[0][0] = 6
    init = [Term(LIA.Int(init_vec), [X(s)])]

    b1 = torch.zeros((N, 1), dtype=torch.int64)
    b1[0][0] = -1
    b2 = torch.zeros((N, 1), dtype=torch.int64)
    b2[1][0] = 1
    eye = torch.eye(N, dtype=torch.int64)

    update = [
        Term(LIA.Linear(eye, b1), [h], [s]),
        Term(LIA.ReLU(), [hr], [h]),
        Term(LIA.Linear(eye, b2), [X(s)], [hr]),
    ]
    return Module.sequential([s], init, update)
