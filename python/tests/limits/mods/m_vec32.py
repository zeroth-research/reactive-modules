"""32-wide state, countdown in slot 0 — the scaling case."""
import torch
from zrth import Module, Term, Wire, Int, Bool, LIA, Var, X

N = 32


def module() -> Module:
    s = Var(Int([N, 1]))
    z11 = torch.zeros((1, 1), dtype=torch.int64)

    init_vec = torch.zeros((N, 1), dtype=torch.int64)
    init_vec[0][0] = 100
    init = [Term(LIA.Int(init_vec), [X(s)])]

    x, zc = Wire(Int([1, 1])), Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    reset, dec = Wire(Int([N, 1])), Wire(Int([N, 1]))

    row0 = torch.zeros((1, N), dtype=torch.int64)
    row0[0][0] = 1
    diag_keep = torch.eye(N, dtype=torch.int64)
    diag_keep[0][0] = 0
    b100 = torch.zeros((N, 1), dtype=torch.int64)
    b100[0][0] = 100
    bneg = torch.zeros((N, 1), dtype=torch.int64)
    bneg[0][0] = -1

    update = [
        Term(LIA.Linear(row0, z11), [x], [s]),
        Term(LIA.Int(torch.tensor([[0]])), [zc]),
        Term(LIA.Eq(), [cond], [x, zc]),
        Term(LIA.Linear(diag_keep, b100), [reset], [s]),
        Term(LIA.Linear(torch.eye(N, dtype=torch.int64), bneg), [dec], [s]),
        Term(LIA.Ite(), [X(s)], [cond, reset, dec]),
    ]
    return Module.sequential([s], init, update)
