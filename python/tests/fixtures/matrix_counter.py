"""Matrix counter reactive module fixture.

State: x ∈ Mat Int 3×1, external input u ∈ Mat Int 2×1.
Init:   x = [[0,0],[1,0],[0,1]] @ u
Update: x' = I @ x + [[1],[0],[0]]  (first component increments unboundedly)
"""
import torch
from zrth import Wire, Term, Module, DType as dt, IType as it


def module() -> Module:
    x = (Wire(dt.Int([3, 1])), Wire(dt.Int([3, 1])))
    u = (Wire(dt.Int([2, 1])), Wire(dt.Int([2, 1])))

    A_wire = Wire(dt.Int([3, 2]))
    init = [
        Term(it.Tensor(torch.tensor([[0, 0], [1, 0], [0, 1]])), [A_wire]),
        Term(it.MatMul(), [x[1]], [A_wire, u[1]]),
    ]

    B_wire = Wire(dt.Int([3, 3]))
    e1_wire = Wire(dt.Int([3, 1]))
    Bx_wire = Wire(dt.Int([3, 1]))
    update = [
        Term(it.Tensor(torch.eye(3, dtype=torch.int64)), [B_wire]),
        Term(it.MatMul(), [Bx_wire], [B_wire, x[0]]),
        Term(it.Tensor(torch.tensor([[1], [0], [0]])), [e1_wire]),
        Term(it.Add(), [x[1]], [Bx_wire, e1_wire]),
    ]
    return Module.sequential(init, update, obs=[x, u])
