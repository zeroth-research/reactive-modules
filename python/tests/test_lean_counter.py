"""Tests for the Module-to-Lean4 functional converter."""

import torch
from zrth import Wire, Term, Module, DType as dt, IType as it, Bool, Int
from zrth.lean.project import (
    create_project,
)

from zrth.expr import Expr, Bool as BoolConst

from os.path import dirname
from pathlib import Path


from .test_lean_diagram import _make_matrix_module


# def _make_P(ctrl: list[tuple[Wire, Wire]]) -> list[Term]:
#     w_out = Wire(Bool(1))
#     t_false = Wire(Bool(1))
#     t1 = Term(it.Tensor(torch.tensor(False)), write=[t_false])
#     # need to use ctrl'
#     t2 = Term(it.Eq(), read=[t_false, ctrl[0][1]], write=[w_out])
#     return [t1, t2]
#

# def _make_ranking(ctrl: list[tuple[Wire, Wire]]) -> list[Term]:
#     terms = []
#     w_out = Wire(Int(1))
#     c2 = Wire(Int(1))
#     c3 = Wire(Int(1))
#     terms.append(Term(it.Tensor(torch.tensor(3)), write=[c3]))
#     terms.append(Term(it.Tensor(torch.tensor(2)), write=[c2]))
#
#     tmp1 = Wire(Int(1))
#     tmp2 = Wire(Int(1))
#     terms.append(
#         Term(
#             it.Mul(),
#             read=[ctrl[1][1], c2],
#             write=[tmp1],
#         )
#     )
#     terms.append(
#         Term(
#             it.Add(),
#             read=[tmp1, ctrl[0][1]],
#             write=[tmp2],
#         )
#     )
#     terms.append(
#         Term(
#             it.Sub(),
#             read=[tmp2, c3],
#             write=[w_out],
#         )
#     )
#     return terms
#


def _make_counter():
    """Counter module using matrix operations.

    Python semantics:
        def init(y0, z0):
            return (0, y0, z0)        # = [[0,0],[1,0],[0,1]] @ [[y0],[z0]]

        def update(x, y, z):
            if x < y or x < z:
                return (x+1, y, z)    # = state + [[1],[0],[0]]
            else:
                return (0, y, z)      # = [[0,0,0],[0,1,0],[0,0,1]] @ state

    ctrl = (x, y, z) as Mat Int 3 1
    extl = (y0, z0) as Mat Int 2 1
    """
    # ctrl: state vector (x, y, z) as 3×1 matrix
    state = (Wire(Int(3, 1)), Wire(Int(3, 1)))
    # extl: external input (y0, z0) as 2×1 matrix
    extl = (Wire(Int(2, 1)), Wire(Int(2, 1)))

    # ── init: return [[0,0],[1,0],[0,1]] @ extl ──
    A_wire = Wire(Int(3, 2))
    init = [
        Term(it.Tensor(torch.tensor([[0, 0], [1, 0], [0, 1]])), write=[A_wire]),
        Term(it.MatMul(), write=[state[1]], read=[A_wire, extl[1]]),
    ]

    # ── update ──
    # Extract scalar components via row-vector multiplication
    # x = [1,0,0] @ state, y = [0,1,0] @ state, z = [0,0,1] @ state
    row_x_wire = Wire(Int(1, 3))
    row_y_wire = Wire(Int(1, 3))
    row_z_wire = Wire(Int(1, 3))
    x_wire = Wire(Int(1, 1))
    y_wire = Wire(Int(1, 1))
    z_wire = Wire(Int(1, 1))

    # Condition: x < y || x < z
    x_lt_y = Wire(Bool(1, 1))
    x_lt_z = Wire(Bool(1, 1))
    cond = Wire(Bool(1, 1))

    # Branch 1 (true): state + [[1],[0],[0]]
    e1_wire = Wire(Int(3, 1))
    result_true = Wire(Int(3, 1))

    # Branch 2 (false): [[0,0,0],[0,1,0],[0,0,1]] @ state
    B_wire = Wire(Int(3, 3))
    result_false = Wire(Int(3, 1))

    update = [
        # Extract x, y, z as scalars
        Term(it.Tensor(torch.tensor([[1, 0, 0]])), write=[row_x_wire]),
        Term(it.MatMul(), write=[x_wire], read=[row_x_wire, state[0]]),
        Term(it.Tensor(torch.tensor([[0, 1, 0]])), write=[row_y_wire]),
        Term(it.MatMul(), write=[y_wire], read=[row_y_wire, state[0]]),
        Term(it.Tensor(torch.tensor([[0, 0, 1]])), write=[row_z_wire]),
        Term(it.MatMul(), write=[z_wire], read=[row_z_wire, state[0]]),
        # Compute condition: x < y || x < z
        Term(it.Lt(), write=[x_lt_y], read=[x_wire, y_wire]),
        Term(it.Lt(), write=[x_lt_z], read=[x_wire, z_wire]),
        Term(it.Or(), write=[cond], read=[x_lt_y, x_lt_z]),
        # True branch: state + [1,0,0]
        Term(it.Tensor(torch.tensor([[1], [0], [0]])), write=[e1_wire]),
        Term(it.Add(), write=[result_true], read=[state[0], e1_wire]),
        # False branch: [[0,0,0],[0,1,0],[0,0,1]] @ state
        Term(
            it.Tensor(torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]])),
            write=[B_wire],
        ),
        Term(it.MatMul(), write=[result_false], read=[B_wire, state[0]]),
        # Select branch
        Term(it.Ite(), write=[state[1]], read=[cond, result_true, result_false]),
    ]

    return Module.sequential(init, update, obs=[state, extl])


def test_inf_counter_generates_lean():
    m = _make_matrix_module()

    _ = create_project(
        output_dir=Path(dirname(__file__)) / "LeanTests",
        module=m,
        project_name="CounterInfCertificate",
        executable=True,
        # p_terms=_make_P(m.ctrl),
        # ranking_terms=_make_ranking(m.ctrl),
    )


def test_counter_generates_lean():
    m = _make_counter()

    _ = create_project(
        output_dir=Path(dirname(__file__)) / "LeanTests",
        module=m,
        project_name="Counter",
        executable=True,
    )
