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


def _make_P(ctrl: list[tuple[Wire, Wire]]) -> list[Term]:
    """Property P: x = 0, i.e., [1,0,0] @ state == [[0]]."""
    s = ctrl[0][1]  # state' : Mat Int 3 1

    row_x = Wire(Int(1, 3))
    x = Wire(Int(1, 1))
    zero = Wire(Int(1, 1))
    out = Wire(Bool(1, 1))

    return [
        Term(it.Tensor(torch.tensor([[1, 0, 0]])), write=[row_x]),
        Term(it.MatMul(), write=[x], read=[row_x, s]),
        Term(it.Tensor(torch.tensor([[0]])), write=[zero]),
        Term(it.Eq(), write=[out], read=[x, zero]),
    ]


def _make_inv(ctrl: list[tuple[Wire, Wire]]) -> list[Term]:
    """Invariant: x >= 0 ∧ (x <= y ∨ x <= z)."""
    s = ctrl[0][1]  # state' : Mat Int 3 1

    # Extract x, y, z
    row_x = Wire(Int(1, 3))
    row_y = Wire(Int(1, 3))
    row_z = Wire(Int(1, 3))
    x = Wire(Int(1, 1))
    y = Wire(Int(1, 1))
    z = Wire(Int(1, 1))

    zero = Wire(Int(1, 1))
    x_ge_0 = Wire(Bool(1, 1))  # x >= 0
    x_le_y = Wire(Bool(1, 1))  # x <= y
    x_le_z = Wire(Bool(1, 1))  # x <= z
    disj = Wire(Bool(1, 1))  # x <= y ∨ x <= z
    out = Wire(Bool(1, 1))  # x >= 0 ∧ (x <= y ∨ x <= z)

    return [
        # Extract x, y, z
        Term(it.Tensor(torch.tensor([[1, 0, 0]])), write=[row_x]),
        Term(it.MatMul(), write=[x], read=[row_x, s]),
        Term(it.Tensor(torch.tensor([[0, 1, 0]])), write=[row_y]),
        Term(it.MatMul(), write=[y], read=[row_y, s]),
        Term(it.Tensor(torch.tensor([[0, 0, 1]])), write=[row_z]),
        Term(it.MatMul(), write=[z], read=[row_z, s]),
        # x >= 0
        Term(it.Tensor(torch.tensor([[0]])), write=[zero]),
        Term(it.Ge(), write=[x_ge_0], read=[x, zero]),
        # x <= y ∨ x <= z
        Term(it.Le(), write=[x_le_y], read=[x, y]),
        Term(it.Le(), write=[x_le_z], read=[x, z]),
        Term(it.Or(), write=[disj], read=[x_le_y, x_le_z]),
        # conjunction
        Term(it.And(), write=[out], read=[x_ge_0, disj]),
    ]


def _make_init_pre(extl: list[tuple[Wire, Wire]]) -> list[Term]:
    """init_pre: y0 >= 0 ∧ z0 >= 0 (external inputs are non-negative)."""
    e = extl[0][1]  # extl' : Mat Int 2 1

    # Extract y0, z0
    row_y0 = Wire(Int(1, 2))
    row_z0 = Wire(Int(1, 2))
    y0 = Wire(Int(1, 1))
    z0 = Wire(Int(1, 1))

    zero = Wire(Int(1, 1))
    y0_ge_0 = Wire(Bool(1, 1))
    z0_ge_0 = Wire(Bool(1, 1))
    out = Wire(Bool(1, 1))

    return [
        Term(it.Tensor(torch.tensor([[1, 0]])), write=[row_y0]),
        Term(it.MatMul(), write=[y0], read=[row_y0, e]),
        Term(it.Tensor(torch.tensor([[0, 1]])), write=[row_z0]),
        Term(it.MatMul(), write=[z0], read=[row_z0, e]),
        Term(it.Tensor(torch.tensor([[0]])), write=[zero]),
        Term(it.Ge(), write=[y0_ge_0], read=[y0, zero]),
        Term(it.Ge(), write=[z0_ge_0], read=[z0, zero]),
        Term(it.And(), write=[out], read=[y0_ge_0, z0_ge_0]),
    ]


def _make_ranking(ctrl: list[tuple[Wire, Wire]]) -> list[Term]:
    """Ranking function: if P(s) then 0 else max(s 1 0, s 2 0) - s 0 0.

    i.e., if x = 0 then 0 else max(y, z) - x.
    """
    s = ctrl[0][1]  # state' : Mat Int 3 1

    # Extract x, y, z
    row_x = Wire(Int(1, 3))
    row_y = Wire(Int(1, 3))
    row_z = Wire(Int(1, 3))
    x = Wire(Int(1, 1))
    y = Wire(Int(1, 1))
    z = Wire(Int(1, 1))

    # P(s): x == 0
    zero = Wire(Int(1, 1))
    p = Wire(Bool(1, 1))

    # max(y, z) = if y >= z then y else z
    y_ge_z = Wire(Bool(1, 1))
    max_yz = Wire(Int(1, 1))

    # max(y, z) - x + 1
    diff = Wire(Int(1, 1))
    one = Wire(Int(1, 1))
    diff_plus_1 = Wire(Int(1, 1))

    # if P then 0 else diff + 1
    ite_result = Wire(Int(1, 1))

    # Extract scalar from Mat Int 1 1 and convert to Nat
    scalar = Wire(Int(1))
    out = Wire(Int(1))

    return [
        # Extract x, y, z
        Term(it.Tensor(torch.tensor([[1, 0, 0]])), write=[row_x]),
        Term(it.MatMul(), write=[x], read=[row_x, s]),
        Term(it.Tensor(torch.tensor([[0, 1, 0]])), write=[row_y]),
        Term(it.MatMul(), write=[y], read=[row_y, s]),
        Term(it.Tensor(torch.tensor([[0, 0, 1]])), write=[row_z]),
        Term(it.MatMul(), write=[z], read=[row_z, s]),
        # P(s): x == 0
        Term(it.Tensor(torch.tensor([[0]])), write=[zero]),
        Term(it.Eq(), write=[p], read=[x, zero]),
        # max(y, z) = if y >= z then y else z
        Term(it.Ge(), write=[y_ge_z], read=[y, z]),
        Term(it.Ite(), write=[max_yz], read=[y_ge_z, y, z]),
        # max(y, z) - x + 1
        Term(it.Sub(), write=[diff], read=[max_yz, x]),
        Term(it.Tensor(torch.tensor([[1]])), write=[one]),
        Term(it.Add(), write=[diff_plus_1], read=[diff, one]),
        # if P then 0 else max(y,z) - x + 1
        Term(it.Ite(), write=[ite_result], read=[p, zero, diff_plus_1]),
        # Extract scalar and convert to Nat
        Term(it.TensorGet(), write=[scalar], read=[ite_result]),
        Term(it.ToUnsigned(), write=[out], read=[scalar]),
    ]


def test_counter_generates_lean():
    m = _make_counter()

    _ = create_project(
        output_dir=Path(dirname(__file__)) / "LeanTests",
        module=m,
        project_name="Counter",
        executable=True,
        p_terms=_make_P(m.ctrl),
        inv_terms=_make_inv(m.ctrl),
        init_pre_terms=_make_init_pre(m.extl),
        ranking_terms=_make_ranking(m.ctrl),
    )
