"""Tests for the Module-to-Lean4 functional converter."""

import torch
from zrth import Wire, Term, Module, LIA as it, Bool, Int
from zrth.lean.project import (
    create_project,
)
from zrth.lean.cert import CertificateData

from os.path import dirname
from pathlib import Path


from .test_lean_diagram import _make_matrix_module


def _make_counter():
    """Counter module with a 3×1 vector state, transitions via LIA.Linear.

    Python semantics (state = (x, y, z) as Mat Int 3 1, extl = (y0, z0)):
        init(y0, z0) = (0, y0, z0)       = [[0,0],[1,0],[0,1]] · (y0, z0)
        update(x, y, z):
            if x < y or x < z: (x+1, y, z)   = I·s + (1,0,0)ᵀ
            else:              (0, y, z)      = [[0,0,0],[0,1,0],[0,0,1]] · s
    """
    state = (Wire(Int([3, 1])), Wire(Int([3, 1])))
    extl = (Wire(Int([2, 1])), Wire(Int([2, 1])))

    zero31 = torch.zeros((3, 1), dtype=torch.int64)
    zero11 = torch.zeros((1, 1), dtype=torch.int64)

    # init: state' = A · extl,  A = [[0,0],[1,0],[0,1]]
    A = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    init = [
        Term(it.Linear(A, zero31), [state[1]], [extl[1]]),
    ]

    # update
    x = Wire(Int([1, 1]))
    y = Wire(Int([1, 1]))
    z = Wire(Int([1, 1]))
    x_lt_y = Wire(Bool([1, 1]))
    x_lt_z = Wire(Bool([1, 1]))
    cond = Wire(Bool([1, 1]))
    result_true = Wire(Int([3, 1]))
    result_false = Wire(Int([3, 1]))

    # scalar extraction via row·state (a 1×3 constant times the 3×1 state)
    row_x = torch.tensor([[1, 0, 0]], dtype=torch.int64)
    row_y = torch.tensor([[0, 1, 0]], dtype=torch.int64)
    row_z = torch.tensor([[0, 0, 1]], dtype=torch.int64)
    e1 = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    diag_yz = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int64)

    update = [
        Term(it.Linear(row_x, zero11), [x], [state[0]]),
        Term(it.Linear(row_y, zero11), [y], [state[0]]),
        Term(it.Linear(row_z, zero11), [z], [state[0]]),
        # cond = x < y ∨ x < z
        Term(it.Lt(), [x_lt_y], [x, y]),
        Term(it.Lt(), [x_lt_z], [x, z]),
        Term(it.Or(), [cond], [x_lt_y, x_lt_z]),
        # true branch: state + (1,0,0)ᵀ  = I·state + e1
        Term(it.Linear(torch.eye(3, dtype=torch.int64), e1), [result_true], [state[0]]),
        # false branch: zero out x  = diag(0,1,1)·state
        Term(it.Linear(diag_yz, zero31), [result_false], [state[0]]),
        Term(it.Ite(), [state[1]], [cond, result_true, result_false]),
    ]

    return Module.sequential(init, update, obs=[state, extl])


def test_inf_counter_generates_lean():
    m = _make_matrix_module()

    _ = create_project(
        output_dir=Path(dirname(__file__)) / "LeanTests",
        module=m,
        project_name="CounterInfCertificate",
        executable=True,
    )


def test_counter_generates_lean():
    m = _make_counter()

    # Certificate predicates over the 3×1 vector state s (x=s[0][0], y=s[1][0],
    # z=s[2][0]) and the 2×1 external input e (y0=e[0][0], z0=e[1][0]), written
    # in the Python-expression DSL with the `v[i][j]` matrix-indexing sugar.
    _ = create_project(
        output_dir=Path(dirname(__file__)) / "LeanTests",
        module=m,
        project_name="Counter",
        executable=True,
        cert_data=CertificateData(
            prp="s[0][0] == 0",
            inv="And(s[0][0] >= 0, Or(s[0][0] <= s[1][0], s[0][0] <= s[2][0]))",
            init_pre="And(e[0][0] >= 0, e[1][0] >= 0)",
            ranking="Ite(s[0][0] == 0, 0, (Ite(s[1][0] >= s[2][0], s[1][0], s[2][0]) - s[0][0]) + 1)",
        ),
    )
