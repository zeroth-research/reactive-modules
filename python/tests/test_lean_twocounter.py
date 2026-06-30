"""Tests for the Module-to-Lean4 functional converter."""

import torch
import pytest
from zrth import Wire, Term, Module, Sort as dt, LIA, BV, Bool, Int
from zrth.lean.project import (
    create_project,
)
from zrth.lean.cert import CertificateData

from zrth.expr import Expr, Bool as BoolConst

from os.path import dirname
from pathlib import Path


def _make_twobitcounter():
    """Bool-only module: two-bit counter with enable."""

    b0 = (Wire(Bool([1, 1])), Wire(Bool([1, 1])))
    b1 = (Wire(Bool([1, 1])), Wire(Bool([1, 1])))
    enable = (Wire(Bool([1, 1])), Wire(Bool([1, 1])))

    not_b0 = Wire(Bool([1, 1]))
    not_b1 = Wire(Bool([1, 1]))
    b0_and_enable = Wire(Bool([1, 1]))

    init = [
        Term(LIA.ConstBool(torch.tensor([[False]])), [b0[1]]),
        Term(LIA.ConstBool(torch.tensor([[False]])), [b1[1]]),
    ]
    update = [
        Term(LIA.Not(), [not_b0], [b0[0]]),
        Term(LIA.Ite(), [b0[1]], [enable[1], not_b0, b0[0]]),
        Term(LIA.And(), [b0_and_enable], [b0[0], enable[1]]),
        Term(LIA.Not(), [not_b1], [b1[0]]),
        Term(LIA.Ite(), [b1[1]], [b0_and_enable, not_b1, b1[0]]),
    ]
    return Module.sequential(init, update, obs=[b0, b1, enable])


def _make_P(ctrl: list[tuple[Wire, Wire]]) -> list[Term]:
    w_out = Wire(Bool([1, 1]))
    t_false = Wire(Bool([1, 1]))
    t1 = Term(LIA.ConstBool(torch.tensor([[False]])), write=[t_false])
    # need to use ctrl'
    t2 = Term(LIA.Eq(), read=[t_false, ctrl[0][1]], write=[w_out])
    return [t1, t2]


def _make_ranking(ctrl: list[tuple[Wire, Wire]]) -> list[Term]:
    terms = []
    w_out = Wire(Int([1, 1]))
    c2 = Wire(Int([1, 1]))
    c3 = Wire(Int([1, 1]))
    terms.append(Term(LIA.ConstInt(torch.tensor([[3]])), write=[c3]))
    terms.append(Term(LIA.ConstInt(torch.tensor([[2]])), write=[c2]))

    tmp1 = Wire(Int([1, 1]))
    tmp2 = Wire(Int([1, 1]))
    terms.append(
        Term(
            BV.Mul(),
            read=[ctrl[1][1], c2],
            write=[tmp1],
        )
    )
    terms.append(
        Term(
            LIA.Add(),
            read=[tmp1, ctrl[0][1]],
            write=[tmp2],
        )
    )
    terms.append(
        Term(
            LIA.Sub(),
            read=[tmp2, c3],
            write=[w_out],
        )
    )
    return terms


@pytest.mark.skip(reason="ranking uses Mul, only in BV theory in main; needs BV re-encoding")
def test_twobitcounter_generates_lean():
    m = _make_twobitcounter()

    _ = create_project(
        output_dir=Path(dirname(__file__)) / "LeanTests",
        module=m,
        project_name="TwoBitCertificate",
        executable=True,
        cert_data=CertificateData(
            prp=_make_P(m.ctrl),
            ranking=_make_ranking(m.ctrl),
        ),
    )
