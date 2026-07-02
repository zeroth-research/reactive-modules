"""Tests for the Module-to-Lean4 functional converter."""

import torch
import pytest
from zrth import Wire, Term, Module, Sort as dt, LIA, Bool, Int
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
    # P: b0' is false. main's LIA.Eq is Int-only, and Bool equality-with-false
    # is just negation, so encode `false = b0'` as ¬b0'.
    w_out = Wire(Bool([1, 1]))
    t = Term(LIA.Not(), read=[ctrl[0][1]], write=[w_out])
    return [t]


def _make_ranking(ctrl: list[tuple[Wire, Wire]]) -> list[Term]:
    # ranking = bit1 * 2 + bit0 - 3, fully in LIA. The bits are Bool, so each is
    # cast to Int with Ite(bit, 1, 0); multiply-by-2 is linear (bit1 + bit1).
    terms = []
    w_out = Wire(Int([1, 1]))
    one = Wire(Int([1, 1]))
    zero = Wire(Int([1, 1]))
    c3 = Wire(Int([1, 1]))
    terms.append(Term(LIA.ConstInt(torch.tensor([[1]])), write=[one]))
    terms.append(Term(LIA.ConstInt(torch.tensor([[0]])), write=[zero]))
    terms.append(Term(LIA.ConstInt(torch.tensor([[3]])), write=[c3]))

    bit1 = Wire(Int([1, 1]))
    bit0 = Wire(Int([1, 1]))
    terms.append(Term(LIA.Ite(), read=[ctrl[1][1], one, zero], write=[bit1]))
    terms.append(Term(LIA.Ite(), read=[ctrl[0][1], one, zero], write=[bit0]))

    tmp1 = Wire(Int([1, 1]))
    tmp2 = Wire(Int([1, 1]))
    terms.append(
        Term(
            LIA.Add(),
            read=[bit1, bit1],
            write=[tmp1],
        )
    )
    terms.append(
        Term(
            LIA.Add(),
            read=[tmp1, bit0],
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
