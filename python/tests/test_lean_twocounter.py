"""Tests for the Module-to-Lean4 functional converter."""

import torch
from zrth import Wire, Term, Module, LIA, Bool, Var, X
from zrth.lean.project import (
    create_project,
)
from zrth.lean.proof.render import lean_certificate
from zrth.lean.core.certificate import Certificate, Claim

from os.path import dirname
from pathlib import Path


def _make_twobitcounter():
    """Bool-only module: two-bit counter with enable."""

    b0 = Var(Bool([1, 1]))
    b1 = Var(Bool([1, 1]))
    enable = Var(Bool([1, 1]))

    not_b0 = Wire(Bool([1, 1]))
    not_b1 = Wire(Bool([1, 1]))
    b0_and_enable = Wire(Bool([1, 1]))

    init = [
        Term(LIA.Bool(torch.tensor([[False]])), [X(b0)]),
        Term(LIA.Bool(torch.tensor([[False]])), [X(b1)]),
    ]
    update = [
        Term(LIA.Not(), [not_b0], [b0]),
        Term(LIA.Ite(), [X(b0)], [X(enable), not_b0, b0]),
        Term(LIA.And(), [b0_and_enable], [b0, X(enable)]),
        Term(LIA.Not(), [not_b1], [b1]),
        Term(LIA.Ite(), [X(b1)], [b0_and_enable, not_b1, b1]),
    ]
    return Module.sequential([b0, b1, enable], init, update)


def test_twobitcounter_generates_lean():
    m = _make_twobitcounter()

    _ = create_project(
        output_dir=Path(dirname(__file__)) / "LeanTests",
        module=m,
        project_name="TwoBitCertificate",
        executable=True,
        # P: b0 is false. ranking = 2*b1 + b0 - 3, each bit read as 0 or 1.
        lean_cert=lean_certificate(
            m,
            Claim(prp="(not s0)"),
            Certificate(ranking="(- (+ (ite s1 1 0) (ite s1 1 0) (ite s0 1 0)) 3)"),
        ),
    )
