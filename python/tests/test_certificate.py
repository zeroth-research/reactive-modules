"""The certificate types (`zrth.lean.certificate`) and the bridge to them."""

from dataclasses import FrozenInstanceError

import pytest

from zrth.lean.cert import CertificateData
from zrth.lean.certificate import Certificate, Claim, LeanAnswer


def test_the_types_are_values():
    """A route is handed these, and handing one back changed is `replace`."""
    for value, field in ((Claim(prp="(= s0 0)"), "prp"),
                         (Certificate(inv="true"), "inv"),
                         (LeanAnswer(inv="fun s => True"), "inv")):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, "x")


def test_the_kind_is_the_proof_rule():
    assert Claim().kind == "buchi" and not Claim().is_safety
    assert Claim(prp="(<= s0 9)", kind="safety").is_safety


def test_an_empty_certificate_has_neither_predicate():
    assert Certificate().empty
    assert Certificate(inv="", ranking="").empty
    assert not Certificate(inv="(>= s0 0)").empty
    assert not Certificate(ranking="s0").empty


def test_split_takes_the_smt_source_over_the_lean_printed_from_it():
    cd = CertificateData(
        prp="(= s0 0)", kind="buchi", init_pre="(>= e0 0)",
        update_pre="(>= e0 1)", inv="fun s => (s 0 0) ≥ 0",
        ranking="fun s => ((s 0 0)).toNat", inv_smt="(>= s0 0)",
        ranking_smt="s0",
    )
    claim, cert = cd.split()
    assert claim == Claim(prp="(= s0 0)", kind="buchi",
                          init_pre="(>= e0 0)", update_pre="(>= e0 1)")
    assert cert == Certificate(inv="(>= s0 0)", ranking="s0")

