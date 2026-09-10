"""Tests for the SMT -> Lean expression translator (`zrth.lean.smt_to_lean`).

Focused on the `min`/`max` peephole. There is no `max` kind to translate
from, so a ReLU unit reaches the translator as `(ite (>= e 0) e 0)`. Left as
an `ite` each unit costs the certificate a `split_ifs` branch, and `hrank`
mentions the ranking twice, so a k-unit net fans one goal out into 2^(2k) --
which is what made a wide net exhaust the heartbeat budget rather than fail
to be provable. `omega` handles `min`/`max` over `Int` with no split at all.
"""

import pytest
import torch

from zrth import Module, Term, Var, X, LIA, LRA, Int, Real
from zrth.lean.cert import CertificateData, smt_predicates_to_lean


def _int_module() -> Module:
    """One 1x1 integer wire; enough to give `s0` a sort."""
    x = Var(Int([1, 1]))
    return Module.sequential(
        [x],
        [Term(LIA.Int(torch.tensor([[0]])), [X(x)])],
        [Term(LIA.Id(), [X(x)], [x])],
    )


def _real_module() -> Module:
    x = Var(Real([1, 1]))
    return Module.sequential(
        [x],
        [Term(LRA.Real(torch.tensor([[0.0]])), [X(x)])],
        [Term(LRA.Id(), [X(x)], [x])],
    )


def _inv(module, smt: str) -> str:
    return smt_predicates_to_lean(CertificateData(inv=smt), module).inv


@pytest.mark.parametrize(
    "smt,expected",
    [
        # a ReLU unit, in the four ways a comparison can be written
        ("(ite (>= s0 0) s0 0)", "max"),
        ("(ite (> s0 0) s0 0)", "max"),
        ("(ite (<= s0 0) 0 s0)", "max"),
        ("(ite (< s0 0) 0 s0)", "max"),
        # branches swapped relative to the comparison: that is a min
        ("(ite (>= s0 0) 0 s0)", "min"),
        ("(ite (<= s0 0) s0 0)", "min"),
        # not just against zero
        ("(ite (>= s0 5) s0 5)", "max"),
    ],
)
def test_comparison_ite_becomes_min_or_max(smt, expected):
    lean = _inv(_int_module(), f"(= {smt} 0)")
    assert expected in lean, lean
    assert "if " not in lean, f"still branching: {lean}"


@pytest.mark.parametrize(
    "smt",
    [
        # branches are not the comparison's operands
        "(ite (>= s0 0) 1 2)",
        # only one branch matches
        "(ite (>= s0 0) s0 1)",
        # the condition is not a comparison at all
        "(ite (= s0 0) s0 0)",
    ],
)
def test_other_ites_are_left_alone(smt):
    lean = _inv(_int_module(), f"(= {smt} 0)")
    assert "if " in lean, f"should still be an ite: {lean}"
    assert "max" not in lean and "min" not in lean


def test_real_keeps_the_ite():
    """`linarith`, which is what a Real goal gets, has no min/max support.

    Folding there would take away the `split_ifs` branch that is the only way
    through, so the peephole is Int-only.
    """
    lean = _inv(_real_module(), "(= (ite (>= s0 0.0) s0 0.0) 0.0)")
    assert "if " in lean, lean
    assert "max" not in lean


def test_a_whole_relu_layer_folds():
    """Every unit of a layer folds, so nothing is left for `split_ifs`."""
    units = " ".join(
        f"(* {w} (ite (>= (+ s0 (- {b})) 0) (+ s0 (- {b})) 0))"
        for w, b in ((3, 0), (2, 1), (1, 2))
    )
    lean = smt_predicates_to_lean(
        CertificateData(ranking=f"(+ {units})"), _int_module()
    ).ranking
    assert lean.count("max") == 3, lean
    assert "if " not in lean, lean
