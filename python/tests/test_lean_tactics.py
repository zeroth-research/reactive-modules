"""Tests for shape-directed certificate tactic generation (`zrth.lean.tactics`).

The plan decides what the three proof obligations are allowed to try. Getting
it wrong is expensive in both directions — a missing prover loses a proof that
used to go through, a superfluous one costs seconds per goal — so the
decisions are pinned here rather than only observed end to end in the slow
Lean tests.
"""

import torch
import pytest

from zrth import Module, Term, Wire, Var, X, LIA, LRA, BV, Int, Real, Bool, BitVec
from zrth.lean.common import LeanContext
from zrth.lean.tactics import is_nonlinear, plan_for


# ── the nonlinearity detector ────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        # A weight times a state element is affine — every layer of a net
        # looks like this, and none of them needs nlinarith.
        ("(2 * (s 0 0))", False),
        ("(2 * (s 0 0)) + (3 * (s 1 0))", False),
        ("(-1 * s0)", False),
        ("(* 2.0 (ite (>= x 0) x 0))", False),
        ("fun s => ((s 0 0) * 4)", False),
        # Two state-dependent operands are not.
        ("(s0 * s0)", True),
        ("((s 0 0) * (s 0 0))", True),
        # An application chain is one operand, not a token ending at a space.
        ("(s.1 0 0 * s.2 0 0)", True),
        ("fun s => (s 0 0 * s 0 0)", True),
    ],
)
def test_is_nonlinear(text, expected):
    assert is_nonlinear(text) is expected


# ── plans ────────────────────────────────────────────────────────────────


def _int_module() -> Module:
    """1x1 integer countdown."""
    x = Var(Int([1, 1]))
    one, hundred = Wire(Int([1, 1])), Wire(Int([1, 1]))
    zero = Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    dec = Wire(Int([1, 1]))
    return Module.sequential(
        [x],
        [Term(LIA.Int(torch.tensor([[100]])), [X(x)])],
        [
            Term(LIA.Int(torch.tensor([[0]])), [zero]),
            Term(LIA.Int(torch.tensor([[1]])), [one]),
            Term(LIA.Int(torch.tensor([[100]])), [hundred]),
            Term(LIA.Eq(), [cond], [x, zero]),
            Term(LIA.Sub(), [dec], [x, one]),
            Term(LIA.Ite(), [X(x)], [cond, hundred, dec]),
        ],
    )


def _real_module() -> Module:
    x = Var(Real([1, 1]))
    one = Wire(Real([1, 1]))
    return Module.sequential(
        [x],
        [Term(LRA.Real(torch.tensor([[5.0]])), [X(x)])],
        [
            Term(LRA.Real(torch.tensor([[1.0]])), [one]),
            Term(LRA.Sub(), [X(x)], [x, one]),
        ],
    )


def _bv_module() -> Module:
    b = Var(BitVec(1, [1, 1]))
    nb = Wire(BitVec(1, [1, 1]))
    return Module.sequential(
        [b],
        [Term(BV.Const(torch.tensor([[0]])), [X(b)])],
        [Term(BV.Not(), [nb], [b]), Term(BV.Id(), [X(b)], [nb])],
    )


def _plan(module, pred_text):
    return plan_for(LeanContext(module), pred_text)


def test_integer_plan_omits_real_and_finite_provers():
    """An integer module pays for neither `decide` nor `nlinarith`.

    `decide` cannot enumerate an unbounded state and `nlinarith` has nothing
    to do on a linear goal; both used to run on every goal of every module.
    """
    plan = _plan(_int_module(), "fun s => ((s 0 0) ≥ 0 ∧ (s 0 0) ≤ 100)")
    assert "omega" in plan.closers
    assert "decide" not in plan.closers
    assert "nlinarith" not in plan.closers
    assert not any("bv_decide" in c for c in plan.closers)


def test_linarith_is_always_available():
    """Gating linarith on Real cost two integer certificates that used it."""
    for module, text in (
        (_int_module(), "fun s => ((s 0 0) = 0)"),
        (_real_module(), "fun s => ((s 0 0) = 0)"),
    ):
        assert "linarith" in _plan(module, text).closers


def test_nonlinear_predicate_pulls_in_nlinarith():
    plan = _plan(_int_module(), "fun s => (((s 0 0) * (s 0 0) : Int)).toNat")
    assert "nlinarith" in plan.closers
    # and the conjunction branch gets it too: an `Int.toNat` comparison
    # unfolds to `a < b ∧ 0 < b`.
    assert any("constructor" in c and "nlinarith" in c for c in plan.closers)


def test_real_state_substitutes_equalities_in_prep():
    """Over the reals the equalities have to go into the goal *before* the
    closers run; omega would have done it for integers."""
    real = _plan(_real_module(), "fun s => ((s 0 0) = 2)")
    assert "simp_all" in real.prep
    integer = _plan(_int_module(), "fun s => ((s 0 0) = 2)")
    assert "simp_all" not in integer.prep


def test_disequality_split_follows_norm_num():
    """`norm_num at *` renormalises `≠` back to `¬ =`, so the split that
    turns it into a usable bound has to come after it."""
    plan = _plan(_int_module(), "fun s => ((s 0 0) = 0)")
    steps = plan.prep
    assert "norm_num at *" in steps
    split = "simp only [← ne_eq, ne_iff_lt_or_gt] at *"
    assert split in steps
    assert steps.index(split) > steps.index("norm_num at *")


def test_finite_state_gets_decide_last():
    """`decide` earns its place on a BitVec state, but it is the most
    expensive thing in the plan, so nothing may follow it."""
    plan = _plan(_bv_module(), "fun s => ((s 0 0) = 1#1)")
    assert plan.closers[-1] == "decide"


def test_flat_predicates_skip_the_splitting_steps():
    """Nothing to split, nothing to case on."""
    x = Var(Int([1, 1]))
    flat = Module.sequential(
        [x],
        [Term(LIA.Int(torch.tensor([[0]])), [X(x)])],
        [Term(LIA.Id(), [X(x)], [x])],
    )
    plan = _plan(flat, "fun s => True")
    assert "split_ifs" not in plan.prep
    assert not any("casesm" in p for p in plan.prep)


def test_budgets_scale_with_the_module():
    """`maxRecDepth` tracks the term count; the heartbeat budget stays low
    unless something in the plan is known to be slow, so failures are fast."""
    small = _plan(_int_module(), "fun s => ((s 0 0) = 0)")
    assert small.max_heartbeats == 400000

    x = Var(Int([64, 1]))
    wide = Module.sequential(
        [x],
        [Term(LIA.Int(torch.zeros((64, 1), dtype=torch.int64)), [X(x)])],
        [Term(LIA.Id(), [X(x)], [x])],
    )
    plan = _plan(wide, "fun s => True")
    assert plan.max_heartbeats > small.max_heartbeats
    assert plan.max_rec_depth >= 4096
