"""Tests for the bounded cvc5 query layer (`zrth.lean.smt_query`).

The contract this file pins down is not "cvc5 gets the right answer" so much
as "nothing here can break generation": every query is bounded, and every
way a query can go wrong -- timeout, incompleteness, an unencodable module,
a cvc5 exception -- comes back as `UNKNOWN` rather than an exception.
"""

import pytest
import torch

from zrth import Module, Term, Var, Wire, X, LIA, Int
from zrth.analyzer import convert_method
from zrth.lean.cert import CertificateData
from zrth.lean.smt_query import (
    ModuleQueries,
    SmtBudget,
    Status,
    pre_check,
)


def countdown() -> Module:
    """`x = 100`, then `x-1` each step, back to 100 at zero."""

    def init():
        return 100

    def update(old_x):
        if old_x == 0:
            return 100
        return old_x - 1

    s = Var(Int([1, 1]))
    return Module.sequential(
        [s],
        convert_method(init, {}, [X(s)], theory=LIA),
        convert_method(update, {"old_x": s}, [X(s)], theory=LIA),
    )


GOOD = CertificateData(
    prp="(= s0 0)", inv="(and (>= s0 0) (<= s0 100))", ranking="s0"
)


def _status(cert, name):
    q = ModuleQueries.build(countdown(), cert)
    assert q is not None
    by_name = {v.name: v for v in q.check_obligations(SmtBudget())}
    return by_name[name]


# ── verdicts ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["init_inv", "step_inv", "hrank"])
def test_a_correct_certificate_holds_everywhere(name):
    assert _status(GOOD, name).status is Status.HOLDS


def test_a_ranking_that_increases_is_refuted_with_a_witness():
    v = _status(
        CertificateData(prp=GOOD.prp, inv=GOOD.inv, ranking="(- 100 s0)"), "hrank"
    )
    assert v.status is Status.REFUTED
    assert "s0 = 1" in v.detail


def test_a_non_inductive_invariant_is_refuted_at_the_step():
    cert = CertificateData(prp=GOOD.prp, inv="(and (>= s0 1) (<= s0 99))", ranking="s0")
    assert _status(cert, "step_inv").status is Status.REFUTED
    # ... and it is not even true at init, where the state is 100
    init = _status(cert, "init_inv")
    assert init.status is Status.REFUTED
    assert "s0 = 100" in init.detail


def test_obligations_needing_absent_predicates_are_not_checked():
    """No ranking and no property, so only the two invariant obligations."""
    q = ModuleQueries.build(countdown(), CertificateData(inv=GOOD.inv))
    assert [v.name for v in q.check_obligations(SmtBudget())] == [
        "init_inv",
        "step_inv",
    ]


def test_no_invariant_means_nothing_to_check():
    q = ModuleQueries.build(countdown(), CertificateData(prp="(= s0 0)"))
    assert q.check_obligations(SmtBudget()) == []


# ── the leash ────────────────────────────────────────────────────────


def test_a_spent_phase_budget_yields_unknown_not_an_exception():
    q = ModuleQueries.build(countdown(), GOOD)
    verdicts = q.check_obligations(SmtBudget(per_call_ms=5000, phase_ms=0))
    assert {v.status for v in verdicts} == {Status.UNKNOWN}
    assert all("budget" in v.detail for v in verdicts)


def test_the_per_call_limit_is_the_smaller_of_the_two():
    b = SmtBudget(per_call_ms=250, phase_ms=100000)
    assert b.next_call_ms() == 250
    b = SmtBudget(per_call_ms=100000, phase_ms=250)
    assert b.next_call_ms() <= 250


def test_a_phase_can_be_refilled():
    b = SmtBudget(per_call_ms=10, phase_ms=0)
    assert b.exhausted
    b.phase_ms = 5000
    b.start_phase()
    assert not b.exhausted


# ── failure is never fatal ───────────────────────────────────────────


def test_an_unparseable_predicate_gives_no_queries_rather_than_raising():
    assert ModuleQueries.build(countdown(), CertificateData(inv="(this is not smt")) is None


def test_an_unencodable_module_gives_no_queries_rather_than_raising():
    """`Uninterpreted` has no SMT translation; `build` must swallow that."""
    x = Var(Int([1, 1]))
    y = Wire(Int([1, 1]))
    module = Module.sequential(
        [x],
        [Term(LIA.Int(torch.tensor([[0]])), [X(x)])],
        # `Uninterpreted` is a lone write (a source) or a lone read.
        [Term(LIA.Uninterpreted("f"), [y]), Term(LIA.Id(), [X(x)], [y])],
    )
    q = ModuleQueries.build(module, GOOD)
    by_name = {v.name: v for v in q.check_obligations(SmtBudget())}
    # Encoding is lazy, so the failure lands on exactly the obligations that
    # touch the untranslatable transition -- `init` does not.
    assert by_name["step_inv"].status is Status.UNKNOWN
    assert "cannot encode" in by_name["step_inv"].detail
    assert by_name["init_inv"].status is not Status.UNKNOWN


def test_pre_check_reports_without_raising_on_a_module_it_cannot_encode():
    lines: list[str] = []
    out = pre_check(countdown(), CertificateData(inv="(bogus"), SmtBudget(), log=lines.append)
    assert out == []
    assert any("skipping" in ln for ln in lines)


def test_pre_check_names_the_refuted_obligations():
    lines: list[str] = []
    pre_check(
        countdown(),
        CertificateData(prp=GOOD.prp, inv=GOOD.inv, ranking="(- 100 s0)"),
        SmtBudget(),
        log=lines.append,
    )
    body = "\n".join(lines)
    assert "REFUTED" in body and "hrank" in body
    assert "counterexample: s0 = 1" in body
