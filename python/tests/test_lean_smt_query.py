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
    predicate_facts,
    solver_hints,
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


def test_pre_check_reports_without_raising_on_a_predicate_it_cannot_parse():
    lines: list[str] = []
    out = pre_check(countdown(), CertificateData(inv="(bogus"), SmtBudget(), log=lines.append)
    assert out == []
    # And it says *what* it could not read: the module encodes fine here, and
    # blaming it would send the reader after the wrong thing.
    assert any("predicates are not SMT-LIB" in ln for ln in lines)
    assert not any("could not encode this module" in ln for ln in lines)


# ── the inferred certificate ─────────────────────────────────────────


INFERRED = CertificateData(
    prp=GOOD.prp,
    # What `--infer ai-cegar` leaves behind: Lean in the fields the project
    # is generated from, the SMT-LIB cvc5 was given beside them.
    inv="fun s => 0 <= s.1 0 0 ∧ s.1 0 0 <= 100",
    ranking="fun s => (s.1 0 0).toNat",
    inv_smt=GOOD.inv,
    ranking_smt=GOOD.ranking,
)


def test_an_inferred_certificate_is_checked_through_its_smt_source():
    """The Lean rendering parses back as nothing; `inv_smt` is the way in."""
    q = ModuleQueries.build(countdown(), INFERRED)
    assert [v.status for v in q.check_obligations(SmtBudget())] == [Status.HOLDS] * 3


def test_an_inferred_certificate_that_is_wrong_is_refuted():
    cert = CertificateData(
        prp=GOOD.prp,
        inv=INFERRED.inv,
        ranking=INFERRED.ranking,
        inv_smt=GOOD.inv,
        ranking_smt="(- 100 s0)",
    )
    lines: list[str] = []
    pre_check(countdown(), cert, SmtBudget(), log=lines.append)
    assert "REFUTED" in "\n".join(lines)


def test_an_inferred_certificate_with_no_smt_source_says_so():
    """`--infer ai` hands back Lean only, and nothing can parse it back."""
    lines: list[str] = []
    out = pre_check(
        countdown(), CertificateData(prp=GOOD.prp, inv=INFERRED.inv), SmtBudget(),
        log=lines.append,
    )
    assert out == []
    assert any("predicates are not SMT-LIB" in ln for ln in lines)


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


# ══════════════════════════════════════════════════════════════════════
# Predicate shape read off the terms
# ══════════════════════════════════════════════════════════════════════


def _facts(**fields):
    return predicate_facts(ModuleQueries.build(countdown(), CertificateData(**fields)))


def _relu(e: str) -> str:
    return f"(ite (>= {e} 0) {e} 0)"


def _dense(layers: int) -> str:
    a, b = "s0", "s0"
    for _ in range(layers):
        a, b = _relu(f"(+ {a} {b})"), _relu(f"(+ {a} (- {b}))")
    return f"(+ {a} {b})"


@pytest.mark.parametrize(
    "ranking,nonlinear",
    [
        ("(* 3 s0)", False),        # an affine layer's weight
        ("(+ s0 s0)", False),
        ("(* s0 s0)", True),
        ("(* (+ s0 1) (- s0 1))", True),
    ],
)
def test_nonlinearity_is_read_off_the_multiplication(ranking, nonlinear):
    assert _facts(ranking=ranking).nonlinear is nonlinear


def test_branch_points_follow_the_min_max_fold():
    """A folded ReLU prints the condition's operands, not both branches.

    So `max e 0` emits `e` once where the `ite` would have emitted it
    twice, and the count has to follow that or a deep net reads as ten
    times more branchy than the goal it produces.
    """
    assert _facts(ranking=_dense(4)).n_branch == 30
    assert _facts(ranking=_dense(5)).n_branch == 62


def test_branch_points_do_not_move_when_the_printer_shares():
    """The whole point of counting on the term rather than on the output.

    `simp` is zeta-reducing, so the goal is the expanded one whether or not
    the printer bound the repeats to names -- and the heartbeat budget is
    calibrated against that goal.
    """
    from zrth.lean.smt_to_lean import smt_to_lean_nat

    q = ModuleQueries.build(countdown(), CertificateData(ranking=_dense(5)))
    unshared = smt_to_lean_nat(q.ranking, q.msmt.ctrl_next, share=False)
    shared = smt_to_lean_nat(q.ranking, q.msmt.ctrl_next, share=True)
    assert unshared.count("(max ") == 62
    assert shared.count("(max ") == 10          # the printer did move
    assert predicate_facts(q).n_branch == 62    # the count did not


def test_repeated_conditions_cost_one_split_between_them():
    """`split_ifs` splits per distinct condition, not per occurrence."""
    twice = "(+ (ite (>= s0 4) 1 2) (ite (>= s0 4) 3 7))"
    f = _facts(ranking=twice)
    assert f.n_branch == 2
    assert f.n_conditions == 1


def test_a_folded_relu_costs_no_split_at_all():
    f = _facts(ranking=_relu("s0"))
    assert f.n_branch == 1
    assert f.n_conditions == 0
    assert f.has_ite is False


def test_structure_comes_from_the_term_not_a_substring_search():
    f = _facts(inv="(and (>= s0 0) (or (<= s0 5) (= s0 9)))")
    assert (f.has_and, f.has_or, f.has_eq) == (True, True, True)


def test_facts_record_which_state_slots_are_mentioned():
    assert _facts(inv="(>= s0 0)").mentioned == frozenset({0})


def test_facts_are_absent_rather_than_wrong_when_cvc5_cannot_parse():
    assert predicate_facts(None) is None


# ══════════════════════════════════════════════════════════════════════
# Solver-informed tactics
# ══════════════════════════════════════════════════════════════════════


def _hints(cert, budget=None):
    q = ModuleQueries.build(countdown(), cert)
    return solver_hints(q, budget or SmtBudget())


def test_a_condition_the_invariant_settles_is_reported():
    """`s0 ≥ 0` is forced by the invariant, so `split_ifs` need not split."""
    h = _hints(
        CertificateData(prp=GOOD.prp, inv=GOOD.inv, ranking="(ite (>= s0 0) s0 1)")
    )
    assert [value for _, value in h.determined] == [True]
    assert "≥" in h.determined[0][0]


def test_a_condition_the_invariant_refutes_is_reported():
    h = _hints(
        CertificateData(prp=GOOD.prp, inv=GOOD.inv, ranking="(ite (< s0 0) 7 s0)")
    )
    assert [value for _, value in h.determined] == [False]


def test_an_undetermined_condition_is_left_to_split_ifs():
    h = _hints(
        CertificateData(prp=GOOD.prp, inv=GOOD.inv, ranking="(ite (>= s0 50) s0 1)")
    )
    assert h.determined == []


def test_conditions_are_rendered_against_the_macro_binder():
    """They are spliced into a tactic, so they read the obligation's state."""
    h = _hints(
        CertificateData(prp=GOOD.prp, inv=GOOD.inv, ranking="(ite (>= s0 0) s0 1)")
    )
    assert "($v)" in h.determined[0][0]


def test_a_folded_relu_is_not_offered_as_a_condition():
    """It becomes `max`, which never splits, so deciding it buys nothing."""
    h = _hints(
        CertificateData(prp=GOOD.prp, inv=GOOD.inv, ranking="(ite (>= s0 0) s0 0)")
    )
    assert h.determined == []


def test_a_precondition_no_refutation_needs_is_reported_as_unused():
    cert = CertificateData(
        prp=GOOD.prp, inv=GOOD.inv, ranking="s0", update_pre="(= 1 1)"
    )
    assert _hints(cert).pre_unused is True


def test_the_refutations_are_recorded_as_linear_when_they_are():
    assert _hints(GOOD).linear_proof is True


def test_no_invariant_means_no_hints_and_no_queries():
    h = _hints(CertificateData(prp="(= s0 0)"))
    assert not h.any


def test_a_spent_budget_yields_no_hints_rather_than_an_exception():
    h = _hints(
        CertificateData(prp=GOOD.prp, inv=GOOD.inv, ranking="(ite (>= s0 0) s0 1)"),
        SmtBudget(per_call_ms=5000, phase_ms=0),
    )
    assert h.determined == []
    assert h.linear_proof is None


def test_hints_survive_a_module_cvc5_cannot_encode():
    assert solver_hints(None, SmtBudget()).any is False


def test_the_hint_log_says_when_there_is_nothing_to_add():
    lines: list[str] = []
    solver_hints(
        ModuleQueries.build(countdown(), CertificateData(prp="(= s0 0)")),
        SmtBudget(),
        log=lines.append,
    )
    assert any("nothing cvc5 can add" in ln for ln in lines)


def test_translation_can_be_asked_not_to_share():
    """`cert_facts` states a condition expanded, so the definition must be.

    A `have` outside the definition cannot name a `let` bound inside it, so
    a shared `if (u3 ≥ 0) …` gives `simp only [if_pos hf0]` nothing to match
    and the branch is never collapsed.
    """
    from zrth.lean.cert import smt_predicates_to_lean

    cert = CertificateData(ranking=_dense(4))
    shared = smt_predicates_to_lean(cert, countdown(), share=True).ranking
    plain = smt_predicates_to_lean(cert, countdown(), share=False).ranking
    assert "let " in shared
    assert "let " not in plain
