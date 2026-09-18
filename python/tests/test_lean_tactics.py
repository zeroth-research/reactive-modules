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
from zrth.lean.tactics import features_for, is_nonlinear, plan_for


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
    turns it into a usable bound has to come after it.

    Only a Real state runs `norm_num` in prep at all now, so that is where
    the ordering can be observed.
    """
    steps = _plan(_real_module(), "fun s => ((s 0 0) = 0)").prep
    assert "norm_num at *" in steps
    split = "simp only [← ne_eq, ne_iff_lt_or_gt] at *"
    assert split in steps
    assert steps.index(split) > steps.index("norm_num at *")


def test_finite_state_gets_decide_last():
    """`decide` earns its place on a BitVec state, but it is the most
    expensive thing in the plan, so nothing may follow it."""
    plan = _plan(_bv_module(), "fun s => ((s 0 0) = 1#1)")
    assert plan.closers[-1] == "decide"


def test_finite_narrow_state_is_enumerated():
    """A two-valued state element gets split before the closers run.

    `decide` is in the plan for a finite state but can never fire while the
    state is a free variable, and a branch that is contradictory only because
    `BitVec 1` has two inhabitants (`x ≠ 0#1` and `x ≠ 1#1`) reduces to a bare
    `False` that nothing else discharges.
    """
    plan = _plan(_bv_module(), "fun s => ((s 0 0) = 1#1)")
    tac = plan.state_case_tactic
    assert "BitVec.eq_zero_or_eq_one" in tac
    assert "($v)" in tac, "the enumeration must take the obligation's binder"
    assert "simp only [h0" in tac, "the two values have to reach the hypotheses"


def test_unbounded_state_is_not_enumerated():
    """Nothing to enumerate over Int or Real, so the step is a no-op."""
    assert _plan(_int_module(), "fun s => True").state_case_tactic == "skip"
    assert _plan(_real_module(), "fun s => True").state_case_tactic == "skip"


def test_wide_finite_state_is_not_enumerated():
    """Each element doubles the fan-out, so a wide state opts out."""
    from zrth.lean.tactics import MAX_ENUMERABLE_SLOTS

    b = Var(BitVec(1, [MAX_ENUMERABLE_SLOTS + 1, 1]))
    wide = Module.sequential(
        [b],
        [Term(BV.Const(torch.zeros((MAX_ENUMERABLE_SLOTS + 1, 1), dtype=torch.int64)),
              [X(b)])],
        [Term(BV.Id(), [X(b)], [b])],
    )
    assert _plan(wide, "fun s => True").state_case_tactic == "skip"


def test_wide_bitvec_is_not_enumerated():
    """`BitVec.eq_zero_or_eq_one` is width-1 only; wider has 2^w values."""
    b = Var(BitVec(8, [1, 1]))
    m = Module.sequential(
        [b],
        [Term(BV.Const(torch.tensor([[0]])), [X(b)])],
        [Term(BV.Id(), [X(b)], [b])],
    )
    assert _plan(m, "fun s => True").state_case_tactic == "skip"


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


def test_budget_scales_with_predicate_branch_points():
    """A net's cost lives in its branch points, not in the module's size.

    A 12-unit ranking net over a one-wire countdown leaves `n_slots` and
    `n_terms` tiny, so the module-size rule alone left it on the base budget
    and it timed out rather than failing to be provable.
    """
    m = _int_module()
    flat = _plan(m, "fun s => ((s 0 0) ≥ 0)")
    assert flat.max_heartbeats == 400000

    mid = _plan(m, "fun s => " + " + ".join(f"(max ((s 0 0) - {k}) 0)" for k in range(20)))
    assert mid.max_heartbeats >= 2000000

    wide = _plan(m, "fun s => " + " + ".join(f"(max ((s 0 0) - {k}) 0)" for k in range(70)))
    assert wide.max_heartbeats > mid.max_heartbeats


def test_branch_points_count_ite_as_well_as_min_max():
    """Over the reals a ReLU stays an `ite`, and costs a `split_ifs` branch."""
    plan = _plan(
        _real_module(),
        "fun s => " + " + ".join(
            f"(if (s 0 0) ≥ {k} then (s 0 0) else 0)" for k in range(20)
        ),
    )
    assert plan.features.n_branch >= 20
    assert plan.max_heartbeats >= 2000000


def test_branchy_real_gets_a_higher_budget():
    """A ReLU over the reals cannot fold to `max`, so it keeps its branches.

    `linarith` has no min/max support, so a Real net still pays a `split_ifs`
    branch per unit. Measured, the base budget makes such a case *fail* in
    106 s where the higher one *succeeds* in 71 s.
    """
    branchy = _plan(_real_module(), "fun s => (if (s 0 0) ≥ 0 then (s 0 0) else 0) = 0")
    assert branchy.max_heartbeats >= 2000000
    flat = _plan(_real_module(), "fun s => ((s 0 0) ≥ 0)")
    assert flat.max_heartbeats == 400000


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


# ══════════════════════════════════════════════════════════════════════
# Features taken from the cvc5 terms rather than from the printed Lean
# ══════════════════════════════════════════════════════════════════════


def test_facts_override_the_text_scan_for_nonlinearity():
    """The regex over-reports by design; the term does not."""
    from zrth.lean.smt_query import PredicateFacts

    ctx = LeanContext(_int_module())
    linear = features_for(ctx, "fun s => ((s 0 0) * (s 0 0))")
    assert linear.nonlinear is True                       # text says yes
    facts = PredicateFacts(nonlinear=False)
    assert features_for(ctx, "fun s => ((s 0 0) * (s 0 0))", facts).nonlinear is False


def test_facts_never_lower_the_branch_count():
    """Terms cover only the SMT-derived fields, so text still counts too."""
    from zrth.lean.smt_query import PredicateFacts

    ctx = LeanContext(_int_module())
    text = "fun s => (max (s 0 0) 0) + (max (s 0 0) 1) + (max (s 0 0) 2)"
    f = features_for(ctx, text, PredicateFacts(n_branch=1))
    assert f.n_branch == 3


def test_facts_raise_the_branch_count_above_what_the_printer_shows():
    """Sharing shrinks the output; the goal the closers see is unchanged."""
    from zrth.lean.smt_query import PredicateFacts

    ctx = LeanContext(_int_module())
    f = features_for(ctx, "fun s => (max (s 0 0) 0)", PredicateFacts(n_branch=62))
    assert f.n_branch == 62


def test_without_facts_everything_falls_back_to_the_text():
    ctx = LeanContext(_int_module())
    f = features_for(ctx, "fun s => ((s 0 0) * (s 0 0)) + (max (s 0 0) 0)")
    assert f.nonlinear is True
    assert f.n_branch == 1
    assert f.n_conditions == 1


# ══════════════════════════════════════════════════════════════════════
# Solver-informed tactics
#
# The contract is that hints only ever *add*: with none, the plan is byte
# for byte the one that was emitted before cvc5 was asked anything.
# ══════════════════════════════════════════════════════════════════════


def _hinted(**fields):
    from zrth.lean.smt_query import SolverHints

    return plan_for(
        LeanContext(_int_module()),
        "fun s => (s 0 0)",
        hints=SolverHints(**fields),
    )


def test_without_hints_the_new_macros_are_inert():
    plan = plan_for(LeanContext(_int_module()), "fun s => (s 0 0)")
    assert plan.facts_tactic == "skip"
    assert plan.clear_tactic == "skip"


def test_a_settled_condition_becomes_a_guarded_have_and_a_rewrite():
    plan = _hinted(determined=[("(($v) 0 0) ≥ 0", True)])
    tac = plan.facts_tactic
    assert tac.startswith("try (have hf0 : (($v) 0 0) ≥ 0 := by ")
    assert "simp only [if_pos hf0]" in tac


def test_a_refuted_condition_is_negated_and_uses_if_neg():
    plan = _hinted(determined=[("(($v) 0 0) ≥ 0", False)])
    tac = plan.facts_tactic
    assert "have hf0 : ¬ ((($v) 0 0) ≥ 0)" in tac
    assert "if_neg hf0" in tac


def test_every_settled_condition_gets_its_own_guarded_step():
    """A `have` the tactics cannot reproduce must cost nothing."""
    plan = _hinted(determined=[("a", True), ("b", False), ("c", True)])
    assert plan.facts_tactic.count("try (have") == 3


def test_an_unused_precondition_is_cleared():
    assert _hinted(pre_unused=True).clear_tactic == "try clear hpre"


def test_product_hints_are_added_before_the_bare_nlinarith():
    """Naming the squares turns nlinarith's search into a check."""
    plan = plan_for(
        LeanContext(_int_module()),
        "fun s => ((s 0 0) * (s 0 0))",       # nonlinear, so nlinarith is in
        hints=_solver_hints(nlinarith_hints=["mul_self_nonneg ((s 0 0) - 1)"]),
    )
    closers = plan.closers
    assert "nlinarith [mul_self_nonneg ((s 0 0) - 1)]" in closers
    assert closers.index("nlinarith [mul_self_nonneg ((s 0 0) - 1)]") < closers.index(
        "nlinarith"
    )


def test_hints_never_remove_a_closer():
    """`first | a | b` costs nothing for `b`, so there is no case for it."""
    ctx = LeanContext(_int_module())
    text = "fun s => ((s 0 0) * (s 0 0))"
    plain = plan_for(ctx, text)
    hinted = plan_for(ctx, text, hints=_solver_hints(linear_proof=True))
    assert set(plain.closers) <= set(hinted.closers)


def test_the_reasons_are_recorded_in_the_generated_comment():
    plan = _hinted(determined=[("(($v) 0 0) ≥ 0", True)], pre_unused=True)
    why = plan.why()
    assert "cvc5: inv forces" in why
    assert "clearing it" in why


def _solver_hints(**fields):
    from zrth.lean.smt_query import SolverHints

    return SolverHints(**fields)


def test_several_settled_conditions_stay_on_one_line():
    """A `;`-separated tactic sequence in a quotation cannot be wrapped.

    Broken across lines, Lean rejects the second step with
    `unexpected token 'try'; expected ')'` -- and only from two conditions
    on, so a single-condition case builds happily and hides it.
    """
    tac = _hinted(determined=[("a", True), ("b", True), ("c", False)]).facts_tactic
    assert "\n" not in tac
    assert tac.count("try (have") == 3


def test_norm_num_is_only_for_real_states():
    """It is 47 s of NN2Deep5's 49 s of tactic execution, and omega does
    not need it: 75.9 s with, 7.0 s without, same proof."""
    int_plan = plan_for(LeanContext(_int_module()), "fun s => (s 0 0)")
    assert not any("norm_num" in p for p in int_plan.prep)
    # still available as a closer for a goal that really does need it
    assert any("norm_num" in c for c in int_plan.closers)


def test_a_real_state_keeps_norm_num_in_prep():
    """Dropping it there leaves two of NN2RealAllPos4's obligations open."""
    plan = plan_for(LeanContext(_real_module()), "fun s => ((s 0 0) ≥ 0)")
    assert "norm_num at *" in plan.prep


# ══════════════════════════════════════════════════════════════════════
# The plan is about the predicates the certificate carries
#
# Both of the plan's inputs are computed before `--infer` runs, which is
# before there is an invariant to compute them from. `main._replan` restates
# them afterwards; without it an inferred certificate is planned for
# predicates it does not have.
# ══════════════════════════════════════════════════════════════════════


class _Settings:
    """The two fields `_replan` reads, without building a whole CLI parse."""

    def __init__(self, smt_tactics="cvc5"):
        from zrth.lean.smt_query import SmtBudget

        self.smt_tactics = smt_tactics
        self.budget = SmtBudget()


def _countdown():
    from zrth.analyzer import convert_method

    s = Var(Int([1, 1]))

    def init():
        return 100

    def update(old_x):
        return 100 if old_x == 0 else old_x - 1

    return Module.sequential(
        [s],
        convert_method(init, {}, [X(s)], theory=LIA),
        convert_method(update, {"old_x": s}, [X(s)], theory=LIA),
    )


def _before_and_after(inv, ranking):
    """The plan inputs as the pipeline has them before a route runs, and as
    `_replan` leaves them once the route has answered."""
    pytest.importorskip("cvc5")
    from zrth.lean.cert import CertificateData, smt_predicates_to_lean
    from zrth.lean.main import _replan

    module = _countdown()
    prp = "(= s0 0)"
    # Before: the property is all there is, so this is what `predicate_facts`
    # had to read the shape of.
    project = smt_predicates_to_lean(CertificateData(prp=prp, kind="buchi"), module)
    before = project.facts

    inferred = CertificateData(prp=prp, kind="buchi", inv=inv, ranking=ranking)
    inferred.inv_smt, inferred.ranking_smt = inv, ranking
    _replan(_Settings(), module, inferred, project)
    return module, before, project


def test_replan_sees_the_nonlinearity_the_route_introduced():
    """`features_for` takes `nonlinear` from the facts *over* the text scan,
    so stale facts do not merely omit a closer -- they remove one the text
    had already found. An inferred `s0 * s0` ranking with the pre-inference
    facts loses `nlinarith` and `positivity` outright."""
    module, before, project = _before_and_after(
        "(and (>= s0 0) (<= s0 100))", "(* s0 s0)"
    )
    assert before.nonlinear is False, "the property alone is linear"
    assert project.facts.nonlinear is True, "the inferred ranking is not"

    from zrth.lean.project import lean_context_for

    ctx = lean_context_for(module, project)
    text = f"{project.prp}\n{project.inv}\n{project.ranking}"
    stale = plan_for(ctx, text, facts=before)
    fresh = plan_for(ctx, text, facts=project.facts)
    assert not any("nlinarith" in c for c in stale.closers)
    assert any("nlinarith" in c for c in fresh.closers)
    assert "positivity" in fresh.closers


def test_replan_asks_cvc5_about_the_invariant_that_now_exists():
    """`solver_hints` returns immediately on a `None` invariant, so before
    inference it answers nothing at all however branchy the certificate ends
    up. The condition below is one the invariant settles."""
    _module, _before, project = _before_and_after(
        "(and (>= s0 0) (<= s0 100))", "(ite (<= s0 200) s0 999)"
    )
    assert project.hints is not None
    assert [value for _text, value in project.hints.determined] == [True]
    assert "≤ 200" in project.hints.determined[0][0]


def test_replan_prints_the_settled_condition_unshared():
    """A `have` outside the definition cannot name a `let` inside it, so the
    predicates are reprinted expanded once a condition is settled -- and all
    of them, since a condition can come from the property."""
    _module, _before, project = _before_and_after(
        "(and (>= s0 0) (<= s0 100))", "(ite (<= s0 200) s0 999)"
    )
    assert project.hints.determined
    assert "let " not in (project.ranking or "")


def test_replan_refreshes_the_facts_without_smt_tactics():
    """Reading the terms' shape is free -- they are parsed either way -- so
    it does not wait for `--smt-tactics cvc5`. Only the queries do."""
    pytest.importorskip("cvc5")
    from zrth.lean.cert import CertificateData, smt_predicates_to_lean
    from zrth.lean.main import _replan

    module = _countdown()
    project = smt_predicates_to_lean(
        CertificateData(prp="(= s0 0)", kind="buchi"), module
    )
    inferred = CertificateData(prp="(= s0 0)", kind="buchi",
                               inv="(and (>= s0 0) (<= s0 100))", ranking="(* s0 s0)")
    inferred.inv_smt, inferred.ranking_smt = inferred.inv, inferred.ranking
    _replan(_Settings(smt_tactics="none"), module, inferred, project)
    assert project.facts.nonlinear is True
    assert project.hints is None, "the queries are what the flag gates"


def test_replan_leaves_the_plan_alone_when_the_route_answered_in_lean():
    """Nothing parses printed Lean back into a term, so a route with no SMT
    source leaves the plan exactly as it was rather than half-restated."""
    pytest.importorskip("cvc5")
    from zrth.lean.cert import CertificateData, smt_predicates_to_lean
    from zrth.lean.main import _replan

    module = _countdown()
    project = smt_predicates_to_lean(
        CertificateData(prp="(= s0 0)", kind="buchi"), module
    )
    before = project.facts
    # Lean, as `--infer ai` hands it back: no `inv_smt` beside it.
    inferred = CertificateData(prp="(= s0 0)", kind="buchi",
                               inv="fun s => (s 0 0) ≥ 0", ranking="fun s => 1")
    _replan(_Settings(), module, inferred, project)
    assert project.facts is before
    assert project.hints is None


# ── floored ranks ────────────────────────────────────────────────────────


def test_a_floored_rank_gets_the_bridge_to_the_real_under_it():
    """`linarith` reads `⌊x⌋` as an opaque `Int` atom.

    The hypotheses that would pin it are about the `Real` `x`, so nothing in
    the closer chain can cross: `0 < s 0 0 ⊢ 0 < 2 + ⌊s 0 0⌋` fails with the
    bracketing lemmas passed in, because half the facts are in the wrong
    sort. `cert_floors` states the bridge in `Int`, where `omega` can use it.
    """
    plan = _plan(_real_module(), "fun s => ((2 + ⌊(s 0 0)⌋ : Int)).toNat")
    assert plan.features.has_floor
    assert plan.features.floor_args == ("(s 0 0)",)
    body = plan.floor_tactic
    assert "Int.floor_nonneg" in body
    assert "($v)" in body, "the macro's own binder, not the printer's `s`"


def test_the_floor_sign_is_split_rather_than_assumed():
    """A `have` whose side condition needs a nested `by` cannot be recovered.

    A nested `by` that fails logs `unsolved goals` at its own position
    instead of raising, so the enclosing `try` does not catch it and the
    failure lands in the build. `by_cases` has no side condition to get
    wrong, and one branch of it is `skip`.
    """
    plan = _plan(_real_module(), "fun s => ((2 + ⌊(s 0 0)⌋ : Int)).toNat")
    assert "by_cases" in plan.floor_tactic
    assert ":= by" not in plan.floor_tactic


def test_a_scaled_floor_distributes_before_the_closers():
    """`norm_num` knows `⌊y - 1⌋ < ⌊y⌋` and not `⌊k*(x - c)⌋ < ⌊k*x⌋`.

    `m_lra_half` ranks by `⌊2*s⌋` and steps by `1/2`, so the scale has to be
    distributed for the two arguments to differ by a literal.
    """
    plan = _plan(_real_module(), "fun s => ((2 + ⌊((2 : Real) * (s 0 0))⌋ : Int)).toNat")
    assert "simp only [mul_sub]" in plan.prep
    assert plan.features.floor_args == ("((2 : Real) * (s 0 0))",)


def test_an_integer_module_pays_nothing_for_floors():
    plan = _plan(_int_module(), "fun s => ((s 0 0) = 2)")
    assert not plan.features.has_floor
    assert plan.floor_tactic == "skip"
    assert "simp only [mul_sub]" not in plan.prep
