"""`--infer vampire`: the certificate Vampire's answer literals derive.

Three layers, and only the last needs the prover.

**Branch splitting** is what makes the question answerable at all, and it is
pure term rewriting: the conditions an `ite` tests, the cases they generate,
and the guard that stands in for each. It runs everywhere.

**The question** is the other half of the same thing -- a cvc5 term with
holes in it, printed by cvc5 as SMT-LIB, and the two things it refuses to
state rather than ask Vampire something it reads and never answers. Also
everywhere.

**The derivations** need the binary and are skipped without one (`$VAMPIRE`,
or `vampire` on PATH). They are the route's whole claim: the coefficients
come back from Vampire, and the certificate built from them satisfies the
module's obligations when cvc5 restates them.

What is *not* here is a derivation for a module past this mechanism's
measured reach -- two holes, which is one component under `--safety`. The
ceiling is pinned by `test_the_reach_is_two_holes` rather than left for a
reader to discover from a timeout.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from zrth.lean.cert import CertificateData
from zrth.lean.common import Refused

LIMITS = Path(__file__).parent / "limits" / "mods"
FIXTURES = Path(__file__).parent / "fixtures"


def module_at(path: Path):
    spec = importlib.util.spec_from_file_location(f"_m_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.module()


def vampire_or_skip() -> str:
    from zrth.lean.houdini_solver import resolve_vampire

    try:
        return resolve_vampire(None)
    except Refused:
        pytest.skip("no Vampire binary: set $VAMPIRE or put `vampire` on PATH")


def parts(name: str, kind: str, prp: str, fixture: Path = LIMITS):
    """A module's context, obligations and branches, without asking anything."""
    from zrth.lean.magic.houdini import Obligations
    from zrth.lean.magic.vampire import Question, split
    from zrth.lean.smt_synth import SynthContext

    cd = CertificateData(prp=prp, kind=kind)
    ctx = SynthContext.build(module_at(fixture / f"{name}.py"), cd,
                             route="vampire",
                             takes=("int", "bool", "bv", "tuple"))
    ob = Obligations(ctx)
    entry = split(ctx, ob.init, "the initial state")
    step = split(ctx, ob.next, "the round")
    return ctx, ob, Question(ctx, ob, entry, step)


def derive(name: str, kind: str, prp: str, *, timeout: float = 30,
           fixture: Path = LIMITS, vampire: "str | None" = None):
    from zrth.lean.magic.vampire import TA2MagicVampire

    magic = TA2MagicVampire(module_at(fixture / f"{name}.py"),
                            vampire=vampire or vampire_or_skip(),
                            timeout=timeout, log=lambda *_: None)
    return magic.infer(CertificateData(prp=prp, kind=kind))


# ══════════════════════════════════════════════════════════════════════════
# Branch splitting
# ══════════════════════════════════════════════════════════════════════════


def test_the_round_is_split_on_the_conditions_its_ites_test():
    """`m_countdown` resets at zero, so its round is one `ite` and two cases,
    each with the condition as a guard rather than inside the term."""
    from zrth.lean.magic.vampire import ite_conditions, split

    ctx, ob, _q = parts("m_countdown", "safety", "(<= s0 100)")
    assert [str(c) for c in ite_conditions(ob.next)] == ["(= v_s0 0)"]

    cases = split(ctx, ob.next, "the round")
    assert len(cases) == 2
    folded = {str(b.guard[0]): str(b.state[0]) for b in cases}
    assert folded == {"(= v_s0 0)": "100", "(not (= v_s0 0))": "(+ (- 1) v_s0)"}
    # Which is the point: nothing left to branch on.
    assert not any(ite_conditions(b.state) for b in cases)


def test_the_initial_state_is_split_apart_from_the_round():
    """They are not quantified over the same things. `init` is a function of
    the inputs alone, so a guard taken from the round -- which tests the
    latched state -- would leave a variable in the entry obligation that
    nothing binds, and Vampire answers that with a type error."""
    ctx, ob, q = parts("m_countdown", "safety", "(<= s0 100)")
    assert len(q.step) == 2         # the round branches
    assert len(q.entry) == 1        # the initial state does not
    assert q.entry[0].guard == ()

    from zrth.lean.magic.vampire import templates

    script = q.conjecture(templates(ctx, ranked=False)[0], ctx.prp)
    # The entry obligation is the one conjunct with no `forall` over it, and
    # it mentions no state variable.
    entry = script[script.index("(and ") : script.index("(forall")]
    assert "s0" not in entry


def test_a_transition_with_too_many_conditions_is_refused_by_name():
    from zrth.lean.magic.vampire import _MAX_CONDITIONS

    assert _MAX_CONDITIONS >= 1
    ctx, ob, _q = parts("m_countdown", "safety", "(<= s0 100)")
    from zrth.lean.magic.vampire import split

    # The bound is on the *count*, so it is checked by lowering it rather
    # than by finding a module with thirty conditionals.
    import zrth.lean.magic.vampire as mv

    old = mv._MAX_CONDITIONS
    try:
        mv._MAX_CONDITIONS = 0
        with pytest.raises(Refused, match="past the .* this route will print"):
            split(ctx, ob.next, "the round")
    finally:
        mv._MAX_CONDITIONS = old


# ══════════════════════════════════════════════════════════════════════════
# The question cvc5 prints
# ══════════════════════════════════════════════════════════════════════════


def test_an_ite_that_survived_the_split_is_refused_rather_than_stated():
    """The one thing this route cannot state. SMT-LIB has `ite` and Vampire
    reads it, so stating it anyway is a question that comes back as a
    timeout rather than as the unsupported shape it is."""
    import cvc5
    from cvc5 import Kind

    from zrth.lean.magic.vampire import refuse_ites

    tm = cvc5.TermManager()
    x = tm.mkConst(tm.getIntegerSort(), "v_s0")
    ite = tm.mkTerm(Kind.ITE, tm.mkTerm(Kind.GT, x, tm.mkInteger(0)),
                    x, tm.mkInteger(0))
    with pytest.raises(Refused, match="still holds an `ite`"):
        refuse_ites(tm.mkTerm(Kind.LEQ, ite, tm.mkInteger(3)))
    refuse_ites(tm.mkTerm(Kind.LEQ, x, tm.mkInteger(3)))    # no ite, no news


def test_a_constant_no_quantifier_binds_is_refused_rather_than_printed():
    """cvc5 prints a stray constant perfectly happily, and then the question
    is over a free symbol nothing declared -- which is a different question
    from the one the module means. The old TPTP printer refused it for free
    by having no variable for it; this says it by name."""
    ctx, ob, q = parts("m_countdown", "safety", "(<= s0 100)")
    import cvc5

    stray = ctx.tm.mkConst(ctx.tm.getIntegerSort(), "elsewhere")
    body = ctx.tm.mkTerm(cvc5.Kind.LEQ, stray, ctx.tm.mkInteger(0))
    with pytest.raises(Refused, match="nothing there binds"):
        q._forall(q.state, body)


def test_the_conjecture_states_every_obligation_under_one_existential():
    ctx, _ob, q = parts("m_countdown", "safety", "(<= s0 100)")
    from zrth.lean.magic.vampire import templates

    tpl = templates(ctx, ranked=False)[0]
    script = q.conjecture(tpl, ctx.prp)

    assert script.startswith("(set-logic ALL)\n(assert-not "
                             "(exists ((A0 Int) (B0 Int)) ")
    assert script.rstrip().endswith("(check-sat)")
    assert "ite" not in script
    # One entry obligation, one step per branch, one for the property; the
    # entry one is over the inputs, which `m_countdown` has none of.
    assert script.count("(forall ((s0 Int))") == 3


def test_buchi_is_two_questions_of_two_holes_not_one_of_four():
    """Two holes is what the answer-literal search closes, so the invariant
    and the rank are asked apart -- the rank over an invariant that is
    already numbers rather than holes."""
    ctx, _ob, q = parts("m_countdown", "buchi", "(= s0 0)")
    from zrth.lean.magic.vampire import templates

    tpl = templates(ctx, ranked=True)[0]
    assert tpl.holes == ("A0", "B0") and tpl.rank == ("R0", "Rc")

    stage1 = q.invariant_conjecture(tpl)
    assert "(exists ((A0 Int) (B0 Int))" in stage1
    # No rank and no property: an inductive invariant is all it asks for.
    assert "R0" not in stage1 and "Rc" not in stage1

    stage2 = q.rank_conjecture(tpl, ctx.prp, {"A0": 0, "B0": 100})
    assert "(exists ((R0 Int) (Rc Int))" in stage2
    assert "A0" not in stage2 and "(<= 0 s0)" in stage2
    # `rank > 0` where the property fails and `rank' < rank`: ite-free, and
    # it implies the clamped `Int.toNat` form `rule_buchi` asks for.
    assert "(+ Rc (* R0 s0))" in stage2
    assert "(> " in stage2 and "(< " in stage2


def test_only_the_question_that_needs_a_window_gets_one():
    """An invariant asked for on its own is loose -- nothing pins the
    interval and the search is over the whole of `Int`. The safety question
    has `inv -> prp` doing that already, and measured, the window costs it
    `m_max` and `m_toward5`. So it is asked for where it is needed."""
    ctx, _ob, q = parts("m_countdown", "safety", "(<= s0 100)")
    from zrth.lean.magic.vampire import templates, window_for

    tpl = templates(ctx, ranked=False)[0]
    window = window_for(ctx)
    assert window >= 100                       # `m_countdown` counts to 100
    assert str(window) not in q.conjecture(tpl, ctx.prp)

    ctx, _ob, q = parts("m_countdown", "buchi", "(= s0 0)")
    tpl = templates(ctx, ranked=True)[0]
    stage1 = q.invariant_conjecture(tpl)
    # cvc5 hoists the repeated `(- 1010)` into a `let`, so the lower bounds
    # read off the binding rather than the numeral.
    assert f"(- {window})" in stage1
    for hole in ("A0", "B0"):
        assert f"(<= {hole} {window})" in stage1


def test_a_template_row_is_one_body_read_two_ways():
    """A row is needed as a cvc5 term in the question and as SMT-LIB in the
    certificate, so it holds which columns it reads rather than text one
    side prints and the other parses back."""
    from zrth.lean.magic.vampire import Row

    assert Row("A0", "B0", (0,), ("s0",)).smt == "s0"
    assert Row("C0_1", "D0_1", (0, 1), ("s0", "s1")).smt == "(- s0 s1)"
    # A column of a matrix-shaped component is written as the selector the
    # certificate carries, not as the component it is an element of.
    sel = "((_ tuple.select 2) s0)"
    assert Row("A2", "B2", (2,), (sel,)).smt == sel


def test_a_matrix_component_is_an_interval_per_element():
    """`m_relu_vec`'s state is one 3x1 wire, and the route refused it.

    A tuple has no order for `lo <= _ <= hi` to bound, so the template is
    built over the *columns* -- one interval per element, one rank
    coefficient per element -- and the obligation names an integer variable
    per element, never the component.

    What is pinned here is that the obligation says the right thing: the
    certificate this module actually has, instantiated into the template,
    is proved by cvc5. Whether Vampire *derives* those numbers is a
    separate matter and on this module it does not -- three ReLUs split the
    round into eight branches against six holes.
    """
    import cvc5
    from cvc5 import Kind

    from zrth.lean.magic.houdini import columns
    from zrth.lean.magic.vampire import template

    ctx, _ob, q = parts("m_relu_vec", "safety",
                        "(<= ((_ tuple.select 0) s0) 3)")
    assert [c.src for c in columns(ctx)] == [
        f"((_ tuple.select {k}) s0)" for k in range(3)]

    tpl = template(ctx, "intervals", ranked=False)
    obs = q._obligations(tpl, ctx.prp, inductive=True, implies=True,
                         ranked=False)
    body = q._and(obs)
    assert "tuple" not in str(body).lower(), "an obligation Vampire can read"

    # `v' = relu(v - 1)` from (3,2,1): every element stays within its start.
    want = {"A0": 0, "B0": 3, "A1": 0, "B1": 2, "A2": 0, "B2": 1}
    body = body.substitute([q._hole(h) for h in want],
                           [ctx.tm.mkInteger(v) for v in want.values()])
    solver = cvc5.Solver(ctx.tm)
    solver.setOption("tlimit", "20000")
    solver.assertFormula(ctx.tm.mkTerm(Kind.NOT, body))
    assert solver.checkSat().isUnsat()


# ══════════════════════════════════════════════════════════════════════════
# Reading an answer back
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(("line", "want"), [
    ("% SZS answers Tuple [[0,100]|_] for cert", ((0, 100),)),
    ("% SZS answers Tuple [([0,100]|[0,100])|_] for cert", ((0, 100),)),
    ("% SZS answers Tuple [[$uminus(3),7]|_] for cert", ((-3, 7),)),
])
def test_an_answer_tuple_is_read_off_vampires_output(line, want):
    from zrth.lean.magic.vampire import Answers

    got = Answers("/nowhere", seconds=1)._read(line, len(want[0]), 0.0)
    assert got is not None and got.answers == want


def test_every_alternative_of_a_disjunctive_answer_is_a_candidate():
    """The alternatives say one of these is a witness, not that each is: a
    split refutation closes each branch under its own hypothesis. Vampire
    answers `m_countdown` with `[1,100]` first and `[0,100]` after it, and
    only the second is an invariant -- `s0 = 1` steps to `0`."""
    from zrth.lean.magic.vampire import Answers

    got = Answers("/nowhere", seconds=1)._read(
        "% SZS answers Tuple [([1,100]|[0,100]|[0,100])|_] for cert", 2, 0.0)
    # Deduplicated, in the order Vampire listed them.
    assert got is not None and got.answers == ((1, 100), (0, 100))


def test_a_hole_the_refutation_never_pinned_down_reads_as_zero():
    """Vampire reports `∀X0.[X0]` for a hole any value serves. Zero is the
    one the certificate reads best, and it is checked with the rest."""
    from zrth.lean.magic.vampire import Answers

    got = Answers("/nowhere", seconds=1)._read(
        "% SZS answers Tuple [[∀X0.[X0],100]|_] for cert", 2, 0.0)
    assert got is not None and got.answers == ((0, 100),)


def test_an_answer_of_the_wrong_width_is_ignored():
    from zrth.lean.magic.vampire import Answers

    said = []
    a = Answers("/nowhere", seconds=1, log=said.append)
    assert a._read("% SZS answers Tuple [[1,2,3]|_] for cert", 2, 0.0) is None
    assert "3 values for 2 holes" in said[0]


def test_no_answer_at_all_is_no_certificate():
    from zrth.lean.magic.vampire import Answers

    assert Answers("/nowhere", seconds=1)._read(
        "% Termination reason: Time limit", 2, 0.0) is None


def test_both_namings_are_asked_and_each_is_told_the_whole_budget(tmp_path):
    """Two things at once, because they are the same decision.

    Vampire slices `--time_limit` between its strategies, so a small one
    runs a *different* search rather than the same one cut short --
    `m_max`'s question reaches the limit at `--time_limit 20` and answers
    after 2.5 s at 25. So the limit it is told is the run's whole budget and
    the wall clock is what actually bounds a call. And naming is asked both
    ways, because neither setting answers what the other does.
    """
    from zrth.lean.magic.vampire import Answers

    fake, log = tmp_path / "vampire", tmp_path / "argv"
    fake.write_text(f'#!/bin/sh\necho "$@" >> {log}\n')
    fake.chmod(0o755)

    asked = Answers(str(fake), seconds=30, log=lambda *_: None)
    assert asked.ask("(check-sat)\n", 2) is None

    lines = log.read_text().splitlines()
    # Both strategies on the short rung, then both on the long one.
    assert len(lines) == 4
    assert [("--naming 0" in ln) for ln in lines] == [False, True, False, True]
    assert all("--time_limit 30" in ln for ln in lines)


# ══════════════════════════════════════════════════════════════════════════
# What the module has to be
# ══════════════════════════════════════════════════════════════════════════


def test_a_real_component_reaches_the_templates():
    """A Real column is bounded by an interval with *integer* endpoints.

    The coefficients stay integers because Vampire reports an answer as a
    literal and this route reads an integer one, but `0 <= s0 <= 9` is a
    true and useful thing to say about a rational column -- so the sorts
    gate lets it through and the search decides, rather than the gate.
    Getting as far as looking for the prover is getting past both gates.
    """
    from zrth.lean.magic.vampire import TA2MagicVampire, template
    from zrth.lean.smt_synth import SynthContext

    magic = TA2MagicVampire(module_at(LIMITS / "m_lra_lin.py"),
                            vampire="/nowhere", log=lambda *_: None)
    ctx = SynthContext.build(magic.module,
                             CertificateData(prp="(= s0 0.0)", kind="buchi"),
                             route="vampire",
                             takes=("int", "bool", "bv", "real", "tuple"))
    magic._check_sorts(ctx)                  # the gate that used to refuse it
    tpl = template(ctx, "intervals", ranked=True)
    assert [r.smt for r in tpl.rows] == ["s0"]


def test_a_bitvector_state_is_refused_by_name():
    """A Real is arithmetic and a bitvector is not: a template row `<=` one
    nowhere, and reading it as `ubv_to_int` would put wraparound into a
    shape that has no word for it."""
    with pytest.raises(Refused, match="states its obligations over numbers"):
        derive("twobit", "buchi", "(and (= s0 (_ bv0 1)) (= s1 (_ bv0 1)))",
               fixture=FIXTURES, vampire="/nowhere")


def test_a_bool_component_is_still_refused_and_says_why():
    """`ite` is the one thing this route cannot print.

    A Bool column's arithmetic reading is `(ite b 1 0)`, and `refuse_ites`
    rejects any question holding one -- measured: a conditional in the goal
    defeats the answer-literal search. So a Bool is not a wiring problem
    here the way a Real was; it needs a template shape of its own.
    """
    with pytest.raises(Refused, match="is Bool"):
        derive("m_boolint", "safety", "(>= s1 0)", vampire="/nowhere")


def test_the_templates_are_tried_smallest_first():
    from zrth.lean.magic.vampire import templates

    ctx, _ob, _q = parts("m_toward2d", "safety", "(<= s0 10)")
    tpls = [t.name for t in templates(ctx, ranked=False)]
    assert tpls == ["intervals", "differences"]
    wide, narrow = templates(ctx, ranked=False)[::-1]
    assert len(wide.all_holes) > len(narrow.all_holes)


# ══════════════════════════════════════════════════════════════════════════
# The derivations
# ══════════════════════════════════════════════════════════════════════════


def test_vampire_derives_a_countdowns_invariant():
    """The route's whole claim, on the case it reaches: nothing proposed
    `0 <= s0 <= 100`, Vampire answered `A0=0, B0=100`."""
    cd = derive("m_countdown", "safety", "(<= s0 100)")
    assert cd.inv_smt == "(and (<= 0 s0) (<= s0 100))"
    assert cd.ranking_smt is None


def test_what_vampire_answers_is_checked_before_it_is_emitted():
    """An answer literal is the substitution *a* refutation used, and this
    route wrote the question. So a certificate that does not satisfy the
    module's own obligations is reported as not found rather than handed on
    -- here by making the check reject whatever came back."""
    import zrth.lean.magic.vampire as mv

    old = mv.TA2MagicVampire._checks_out
    try:
        mv.TA2MagicVampire._checks_out = lambda *_a, **_k: False
        with pytest.raises(Refused, match="derived no inductive invariant"):
            derive("m_countdown", "safety", "(<= s0 100)", timeout=10)
    finally:
        mv.TA2MagicVampire._checks_out = old


def test_vampire_derives_a_countdowns_invariant_and_ranking():
    """The Buchi claim, and the reason it is two questions: nothing proposed
    `0 <= s0 <= 100` or `1 + s0`, and the joint four-hole question for this
    module answers neither."""
    cd = derive("m_countdown", "buchi", "(= s0 0)", timeout=40)
    assert cd.inv_smt == "(and (<= 0 s0) (<= s0 100))"
    assert cd.ranking_smt is not None
    # Any `c + s0` ranks it; which constant Vampire picks is its business.
    assert cd.ranking_smt.endswith(" s0)") and cd.ranking_smt.startswith("(+ ")


def test_the_reach_is_two_holes():
    """The measured ceiling, pinned so it is a fact rather than a timeout.

    Vampire's answer-literal search closes the two-hole question for one
    component and does not close a three-hole one: `m_twovars` is two
    components, so its narrowest template is four holes, and it comes back
    with nothing inside a limit that answers `m_countdown` in under a
    second. `--infer houdini` is what searches a module this wide.
    """
    with pytest.raises(Refused, match="derived no inductive invariant"):
        derive("m_twovars", "safety", "(<= s0 10)", timeout=8)
