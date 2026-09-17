"""`--infer vampire`: the certificate Vampire's answer literals derive.

Three layers, and only the last needs the prover.

**Branch splitting** is what makes the question answerable at all, and it is
pure term rewriting: the conditions an `ite` tests, the cases they generate,
and the guard that stands in for each. It runs everywhere.

**The TPTP printer** is the other half of the same thing -- what a cvc5 term
looks like as a conjecture with holes in it, and what it refuses to print
rather than ask Vampire a question it cannot read. Also everywhere.

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
    from zrth.lean.magic_houdini import Obligations
    from zrth.lean.magic_vampire import Question, split
    from zrth.lean.smt_synth import SynthContext

    cd = CertificateData(prp=prp, kind=kind)
    ctx = SynthContext.build(module_at(fixture / f"{name}.py"), cd,
                             route="vampire")
    ob = Obligations(ctx)
    entry = split(ctx, ob.init, "the initial state")
    step = split(ctx, ob.next, "the round")
    return ctx, ob, Question(ctx, ob, entry, step)


def derive(name: str, kind: str, prp: str, *, timeout: float = 30,
           fixture: Path = LIMITS, vampire: "str | None" = None):
    from zrth.lean.magic_vampire import TA2MagicVampire

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
    from zrth.lean.magic_vampire import ite_conditions, split

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

    from zrth.lean.magic_vampire import templates

    script = q.conjecture(templates(ctx, ranked=False)[0], ctx.prp, safety=True)
    entry = script[script.index("(") : script.index("![")]
    assert "S0" not in entry


def test_a_transition_with_too_many_conditions_is_refused_by_name():
    from zrth.lean.magic_vampire import _MAX_CONDITIONS

    assert _MAX_CONDITIONS >= 1
    ctx, ob, _q = parts("m_countdown", "safety", "(<= s0 100)")
    from zrth.lean.magic_vampire import split

    # The bound is on the *count*, so it is checked by lowering it rather
    # than by finding a module with thirty conditionals.
    import zrth.lean.magic_vampire as mv

    old = mv._MAX_CONDITIONS
    try:
        mv._MAX_CONDITIONS = 0
        with pytest.raises(Refused, match="past the .* this route will print"):
            split(ctx, ob.next, "the round")
    finally:
        mv._MAX_CONDITIONS = old


# ══════════════════════════════════════════════════════════════════════════
# TPTP
# ══════════════════════════════════════════════════════════════════════════


def test_a_term_prints_as_tptp_over_the_names_it_was_given():
    import cvc5
    from cvc5 import Kind

    from zrth.lean.magic_vampire import Tptp

    tm = cvc5.TermManager()
    x = tm.mkConst(tm.getIntegerSort(), "v_s0")
    y = tm.mkConst(tm.getIntegerSort(), "v_s1")
    p = Tptp({"v_s0": "S0", "v_s1": "S1"})

    assert p(tm.mkTerm(Kind.ADD, x, y)) == "$sum(S0,S1)"
    assert p(tm.mkTerm(Kind.SUB, x, y)) == "$difference(S0,S1)"
    assert p(tm.mkTerm(Kind.LEQ, x, y)) == "$lesseq(S0,S1)"
    assert p(tm.mkTerm(Kind.EQUAL, x, y)) == "(S0 = S1)"
    assert p(tm.mkInteger(-3)) == "$uminus(3)"
    assert p(tm.mkTerm(Kind.NOT, tm.mkTerm(Kind.GT, x, tm.mkInteger(0)))) == (
        "~($greater(S0,0))")
    # n-ary in cvc5, binary in TPTP.
    assert p(tm.mkTerm(Kind.ADD, x, y, tm.mkInteger(1))) == (
        "$sum($sum(S0,S1),1)")


def test_an_ite_that_survived_the_split_is_refused_rather_than_printed():
    """The one thing this route cannot state. Printing it anyway would be a
    question Vampire reads and never answers, which reads as a timeout
    rather than as the unsupported shape it is."""
    import cvc5
    from cvc5 import Kind

    from zrth.lean.magic_vampire import Tptp

    tm = cvc5.TermManager()
    x = tm.mkConst(tm.getIntegerSort(), "v_s0")
    ite = tm.mkTerm(Kind.ITE, tm.mkTerm(Kind.GT, x, tm.mkInteger(0)),
                    x, tm.mkInteger(0))
    with pytest.raises(Refused, match="still holds an `ite`"):
        Tptp({"v_s0": "S0"})(ite)


def test_a_symbol_with_no_variable_is_refused_rather_than_invented():
    """A constant this route did not put there would be printed as a fresh
    TPTP symbol and quantified over, which silently changes the question."""
    import cvc5

    from zrth.lean.magic_vampire import Tptp

    tm = cvc5.TermManager()
    stray = tm.mkConst(tm.getIntegerSort(), "elsewhere")
    with pytest.raises(Refused, match="no TPTP variable for it"):
        Tptp({})(stray)


def test_the_conjecture_states_every_obligation_under_one_existential():
    ctx, _ob, q = parts("m_countdown", "safety", "(<= s0 100)")
    from zrth.lean.magic_vampire import templates

    tpl = templates(ctx, ranked=False)[0]
    script = q.conjecture(tpl, ctx.prp, safety=True)

    assert script.startswith("tff(cert, conjecture, ?[A0:$int,B0:$int]:")
    assert script.rstrip().endswith(").")
    assert "$ite" not in script
    # One entry obligation, one step per branch, one for the property.
    assert script.count("![S0:$int]") == 3


def test_a_buchi_conjecture_asks_for_the_ranking_in_the_same_question():
    ctx, _ob, q = parts("m_countdown", "buchi", "(= s0 0)")
    from zrth.lean.magic_vampire import templates

    tpl = templates(ctx, ranked=True)[0]
    assert tpl.all_holes == ("A0", "B0", "R0", "Rc")
    script = q.conjecture(tpl, ctx.prp, safety=False)
    # `rank > 0` where the property fails and `rank' < rank`: ite-free, and
    # it implies the clamped `Int.toNat` form `rule_buchi` asks for.
    assert "$greater($sum(Rc,$product(R0,S0)),0)" in script
    assert "$less(" in script


# ══════════════════════════════════════════════════════════════════════════
# Reading an answer back
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(("line", "want"), [
    ("% SZS answers Tuple [[0,100]|_] for cert", (0, 100)),
    ("% SZS answers Tuple [([0,100]|[0,100])|_] for cert", (0, 100)),
    ("% SZS answers Tuple [[$uminus(3),7]|_] for cert", (-3, 7)),
])
def test_an_answer_tuple_is_read_off_vampires_output(line, want):
    from zrth.lean.magic_vampire import Answers

    got = Answers("/nowhere", seconds=1)._read(line, len(want), 0.0)
    assert got is not None and got.values == want


def test_a_hole_the_refutation_never_pinned_down_reads_as_zero():
    """Vampire reports `∀X0.[X0]` for a hole any value serves. Zero is the
    one the certificate reads best, and it is checked with the rest."""
    from zrth.lean.magic_vampire import Answers

    got = Answers("/nowhere", seconds=1)._read(
        "% SZS answers Tuple [[∀X0.[X0],100]|_] for cert", 2, 0.0)
    assert got is not None and got.values == (0, 100)


def test_an_answer_of_the_wrong_width_is_ignored():
    from zrth.lean.magic_vampire import Answers

    said = []
    a = Answers("/nowhere", seconds=1, log=said.append)
    assert a._read("% SZS answers Tuple [[1,2,3]|_] for cert", 2, 0.0) is None
    assert "3 values for 2 holes" in said[0]


def test_no_answer_at_all_is_no_certificate():
    from zrth.lean.magic_vampire import Answers

    assert Answers("/nowhere", seconds=1)._read(
        "% Termination reason: Time limit", 2, 0.0) is None


# ══════════════════════════════════════════════════════════════════════════
# What the module has to be
# ══════════════════════════════════════════════════════════════════════════


def test_a_real_component_is_refused_by_name():
    """TPTP has `$real` and Vampire reads it, but the templates here are
    integer intervals and integer coefficients -- so this route takes the
    same gate `--infer smt-linear` does, before anything is printed."""
    with pytest.raises(Refused, match="s0 is Real"):
        derive("m_lra_lin", "buchi", "(= s0 0.0)", vampire="/nowhere")


def test_a_bitvector_state_is_refused_by_name():
    with pytest.raises(Refused, match="integers and no bitvectors"):
        derive("twobit", "buchi", "(and (= s0 (_ bv0 1)) (= s1 (_ bv0 1)))",
               fixture=FIXTURES, vampire="/nowhere")


def test_the_templates_are_tried_smallest_first():
    from zrth.lean.magic_vampire import templates

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
    import zrth.lean.magic_vampire as mv

    old = mv.TA2MagicVampire._checks_out
    try:
        mv.TA2MagicVampire._checks_out = lambda *_a, **_k: False
        with pytest.raises(Refused, match="derived no inductive invariant"):
            derive("m_countdown", "safety", "(<= s0 100)", timeout=10)
    finally:
        mv.TA2MagicVampire._checks_out = old


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
