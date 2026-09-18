"""`--infer houdini`: Houdini and a ranking search, put to either solver.

Three layers, and they are tested apart.

The half that needs no solver runs everywhere: the time-limit ladders, where
the Vampire binary is looked for, the evaluator the simulation runs on
(SMT-LIB's division is Euclidean, and Python's `//` is not), and the
simulation refuting a property the module violates before any solver is
started.

The *seam* -- :mod:`zrth.lean.houdini_solver` -- is tested against cvc5
alone, because what it is for is the thing only cvc5 does: answer a failed
proof with a counterexample, name the facts that counterexample breaks, and
settle a candidate for good rather than until the next rung.

The searches themselves run on every solver that is there. cvc5 always is;
Vampire is skipped when no binary can be found -- pass it as `$VAMPIRE`, or
put `vampire` on PATH. What they pin is what the route is for, and it does
not depend on who proved it: the certificate that comes back is the small
one the proofs used, not every fact Houdini kept.
"""

from __future__ import annotations

import importlib.util
from fractions import Fraction
from dataclasses import replace
from pathlib import Path

import pytest

from zrth.lean.cert import CertificateData
from zrth.lean.common import Refused
from zrth.lean.houdini_solver import SolverSpec

LIMITS = Path(__file__).parent / "limits" / "mods"
FIXTURES = Path(__file__).parent / "fixtures"


def module_at(path: Path):
    spec = importlib.util.spec_from_file_location(f"_m_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.module()


def solver_or_skip(kind: str):
    """The solver `kind` names, or a skip when its binary is not here.

    `--houdini-solver cvc5` resolves to nothing to look for, so the cvc5 half
    of every parametrised case below always runs.
    """
    from zrth.lean.houdini_solver import resolve_solver

    try:
        return resolve_solver(kind)
    except Refused:
        pytest.skip(f"no {kind}: set $VAMPIRE or put `vampire` on PATH")


# Every search below is run once per solver. Naming them here rather than in
# each `parametrize` keeps "both solvers agree" the default a case has to opt
# out of, which is the property the seam exists to hold.
BOTH = pytest.mark.parametrize("solver", ["cvc5", "vampire"])


def infer(name: str, kind: str, prp: str, *, solver="cvc5",
          timeout: float = 60, fixture: Path = LIMITS, log=None):
    from zrth.lean.magic.houdini import TA2MagicHoudini

    magic = TA2MagicHoudini(module_at(fixture / f"{name}.py"), solver=solver,
                            timeout=timeout, log=log or (lambda *_: None))
    return magic.infer(CertificateData(prp=prp, kind=kind))


def a_search(name: str, kind: str, prp: str, fixture: Path = LIMITS):
    """A route mid-flight: its context, obligations and candidate parser,
    without a search having been run. What the seam's cases ask about."""
    from zrth.lean.magic.houdini import Obligations, TA2MagicHoudini
    from zrth.lean.smt_synth import SynthContext

    cd = CertificateData(prp=prp, kind=kind)
    magic = TA2MagicHoudini(module_at(fixture / f"{name}.py"),
                            log=lambda *_: None)
    magic.ctx = SynthContext.build(magic.module, cd, route="houdini",
                                   reals=True)
    return magic, Obligations(magic.ctx)


# ══════════════════════════════════════════════════════════════════════════
# Without the prover
# ══════════════════════════════════════════════════════════════════════════


def test_vampires_time_limit_climbs_and_ends_at_the_budget():
    """Four rungs, because its portfolio schedules a short limit differently
    rather than cutting a long one short."""
    from zrth.lean.houdini_solver import VampireSolver

    v = VampireSolver("/nowhere", seconds=1)
    assert v.ladder(120) == (2, 10, 60, 120)
    assert v.ladder(60) == (2, 10, 60)
    assert v.ladder(5) == (2, 5)
    assert v.ladder(1) == (1,)


def test_cvc5s_ladder_is_a_cap_rather_than_a_schedule():
    """Its search at a longer limit contains the shorter one, so there is
    nothing to be had from re-running one at a bigger limit -- the rungs are
    there only so a single query cannot eat the whole budget."""
    import cvc5

    from zrth.lean.houdini_solver import Cvc5Solver

    c = Cvc5Solver(cvc5.TermManager(), seconds=1)
    assert c.ladder(120) == (2, 120)
    assert c.ladder(1) == (1,)
    assert c.short == 2


def test_a_vampire_path_that_is_not_there_is_refused(tmp_path):
    from zrth.lean.houdini_solver import resolve_vampire

    with pytest.raises(Refused, match="--vampire: no such file"):
        resolve_vampire(str(tmp_path / "vampire"))


def test_a_directory_holding_vampire_is_accepted(tmp_path):
    """The release zip unpacks to a directory with the binary in it."""
    from zrth.lean.houdini_solver import resolve_vampire

    exe = tmp_path / "vampire"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert resolve_vampire(str(tmp_path)) == str(exe)


@pytest.mark.parametrize("a", [-7, -6, 0, 5, 7])
@pytest.mark.parametrize("b", [-3, 2, 3])
def test_the_evaluator_divides_the_way_smt_lib_does(a, b):
    """Euclidean: the remainder is never negative, whatever the signs --
    checked against cvc5's own rewriter, which is the semantics every
    obligation is stated in."""
    import cvc5
    from cvc5 import Kind

    from zrth.lean.magic.houdini import Evaluator

    tm = cvc5.TermManager()
    solver = cvc5.Solver(tm)
    x = tm.mkConst(tm.getIntegerSort(), "x")
    ev = Evaluator(tm)
    for kind in (Kind.INTS_DIVISION, Kind.INTS_MODULUS):
        term = tm.mkTerm(kind, x, tm.mkInteger(b))
        want = solver.simplify(term.substitute([x], [tm.mkInteger(a)]))
        assert ev(term, {"x": a}) == want.getIntegerValue()


def test_a_property_the_module_violates_is_refuted_by_running_it():
    """`m_countdown` starts at 100: `s0 <= 50` fails on round zero, and no
    solver is needed -- the Vampire binary named here does not exist, and
    the refusal comes before anything would have started it."""
    with pytest.raises(Refused, match="does not hold.*s0 = 100"):
        infer("m_countdown", "safety", "(<= s0 50)",
              solver=SolverSpec("vampire", "/nowhere"))


@pytest.mark.parametrize(("solver", "why"), [
    ("vampire", "no bitvector theory"),
    # cvc5 has one; what it has no candidate shapes for is the point.
    ("cvc5", "the candidate shapes here do not"),
])
def test_a_bitvector_state_is_refused_by_name(solver, why):
    with pytest.raises(Refused, match=why):
        infer("twobit", "buchi", "(and (= s0 (_ bv0 1)) (= s1 (_ bv0 1)))",
              solver=solver, fixture=FIXTURES)


@pytest.mark.parametrize(("value", "written"), [
    ("1/2", "0.5"),
    ("-1/4", "(- 0.25)"),
    ("3", "3.0"),
    ("-2", "(- 2.0)"),
    ("5/8", "0.625"),
    # No decimal says a third, and one that rounded it would state a
    # different obligation than the module's.
    ("1/3", "(/ 1.0 3.0)"),
    ("-1/3", "(- (/ 1.0 3.0))"),
])
def test_a_real_literal_is_written_as_a_decimal_where_one_is_exact(value, written):
    from fractions import Fraction

    from zrth.lean.magic.houdini import smt_real

    assert smt_real(Fraction(value)) == written


def test_cvc5s_rationals_are_rewritten_on_the_way_into_a_vampire_script():
    """cvc5 prints one half as `(/ 1 2)`, whose arguments are Int-sorted;
    Vampire answers that with `invalid sort $int for interpretation /`."""
    from zrth.lean.magic.houdini import decimals

    assert decimals("(- v_s0 (/ 1 2))") == "(- v_s0 0.5)"
    assert decimals("(* (/ (- 1) 4) x)") == "(* (- 0.25) x)"
    assert decimals("(div a 2)") == "(div a 2)"


def test_the_evaluator_floors_a_real_the_way_to_int_does():
    """The scale a real ranking function is read at exists because of this:
    a quantity that falls by a half need not floor to a smaller number."""
    import cvc5
    from cvc5 import Kind

    from zrth.lean.magic.houdini import Evaluator

    tm = cvc5.TermManager()
    x = tm.mkConst(tm.getRealSort(), "x")
    ev = Evaluator(tm)
    floor = tm.mkTerm(Kind.TO_INTEGER, x)
    assert [ev(floor, {"x": Fraction(k, 2)}) for k in range(5)] == [0, 0, 1, 1, 2]
    scaled = tm.mkTerm(Kind.TO_INTEGER, tm.mkTerm(Kind.MULT, tm.mkReal(2), x))
    assert [ev(scaled, {"x": Fraction(k, 2)}) for k in range(5)] == [0, 1, 2, 3, 4]


def test_a_real_module_is_no_longer_refused_for_its_sort():
    """It is the shapes that have to answer for a Real component now, not a
    sort check: `SynthContext.build` still refuses one for every other route."""
    from zrth.lean.smt_synth import SynthContext

    cd = CertificateData(prp="(= s0 0.0)", kind="buchi")
    module = module_at(LIMITS / "m_lra_lin.py")
    with pytest.raises(Refused, match="no integer reading"):
        SynthContext.build(module, cd, route="smt-linear")
    ctx = SynthContext.build(module, cd, route="houdini", reals=True)
    assert [str(s) for s in ctx.env.state_sorts] == ["Real"]


# ══════════════════════════════════════════════════════════════════════════
# The seam: what only a deciding solver gives back
# ══════════════════════════════════════════════════════════════════════════


def test_a_false_obligation_comes_back_refuted_with_the_facts_it_breaks():
    """The one thing Vampire cannot do, and the whole reason cvc5 is here.

    `m_countdown` counts down from 100. `s0 >= 50` holds at entry and is
    false a round later; `s0 >= 0` and `s0 <= 100` are preserved. One query
    about all three comes back refuted, and its model names the one that
    broke -- which is Houdini's next step, taken from one call rather than
    from one call per fact.
    """
    import cvc5

    from zrth.lean.houdini_solver import Cvc5Solver, Verdict

    magic, ob = a_search("m_countdown", "buchi", "(= s0 0)")
    kept = magic._parse(["(<= 0 s0)", "(<= s0 100)", "(<= 50 s0)"])
    assert len(kept) == 3
    solver = Cvc5Solver(magic.ctx.tm, seconds=30)
    answer = solver.prove(ob.preserved(kept, kept), 5, model=True)
    assert answer.verdict is Verdict.REFUTED
    assert answer.broken == frozenset({"g2"})       # only `50 <= s0`
    assert isinstance(magic.ctx.tm, cvc5.TermManager)


def test_a_refuted_answer_is_kept_forever_and_an_unknown_only_per_limit():
    """`REFUTED` holds at every time limit there will ever be, so it is
    cached like a proof; an `UNKNOWN` is only ever "not at *that* limit"."""
    from zrth.lean.houdini_solver import Answer, Query, Solver, Verdict

    q = Query(hyps=(), goal="g")
    for verdict, limits in ((Verdict.PROVED, [2]), (Verdict.REFUTED, [2]),
                            (Verdict.UNKNOWN, [2, 10])):
        asked = []

        class Once(Solver):
            def _run(self, query, budget, core, model):
                asked.append(budget)
                return Answer(verdict)

        solver = Once(seconds=30)
        solver.prove(q, 2)
        solver.prove(q, 2)
        solver.prove(q, 10)
        # A settled answer is asked once and never again. An unknown is
        # re-asked at a higher limit and not at one it already gave up on,
        # which is what makes repeating a whole pass nearly free.
        assert asked == limits


def test_a_cached_proof_is_asked_again_only_when_a_core_is_wanted():
    """A proof already in hand serves a question that wants no core, and is
    re-asked when one does -- the same rule the counter-model follows."""
    from zrth.lean.houdini_solver import Answer, Query, Solver, Verdict

    asked = []

    class Bare(Solver):
        def _run(self, query, budget, core, model):
            asked.append(core)
            return Answer(Verdict.PROVED, core=frozenset({"h0"}) if core else None)

    q = Query(hyps=(), goal="g")
    solver = Bare(seconds=30)
    assert solver.prove(q, 2).core is None
    assert solver.prove(q, 2).core is None           # served from the cache
    assert solver.prove(q, 2, core=True).core == frozenset({"h0"})
    assert solver.prove(q, 2, core=True).core == frozenset({"h0"})
    assert asked == [False, True]


def test_entry_is_narrowed_until_it_is_proved_not_once_per_model():
    """One counter-model names the facts *that* initial state breaks, and a
    different one may break others -- so the entry check is a fixpoint, not
    a single drop.

    Driven by a solver scripted to witness one fact at a time, which is what
    a module whose `init` reads an input genuinely does. Nothing downstream
    asks `init_inv` again, so a fact left in here is a Lean failure later,
    and this loop is the only thing between the two.
    """
    from zrth.lean.houdini_solver import Answer, Solver, Verdict

    magic, ob = a_search("m_countdown", "buchi", "(= s0 0)")
    magic.ob = ob
    facts = magic._parse(["(<= 0 s0)", "(= s0 0)", "(<= s0 50)", "(<= s0 100)"])
    assert len(facts) == 4
    asked = []

    class Scripted(Solver):
        """Answers down a list, and records how many facts it was asked
        about -- the probes are positional, so `g1` is the second one of
        whatever list the caller currently holds."""

        def __init__(self, answers, **kw):
            super().__init__(**kw)
            self.answers = list(answers)

        def _run(self, query, budget, core, model):
            asked.append(len(query.probes))
            return self.answers.pop(0)

    one = frozenset({"g1"})
    solver = Scripted([Answer(Verdict.REFUTED, broken=one),
                       Answer(Verdict.REFUTED, broken=one),
                       Answer(Verdict.PROVED)], seconds=30)
    kept = magic._at_entry(solver, facts, 5)

    assert [f.src for f in kept] == ["(<= 0 s0)", "(<= s0 100)"]
    # Four facts, then three, then two: it asked again after each model
    # rather than taking the first one's word for the whole set.
    assert asked == [4, 3, 2]
    assert not solver.answers


def test_entry_stops_when_a_model_names_nothing_to_drop():
    """An unreadable model is not evidence that every fact is fine, so the
    loop ends rather than spinning on an answer it cannot act on."""
    from zrth.lean.houdini_solver import Answer, Solver, Verdict

    magic, ob = a_search("m_countdown", "buchi", "(= s0 0)")
    magic.ob = ob
    facts = magic._parse(["(<= 0 s0)", "(<= s0 100)"])
    calls = []

    class Mute(Solver):
        def _run(self, query, budget, core, model):
            calls.append(len(query.probes) or 1)
            # Refuted, no model, and each fact on its own proves: exactly
            # what Vampire's silence looks like on a conjunction it timed
            # out on.
            return (Answer(Verdict.PROVED) if not query.probes
                    else Answer(Verdict.UNKNOWN))

    kept = magic._at_entry(Mute(seconds=30), facts, 5)
    assert [f.src for f in kept] == ["(<= 0 s0)", "(<= s0 100)"]
    assert calls == [2, 1, 1]           # the batch, then fact by fact


def test_a_solver_with_no_model_falls_back_to_asking_fact_by_fact():
    """`_survivors` is the fork. Handed an answer carrying no `broken` it
    asks about each fact on its own, which is the only thing a refutation
    prover leaves to do -- and it must reach the same set."""
    from zrth.lean.houdini_solver import Cvc5Solver

    magic, ob = a_search("m_countdown", "buchi", "(= s0 0)")
    magic.ob = ob
    kept = magic._parse(["(<= 0 s0)", "(<= s0 100)", "(<= 50 s0)"])
    solver = Cvc5Solver(magic.ctx.tm, seconds=30)

    with_model = solver.prove(ob.preserved(kept, kept), 5, model=True)
    assert with_model.broken                        # cvc5 gave one
    from_model = magic._survivors(solver, with_model, kept, None, 5)

    blind = replace(with_model, broken=None)        # as Vampire would answer
    one_at_a_time = magic._survivors(
        solver, blind, kept,
        lambda fs: [ob.preserved(fs, [f]) for f in fs], 5)

    assert [f.src for f in from_model] == [f.src for f in one_at_a_time]
    assert [f.src for f in from_model] == ["(<= 0 s0)", "(<= s0 100)"]


def test_a_query_carries_its_transition_once_and_names_every_fact():
    """What the seam promises a solver: the round stated once as `defines`,
    the hypotheses named for a core, the goal's conjuncts named for a model.
    A goal that is one thing gets no probes -- `REFUTED` already says it."""
    magic, ob = a_search("m_countdown", "buchi", "(= s0 0)")
    kept = magic._parse(["(<= 0 s0)", "(<= s0 100)"])

    q = ob.preserved(kept, kept)
    assert len(q.defines) == len(magic.ctx.state)
    assert [n for n, _t in q.hyps] == ["h0", "h1", "pre"]
    assert [n for n, _t in q.probes] == ["g0", "g1"]

    assert ob.preserved(kept, kept[:1]).probes == ()
    rank = magic._parse(["s0"])[0]
    assert ob.drops(kept, rank).probes == ()


def test_the_vampire_script_states_the_round_once_and_names_the_goal():
    """The printed half, checked without a binary: a `define-fun` per
    component, `:named` on every hypothesis, and the goal asserted negated
    under a name of its own so a core is never empty by accident."""
    from zrth.lean.houdini_solver import VampireSolver

    magic, ob = a_search("m_lra_lin", "buchi", "(= s0 0.0)")
    kept = magic._parse(["(<= 0.0 s0)", "(<= s0 5.0)"])
    script = VampireSolver._script(ob.preserved(kept, kept))

    assert script.startswith("(set-logic ALL)")
    assert script.count("(define-fun v_sp0") == 1
    assert ":named h0" in script and ":named h1" in script
    assert ":named pre" in script
    assert "(assert (! (not " in script and ":named goal)" in script
    assert script.rstrip().endswith("(check-sat)")
    # Every literal a decimal: Vampire reads no `(/ 1 2)` and no bare `3`
    # where a Real belongs.
    assert "(/ 1 2)" not in script


def test_vampires_answer_is_read_off_its_output(tmp_path):
    """The subprocess half, against a stub that prints what Vampire prints.

    `unsat` followed by the names of the assertions a refutation used is a
    proof plus a core; anything else is `UNKNOWN` and never `REFUTED`, which
    is the whole of what this solver cannot tell you.
    """
    from zrth.lean.houdini_solver import VampireSolver, Verdict

    def stub(output: str):
        exe = tmp_path / f"vampire{abs(hash(output))}"
        exe.write_text(f"#!/bin/sh\ncat <<'OUT'\n{output}\nOUT\n")
        exe.chmod(0o755)
        return VampireSolver(str(exe), seconds=30, cores=1)

    magic, ob = a_search("m_countdown", "buchi", "(= s0 0)")
    kept = magic._parse(["(<= 0 s0)", "(<= s0 100)"])
    q = ob.preserved(kept, kept)

    proof = stub("unsat\n(\nh1\ngoal\n)").prove(q, 5, core=True)
    assert proof.verdict is Verdict.PROVED
    assert proof.core == frozenset({"h1"})          # `goal` is not a hypothesis

    # No model, ever: a query's probes are ignored and `broken` stays `None`,
    # which is what sends `_survivors` down the fact-by-fact path.
    assert proof.broken is None
    assert stub("% Time limit reached\n").prove(q, 5, model=True).verdict is (
        Verdict.UNKNOWN)

    # An encoding it cannot read is a refusal naming the other solver, not a
    # silent failure to prove.
    with pytest.raises(Refused, match="cannot read this module's encoding"):
        stub("User error: invalid sort $int for interpretation /").prove(q, 5)


# ══════════════════════════════════════════════════════════════════════════
# The searches, on every solver that is here
#
# One body per case, run once per solver. The certificates are the route's
# contract and they do not depend on who proved them -- which is the claim
# these cases are here to keep true as the seam changes.
# ══════════════════════════════════════════════════════════════════════════


@BOTH
def test_a_countdown_is_ranked(solver):
    cd = infer("m_countdown", "buchi", "(= s0 0)", solver=solver_or_skip(solver))
    assert cd.ranking_smt == "s0"
    assert "(<= 0 s0)" in cd.inv_smt


@BOTH
def test_a_safety_invariant_is_cut_to_what_its_proofs_use(solver):
    """Houdini keeps `0 <= s0` and `s0 <= 100`; the property is the second,
    and it is inductive on its own, so the core leaves the first out."""
    cd = infer("m_countdown", "safety", "(<= s0 100)",
               solver=solver_or_skip(solver))
    assert cd.inv_smt == "(<= s0 100)"


@BOTH
def test_the_congruence_a_parity_property_needs_is_kept(solver):
    """`m_step2` counts 0, 2, ..., 10: `s0 != 5` is not inductive (3 steps
    to 5), and `s0` being even is the fact that makes it so."""
    cd = infer("m_step2", "safety", "(not (= s0 5))",
               solver=solver_or_skip(solver))
    assert "(= (mod s0 2) 0)" in cd.inv_smt


@BOTH
def test_a_real_invariant_is_the_values_a_run_takes(solver):
    """`m_lra_lin` steps `x' = x - 1` while `x > 0`, so `0 <= x <= 5` admits
    `x = 1/2` and steps it out: over the reals it is the value set, not the
    interval, that is inductive. The pair `tests/limits` carries by hand."""
    cd = infer("m_lra_lin", "buchi", "(= s0 0.0)", solver=solver_or_skip(solver))
    assert cd.inv_smt == ("(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) "
                          "(= s0 4.0) (= s0 5.0))")
    assert cd.ranking_smt == "(to_int s0)"


@BOTH
def test_a_real_ranking_function_is_scaled_before_it_is_floored(solver):
    """`m_lra_half` steps by a half, where `to_int s0` repeats -- 3.0 and 2.5
    both floor to 3 -- so the scale is the denominator the program writes.
    Either solver proves the floored obligation directly."""
    cd = infer("m_lra_half", "buchi", "(= s0 0.0)", solver=solver_or_skip(solver))
    assert cd.ranking_smt == "(to_int (* 2.0 s0))"
    assert "(= s0 0.5)" in cd.inv_smt


@BOTH
def test_two_real_components_are_ranked_by_the_one_that_falls(solver):
    """`m_lra_conv`: `x` converges to 0 and `y` to 2, two rounds apart. The
    relation between them is a fact, so `y`'s range carries `x`'s rank."""
    cd = infer("m_lra_conv", "buchi", "(and (= s0 0.0) (= s1 2.0))",
               solver=solver_or_skip(solver))
    assert cd.ranking_smt == "(to_int s0)"
    assert "(= (- s0 s1) (- 2.0))" in cd.inv_smt


# ══════════════════════════════════════════════════════════════════════════
# The flags
# ══════════════════════════════════════════════════════════════════════════


def test_cvc5_is_the_default_and_resolves_without_a_binary():
    """The reason it is the default: nothing to look for, and nothing to
    fail at parse time on a machine with no prover installed."""
    from zrth.lean.houdini_solver import DEFAULT_SOLVER, resolve_solver

    assert DEFAULT_SOLVER == "cvc5"
    assert resolve_solver("cvc5") == SolverSpec("cvc5", None, 4)


def test_a_vampire_flag_without_the_vampire_solver_is_refused():
    """A flag that quietly did nothing would be worse than one that is an
    error, which is what the route's `extra_check` is for."""
    from zrth.lean.infer_route import route_by_name

    check = route_by_name("houdini").extra_check
    opts = {"houdini_solver": "cvc5", "vampire": "/x", "vampire_cores": None}
    assert "starts no binary" in check(None, opts)
    assert check(None, {**opts, "vampire": None}) is None
    assert check(None, {**opts, "houdini_solver": "vampire"}) is None
