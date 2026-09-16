"""The two searching routes (`--infer sygus`, `--infer smt-linear`).

These run cvc5, which is what makes them worth testing without an API key:
every case here is a real search over a real fixture, and the answers are
the ones the design was measured against -- `m_step2`'s invariant is a
congruence and only the grammar with `mod` in it can say so; `m_toward5` has
no linear ranking function and the point of the route is that cvc5 *proves*
that rather than failing to find one.

The other half is the handoff. A run leaves what it found and what it ruled
out in `artifacts/`, and the next run picks both up -- an invariant as a
fixed starting point, an empty space as a line in the LLM prompt. That is
checked here end to end, because it is the part no single route owns.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from zrth.lean.artifacts import ArtifactStore, module_digest
from zrth.lean.cert import CertificateData
from zrth.lean.common import Refused

MODS = Path(__file__).parent / "limits" / "mods"


def module_of(name: str):
    """One `tests/limits` fixture module, loaded by path."""
    spec = importlib.util.spec_from_file_location(f"_m_{name}", MODS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.module()


def store_for(tmp_path: Path, name: str, *, kind: str, prp: str,
              producer: str = "test") -> ArtifactStore:
    return ArtifactStore(
        dir=tmp_path / "artifacts",
        module_digest=module_digest(module_of(name)),
        kind=kind,
        prp=prp,
        producer=producer,
    )


# ══════════════════════════════════════════════════════════════════════════
# --infer sygus
# ══════════════════════════════════════════════════════════════════════════


def test_sygus_finds_the_congruence_no_lattice_can_state():
    """`m_step2` steps by two: the invariant is that `x` is even.

    The limit matrix records this as the one Buchi case `--infer nuterm`
    misses on the certificate rather than on reach, and the reason is that
    Houdini's candidates are signs and pairwise relations. A grammar with
    `mod` in it says it in milliseconds.
    """
    from zrth.lean.magic_sygus import TA2MagicSygus

    cd = CertificateData(prp="(not (= s0 1))", kind="safety")
    out = TA2MagicSygus(module_of("m_step2"), log=lambda *_: None).infer(cd)
    assert "mod" in out.inv_smt


def test_a_synthesised_invariant_that_prints_with_let_is_kept():
    """cvc5 `let`-binds a repeated subterm when it prints a solution.

    The route used to refuse such an invariant as one "the certificate's
    definition cannot carry". It can: the source is parsed back before it is
    rendered, and `smt_to_lean` binds what is shared in Lean. This one --
    `(let ((_let_1 (* (- 1) s1))) ...)` -- was found and thrown away."""
    from benchmarks.svcomp import discover
    from zrth.lean.magic_sygus import TA2MagicSygus

    bench = next(b for b in discover() if b.name == "AliasDarteFeautrierGonnord-SAS2010-easy2-2")
    prp = ("(and (>= s0 0) (<= s1 0) (>= (+ s0 (* (- 1) s1)) 0) (>= (+ s0 s1) 0) "
           "(<= (+ s0 s1) 0) (= (+ s0 s1) 0) (>= s0 (- 1)) (<= s1 1))")
    out = TA2MagicSygus(bench.build()[0], log=lambda *_: None).infer(
        CertificateData(prp=prp, kind="safety"))
    assert out.inv_smt

def test_sygus_without_congruences_cannot_state_it():
    """The same module under `--sygus-grammar linear`.

    The option is not a knob for its own sake: it is what shows that the
    congruence is doing the work rather than the search.
    """
    from zrth.lean.magic_sygus import TA2MagicSygus

    cd = CertificateData(prp="(not (= s0 1))", kind="safety")
    magic = TA2MagicSygus(
        module_of("m_step2"), grammar="linear", log=lambda *_: None
    )
    with pytest.raises(Refused, match="no invariant"):
        magic.infer(cd)


def test_sygus_leaves_the_invariant_where_the_next_run_resumes_from(tmp_path):
    """Found, and then *resumable*: `proved` and SMT-LIB, which is what
    `infer_route.FIXABLE` and `_resume_inv`'s filter ask for."""
    from zrth.lean.magic_sygus import TA2MagicSygus

    store = store_for(tmp_path, "m_step2", kind="safety", prp="(not (= s0 1))",
                      producer="sygus")
    cd = CertificateData(prp="(not (= s0 1))", kind="safety")
    TA2MagicSygus(
        module_of("m_step2"), artifacts=store, log=lambda *_: None
    ).infer(cd)
    kept = store.usable("inv", languages=("smt",), statuses=("proved",))
    assert len(kept) == 1
    assert store.read(kept[0]) == cd.inv_smt


def test_a_real_state_is_refused_by_name_by_both_routes():
    """No integer reading, and a ranking function has to land in `Nat`.

    Refused where the sorts are read rather than inside a grammar rule or a
    template, so the message names the component instead of a cvc5 error.
    """
    from zrth.lean.magic_linear import TA2MagicLinear
    from zrth.lean.magic_sygus import TA2MagicSygus

    cd = CertificateData(prp="(>= s0 0.0)", kind="safety")
    for magic in (TA2MagicSygus(module_of("m_lra_lin"), log=lambda *_: None),
                  TA2MagicLinear(module_of("m_lra_lin"), log=lambda *_: None)):
        with pytest.raises(Refused, match="not weighable"):
            magic.infer(cd)


def test_sygus_sends_a_bool_state_to_the_route_that_weighs_it():
    """The grammar's `synthFun` takes integer arguments, so a Bool column has
    nowhere to go -- and the refusal names the route that does read it."""
    from zrth.lean.magic_sygus import TA2MagicSygus

    cd = CertificateData(prp="(>= s1 0)", kind="safety")
    with pytest.raises(Refused, match="--infer smt-linear"):
        TA2MagicSygus(module_of("m_boolint"), log=lambda *_: None).infer(cd)


def test_sygus_is_a_safety_route_in_the_class_as_well_as_the_row():
    from zrth.lean.magic_sygus import TA2MagicSygus

    cd = CertificateData(prp="(= s0 0)", kind="buchi")
    with pytest.raises(Refused, match="ranking function"):
        TA2MagicSygus(module_of("m_countdown"), log=lambda *_: None).infer(cd)


# ══════════════════════════════════════════════════════════════════════════
# --infer smt-linear
# ══════════════════════════════════════════════════════════════════════════


def test_smt_linear_finds_a_rank_when_the_shape_has_one():
    from zrth.lean.magic_linear import TA2MagicLinear

    cd = CertificateData(
        prp="(= s0 0)", kind="buchi", inv="(and (>= s0 0) (<= s0 100))"
    )
    out = TA2MagicLinear(module_of("m_countdown"), log=lambda *_: None).infer(cd)
    assert out.ranking_smt == "s0"


def test_smt_linear_ranks_a_loop_whose_exit_states_are_unbounded():
    """`while (y >= 0) y = y - 1`: `rule_buchi` asks the rank to fall only
    where the property fails, and `1 + y` does, over the invariant `true`.

    The route used to demand `rank >= 0` over every state as well. Where the
    property holds `y` runs to minus infinity, so no linear rank survived that,
    and the empty space was reported as a *proof* that none exists."""
    from benchmarks.svcomp import discover
    from zrth.lean.magic_linear import TA2MagicLinear

    bench = next(b for b in discover() if b.name == "PodelskiRybalchenko-TACAS2011-Fig1")
    cd = CertificateData(prp="(not (<= 0 s0))", kind="buchi")
    out = TA2MagicLinear(bench.build()[0], log=lambda *_: None).infer(cd)
    assert out.ranking_smt is not None

def test_smt_linear_proves_a_shape_empty_rather_than_failing_to_search_it():
    """`m_toward5` moves toward 5 from both sides, so a rank has to branch.

    The refusal says so as a *proof*, which is the difference between this
    route and an LLM that tried linear candidates until its attempts ran
    out.
    """
    from zrth.lean.magic_linear import TA2MagicLinear

    cd = CertificateData(
        prp="(= s0 5)", kind="buchi", inv="(and (>= s0 0) (<= s0 10))"
    )
    with pytest.raises(Refused) as e:
        TA2MagicLinear(module_of("m_toward5"), log=lambda *_: None).infer(cd)
    assert "proof that the space is empty" in str(e.value)


def test_smt_linear_finds_a_one_row_safety_invariant():
    from zrth.lean.magic_linear import TA2MagicLinear

    cd = CertificateData(prp="(<= s0 100)", kind="safety")
    out = TA2MagicLinear(module_of("m_countdown"), log=lambda *_: None).infer(cd)
    # Divided through by the gcd: the same predicate, and the one the
    # certificate should carry.
    assert out.inv_smt == "(>= (+ 100 (- s0)) 0)"


def test_smt_linear_strengthens_a_supplied_invariant_rather_than_replacing_it():
    from zrth.lean.magic_linear import TA2MagicLinear

    cd = CertificateData(prp="(<= s0 100)", kind="safety", inv="(>= s0 0)")
    out = TA2MagicLinear(module_of("m_countdown"), log=lambda *_: None).infer(cd)
    assert out.inv_smt.startswith("(and (>= s0 0)")


def test_smt_linear_writes_what_it_ruled_out(tmp_path):
    from zrth.lean.magic_linear import TA2MagicLinear

    store = store_for(tmp_path, "m_toward5", kind="buchi", prp="(= s0 5)",
                      producer="smt-linear")
    cd = CertificateData(
        prp="(= s0 5)", kind="buchi", inv="(and (>= s0 0) (<= s0 10))"
    )
    with pytest.raises(Refused):
        TA2MagicLinear(
            module_of("m_toward5"), artifacts=store, log=lambda *_: None
        ).infer(cd)
    notes = store.usable("note", languages=("md",), statuses=("no_solution",))
    assert len(notes) == 1
    text = store.read(notes[0])
    assert "No ranking function linear in the state" in text
    assert "(and (>= s0 0) (<= s0 10))" in text, "the invariant it searched under"


def test_a_search_that_times_out_is_not_a_proof_of_absence(tmp_path):
    """The distinction the whole design rests on.

    A budget of one millisecond decides nothing, and what it writes has to
    say so: `unknown`, not `no_solution`. A consumer that confused the two
    would state a falsehood in an LLM prompt.
    """
    from zrth.lean.magic_linear import TA2MagicLinear
    from zrth.lean.smt_query import SmtBudget

    store = store_for(tmp_path, "m_lex", kind="safety", prp="(<= s0 3)",
                      producer="smt-linear")
    cd = CertificateData(prp="(<= s0 3)", kind="safety")
    with pytest.raises(Refused):
        TA2MagicLinear(
            module_of("m_lex"),
            rows=3,
            budget=SmtBudget(per_call_ms=1, phase_ms=1),
            artifacts=store,
            log=lambda *_: None,
        ).infer(cd)
    assert not store.usable("note", statuses=("no_solution",))
    undecided = store.usable("note", statuses=("unknown",))
    assert len(undecided) == 1
    assert "did not finish" in store.read(undecided[0])


# ══════════════════════════════════════════════════════════════════════════
# The handoff
# ══════════════════════════════════════════════════════════════════════════


def test_what_one_run_ruled_out_reaches_the_next_run_s_prompt():
    """`--infer smt-linear` refutes a shape; `--infer ai-cegar` is told.

    The note is written to be read by a model, so what is checked is that it
    arrives in the message verbatim and is framed as established rather than
    as feedback about an attempt.
    """
    from zrth.lean.smt_module import ModuleSMT
    from zrth.lean.smt_prompt import CegarPromptEnv, prompt_inv_ranking
    import cvc5

    tm = cvc5.TermManager()
    env = CegarPromptEnv(ModuleSMT(tm=tm, module=module_of("m_countdown")))
    seen: list[str] = []

    def chat(system: str, user: str) -> str:
        seen.append(user)
        return "INVARIANT: (and (>= s0 0) (<= s0 100))\nRANKING: s0"

    ruled_out = "No ranking function linear in the state satisfies the obligations."
    prompt_inv_ranking(
        env, chat, "source", "(= s0 0)", "", None, known=(ruled_out,)
    )
    assert ruled_out in seen[0]
    assert "decision procedure" in seen[0]
    assert "do not propose anything of a shape ruled out" in seen[0]


def test_an_empty_ranking_is_a_parse_error_the_loop_can_answer():
    """`RANKING:` with nothing after it is feedback, not a cvc5 crash.

    cvc5 parses an empty source to a null term, and `getSort()` on one raises
    a `RuntimeError` from inside the binding -- which `TA2MagicCEGAR`'s loop does
    not catch, so the whole route died on one bad reply."""
    from zrth.lean.smt_module import ModuleSMT
    from zrth.lean.smt_prompt import CegarPromptEnv, PromptParseError, prompt_inv_ranking
    import cvc5

    tm = cvc5.TermManager()
    env = CegarPromptEnv(ModuleSMT(tm=tm, module=module_of("m_countdown")))

    def chat(system: str, user: str) -> str:
        return "INVARIANT: (and (>= s0 0) (<= s0 100))\nRANKING:"

    with pytest.raises(PromptParseError, match="RANKING"):
        prompt_inv_ranking(env, chat, "source", "(= s0 0)", "", None)

def test_only_a_proof_of_absence_reaches_the_prompt(tmp_path):
    """`_ruled_out` filters on `no_solution`, so a timed-out search stays out."""
    from zrth.lean.infer_route import InferInput, ProjectHandle, _ruled_out

    store = store_for(tmp_path, "m_countdown", kind="buchi", prp="(= s0 0)")
    store.note("proved empty\n", status="no_solution", what="x")
    store.note("ran out of budget\n", status="unknown", what="x")
    handle = ProjectHandle(
        dir=tmp_path,
        name="Rea",
        module=module_of("m_countdown"),
        ctx=None,
        reads=frozenset(),
        owns=(),
        resumes=frozenset({"note"}),
        artifacts=store,
    )
    known = _ruled_out(
        InferInput(
            project=handle,
            module=None,
            cert_data=CertificateData(prp="(= s0 0)", kind="buchi"),
            opts={},
            log=lambda *_: None,
        )
    )
    assert known == ("proved empty\n",)


def test_a_note_about_another_property_is_not_carried_over(tmp_path):
    """The store's staleness rule is what stops the handoff being a trap: a
    shape refuted for one property says nothing about another."""
    from zrth.lean.infer_route import InferInput, ProjectHandle, _ruled_out

    store = store_for(tmp_path, "m_countdown", kind="buchi", prp="(= s0 0)")
    store.note("proved empty\n", status="no_solution", what="x")
    later = store_for(tmp_path, "m_countdown", kind="buchi", prp="(= s0 7)")
    handle = ProjectHandle(
        dir=tmp_path,
        name="Rea",
        module=module_of("m_countdown"),
        ctx=None,
        reads=frozenset(),
        owns=(),
        resumes=frozenset({"note"}),
        artifacts=later,
    )
    assert _ruled_out(
        InferInput(
            project=handle,
            module=None,
            cert_data=CertificateData(prp="(= s0 7)", kind="buchi"),
            opts={},
            log=lambda *_: None,
        )
    ) == ()


# ══════════════════════════════════════════════════════════════════════════
# What the searches return, put back to the obligations
# ══════════════════════════════════════════════════════════════════════════


def obligations_hold(name: str, cd: CertificateData) -> list[str]:
    """The obligation names `smt_query` does not certify for `cd`.

    A second opinion, and deliberately not the query that produced the
    answer: `check_obligations` states `init_inv`, `step_inv` and
    `inv_imp_P` the way `--pre-check cvc5` does, from the predicates alone.
    """
    from zrth.lean.smt_query import ModuleQueries, SmtBudget, Status

    q = ModuleQueries.build(module_of(name), cd)
    assert q is not None, "the predicates should be readable for this module"
    return [
        v.name
        for v in q.check_obligations(SmtBudget())
        if v.status is not Status.HOLDS
    ]


def test_a_synthesised_invariant_passes_the_obligations_cvc5_states():
    from zrth.lean.magic_sygus import TA2MagicSygus

    cd = CertificateData(prp="(not (= s0 1))", kind="safety")
    out = TA2MagicSygus(module_of("m_step2"), log=lambda *_: None).infer(cd)
    assert obligations_hold("m_step2", out) == []


def test_sygus_reads_a_module_with_an_external_input():
    """`pre` and `trans` are predicates on the state alone, so an input is
    existentially quantified inside them -- `s` is initial when *some* input
    makes `init` produce it. If that were the wrong quantifier the invariant
    would come back and `step_inv` would then refute it, which is what this
    checks rather than merely that something was returned."""
    from zrth.lean.magic_sygus import TA2MagicSygus

    cd = CertificateData(prp="(>= s0 0)", kind="safety")
    out = TA2MagicSygus(module_of("m_relu_input"), log=lambda *_: None).infer(cd)
    assert out.inv_smt
    assert obligations_hold("m_relu_input", out) == []


def test_a_linear_certificate_passes_the_obligations_too():
    from zrth.lean.magic_linear import TA2MagicLinear

    cd = CertificateData(
        prp="(= s0 0)", kind="buchi", inv="(and (>= s0 0) (<= s0 100))"
    )
    out = TA2MagicLinear(module_of("m_countdown"), log=lambda *_: None).infer(cd)
    assert obligations_hold("m_countdown", out) == []


def test_sygus_writes_a_proof_of_absence_when_its_space_is_finite_and_empty(tmp_path):
    """The bound is what makes the failure worth writing down.

    `--sygus-conjuncts` makes the grammar a finite space, so cvc5 answers
    `hasNoSolution` rather than `unknown` -- and only then may the note say
    the space is empty. Without congruences `m_step2` has no invariant in
    it, which is the case that shows both halves at once.
    """
    from zrth.lean.magic_sygus import TA2MagicSygus

    store = store_for(tmp_path, "m_step2", kind="safety", prp="(not (= s0 1))",
                      producer="sygus")
    cd = CertificateData(prp="(not (= s0 1))", kind="safety")
    with pytest.raises(Refused):
        TA2MagicSygus(
            module_of("m_step2"), grammar="linear", artifacts=store,
            log=lambda *_: None,
        ).infer(cd)
    notes = store.usable("note", languages=("md",), statuses=("no_solution",))
    assert len(notes) == 1
    text = store.read(notes[0])
    assert "proof that the space is empty" in text
    assert "the grammar is finite" in text


def test_a_bound_the_program_mentions_is_seeded_the_other_way_round():
    """`m_countdown` counts down from 100 and its invariant is `s0 <= 100`.

    The grammar's atoms are `c0 + c1*s0 + ... <= 0`, so that invariant is
    spelled `-100 + s0 <= 0` -- and a constant set carrying `100` without
    `-100` cannot say it. Before the negations were seeded this came back as
    a *proof* that the space was empty, which is the worst way for a seeding
    gap to show up: not a missing answer but a confident wrong one.
    """
    from zrth.lean.magic_sygus import TA2MagicSygus

    cd = CertificateData(prp="(<= s0 100)", kind="safety")
    out = TA2MagicSygus(module_of("m_countdown"), log=lambda *_: None).infer(cd)
    assert obligations_hold("m_countdown", out) == []


def test_smt_linear_adds_to_an_invariant_another_run_left(tmp_path):
    """The workspace in the other direction: `--infer sygus` writes an `inv`,
    and a later `--infer smt-linear --safety` starts from it.

    `too_weak` is the case that matters -- inductive and not yet implying
    the property is exactly what "find the missing conjuncts" wants, and it
    is what `_resume_inv` (which takes an invariant as *given*) refuses.
    """
    from zrth.lean.infer_route import (
        InferInput,
        ProjectHandle,
        _resume_inv_to_strengthen,
        _take_inv,
    )

    store = store_for(tmp_path, "m_countdown", kind="safety", prp="(<= s0 100)")
    store.put("inv", "(>= s0 0)", status="too_weak", language="smt",
              what="x", why="does not imply the property")
    handle = ProjectHandle(
        dir=tmp_path,
        name="Rea",
        module=module_of("m_countdown"),
        ctx=None,
        reads=frozenset(),
        owns=(),
        resumes=frozenset({"inv"}),
        artifacts=store,
    )
    inp = InferInput(
        project=handle,
        module=None,
        cert_data=CertificateData(prp="(<= s0 100)", kind="safety"),
        opts={},
        log=lambda *_: None,
    )
    _take_inv(inp, _resume_inv_to_strengthen(inp), "adding to it")
    assert inp.cert_data.inv == "(>= s0 0)"

    out = TA2MagicLinearFor(inp.cert_data)
    assert out.inv_smt.startswith("(and (>= s0 0)"), "kept, and added to"
    assert obligations_hold("m_countdown", out) == []


def TA2MagicLinearFor(cd: CertificateData) -> CertificateData:
    from zrth.lean.magic_linear import TA2MagicLinear

    return TA2MagicLinear(module_of("m_countdown"), log=lambda *_: None).infer(cd)


def test_an_explicit_invariant_is_not_overridden_by_a_resumed_one(tmp_path):
    """The rule `_resume_inv` already states, on the other resume path: what
    the user passed wins, or two identical command lines mean two things."""
    from zrth.lean.infer_route import InferInput, ProjectHandle, _resume_inv_to_strengthen

    store = store_for(tmp_path, "m_countdown", kind="safety", prp="(<= s0 100)")
    store.put("inv", "(>= s0 0)", status="proved", language="smt", what="x")
    handle = ProjectHandle(
        dir=tmp_path, name="Rea", module=module_of("m_countdown"), ctx=None,
        reads=frozenset(), owns=(), resumes=frozenset({"inv"}), artifacts=store,
    )
    inp = InferInput(
        project=handle,
        module=None,
        cert_data=CertificateData(prp="(<= s0 100)", kind="safety",
                                  inv="(<= s0 50)"),
        opts={},
        log=lambda *_: None,
    )
    assert _resume_inv_to_strengthen(inp) is None


# ══════════════════════════════════════════════════════════════════════════
# The sorts a template can weigh
# ══════════════════════════════════════════════════════════════════════════


def bv_counter(width: int, top: int):
    """A BitVec counter: 0, 1, … top, then back to 0.

    Built here rather than in `tests/limits/mods` because it is this file's
    fixture and the limit matrix's cases carry a baseline.
    """
    import torch
    from zrth import BV, BitVec, Module, Term, Var, Wire, X

    s = Var(BitVec(width, [1, 1]))
    one, cap, zero = (Wire(BitVec(width, [1, 1])) for _ in range(3))
    at_top = Wire(BitVec(1, [1, 1]))     # BV has no Bool wires: one bit
    bumped = Wire(BitVec(width, [1, 1]))
    return Module.sequential(
        [s],
        [Term(BV.Const(torch.tensor([[0]])), [X(s)])],
        [
            Term(BV.Const(torch.tensor([[1]])), [one]),
            Term(BV.Const(torch.tensor([[top]])), [cap]),
            Term(BV.Const(torch.tensor([[0]])), [zero]),
            Term(BV.Eq(), [at_top], [s, cap]),
            Term(BV.Add(), [bumped], [s, one]),
            Term(BV.Ite(), [X(s)], [at_top, zero, bumped]),
        ],
    )


def obligations_of(module, cd: CertificateData) -> list[str]:
    from zrth.lean.smt_query import ModuleQueries, SmtBudget, Status

    q = ModuleQueries.build(module, cd)
    assert q is not None
    return [v.name for v in q.check_obligations(SmtBudget())
            if v.status is not Status.HOLDS]


def test_a_bool_component_is_weighed_as_zero_or_one():
    """`m_boolint` has a Bool beside its Int, which `--infer nuterm` refuses.

    The Bool column enters the template as `(ite s0 1 0)`, so a row can say
    `b`, `¬b`, or a bound that mixes the flag with the counter -- and the
    invariant that comes back is put to the obligations like any other.
    """
    from zrth.lean.magic_linear import TA2MagicLinear

    module = module_of("m_boolint")
    cd = CertificateData(prp="(and (>= s1 0) (<= s1 5))", kind="safety")
    out = TA2MagicLinear(module, log=lambda *_: None).infer(cd)
    assert "(ite s0 1 0)" in out.inv_smt
    assert obligations_of(module, out) == []


def test_a_bitvector_component_is_weighed_as_its_unsigned_value():
    """And the quantified query cannot be used for it, so the loop is.

    cvc5 answers the `exists c. forall s.` form `unknown (INCOMPLETE)` in a
    millisecond once a bitvector is in it -- not a timeout, a refusal to
    state it. The counterexample loop asks the same question with both
    halves quantifier-free, and this checks that the route reaches the same
    place by that road: a certificate the obligations accept.
    """
    from zrth.lean.magic_linear import TA2MagicLinear

    module = bv_counter(8, 10)
    said: list[str] = []
    cd = CertificateData(prp="(<= (ubv_to_int s0) 10)", kind="safety")
    out = TA2MagicLinear(module, log=said.append).infer(cd)
    assert "ubv_to_int s0" in out.inv_smt
    assert obligations_of(module, out) == []
    assert any("counterexample" in line for line in said), (
        "a bitvector column should fall back to the loop"
    )


def test_an_integer_state_does_not_pay_for_the_fallback():
    """The loop is a fallback, not the engine: an integer column is decided
    by the one quantified query, which is the stronger answer as well as the
    faster one -- it is about every integer coefficient, not a bounded box."""
    from zrth.lean.magic_linear import TA2MagicLinear

    said: list[str] = []
    cd = CertificateData(prp="(<= s0 100)", kind="safety")
    TA2MagicLinear(module_of("m_countdown"), log=said.append).infer(cd)
    assert not any("counterexample" in line for line in said)


def test_a_bounded_refutation_says_it_is_bounded(tmp_path):
    """What the loop proves is narrower than what the query proves, and the
    note has to carry the difference: its `unsat` is about the coefficients
    it was allowed to try, not about every integer."""
    from zrth.lean.artifacts import ArtifactStore, module_digest
    from zrth.lean.magic_linear import TA2MagicLinear

    module = bv_counter(4, 10)
    store = ArtifactStore(dir=tmp_path / "artifacts",
                          module_digest=module_digest(module),
                          kind="safety", prp="(<= (ubv_to_int s0) 5)",
                          producer="smt-linear")
    # False: the counter reaches 10, so no invariant implies `s0 <= 5`.
    cd = CertificateData(prp="(<= (ubv_to_int s0) 5)", kind="safety")
    with pytest.raises(Refused):
        TA2MagicLinear(module, rows=1, artifacts=store,
                       log=lambda *_: None).infer(cd)
    notes = store.usable("note", languages=("md",), statuses=("no_solution",))
    if notes:                       # the loop decided it: then say how
        text = store.read(notes[0])
        assert "coefficients were searched in" in text
        assert "counterexample" in text
