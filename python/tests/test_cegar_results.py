"""`_run_query` must survive every answer cvc5 can give.

The driver read model values whenever the result was not UNSAT, but
`unknown` carries no model, so `getValue` raised and took the whole CEGAR
run down. Reproduced here with a stub solver rather than by provoking a real
timeout, so the test is fast and deterministic.

The second half drives the whole loop against a scripted LLM: real cvc5,
real obligations, no network.
"""

import cvc5

from zrth import Module, Int, LIA, Var, X
from zrth.analyzer import convert_method
from zrth.lean import magic_cegar
from zrth.lean.cert import CertificateData
from zrth.lean.magic_cegar import TA2MagicCEGAR
from zrth.lean.smt_prompt import CEGAR_GENERATE_SYSTEM, CEGAR_SAFETY_SYSTEM


class _Result:
    def __init__(self, kind):
        self._kind = kind

    def isUnsat(self):
        return self._kind == "unsat"

    def isSat(self):
        return self._kind == "sat"

    def isUnknown(self):
        return self._kind == "unknown"

    def getUnknownExplanation(self):
        assert self._kind == "unknown", "only an unknown result has an explanation"
        return "INCOMPLETE"


class _Solver:
    """Enough of cvc5.Solver for `_run_query`, with a scripted answer."""

    def __init__(self, kind):
        self._kind = kind
        self.get_value_calls = 0

    def setLogic(self, _):
        pass

    def setOption(self, *_):
        pass

    def assertFormula(self, _):
        pass

    def checkSat(self):
        return _Result(self._kind)

    def getValue(self, term):
        self.get_value_calls += 1
        if self._kind != "sat":
            # what cvc5 actually raises when there is no model (verified
            # against a real solver: a plain RuntimeError)
            raise RuntimeError("cannot get value unless satisfiable")
        return term


class _Env:
    def __init__(self):
        self.tm = cvc5.TermManager()


def _run(monkeypatch, kind):
    """Call `_run_query` against a solver scripted to answer `kind`."""
    env = _Env()
    solver = _Solver(kind)
    monkeypatch.setattr(magic_cegar.cvc5, "Solver", lambda _tm: solver)
    var = env.tm.mkConst(env.tm.getIntegerSort(), "s0")
    query = env.tm.mkTerm(cvc5.Kind.EQUAL, var, env.tm.mkInteger(0))
    result = TA2MagicCEGAR._run_query(
        None, "inv-step", query, env, [("s", [var])]
    )
    return result, solver


def test_unsat_means_the_obligation_holds(monkeypatch):
    result, solver = _run(monkeypatch, "unsat")
    assert result.ok is True
    assert result.counterexample is None
    assert solver.get_value_calls == 0


def test_sat_reports_a_counterexample(monkeypatch):
    result, solver = _run(monkeypatch, "sat")
    assert result.ok is False
    assert "Counterexample" in result.counterexample
    assert solver.get_value_calls == 1, "the model should be read exactly once"


def test_unknown_reports_instead_of_crashing(monkeypatch):
    """The regression: `unknown` fell into the model-reading path."""
    result, solver = _run(monkeypatch, "unknown")
    assert result.ok is False, "unknown must not be reported as proved"
    assert "unknown" in result.counterexample
    assert "INCOMPLETE" in result.counterexample, "the solver's reason is dropped"
    assert solver.get_value_calls == 0, "no model exists; getValue must not be called"


def test_unknown_is_distinguishable_from_a_real_counterexample(monkeypatch):
    """CEGAR feedback must not present 'unknown' as a refutation."""
    unknown, _ = _run(monkeypatch, "unknown")
    sat, _ = _run(monkeypatch, "sat")
    assert "Counterexample" not in unknown.counterexample
    assert "unknown" not in sat.counterexample


# ══════════════════════════════════════════════════════════════════════
# The loop, against a scripted LLM
# ══════════════════════════════════════════════════════════════════════


def _counter() -> Module:
    """`x = 0`, `+1` each step, back to 0 at 10 -- so `x` stays in [0, 9]."""

    def init():
        return 0

    def update(old_x):
        x = old_x + 1
        if x == 10:
            return 0
        return x

    s = Var(Int([1, 1]))
    return Module.sequential(
        [s],
        convert_method(init, {}, [X(s)], theory=LIA),
        convert_method(update, {"old_x": s}, [X(s)], theory=LIA),
    )


def _infer(monkeypatch, cd, replies):
    """Run the real CEGAR loop, answering each prompt from `replies`.

    Returns `(cert_data, prompts)` -- the prompts being `(system, user)`
    pairs, so a test can say which prompt the route actually sent.
    """
    prompts: list[tuple[str, str]] = []
    pending = list(replies)

    def chat(system, user):
        prompts.append((system, user))
        return pending.pop(0)

    monkeypatch.setattr(magic_cegar, "_make_client", lambda base_url, model: chat)
    magic = TA2MagicCEGAR("", _counter())
    return magic.infer(cd), prompts


def test_safety_inference_asks_for_an_invariant_alone(monkeypatch):
    """`rule_globally` takes no ranking function, so none is requested, none
    is parsed, and none ends up on the certificate."""
    cd, prompts = _infer(
        monkeypatch,
        CertificateData(kind="safety", prp="(<= s0 9)"),
        ["INVARIANT: (and (>= s0 0) (<= s0 9))"],
    )
    assert prompts[0][0] is CEGAR_SAFETY_SYSTEM
    assert "RANKING" not in prompts[0][1]
    assert cd.inv is not None
    assert cd.ranking is None
    # The SMT source is kept beside the Lean, for `--pre-check`.
    assert cd.inv_smt == "(and (>= s0 0) (<= s0 9))"


def test_an_invariant_too_weak_for_the_property_comes_back_as_feedback(monkeypatch):
    """`(>= s0 0)` is inductive and true at init, and a Buchi certificate
    would accept it. It does not imply `s0 <= 9`, which is the obligation
    that only exists on the safety route."""
    cd, prompts = _infer(
        monkeypatch,
        CertificateData(kind="safety", prp="(<= s0 9)"),
        [
            "INVARIANT: (>= s0 0)",
            "INVARIANT: (and (>= s0 0) (<= s0 9))",
        ],
    )
    assert len(prompts) == 2, "the first invariant should have been rejected"
    assert "inv_imp_P" in prompts[1][1]
    assert cd.inv_smt == "(and (>= s0 0) (<= s0 9))"


def test_a_buchi_property_still_asks_for_both(monkeypatch):
    cd, prompts = _infer(
        monkeypatch,
        CertificateData(prp="(= s0 0)"),
        ["INVARIANT: (and (>= s0 0) (<= s0 9))\nRANKING: (ite (= s0 0) 0 (- 10 s0))"],
    )
    assert prompts[0][0] is CEGAR_GENERATE_SYSTEM
    assert cd.inv is not None and cd.ranking is not None
    assert cd.ranking_smt == "(ite (= s0 0) 0 (- 10 s0))"
