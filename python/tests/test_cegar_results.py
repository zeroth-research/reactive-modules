"""`_run_query` must survive every answer cvc5 can give.

The driver read model values whenever the result was not UNSAT, but
`unknown` carries no model, so `getValue` raised and took the whole CEGAR
run down. Reproduced here with a stub solver rather than by provoking a real
timeout, so the test is fast and deterministic.
"""

import cvc5
import pytest

from zrth.lean import magic_cegar
from zrth.lean.magic_cegar import TA2MagicCEGAR


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
