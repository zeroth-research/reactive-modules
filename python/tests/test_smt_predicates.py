"""`parse_predicate`: the Python-expression DSL is read, never `eval`-ed.

The source of a predicate is a CLI flag or a line of an LLM reply, so
whatever it says must not run as Python. The DSL is walked over a whitelist
instead, and anything outside it falls through to the SMT-LIB parser.
"""

import os
from pathlib import Path

import cvc5
import pytest

from zrth.lean.smt.module import ModuleSMT
from zrth.lean.inputs.python import load_module
from zrth.lean.smt.predicates import (
    PredicateParser,
    _NotPredicate,
    _pyeval,
    _pyeval_namespace,
    parse_predicate,
)

MODS = Path(__file__).parent / "limits" / "mods"


def _env(name: str) -> PredicateParser:
    """A parser context for one `tests/limits` fixture module."""
    module = load_module(MODS / f"{name}.py")
    return PredicateParser(ModuleSMT(tm=cvc5.TermManager(), module=module))


@pytest.mark.parametrize(
    "mod, python, smtlib",
    [
        ("m_countdown", "s[0][0] == 0", "(= s0 0)"),
        ("m_countdown", " s[0][0] == 0", "(= s0 0)"),
        ("m_countdown", "Ite(s[0][0] == 0, 0, s[0][0])", "(ite (= s0 0) 0 s0)"),
        (
            "m_boolint",
            "And(s1[0][0] >= 0, s1[0][0] <= 5, Implies(s0[0][0], s1[0][0] <= 5))",
            "(and (>= s1 0) (<= s1 5) (=> s0 (<= s1 5)))",
        ),
        (
            "m_relu_vec",
            "s[1, 0] + 1 > s[2][0]",
            "(> (+ ((_ tuple.select 1) s0) 1) ((_ tuple.select 2) s0))",
        ),
        # Python's own reading of these asks whether a term is truthy.
        ("m_countdown", "0 <= s[0][0] <= 5", "(and (>= s0 0) (<= s0 5))"),
        (
            "m_boolint",
            "not s0[0][0] and s1[0][0] == 0",
            "(and (not s0) (= s1 0))",
        ),
        ("m_boolint", "s1[0][0] if s0[0][0] else 0", "(ite s0 s1 0)"),
    ],
)
def test_the_python_dsl_reads_as_the_smtlib_it_means(mod, python, smtlib):
    env = _env(mod)
    assert parse_predicate(env, python) == parse_predicate(env, smtlib)


def test_a_reply_that_escapes_an_eval_sandbox_runs_nothing(monkeypatch):
    """Under `eval` with `{"__builtins__": {}}` this sets the variable: the
    class graph leads from `()` to `os`. Read as the DSL, it is not a
    predicate, and the SMT-LIB parser cannot read it either."""
    monkeypatch.delenv("ZRTH_RT11_ESCAPE", raising=False)
    src = (
        "[c for c in ().__class__.__base__.__subclasses__() "
        "if c.__name__ == '_wrap_close'][0].__init__.__globals__"
        "['environ'].__setitem__('ZRTH_RT11_ESCAPE', '1')"
    )
    with pytest.raises(RuntimeError):
        parse_predicate(_env("m_countdown"), src)
    assert "ZRTH_RT11_ESCAPE" not in os.environ


@pytest.mark.parametrize(
    "src",
    [
        "s0.t",                       # attribute access
        "__builtins__",               # dunder name
        "_E",                         # private name, and not bound
        "x",                          # unbound name
        "(lambda: s0)()",
        "[s0 for _ in (0,)][0]",
        "'s0'",
        "None",
        "2 ** 10",
        "+s0",
        "s0 / 2",
        "(x := s0)",
        "Ite(s0 == 0, 0, e=1)",
        "And(*(s0,))",
        "s0(0)",                      # a bound name that is not a connective
        "s[s0][0]",                   # an index that is not an int
    ],
)
def test_python_outside_the_dsl_is_not_read(src):
    ns = _pyeval_namespace(_env("m_countdown"))
    with pytest.raises((_NotPredicate, SyntaxError)):
        _pyeval(src, ns)


def test_a_bare_numeral_still_falls_through_to_smtlib():
    """`0` reads as a Python int, which is not a term; SMT-LIB makes it one."""
    term = parse_predicate(_env("m_countdown"), "0")
    assert term.getSort().isInteger()


@pytest.mark.parametrize(
    "src, why",
    [
        ("s1[0][0] + s0[0][0] > 0", "arithmetic"),       # Int + Bool
        ("s1[5][0] > 0", "out of bounds"),
        ("Ite(s0[0][0], 1)", "missing"),
    ],
)
def test_the_dsl_says_what_is_wrong_with_a_predicate_in_it(src, why):
    """Python syntax that is not a predicate of the module is refused in
    the DSL's terms, rather than handed to the SMT-LIB parser as text it
    was never written as."""
    with pytest.raises(RuntimeError, match=f"is predicate syntax.*{why}"):
        parse_predicate(_env("m_boolint"), src)


def test_a_fault_in_the_dsl_itself_is_not_a_parse_error(monkeypatch):
    """A bug raises as itself instead of reading as "could not parse"."""
    import zrth.lean.smt.predicates as predicates

    def broken(src, ns):
        raise AttributeError("a bug")

    monkeypatch.setattr(predicates, "_pyeval", broken)
    with pytest.raises(AttributeError, match="a bug"):
        parse_predicate(_env("m_countdown"), "s[0][0] == 0")

