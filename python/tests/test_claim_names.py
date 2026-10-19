"""Claims in the reactive module's own terms: `zrth.lean.inputs.names`."""

import shlex
import subprocess
from pathlib import Path

import pytest

from zrth.lean.inputs import ClaimError, Names, load, to_internal, to_surface

FIXTURES = Path(__file__).parent / "fixtures"
PKG_ROOT = Path(__file__).parent.parent

NAMED = Names(("x", None), ("u",))
POSITIONAL = Names((None, None), (None,))


@pytest.mark.parametrize("surface, internal", [
    ("(>= (X x) 0)", "(>= s0 0)"),
    ("(>= (X @intf.1) (X  |x| ))", "(>= s1 s0)"),
    ("(and u (X u) (not @extl.0))", "(and el0 e0 (not el0))"),
    ("(ite (= (X x) 0) 0 (X x))", "(ite (= s0 0) 0 s0)"),
    ("(- 1)", "(- 1)"),
    ("true", "true"),
    # The Python DSL: `X(v)`, and the text around the names kept as written.
    ("X(x)[0][0] >= 0 and X(@intf.1)[0][0] < 3", "s0[0][0] >= 0 and s1[0][0] < 3"),
    ("And(X(x)[0][0] >= 0, u)", "And(s0[0][0] >= 0, el0)"),
])
def test_a_claim_reads_in_internal_names(surface, internal):
    assert to_internal(surface, NAMED, "--safety") == internal


@pytest.mark.parametrize("surface, why", [
    ("(>= x 0)", "interface variable 0's latched value"),
    ("(>= s0 0)", r"positional names are `\(X @intf.0\)` for `s0`"),
    ("(= el0 1)", "`@extl.0` for `el0`"),
    ("(X (+ x 1))", r"`X` applies to a variable: `\(X v\)`"),
    ("X(x+1)", r"`X` applies to a variable: `X\(v\)`"),
    ("(>= (X @intf.5) 0)", r"the module has 2 interface variable\(s\)"),
    ("(X y)", "`y` is not a variable"),
    ("(>= @foo 1)", "`@foo` is not a positional name"),
])
def test_a_claim_that_names_nothing_is_refused(surface, why):
    with pytest.raises(ClaimError, match=r"^--safety: .*" + why):
        to_internal(surface, NAMED, "--safety")


def test_an_operator_is_never_read_as_a_variable():
    names = Names(("and", None), ())
    assert to_internal("(and (X and) (X @intf.1))", names, "--safety") == \
        "(and s0 s1)"


@pytest.mark.parametrize("internal, named, positional", [
    ("(and (>= s0 0) (<= s1 el0) e0)",
     "(and (>= (X x) 0) (<= (X @intf.1) u) (X u))",
     "(and (>= (X @intf.0) 0) (<= (X @intf.1) @extl.0) (X @extl.0))"),
    ("s0[0][0] >= 0 and el0[0][0] == 1",
     "X(x)[0][0] >= 0 and u[0][0] == 1",
     "X(@intf.0)[0][0] >= 0 and @extl.0[0][0] == 1"),
])
def test_a_predicate_is_shown_in_surface_names(internal, named, positional):
    assert to_surface(internal, NAMED) == named
    assert to_surface(internal) == positional
    assert to_internal(named, NAMED, "rt") == internal
    assert to_internal(positional, POSITIONAL, "rt") == internal


@pytest.mark.parametrize("bad", [("X",), ("@a",), ("a b",), ("x", "x")])
def test_a_name_a_claim_could_not_use_is_refused(bad):
    with pytest.raises(ClaimError):
        Names(bad, ())


def test_a_named_python_module_carries_its_names():
    problem = load(FIXTURES / "twobit_named.py")
    assert problem.names == Names(("lo", "hi"), ("enable",))
    assert load(FIXTURES / "twobit_lia.py").names == Names((None, None), (None,))


def _verith(*args):
    return subprocess.run(["uv", "run", "verith", *args], capture_output=True,
                          text=True, cwd=PKG_ROOT)


def test_a_claim_in_names_is_the_claim_by_position(tmp_path):
    """The names are spelling only: the project is byte for byte the one
    the positional claim generates."""
    named = _verith(str(FIXTURES / "twobit_named.py"),
                    "--buchi", "(and (not (X lo)) (not (X hi)))",
                    "--pre", "(or (X enable) enable)",
                    "--invariant", "true",
                    "--cert-file", str(tmp_path / "named.lean"))
    positional = _verith(str(FIXTURES / "twobit_lia.py"),
                         "--buchi", "(and (not (X @intf.0)) (not (X @intf.1)))",
                         "--pre", "(or (X @extl.0) @extl.0)",
                         "--invariant", "true",
                         "--cert-file", str(tmp_path / "positional.lean"))
    assert named.returncode == 0, named.stderr
    assert positional.returncode == 0, positional.stderr
    assert (tmp_path / "named.lean").read_bytes() == \
        (tmp_path / "positional.lean").read_bytes()
    assert ".. --buchi, in internal names: (and (not s0) (not s1))" in named.stdout
    assert ".. --pre, in internal names: (or e0 el0)" in named.stdout


def test_a_bad_claim_or_name_is_an_error_line(tmp_path):
    for args in (
        (str(FIXTURES / "twobit_lia.py"), "--buchi", "(not s0)"),
        (str(FIXTURES / "twobit_named.py"), "--buchi", "(not lo)"),
        (str(FIXTURES / "twobit_named.py"), "-d", "stray", "--buchi", "true"),
    ):
        r = _verith(*args, "-o", str(tmp_path))
        assert r.returncode != 0, args
        assert "Traceback" not in r.stderr, r.stderr
        assert "error:" in r.stderr, r.stderr


@pytest.mark.parametrize("internal, surface", [
    ("s0", "(X @intf.0)"),
    ("(not el0)", "(not @extl.0)"),
    ("(s0 >= 0)", "(X(@intf.0) >= 0)"),
])
def test_text_both_dialects_read_is_shown_as_smtlib(internal, surface):
    assert to_surface(internal) == surface


def test_a_found_certificate_is_shown_as_the_flags_take_it(tmp_path):
    """What one run found, the next can be given: printed in the input's
    names, quoted for a shell."""
    r = _verith(str(FIXTURES / "counter.py"), "--buchi", "(= (X @intf.0) 0)",
                "--infer", "houdini", "-o", str(tmp_path))
    assert r.returncode == 0, r.stderr
    shown = r.stdout.split(".. The certificate, in the input's names:\n")[1]
    flags = [shlex.split(line) for line in shown.splitlines()[:2]]
    assert [f for f, _ in flags] == ["--invariant", "--ranking"]
    again = _verith(str(FIXTURES / "counter.py"), "--buchi", "(= (X @intf.0) 0)",
                    *(word for pair in flags for word in pair),
                    "--pre-check", "cvc5", "--cert-file", str(tmp_path / "C.lean"))
    assert again.returncode == 0, again.stderr
