"""An SMV model's own specifications as the claim."""

import subprocess
from pathlib import Path

import pytest

from zrth.lean import inputs
from zrth.smv import SMVRefused, parse_smv, read_smv

SMV = Path(__file__).parent / "smv_fixtures"
PKG_ROOT = Path(__file__).parent.parent
MOD10 = (SMV / "mod10.smv").read_text()


def _with(spec: str) -> str:
    return MOD10 + spec + "\n"


@pytest.mark.parametrize("spec, kind", [
    ("INVARSPEC x <= 9;", "safety"),
    ("LTLSPEC G x <= 9", "safety"),
    ("LTLSPEC G F x = 0;", "buchi"),
    ("LTLSPEC G (F (x = 0))", "buchi"),
    ("LTLSPEC NAME back := G F x = 0", "buchi"),
])
def test_a_spec_reads_as_a_claim(spec, kind):
    _, _, specs = read_smv(_with(spec))
    assert [s.kind for s in specs] == [kind]


@pytest.mark.parametrize("spec", [
    "LTLSPEC F x = 0",          # eventually, once: no proof rule here
    "LTLSPEC x = 0",            # the initial state only
    "LTLSPEC G X x = 0",
    "INVARSPEC inc",            # an input
])
def test_any_other_spec_is_refused(spec):
    with pytest.raises(SMVRefused):
        read_smv(_with(spec))


def test_specs_leave_the_module_alone():
    import re

    plain, _ = parse_smv(MOD10, strict=True)
    stated, _ = parse_smv(_with("INVARSPEC x <= 9;"), strict=True)
    ids = re.compile(r"#\d+")
    assert ids.sub("#N", str(plain)) == ids.sub("#N", str(stated))


def test_a_stated_claim_is_in_internal_names():
    problem = inputs.load(SMV / "mod10_specs.smv")
    assert [(c.kind, c.where, c.text) for c in problem.claims] == [
        ("safety", "line 14: INVARSPEC", "(and (<= 0 s0) (<= s0 9))"),
        ("buchi", "line 15: LTLSPEC", "(= s0 0)"),
    ]


def _verith(*args):
    return subprocess.run(["uv", "run", "verith", *args], capture_output=True,
                          text=True, cwd=PKG_ROOT)


def test_the_chosen_spec_is_the_claim(tmp_path):
    r = _verith(str(SMV / "mod10_specs.smv"), "--spec", "1",
                "--infer", "houdini", "--pre-check", "cvc5",
                "-o", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert ".. the claim is the input's line 14: INVARSPEC (--safety)" in r.stdout
    assert ".. --safety, in internal names: (and (<= 0 s0) (<= s0 9))" in r.stdout


@pytest.mark.parametrize("args, why", [
    ((), "states 2 properties"),
    (("--spec", "3"), "--spec 3: the input states 2 properties"),
    (("--safety", "(<= (X x) 9)"), "states its own property"),
    (("--spec", "1", "--safety", "(<= (X x) 9)"), "--safety/--buchi already"),
])
def test_an_ambiguous_claim_is_refused(tmp_path, args, why):
    r = _verith(str(SMV / "mod10_specs.smv"), *args, "-o", str(tmp_path))
    assert r.returncode != 0
    assert "Traceback" not in r.stderr, r.stderr
    assert why in r.stderr, r.stderr


def test_spec_is_only_meaningful_for_smv_input(tmp_path):
    r = _verith(str(Path(__file__).parent / "fixtures" / "counter.py"),
                "--spec", "1", "-o", str(tmp_path))
    assert r.returncode != 0
    assert "--spec is only meaningful for smv input" in r.stderr
