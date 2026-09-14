"""Generated Lean must not depend on the interpreter's string hash seed.

`visit_If` merged the two branch scopes through a `set` of variable-*name*
strings, so the `Ite` terms — and every `let` the backends emit from them —
came out in a per-process order.  The mismatch only shows up across
processes, so each case is generated under two different `PYTHONHASHSEED`s
and the digests compared.
"""

import json
import subprocess
import sys
from os.path import dirname
from pathlib import Path

import pytest

TESTS = Path(dirname(__file__))

# Every module whose update block writes two or more ctrl wires under one
# branch: the shape that made the merge order observable.
CASES = [
    ("limits/mods", "m_lex"),
    ("limits/mods", "m_nonlin"),
    ("limits/mods", "m_twovars"),
    ("fixtures", "svcomp_gcd"),
    ("fixtures", "svcomp_nested"),
    ("fixtures", "svcomp_twovars"),
]

ENCODINGS = ["to_lean_functional", "to_lean_scalar", "to_lean_circ", "to_lean_rel"]

_SCRIPT = """
import hashlib, importlib, json, sys
cases, encodings = json.loads(sys.argv[1])
sys.path[:0] = sorted({c[0] for c in cases})
from zrth.lean import ModuleToLean4

out = {}
for pkg, name in cases:
    module = importlib.import_module(name).module()
    for enc in encodings:
        lean = getattr(ModuleToLean4(module), enc)()
        out[name + ":" + enc] = hashlib.sha256(lean.encode()).hexdigest()
print(json.dumps(out))
"""


def _digests(seed):
    cases = [[str(TESTS / pkg), name] for pkg, name in CASES]
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, json.dumps([cases, ENCODINGS])],
        capture_output=True,
        text=True,
        cwd=TESTS,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def digests():
    """Two independent interpreters, two different string hash seeds."""
    return _digests("0"), _digests("1")


@pytest.mark.parametrize("enc", ENCODINGS)
@pytest.mark.parametrize("pkg,name", CASES, ids=[c[1] for c in CASES])
def test_generation_is_hash_seed_independent(digests, pkg, name, enc):
    seed0, seed1 = digests
    key = f"{name}:{enc}"
    assert seed0[key] == seed1[key]
