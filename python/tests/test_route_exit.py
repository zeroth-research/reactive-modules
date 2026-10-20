"""A process that ran every route exits 0, and a bare import loads no cvc5.

The cvc5 bindings segfault at interpreter shutdown when a `TermManager` is
finalised out of order with the solvers and terms made from it, which is
what `smt.module.keep_alive` is for. That failure is invisible in process:
the run has printed everything and the certificate is on disk, and only the
exit status says -11. So the routes are run one after another in one child
process, which then exits normally, and its status is the check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cvc5")

FIXTURES = Path(__file__).parent / "fixtures"

_DRIVER = r"""
import gc
import json
import sys

import zrth.lean.main as entry

print("cvc5 loaded by import:", "cvc5" in sys.modules, flush=True)

from zrth.lean.magic import ai, cegis

def client(base_url, model):
    def chat(system, user):
        if system is ai.VERIFY_SYSTEM:
            return "CORRECT"
        return ("INVARIANT: fun s => (s 0 0) ≥ 0 ∧ (s 0 0) ≤ 9\n"
                "RANKING: fun s => (if (s 0 0) = 0 then 0 else 10 - (s 0 0)).toNat")
    return chat

ai.make_client = client
cegis.make_client = client

for argv in json.loads(sys.argv[1]):
    sys.argv = ["verith", *argv]
    try:
        entry.main()
        print("== exit 0", flush=True)
    except SystemExit as e:
        print(f"== exit {e.code}", flush=True)
gc.collect()
"""

PRE = ["--pre-check", "cvc5", "--smt-tactics", "cvc5"]


def _cases(out: Path) -> list[list[str]]:
    from zrth.lean.errors import Refused
    from zrth.lean.smt.vampire import resolve_vampire

    counter = str(FIXTURES / "counter.py")
    countdown = str(FIXTURES / "svcomp_countdown.py")
    safe = ["--safety", "(<= (X @intf.0) 100)"]
    runs = [
        (counter, ["--buchi", "(= (X @intf.0) 0)", "--invariant",
                   "(and (>= (X @intf.0) 0) (<= (X @intf.0) 9))", "--ranking",
                   "(ite (= (X @intf.0) 0) 0 (- 10 (X @intf.0)))", *PRE]),
        (countdown, ["--buchi", "(= (X @intf.0) 0)", "--invariant",
                     "(and (>= (X @intf.0) 0) (<= (X @intf.0) 100))", "--infer", "smt-linear",
                     *PRE]),
        (countdown, [*safe, "--infer", "sygus", *PRE]),
        (countdown, [*safe, "--infer", "houdini", *PRE]),
        (countdown, ["--buchi", "(= (X @intf.0) 0)", "--infer", "nuterm", *PRE]),
        (counter, ["--buchi", "(= (X @intf.0) 0)", "--invariant",
                   "(and (>= (X @intf.0) 0) (<= (X @intf.0) 9))", "--ranking",
                   "(ite (= (X @intf.0) 0) 0 (- 10 (X @intf.0)))", "--infer", "ai-cegis", *PRE]),
        (counter, ["--buchi", "(= (X @intf.0) 0)", "--infer", "ai", *PRE]),
    ]
    try:
        resolve_vampire(None)
        runs += [
            (countdown, [*safe, "--infer", "vampire", "--vampire-timeout", "10",
                         *PRE]),
            (countdown, [*safe, "--infer", "houdini", "--houdini-solver",
                         "vampire", *PRE]),
        ]
    except Refused:
        pass
    return [[module, *args, "-o", str(out / str(i)), "-p", "G"]
            for i, (module, args) in enumerate(runs)]


def test_every_route_exits_cleanly(tmp_path):
    cases = _cases(tmp_path)
    r = subprocess.run([sys.executable, "-c", _DRIVER, json.dumps(cases)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (
        f"exit status {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    assert "cvc5 loaded by import: False" in r.stdout
    exits = [ln for ln in r.stdout.splitlines() if ln.startswith("== exit")]
    assert exits == ["== exit 0"] * len(cases), r.stdout[-3000:]


def test_a_predicate_refusal_is_shown_as_its_flag():
    """The layers below the CLI name a predicate by role; the refusal is
    read where the user can act on it, as the flag it came from."""
    from zrth.lean.errors import PredicateRefused

    assert str(PredicateRefused("property", "x", kind="safety")) == "--safety: x"
    assert str(PredicateRefused("property", "x")) == "--buchi: x"
    assert str(PredicateRefused("pre", "x")) == "--pre: x"
    assert str(PredicateRefused("branch condition", "x")) == "branch condition: x"
