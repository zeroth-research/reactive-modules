"""Tests for the kernel-check driver (``benchmarks.svcomp._lean_check``).

The outcome taxonomy is what the runner's summary reports, so each class is
pinned against a file that actually provokes it from the Lean toolchain.
"""
import shutil

import pytest

from benchmarks.svcomp import _lean_check as lc

pytestmark = pytest.mark.skipif(shutil.which("lake") is None,
                                reason="no Lean toolchain")

_CASES = {
    "CHECKED": "theorem ok : (1 : Int) + 1 = 2 := by omega",
    "OMEGA": "theorem bad (x : Int) : x = 0 := by omega",
    "HEARTBEAT": ("set_option maxHeartbeats 1\n"
                  "theorem slow (x : Int) (h : x > 0) : x + 0 > 0 := by omega"),
    "ERROR": "theorem broken : (1 : Int) = 1 := by exact notATactic",
}


@pytest.fixture(scope="module", autouse=True)
def _substrate():
    lc.build_substrate()


def _write(name: str, body: str):
    d = lc.LEAN_DIR / "proofs" / f"_t_{name}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "program.lean"
    f.write_text(f"import Coverage\nnamespace Matrix\n{body}\nend Matrix\n")
    return f, d


@pytest.mark.parametrize("expected", list(_CASES))
def test_outcome_classification(expected):
    f, d = _write(expected.lower(), _CASES[expected])
    try:
        outcome, detail = lc.check_file(f)
        assert outcome == expected, f"got {outcome} ({detail})"
        assert (detail == "") == (expected == "CHECKED")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_timeout_is_reported():
    f, d = _write("timeout", _CASES["CHECKED"])
    try:
        outcome, detail = lc.check_file(f, timeout=1e-6)
        assert outcome == "TIMEOUT" and "exceeded" in detail
    finally:
        shutil.rmtree(d, ignore_errors=True)
