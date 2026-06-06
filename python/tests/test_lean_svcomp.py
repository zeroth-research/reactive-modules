"""SV-COMP style termination verification tests.

Each test creates a reactive module, provides a correct invariant and ranking
function, and generates a Lean 4 project. The zeroth_hammer should close
all proof obligations including hrank.

To verify proofs actually close (slow, requires lake + mathlib):
    .venv/bin/python -m pytest tests/test_lean_svcomp.py -m slow -v
"""

import subprocess
import pytest
from os.path import dirname
from pathlib import Path

from zrth import Wire, Module, DType as dt
from zrth.analyzer import convert_method
from zrth.lean.project import create_project, write_data_lean
from zrth.lean.cert import CertificateData, smt_predicates_to_lean

OUTPUT_DIR = Path(dirname(__file__)) / "LeanTests"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _gen(name, module, prp, inv, ranking, pre=None):
    """Generate a Lean project with certificate and return the project dir."""
    cert = CertificateData(prp=prp, inv=inv, ranking=ranking)
    if pre is not None:
        cert.init_pre = pre
        cert.update_pre = pre
    lean_cert = smt_predicates_to_lean(cert, module)
    project_dir = create_project(
        output_dir=OUTPUT_DIR,
        module=module,
        project_name=name,
    )
    write_data_lean(project_dir, name, module, lean_cert)
    return project_dir


def _lake_build(project_dir):
    """Run lake build Certificate and check for sorry/errors."""
    r = subprocess.run(
        ["lake", "update"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, f"lake update failed: {r.stderr[-300:]}"

    r = subprocess.run(
        ["lake", "build", "Certificate"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    sorry_lines = [
        l for l in r.stdout.split("\n") if "sorry" in l and "Certificate/" in l
    ]
    error_lines = [l for l in r.stdout.split("\n") if l.startswith("error:")]
    assert r.returncode == 0, r"lake build failed:\n" + "\n".join(error_lines[:5])
    assert len(sorry_lines) == 0, "Certificate has sorry:\n" + "\n".join(sorry_lines)


# ──────────────────────────────────────────────────────────────
# Test modules
# ──────────────────────────────────────────────────────────────


def _make_countdown():
    """x starts at 100, decrements to 0, resets."""

    def init():
        return 100

    def update(old_x):
        if old_x == 0:
            return 100
        return old_x - 1

    s = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    return Module.sequential(
        convert_method(init, {}, [s[1]]),
        convert_method(update, {"old_x": s}, [s[1]]),
        obs=[s],
    )


def _make_twovars():
    """x increments toward y, resets when equal."""

    def init():
        return 0, 10

    def update(old_x, old_y):
        if old_x < old_y:
            return old_x + 1, old_y
        return 0, 10

    x = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    y = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    return Module.sequential(
        convert_method(init, {}, [x[1], y[1]]),
        convert_method(update, {"old_x": x, "old_y": y}, [x[1], y[1]]),
        obs=[x, y],
    )


def _make_collatz_bounded():
    """x starts at 7, steps down by 3 or 1, resets at 1."""

    def init():
        return 7

    def update(old_x):
        if old_x == 1:
            return 7
        if old_x > 4:
            return old_x - 3
        if old_x > 1:
            return old_x - 1
        return old_x

    s = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    return Module.sequential(
        convert_method(init, {}, [s[1]]),
        convert_method(update, {"old_x": s}, [s[1]]),
        obs=[s],
    )


def _make_nested():
    """Nested i/j counters: j counts to 3, then i increments, both reset at (3,3)."""

    def init():
        return 0, 0

    def update(old_i, old_j):
        if old_j < 3:
            return old_i, old_j + 1
        if old_i < 3:
            return old_i + 1, 0
        return 0, 0

    i = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    j = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    return Module.sequential(
        convert_method(init, {}, [i[1], j[1]]),
        convert_method(update, {"old_i": i, "old_j": j}, [i[1], j[1]]),
        obs=[i, j],
    )


def _make_gcd():
    """Euclidean GCD: a and b subtract until equal, then reset."""

    def init():
        return 12, 8

    def update(old_a, old_b):
        if old_a > old_b:
            return old_a - old_b, old_b
        if old_b > old_a:
            return old_a, old_b - old_a
        return 12, 8

    a = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    b = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    return Module.sequential(
        convert_method(init, {}, [a[1], b[1]]),
        convert_method(update, {"old_a": a, "old_b": b}, [a[1], b[1]]),
        obs=[a, b],
    )


# ──────────────────────────────────────────────────────────────
# Generation tests (fast — just check project is created)
# ──────────────────────────────────────────────────────────────


def test_countdown_generates():
    m = _make_countdown()
    p = _gen(
        "SvcompCountdown",
        m,
        prp="(= s0 0)",
        inv="(and (>= s0 0) (<= s0 100))",
        ranking="(ite (= s0 0) 0 s0)",
    )
    assert (p / "Certificate" / "Certificate.lean").exists()


def test_twovars_generates():
    m = _make_twovars()
    p = _gen(
        "SvcompTwovars",
        m,
        prp="(= s0 s1)",
        inv="(and (>= s0 0) (<= s0 s1) (= s1 10))",
        ranking="(ite (= s0 s1) 0 (- s1 s0))",
    )
    assert (p / "Certificate" / "Certificate.lean").exists()


def test_collatz_bounded_generates():
    m = _make_collatz_bounded()
    p = _gen(
        "SvcompCollatz",
        m,
        prp="(= s0 1)",
        inv="(and (>= s0 1) (<= s0 8))",
        ranking="(ite (= s0 1) 0 (- s0 1))",
    )
    assert (p / "Certificate" / "Certificate.lean").exists()


def test_nested_generates():
    m = _make_nested()
    p = _gen(
        "SvcompNested",
        m,
        prp="(and (= s0 0) (= s1 0))",
        inv="(and (>= s0 0) (<= s0 3) (>= s1 0) (<= s1 3))",
        ranking="(ite (and (= s0 0) (= s1 0)) 0 (+ (* (- 3 s0) 4) (- 3 s1)))",
    )
    assert (p / "Certificate" / "Certificate.lean").exists()


def test_gcd_generates():
    m = _make_gcd()
    p = _gen(
        "SvcompGcd",
        m,
        prp="(= s0 s1)",
        inv="(and (>= s0 1) (>= s1 1))",
        ranking="(ite (= s0 s1) 0 (+ (- s0 1) (- s1 1)))",
    )
    assert (p / "Certificate" / "Certificate.lean").exists()


# ──────────────────────────────────────────────────────────────
# Proof verification tests (slow — require lake + mathlib)
# ──────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_countdown_proofs():
    m = _make_countdown()
    p = _gen(
        "SvcompCountdown",
        m,
        prp="(= s0 0)",
        inv="(and (>= s0 0) (<= s0 100))",
        ranking="(ite (= s0 0) 0 s0)",
    )
    _lake_build(p)


@pytest.mark.slow
def test_twovars_proofs():
    m = _make_twovars()
    p = _gen(
        "SvcompTwovars",
        m,
        prp="(= s0 s1)",
        inv="(and (>= s0 0) (<= s0 s1) (= s1 10))",
        ranking="(ite (= s0 s1) 0 (- s1 s0))",
    )
    _lake_build(p)


@pytest.mark.slow
def test_collatz_bounded_proofs():
    m = _make_collatz_bounded()
    p = _gen(
        "SvcompCollatz",
        m,
        prp="(= s0 1)",
        inv="(and (>= s0 1) (<= s0 8))",
        ranking="(ite (= s0 1) 0 (- s0 1))",
    )
    _lake_build(p)
