"""
Lean compilation tests for the tests/lean/ lake project.

Fast tests (default):
    just pytest tests/lean/            # checks generated files exist

Slow tests (require lake + Mathlib cache):
    just pytest tests/lean/ -m slow    # runs lake build

Targets verified by the slow tests:
  ZerothHammerTests  — individual tactic examples (manually written)
  ZerothHammer       — standalone zeroth_hammer tactic (generated)
  Certs              — self-contained module certificates (generated)
  ManualTests        — zeroth_hammer on standalone goal shapes (manually written)
"""
import subprocess
from pathlib import Path

import pytest

_LEAN_DIR = Path(__file__).parent


# ──────────────────────────────────────────────────────────────
# Fast tests — just verify file presence after fixture generation
# ──────────────────────────────────────────────────────────────


def test_core_files_present(sync_core_templates):
    """Core/ template files were copied successfully."""
    from zrth.lean.project import CORE_FILES

    for name in CORE_FILES:
        assert (_LEAN_DIR / "Core" / name).exists(), f"Core/{name} missing after sync"


def test_generated_files_present(generate_lean_files):
    """ZerothHammer.lean and Certs/*.lean exist after generation fixture runs."""
    assert (_LEAN_DIR / "ZerothHammer.lean").exists(), "ZerothHammer.lean not generated"
    for name in ("Countdown", "TwoVars", "Collatz"):
        path = _LEAN_DIR / "Certs" / f"{name}.lean"
        assert path.exists(), f"Certs/{name}.lean not generated"


# ──────────────────────────────────────────────────────────────
# Slow tests — full lake build
# ──────────────────────────────────────────────────────────────


def _lake_build(*targets: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["lake", "build", *targets],
        cwd=_LEAN_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.slow
def test_zeroth_hammer_proofs(generate_lean_files):
    """ZerothHammerTests.lean compiles — every example in the file type-checks."""
    r = _lake_build("ZerothHammerTests")
    assert r.returncode == 0, (
        f"lake build ZerothHammerTests failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert "sorry" not in r.stdout and "warning: declaration uses 'sorry'" not in r.stderr, (
        "Build succeeded but some proof used sorry."
    )


@pytest.mark.slow
def test_manual_tests_build(generate_lean_files):
    """ManualTests/Basic.lean compiles — zeroth_hammer closes each standalone example."""
    r = _lake_build("ManualTests")
    assert r.returncode == 0, (
        f"lake build ManualTests failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "ManualTests" in l]
    assert not sorry_lines, "ManualTests proof used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_playground_build(generate_lean_files):
    """Playground.lean compiles — proof-pattern experiments and regression tests."""
    r = _lake_build("Playground")
    assert r.returncode == 0, (
        f"lake build Playground failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Playground" in l]
    assert not sorry_lines, "Playground proof used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_hugecounter_build(generate_lean_files):
    """Certs/HugeCounter.lean: a 32-wide state whose transitions are 32-wide
    contractions. Pre-contraction keeps this tractable where a dense matrix
    literal + symbolic sum-expansion blows the heartbeat budget."""
    r = _lake_build("Certs.HugeCounter")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.HugeCounter failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, "HugeCounter certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_bigcounter_build(generate_lean_files):
    """Certs/BigCounter.lean: a 6-vector state whose every transition is a 6-wide
    MatMul contraction — exercises Fin.sum_univ_succ expansion at scale."""
    r = _lake_build("Certs.BigCounter")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.BigCounter failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, "BigCounter certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_counter_build(generate_lean_files):
    """Certs/Counter.lean: a 3×1 vector-state module with LIA.Linear transitions
    and tuple-select (s[i][j]) predicates — zeroth_hammer closes all obligations."""
    r = _lake_build("Certs.Counter")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.Counter failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, "Counter certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_countdown_build(generate_lean_files):
    """Certs/Countdown.lean: zeroth_hammer closes all proof obligations."""
    r = _lake_build("Certs.Countdown")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.Countdown failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert not sorry_lines, "Countdown certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_twovars_build(generate_lean_files):
    """Certs/TwoVars.lean: zeroth_hammer closes all proof obligations."""
    r = _lake_build("Certs.TwoVars")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.TwoVars failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert not sorry_lines, "TwoVars certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_collatz_build(generate_lean_files):
    """Certs/Collatz.lean: zeroth_hammer closes all proof obligations."""
    r = _lake_build("Certs.Collatz")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.Collatz failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert not sorry_lines, "Collatz certificate has sorry:\n" + "\n".join(sorry_lines)
