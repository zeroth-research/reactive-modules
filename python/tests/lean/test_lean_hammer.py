"""
Lean compilation test for ZerothHammerTests.lean.

The conftest.py in this directory syncs Core/ from the templates automatically
before this test runs, so the Lean project always sees the latest Core files.

Run fast (generation only):
    just pytest tests/lean/

Run slow (includes `lake build`, requires lake + Mathlib cache):
    just pytest tests/lean/ -m slow
"""
import subprocess
from pathlib import Path

import pytest

_LEAN_DIR = Path(__file__).parent


def test_core_files_present(sync_core_templates):
    """Core files were copied from templates — verify they exist."""
    from zrth.lean.project import CORE_FILES

    for name in CORE_FILES:
        assert (_LEAN_DIR / "Core" / name).exists(), f"Core/{name} missing after sync"


@pytest.mark.slow
def test_zeroth_hammer_proofs(sync_core_templates):
    """
    Compile ZerothHammerTests.lean with `lake build`.

    Verifies that every `example` in ZerothHammerTests.lean type-checks,
    confirming that each new zeroth_hammer phase handles its target goal shape.
    """
    r = subprocess.run(
        ["lake", "build", "ZerothHammerTests"],
        cwd=_LEAN_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, (
        f"lake build failed.\nstdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    # Fail if any proof fell through to sorry
    assert "sorry" not in r.stdout and "warning: declaration uses 'sorry'" not in r.stderr, (
        "Build succeeded but some proof used sorry — a phase may be incomplete."
    )
