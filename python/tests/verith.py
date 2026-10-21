"""`uv run verith`, run as a user runs it, for the tests that check what it
prints and writes."""

import subprocess
from pathlib import Path

# Root of the python package (where pyproject.toml lives).
PKG_ROOT = Path(__file__).parent.parent


def verith(*args) -> subprocess.CompletedProcess:
    """`uv run verith *args` from the package root, its output captured."""
    return subprocess.run(
        ["uv", "run", "verith", *args],
        capture_output=True,
        text=True,
        cwd=PKG_ROOT,
    )
