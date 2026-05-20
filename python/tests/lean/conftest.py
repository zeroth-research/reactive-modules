"""
Auto-sync Core/ from the canonical templates before any test in this directory runs.

The `TEMPLATE_DIR` and `CORE_FILES` constants live in `zrth.lean.project` — the
same source used when generating full Lean projects — so this directory always
reflects the current templates without manual maintenance.
"""
import shutil
from pathlib import Path

import pytest

from zrth.lean.project import CORE_FILES, TEMPLATE_DIR

_LEAN_DIR = Path(__file__).parent
_CORE_DIR = _LEAN_DIR / "Core"


@pytest.fixture(scope="session", autouse=True)
def sync_core_templates() -> None:
    """Copy the current template files into Core/ before the test session."""
    _CORE_DIR.mkdir(exist_ok=True)
    for name in CORE_FILES:
        src = TEMPLATE_DIR / name
        dst = _CORE_DIR / name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
