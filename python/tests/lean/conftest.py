"""
Session-scoped fixtures that prepare the tests/lean/ lake project before any
test in this directory runs:

1. ``sync_core_templates`` — copies template files into Core/ (same as before).
2. ``generate_lean_files`` — generates:
     • ZerothHammer.lean  (the zeroth_hammer tactic, standalone)
     • Certs/Countdown.lean  (self-contained certificate, inline module)
     • Certs/TwoVars.lean
     • Certs/Collatz.lean

The generated files import ZerothHammer for the tactic and define their own
module-specific ``simp_mat`` / ``simp_defs`` / ``mat_collapse`` macros.
"""
import shutil
from pathlib import Path

import pytest

from zrth import DType as dt
from zrth import Module, Wire
from zrth.analyzer import convert_method
from zrth.lean.cert import CertificateData, generate_zeroth_hammer_lean, smt_predicates_to_lean
from zrth.lean.project import CORE_FILES, TEMPLATE_DIR, generate_standalone_cert_lean

_LEAN_DIR = Path(__file__).parent
_CORE_DIR = _LEAN_DIR / "Core"
_CERTS_DIR = _LEAN_DIR / "Certs"


# ──────────────────────────────────────────────────────────────
# Module factories (same modules as test_lean_svcomp.py)
# ──────────────────────────────────────────────────────────────


def _make_countdown() -> Module:
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


def _make_twovars() -> Module:
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


def _make_collatz() -> Module:
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


_CERT_SPECS = [
    (
        "Countdown",
        _make_countdown,
        CertificateData(
            prp="(= s0 0)",
            inv="(and (>= s0 0) (<= s0 100))",
            ranking="(ite (= s0 0) 0 s0)",
        ),
    ),
    (
        "TwoVars",
        _make_twovars,
        CertificateData(
            prp="(= s0 s1)",
            inv="(and (>= s0 0) (<= s0 s1) (= s1 10))",
            ranking="(ite (= s0 s1) 0 (- s1 s0))",
        ),
    ),
    (
        "Collatz",
        _make_collatz,
        CertificateData(
            prp="(= s0 1)",
            inv="(and (>= s0 1) (<= s0 8))",
            ranking="(ite (= s0 1) 0 (- s0 1))",
        ),
    ),
]


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


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


@pytest.fixture(scope="session", autouse=True)
def generate_lean_files(sync_core_templates) -> None:
    """Generate ZerothHammer.lean and Certs/*.lean for the lake project."""
    # ZerothHammer.lean — standalone tactic
    hammer_file = _LEAN_DIR / "ZerothHammer.lean"
    hammer_file.write_text(generate_zeroth_hammer_lean())

    # Certs/ — one self-contained certificate per module + scalar encoding file
    _CERTS_DIR.mkdir(exist_ok=True)
    for name, make_module, cert_data in _CERT_SPECS:
        module = make_module()
        lean_cert = smt_predicates_to_lean(cert_data, module)
        content = generate_standalone_cert_lean(
            name,
            module,
            lean_cert,
            hammer_import="ZerothHammer",
        )
        (_CERTS_DIR / f"{name}.lean").write_text(content)
