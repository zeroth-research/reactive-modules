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
import torch

from zrth import Sort as dt
from zrth import Module, Wire, Term, LIA, Int, Bool
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

    s = (Wire(dt.Int([1, 1])), Wire(dt.Int([1, 1])))
    return Module.sequential(
        convert_method(init, {}, [s[1]], theory=LIA),
        convert_method(update, {"old_x": s}, [s[1]], theory=LIA),
        obs=[s],
    )


def _make_twovars() -> Module:
    def init():
        return 0, 10

    def update(old_x, old_y):
        if old_x < old_y:
            return old_x + 1, old_y
        return 0, 10

    x = (Wire(dt.Int([1, 1])), Wire(dt.Int([1, 1])))
    y = (Wire(dt.Int([1, 1])), Wire(dt.Int([1, 1])))
    return Module.sequential(
        convert_method(init, {}, [x[1], y[1]], theory=LIA),
        convert_method(update, {"old_x": x, "old_y": y}, [x[1], y[1]], theory=LIA),
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

    s = (Wire(dt.Int([1, 1])), Wire(dt.Int([1, 1])))
    return Module.sequential(
        convert_method(init, {}, [s[1]], theory=LIA),
        convert_method(update, {"old_x": s}, [s[1]], theory=LIA),
        obs=[s],
    )


def _make_counter() -> Module:
    """3×1 vector-state counter with LIA.Linear transitions (matrix cert).

    state = (x, y, z); init = (0, y0, z0); update increments x while x < y or
    x < z, else resets x to 0. Exercises the Linear affine map plus tuple-select
    (`s[i][j]`) predicates end-to-end through zeroth_hammer.
    """
    state = (Wire(Int([3, 1])), Wire(Int([3, 1])))
    extl = (Wire(Int([2, 1])), Wire(Int([2, 1])))
    zero31 = torch.zeros((3, 1), dtype=torch.int64)
    zero11 = torch.zeros((1, 1), dtype=torch.int64)

    A = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    init = [Term(LIA.Linear(A, zero31), [state[1]], [extl[1]])]

    x, y, z = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))
    x_lt_y, x_lt_z, cond = Wire(Bool([1, 1])), Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    result_true, result_false = Wire(Int([3, 1])), Wire(Int([3, 1]))
    row_x = torch.tensor([[1, 0, 0]], dtype=torch.int64)
    row_y = torch.tensor([[0, 1, 0]], dtype=torch.int64)
    row_z = torch.tensor([[0, 0, 1]], dtype=torch.int64)
    e1 = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    diag_yz = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int64)
    update = [
        Term(LIA.Linear(row_x, zero11), [x], [state[0]]),
        Term(LIA.Linear(row_y, zero11), [y], [state[0]]),
        Term(LIA.Linear(row_z, zero11), [z], [state[0]]),
        Term(LIA.Lt(), [x_lt_y], [x, y]),
        Term(LIA.Lt(), [x_lt_z], [x, z]),
        Term(LIA.Or(), [cond], [x_lt_y, x_lt_z]),
        Term(LIA.Linear(torch.eye(3, dtype=torch.int64), e1), [result_true], [state[0]]),
        Term(LIA.Linear(diag_yz, zero31), [result_false], [state[0]]),
        Term(LIA.Ite(), [state[1]], [cond, result_true, result_false]),
    ]
    return Module.sequential(init, update, obs=[state, extl])


_CERT_SPECS = [
    (
        "Counter",
        _make_counter,
        CertificateData(
            prp="s[0][0] == 0",
            inv="And(s[0][0] >= 0, Or(s[0][0] <= s[1][0], s[0][0] <= s[2][0]))",
            init_pre="And(e[0][0] >= 0, e[1][0] >= 0)",
            ranking="Ite(s[0][0] == 0, 0, (Ite(s[1][0] >= s[2][0], s[1][0], s[2][0]) - s[0][0]) + 1)",
        ),
    ),
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
        src = TEMPLATE_DIR / "Core" / name
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
        content = generate_standalone_cert_lean(module, lean_cert)
        (_CERTS_DIR / f"{name}.lean").write_text(content)
