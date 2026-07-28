"""Kernel-check emitted proofs with the Lean toolchain.

``certify`` emits ``program.lean`` under ``lean/proofs/<name>/`` from a
verification's Farkas certificates and compiles it against the vendored
substrate, returning one :class:`CheckResult`. The outcome separates the ways a
proof can fail to compile, because they call for different work:

  ``CHECKED``    the proof compiled: the program's termination is kernel-verified
  ``HEARTBEAT``  the elaborator hit its heartbeat budget (coverage ``omega`` cost)
  ``SCALE``      the elaborator hit its recursion depth (term too large)
  ``OMEGA``      a goal ``omega`` cannot prove — the coverage cells do not tile
  ``ERROR``      any other compile error
  ``TIMEOUT``    the compile exceeded the wall-clock budget
  ``UNVERIFIED`` the Farkas verifier produced no certificate, so nothing was emitted
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ._lean import write_program_proof

LEAN_DIR = Path(__file__).resolve().parent / "lean"

CHECK_TIMEOUT = 1200.0

# stdout marker -> outcome, most specific first
_MARKERS = (
    ("maximum number of heartbeats", "HEARTBEAT"),
    ("maximum recursion depth", "SCALE"),
    ("omega could not prove", "OMEGA"),
)


@dataclass
class CheckResult:
    name: str
    outcome: str
    n_paths: int = 0
    n_cells: int = 0
    n_invariants: int = 0
    detail: str = ""
    train_s: float = 0.0        # training, including the verification that accepted V
    check_s: float = 0.0        # emitting and compiling the proof

    @property
    def checked(self) -> bool:
        return self.outcome == "CHECKED"


def toolchain_available() -> bool:
    return shutil.which("lake") is not None


def build_substrate(timeout: float = 600.0) -> None:
    """Compile the substrate libraries, so per-proof checks only elaborate the
    proof itself. Raises ``RuntimeError`` if the build fails."""
    r = subprocess.run(["lake", "build"], cwd=LEAN_DIR,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"lake build failed:\n{r.stdout}{r.stderr}")


def check_file(path: Path, timeout: float = CHECK_TIMEOUT) -> tuple[str, str]:
    """Compile one emitted proof. Returns ``(outcome, first error line)``.

    ``-j1`` keeps Lean to one thread: a proof is one file, and the runner gets
    its throughput from checking several benchmarks at once, so threads inside a
    single elaboration only contend with the other benchmarks in flight."""
    cmd = ["lake", "env", "lean", "-j1", str(path.relative_to(LEAN_DIR))]
    try:
        r = subprocess.run(cmd, cwd=LEAN_DIR, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"exceeded {timeout:g}s"
    out = r.stdout + r.stderr
    if r.returncode == 0 and not r.stdout.strip():
        return "CHECKED", ""
    for marker, outcome in _MARKERS:
        if marker in out:
            return outcome, _first_error(out)
    return "ERROR", _first_error(out)


def _first_error(out: str) -> str:
    for line in out.splitlines():
        if "error" in line.lower():
            return line.strip()
    return out.strip().splitlines()[0] if out.strip() else ""


def certify(name: str, obligation, paths, timeout: float = CHECK_TIMEOUT) -> CheckResult:
    """Emit the proof for ``paths`` (the per-path Farkas certificates a
    ``farkas_cell`` run produced on ``obligation``) and compile it."""
    t0 = time.perf_counter()
    out = write_program_proof(name, obligation, paths, LEAN_DIR / "proofs")
    outcome, detail = check_file(out, timeout)
    return CheckResult(name, outcome, n_paths=len(paths),
                       n_cells=sum(len(p.cells) for p in paths),
                       n_invariants=len(obligation.invariants), detail=detail,
                       check_s=time.perf_counter() - t0)
