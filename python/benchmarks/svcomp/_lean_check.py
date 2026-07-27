"""Kernel-check emitted proofs with the Lean toolchain.

``certify(bench, layers)`` runs the Farkas verifier, emits ``program.lean`` under
``lean/proofs/<name>/`` and compiles it against the vendored substrate, returning
one :class:`CheckResult`. The outcome separates the ways a proof can fail to
compile, because they call for different work:

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
from dataclasses import dataclass
from pathlib import Path

from ._bench import Bench
from ._invariants import infer_invariants
from ._lean import write_program_proof
from ._verify_ranking import build_obligation, farkas_cell

LEAN_DIR = Path(__file__).resolve().parent / "lean"

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


def check_file(path: Path, timeout: float = 300.0) -> tuple[str, str]:
    """Compile one emitted proof. Returns ``(outcome, first error line)``."""
    try:
        r = subprocess.run(["lake", "env", "lean", str(path.relative_to(LEAN_DIR))],
                           cwd=LEAN_DIR, capture_output=True, text=True, timeout=timeout)
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


def certify(bench: Bench, layers, delta: float = 1.0,
            timeout: float = 300.0) -> CheckResult:
    """Verify ``layers`` for ``bench`` with Farkas, emit the proof, and check it."""
    ob = build_obligation(bench, layers, delta, infer_invariants(bench))
    res = farkas_cell(ob)
    if not res.verified:
        return CheckResult(bench.name, "UNVERIFIED", detail=res.status,
                           n_invariants=len(ob.invariants))
    paths = res.certificate
    counts = dict(n_paths=len(paths), n_cells=sum(len(p.cells) for p in paths),
                  n_invariants=len(ob.invariants))
    out = write_program_proof(bench.name, ob, paths, LEAN_DIR / "proofs")
    outcome, detail = check_file(out, timeout)
    return CheckResult(bench.name, outcome, detail=detail, **counts)
