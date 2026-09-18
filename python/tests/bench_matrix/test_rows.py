"""Check that the `hybrid` and `petri` rows say true things about their modules.

Four of the matrix's suites transcribe `Row.truth` from something upstream
that already paid for it -- an fbk probe's `expect`, a benchmark's own Houdini
invariant. These two have no upstream, so this is it: every row is simulated
and its claim compared with what the run does (`sim.py` spells out exactly
what that establishes -- a `fails` row is *proved* by the counterexample, a
`holds` row is only not refuted).

The point is not to verify the properties, which is the matrix's whole job.
It is to keep a row from quietly stating something false: a wrong `truth`
turns every honest `REFUTED` into what looks like a route bug, and a session
can lose an afternoon to it.
"""
import os
import sys
from pathlib import Path

import pytest

# The harness's own modules are scripts, imported by directory rather than as
# a package; pytest's `importlib` mode does not put a test file's directory on
# the path, so this does.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim                                                      # noqa: E402
from suites import SUITES                                       # noqa: E402

from zrth.lean.project import load_module_from_file

ROWS = [r for name in ("hybrid", "petri") for r in SUITES[name]()]


def _load(row):
    """`row`'s module, with the environment the row asks for."""
    saved = {k: os.environ.get(k) for k in row.env}
    os.environ.update(row.env)
    try:
        return load_module_from_file(str(row.module))
    finally:
        for k, v in saved.items():
            os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)


@pytest.mark.parametrize("row", ROWS, ids=[r.key for r in ROWS])
def test_declared_truth(row):
    seen = sim.observe(_load(row), row.prop, row.kind, row.pre)
    assert seen.verdict == row.truth, (
        f"{row.key}: declared {row.truth}, the runs say {seen.verdict}.\n"
        f"  property {row.kind}  {row.prop}\n"
        f"  pre      {row.pre or '(none)'}\n"
        + (f"  witness  step {seen.step}, state {seen.witness}\n"
           if seen.refuted else "")
        + f"  bounds   {[(round(a, 4), round(b, 4)) for a, b in seen.bounds]}"
    )


def test_every_module_has_a_refutable_row():
    """Each module carries at least one property that fails.

    A suite of only-true properties measures half a route: nothing in it can
    show that a `REFUTED` was *right*, so a route that answered `VERIFIED` to
    every question would score perfectly."""
    def net_of(bench):
        """The net a variant belongs to. A `-rr` variant is the same net under
        another scheduler and a `-cont` one the same net over the reals, so a
        counterexample against any of them covers the family."""
        return bench.removesuffix("-rr").removesuffix("-cont")

    fails = {net_of(r.bench) for r in ROWS if r.truth == "fails"}
    missing = sorted({net_of(r.bench) for r in ROWS} - fails)
    assert not missing, f"no failing property for: {', '.join(missing)}"
