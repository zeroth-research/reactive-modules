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


# ══════════════════════════════════════════════════════════════════════════
# The recurrence rule itself
#
# `m_countdown` is the module the old tail count got wrong -- it counts 100
# down to 0 and resets, so its period is 101 and it satisfies `s0 = 0` once
# in a 50-tick tail. Both cases below are that one module, which is what
# makes them a test of the rule rather than of a benchmark: the tail count
# cannot tell them apart and answers `fails` to both.
# ══════════════════════════════════════════════════════════════════════════

COUNTDOWN = Path(__file__).resolve().parents[1] / "limits" / "mods" / "m_countdown.py"


def test_a_recurrence_longer_than_the_tail_is_not_refuted():
    """`s0 = 0` recurs once per 101 ticks, which is a property that holds."""
    seen = sim.observe(load_module_from_file(str(COUNTDOWN)), "(= s0 0)", "buchi")

    assert not seen.refuted
    # And the absence has force: a cycle *was* closed, so the property was
    # tested against one rather than merely never contradicted.
    assert seen.looped


def test_a_cycle_the_property_never_holds_in_refutes_it():
    """`s0 = -1` is false at every state of the same cycle, and the cycle is
    the refutation: its length is the module's real period, and replaying
    the inputs of those 101 ticks repeats it for ever."""
    seen = sim.observe(load_module_from_file(str(COUNTDOWN)), "(= s0 (- 1))",
                       "buchi")

    assert seen.refuted and seen.verdict == "fails"
    assert seen.period == 101
    assert seen.witness == (100,)


def test_only_consecutive_returns_to_a_configuration_are_compared():
    """A loop spanning three visits is property-free only if both halves
    are, and the first half is checked when it closes -- so comparing
    neighbours misses no refutation, and `lasso` reports the earliest."""
    # Three visits to the configuration `s0 = 0`, at ticks 0, 2 and 4, and
    # no inputs. A run is written here rather than stepped, so that what is
    # under test is the scan and not a module.
    dtypes = [sim.Int([1, 1])]
    at = sim._consts("s", dtypes)
    run = [([sim._tensor(v, dtypes[0])], []) for v in (0, 1, 0, 2, 0)]

    # `s0 = 1` holds in the first stretch and not the second, so the second
    # is the refutation; a property in neither stretch is refuted by the
    # first, which is the earlier one.
    assert sim.lasso(run, sim.parse("(= s0 1)", at), at, dtypes) == (2, 4)
    assert sim.lasso(run, sim.parse("(= s0 9)", at), at, dtypes) == (0, 2)
    # And a property that holds somewhere in every stretch is not refuted.
    assert sim.lasso(run, sim.parse("(<= s0 1)", at), at, dtypes) is None
