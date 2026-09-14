"""Tests for the corpus runner (``benchmarks.svcomp.__main__``).

The property under test is that a benchmark's result does not depend on what ran
before it. Z3's context is global and accumulates every term created in the
process, and the CEGAR loop's choice of uncovered witness depends on it, so
running benchmarks in one process makes the cell decomposition — and with it the
emitted proof — a function of the whole sequence. The runner gives each benchmark
its own process to remove that coupling.
"""
import shutil

import pytest

from benchmarks.svcomp import __main__ as runner
from benchmarks.svcomp import discover
from benchmarks.svcomp.__main__ import _spawn, certify_one


@pytest.mark.parametrize("argv, jobs", [
    (["--lean"], runner._default_jobs()),
    (["--lean", "--jobs", "3"], 3),
])
def test_jobs_flag(argv, jobs, monkeypatch):
    seen = {}
    monkeypatch.setattr(runner, "_run_lean",
                        lambda benches, jobs: seen.setdefault("jobs", jobs) or 0)
    assert runner.main(argv) == 0
    assert seen["jobs"] == jobs


@pytest.mark.parametrize("argv", [["--lean", "--jobs"], ["--lean", "--jobs", "0"],
                                  ["--lean", "--jobs", "x"]])
def test_jobs_flag_rejects_bad_values(argv):
    assert runner.main(argv) == 2


needs_lean = pytest.mark.skipif(shutil.which("lake") is None,
                                reason="no Lean toolchain")

# Ex2.07's decomposition is sensitive to preceding work: 4 cells on its own, 3
# after other benchmarks have run in the same process.
TARGET = "ChenFlurMukhopadhyay-SAS2012-Ex2.07"
PERTURBER = "AliasDarteFeautrierGonnord-SAS2010-ndecr"


def _bench(name):
    return next(b for b in discover() if b.name == name)


def _shape(res):
    return res.outcome, res.n_paths, res.n_cells


@needs_lean
def test_worker_result_is_independent_of_this_process():
    """Perturbing the parent's Z3 context must not change what a worker reports."""
    before = _spawn(TARGET)
    certify_one(_bench(PERTURBER))          # adds terms to this process's context
    after = _spawn(TARGET)
    assert _shape(before) == _shape(after)
    assert before.outcome == "CHECKED"


@needs_lean
def test_spawn_reports_a_missing_benchmark_as_error():
    res = _spawn("no-such-benchmark")
    assert res.outcome == "ERROR" and res.detail
