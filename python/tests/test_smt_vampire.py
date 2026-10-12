"""The one Vampire driver: `--houdini-solver vampire` and `--infer vampire`
both run the prover through `smt.vampire.run_vampire`. Stubs stand in for
the binary, so nothing here needs Vampire."""

from __future__ import annotations

import os
import time
from pathlib import Path

from zrth.lean.smt.vampire import run_vampire


def _stub(tmp_path: Path, body: str) -> str:
    exe = tmp_path / "vampire"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)
    return str(exe)


def test_the_script_is_the_last_argument_and_is_removed_after(tmp_path):
    exe = _stub(tmp_path, 'for a; do last=$a; done\necho "$@"\ncat "$last"\n')
    run = run_vampire(exe, "(check-sat)\n", ["--mode", "portfolio"],
                      timeout=10)
    args, script = run.out.split("\n", 1)
    assert args.split()[:4] == ["--input_syntax", "smtlib2", "--mode",
                                "portfolio"]
    assert script == "(check-sat)\n"
    assert not Path(args.split()[-1]).exists()


def test_a_parse_error_is_the_first_line_that_says_so(tmp_path):
    exe = _stub(tmp_path, "echo '% ok'\necho '  User error: invalid sort'\n")
    run = run_vampire(exe, "", [], timeout=10)
    assert run.error == "User error: invalid sort"
    assert run_vampire(_stub(tmp_path, "echo unsat\n"), "", [],
                       timeout=10).error is None


def test_a_run_past_its_wall_clock_is_killed_with_what_it_forked(tmp_path):
    """The portfolio forks a process per strategy; a run stopped from
    outside takes them all with it."""
    child = tmp_path / "child"
    exe = _stub(tmp_path, f"sleep 30 &\necho $! > {child}\nwait\n")
    t0 = time.monotonic()
    run = run_vampire(exe, "", [], timeout=0.5)
    assert run.out == "" and time.monotonic() - t0 < 10
    pid = int(child.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid}, forked by the stub, outlived the run")
