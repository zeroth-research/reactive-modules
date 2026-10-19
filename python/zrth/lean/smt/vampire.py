"""Vampire: where the binary is, and one run of it on a script.

Both routes that use it -- `--houdini-solver vampire` and `--infer vampire`
-- run it the same way: a script in a temporary file, the prover in its
own process session, and the whole group killed when the wall clock runs
out. What the output means, and how a failure is worded, is the caller's.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ..errors import Refused

ENV = "VAMPIRE"


def resolve_vampire(spec: "str | None") -> str:
    """`--vampire` as an executable, else `$VAMPIRE`, else `vampire` on PATH.

    Resolved at parse time, so a run that cannot start the prover says so
    before a project is generated rather than after. A directory is accepted
    when it holds a `vampire` -- the release zip unpacks to one.
    """
    for source, value in (("--vampire", spec), (f"${ENV}", os.environ.get(ENV))):
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_dir():
            path = path / "vampire"
        if path.is_file():
            if not os.access(path, os.X_OK):
                raise Refused(f"{source}: {path} is not executable")
            return str(path)
        found = shutil.which(value)
        if found:
            return found
        raise Refused(
            f"{source}: no such file: {path}. Pass the Vampire binary, a "
            f"directory holding one, or a name on PATH."
        )
    found = shutil.which("vampire")
    if found:
        return found
    raise Refused(
        "--houdini-solver vampire needs the Vampire prover: pass --vampire "
        f"PATH, set ${ENV}, or put `vampire` on PATH. Builds for Linux and "
        "macOS are at https://github.com/vprover/vampire/releases\n"
        "`--houdini-solver cvc5` needs no binary."
    )


def kill_group(proc: subprocess.Popen) -> None:
    """Kill `proc`'s process group, if it is still running, and reap it."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()


@dataclass(frozen=True)
class VampireRun:
    """What one run printed, stdout and stderr together, and how long it took.

    `out` is empty when the wall clock ran out first."""

    out: str
    secs: float

    @property
    def lines(self) -> list[str]:
        return [ln.strip() for ln in self.out.splitlines()]

    @property
    def error(self) -> "str | None":
        """The first line saying Vampire could not read the script."""
        return next((ln for ln in self.lines
                     if "User error" in ln or "Parsing Error" in ln), None)


def run_vampire(exe: str, script: str, args, *, timeout: float,
                prefix: str = "verith-vampire-") -> VampireRun:
    """`exe` on `script` with `args`, stopped after `timeout` seconds.

    Its own session: the portfolio forks a process per strategy, and a run
    killed from outside has to take them all with it.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".smt2", prefix=prefix, delete=False
    ) as f:
        f.write(script)
        path = f.name
    cmd = [exe, "--input_syntax", "smtlib2", *args, path]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_group(proc)
        out = ""
    finally:
        kill_group(proc)
        Path(path).unlink(missing_ok=True)
    return VampireRun(out, time.perf_counter() - t0)
