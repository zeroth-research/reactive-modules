"""`lake build` of a generated project: its output streamed as it comes,
and a failure read down to the errors that caused it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


from .names import (
    CERT_LIB,
)


class LakeBuildError(RuntimeError):
    """`lake` could not build the generated project's certificate."""


def stream(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    """Run `cmd` in `cwd`, echoing its output as it arrives.

    Returns the exit code *and* the output rather than raising, so each
    caller can say what went wrong in its own vocabulary -- `lake` and
    the tool a route runs do not fail in the same words. `FileNotFoundError`
    propagates: a missing executable is the caller's to name.
    """
    print(f"$ {' '.join(cmd)}")
    chunks: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        chunks.append(line)
        print(line, end="")
    return proc.wait(), "".join(chunks)


def build_certificate(project_dir: Path) -> None:
    """`lake build` the certificate of a generated project.

    `lake update` runs first: a freshly generated project has no manifest at
    all, and one left by an earlier run predates this run's lakefile -- a
    route's lakefile can add a require the manifest has never seen, and
    `lake build` resolves neither case.

    A build that succeeds while the certificate still holds `sorry` is not a
    proof, so that is an error too. `zeroth_hammer` leaves a `sorry` where it
    cannot close an obligation, and lake reports those as warnings and exits
    0 -- which would make "built" mean nothing.
    """
    for cmd in (["lake", "update"], ["lake", "build", CERT_LIB]):
        try:
            rc, out = stream(cmd, cwd=project_dir)
        except FileNotFoundError as e:
            raise LakeBuildError(f"cannot run {cmd[0]}: {e}") from e
        if rc != 0:
            raise LakeBuildError(
                f"`{' '.join(cmd)}` failed (exit {rc}) in {project_dir}\n"
                + _build_why(out)
            )

    unproved = [l for l in out.splitlines()
                if "sorry" in l and f"{CERT_LIB}/" in l]
    if unproved:
        raise LakeBuildError(
            f"the certificate compiles but is not proved: "
            f"{len(unproved)} obligation(s) left as `sorry`\n"
            + "\n".join(f"  {l.strip()}" for l in unproved[:5])
        )
    print(f"Certificate built: lake build Certificate in {project_dir}")


# A declaration whose elaboration fails is never added to the environment,
# so the next one that mentions it fails again -- in the kernel this time,
# as `(kernel) unknown constant 'hrank'` after `hrank` itself timed out at
# `whnf`. The second error is the first one's shadow: it names a symbol
# where the first named a cause, and it is the one that reads like a
# codegen bug rather than a certificate that could not be proved. Over the
# matrix every occurrence of it follows an earlier error.
_CASCADE = re.compile(r"\(kernel\) unknown constant")


def without_cascades(errors: list) -> list:
    """`errors`, without the ones an earlier error already accounts for.

    Only ever drops a *later* kernel `unknown constant`. A build whose
    first complaint is one is left alone: nothing came before it to be a
    shadow of, and then something really is missing.
    """
    out: list = []
    for e in errors:
        if out and _CASCADE.search(str(e)):
            continue
        out.append(e)
    return out


def _build_why(out: str) -> str:
    """The reason lake refused, restated where the user will see it.

    Its own diagnosis is accurate but scrolls past behind thousands of
    replayed Mathlib jobs, leaving a bare "failed (exit 1)" as the last line.
    """
    # Lake's own two wrapper lines say only that something failed, which is
    # what this is already replacing; the diagnostics are the lines between.
    noise = ("error: build failed", "error: Lean exited with code")
    errors = without_cascades([
        l
        for l in out.splitlines()
        if l.startswith("error:") and not l.startswith(noise)
    ])
    if not errors:
        return "  (see the output above)"
    return (
        "\n".join(f"  {l}" for l in errors[:5])
        + "\n  An unresolved import means the project cannot see what the "
        "certificate needs.\n  A failed tactic means an obligation "
        "`zeroth_hammer` could not close: the invariant\n  or the ranking "
        "function is wrong, or too weak to be inductive."
    )
