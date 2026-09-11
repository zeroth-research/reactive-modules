"""The ``--fbk-proveit`` route: certify via ``lean-ltl-certifying``.

Instead of verith's own invariant/ranking machinery, this hands the module
to the ``proveit.py`` driver of the ``lean-ltl-certifying`` repository:

    module.py ──verith──▶ <Proj>/ProveIt/<Proj>NA.lean   (this file's job)
                              │
                              ▼  proveit.py
                    lean2vmt ─▶ ic3ia ─▶ vmt2lean
                              │
                              ▼
              <Proj>/ProveIt/<Proj>Cert.lean   (raw proveit.py output)
                              │
                              ▼  "processing" — a copy, for now
              <Proj>/Certificate/Certificate.lean   (the project's own
                                    certificate, which `Certificate.lean`
                                    imports and `lake build` reaches)

The Lean project verith generates is *bare* of an invariant — it comes from
ic3ia, so `--infer`, `--invariant` and `--ranking` are rejected rather than
quietly ignored, and the property has to be `--safety`: ic3ia decides
reachability of `¬P`, which is not a question about recurrence — but it is not unaware of the route: `create_project` is
given the checkout, so the lakefile requires `LTL_Certifying` and declares
the NA model's lean_lib.  Without those two the installed certificate is a
file lake reaches and cannot elaborate.

Nothing here degrades silently.  Every step that cannot be carried out —
an unsupported module shape, a missing tool, a failing subprocess — raises
:class:`ProveItError`, which `main` turns into a non-zero exit.  The
certificate is still only *written* by default, not compiled, because
compiling it means resolving cslib, Mathlib, lean-smt and cvc5 in the
generated project; ``--build-cert`` asks for that build (``lake update``
then ``lake build Certificate``) and makes a certificate that does not
compile an error.  ``README.md`` spells the whole envelope out.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .common import LeanContext
from .project import PROVEIT_DIR, na_module_name, stream
from .translate.na import (
    NA_IMPORTS,
    NAUnsupported,
    atom_to_lean_na,
    check_na_supported,
)


class ProveItError(RuntimeError):
    """A step of the proveit route could not be carried out."""


# Files that identify a `lean-ltl-certifying` checkout.
_REQUIRED = (
    Path("proveit.py"),
    Path("vmt2lean.py"),
    Path("lakefile.toml"),
    Path("LTLCertifying") / "lean2vmt.lean",
)

# `lean2vmt` elaborates the model with `processHeader`, so the model's
# imports have to be on `LEAN_PATH` as oleans first — `lake exe lean2vmt`
# builds only the executable, whose own import list is just `Lean`.
# Building the model's *imports* rather than a lib target keeps the two in
# step, and is also what works: the `LTLCertifying` lean_lib globs its root
# module alone, so `lake build LTLCertifying` compiles `LTLCertifying.lean`
# and nothing the model actually names.
_LAKE_TARGETS = list(NA_IMPORTS)


def resolve_project(path: str | Path) -> Path:
    """Validate that `path` is a `lean-ltl-certifying` checkout."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ProveItError(f"--fbk-proveit: not a directory: {root}")
    missing = [str(p) for p in _REQUIRED if not (root / p).exists()]
    if missing:
        raise ProveItError(
            f"--fbk-proveit: {root} does not look like a lean-ltl-certifying "
            f"checkout (missing: {', '.join(missing)})"
        )
    return root


def check_toolchain(python: str) -> None:
    """Fail early on the tools `proveit.py` will need but not report well.

    `vmt2lean.py` starts with ``from mathsat import *``; run under an
    interpreter without the MathSAT bindings it dies three steps into the
    pipeline, after the ic3ia run.  ic3ia is handled by
    :func:`resolve_ic3ia`, which reports better than `proveit.py` does.
    """
    if shutil.which("lake") is None:
        raise ProveItError(
            "--fbk-proveit: `lake` not found on PATH; it is needed to build "
            "lean-ltl-certifying and to run its lean2vmt executable"
        )
    probe = subprocess.run(
        [python, "-c", "import mathsat"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise ProveItError(
            f"--fbk-proveit: `{python}` cannot import `mathsat`, which "
            "lean-ltl-certifying's vmt2lean.py needs to read the ic3ia "
            "witness.\nPoint PYTHONPATH at the MathSAT Python bindings "
            "built for this interpreter, e.g.\n"
            "  PYTHONPATH=/path/to/mathsat/python uv run verith ..."
        )


def resolve_ic3ia(spec: str | None) -> str | None:
    """Turn `--ic3ia` into a path `proveit.py` will accept, or abort.

    `proveit.py` resolves it as `which(spec) or spec-if-a-file`, so a
    directory -- the build directory, the obvious thing to reach for --
    fails both tests and is reported as "executable not found" with the
    binary sitting inside it. Do the resolution here, where it can say
    what it actually found.

    `None` is passed through: `proveit.py` then falls back to `$IC3IA`
    and then to `ic3ia` on PATH, and reports that itself.
    """
    if spec is None:
        return None
    path = Path(spec).expanduser()

    if path.is_dir():
        candidate = path / "ic3ia"
        if os.access(candidate, os.X_OK):
            print(f".. --ic3ia is a directory; using {candidate}")
            return str(candidate)
        raise ProveItError(
            f"--ic3ia: {path} is a directory and holds no executable "
            "`ic3ia`. Pass the binary itself, e.g. "
            f"{path / 'ic3ia'}"
        )

    if path.is_file():
        if not os.access(path, os.X_OK):
            raise ProveItError(f"--ic3ia: {path} is not executable")
        return str(path)

    found = shutil.which(spec)
    if found:
        return found
    raise ProveItError(
        f"--ic3ia: no such file: {path}. Pass the ic3ia binary, a "
        "directory containing it, or a name on PATH."
    )


def property_to_bool_lean(module, property_smt: str, n_state: int) -> str:
    """Translate the `--safety` SMT source to Bool-valued Lean.

    The result reads state through `(var_i state)`, exactly as
    ``translate/na.py`` binds it.
    """
    import cvc5  # lazy: keeps cvc5 off the critical path of a bare run

    from .smt_module import ModuleSMT
    from .smt_prompt import CegarPromptEnv, parse_predicate
    from .smt_to_lean import smt_to_lean_bool

    tm = cvc5.TermManager()
    env = CegarPromptEnv(ModuleSMT(tm=tm, module=module))
    term = parse_predicate(env, property_smt)
    if not term.getSort().isBoolean():
        raise ProveItError(f"--safety must have sort Bool, got {term.getSort()}")
    accessors = {f"s{i}": f"(var_{i} state)" for i in range(n_state)}
    try:
        return smt_to_lean_bool(term, accessors)
    except ValueError as e:
        raise ProveItError(f"--fbk-proveit: cannot encode --safety: {e}") from e


def write_na_model(
    project_dir: Path,
    project_name: str,
    ctx: LeanContext,
    property_lean: str,
) -> Path:
    """Write the NA model `proveit.py` consumes and return its path.

    The file name is the Lean module name `proveit.py` will import the
    model under (`resolve_model` uses the stem for an out-of-tree model),
    so it has to be a valid Lean identifier -- and it is the same name the
    generated lakefile's lean_lib declares, which is why both come from
    `project.na_module_name`.
    """
    out_dir = project_dir / PROVEIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model = out_dir / f"{na_module_name(project_name)}.lean"
    model.write_text(atom_to_lean_na(ctx, property_lean, module_name=project_name))
    print(f"Wrote NA model for proveit.py: {model}")
    return model


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    what: str,
    why: Callable[[str], str] | None = None,
) -> str:
    """Run `cmd`, streaming its output; raise `ProveItError` if it fails.

    Returns the output as well as streaming it, so the caller can say what
    went wrong instead of only that something did.  `why` reads that output
    and restates the reason; it is per step rather than global because each
    step fails in its own vocabulary -- a Lean log saying `unknown
    identifier` is not ic3ia reporting `unknown`.
    """
    try:
        rc, out = stream(cmd, cwd=cwd)
    except FileNotFoundError as e:
        raise ProveItError(f"--fbk-proveit: cannot run {cmd[0]}: {e}") from e
    if rc != 0:
        reason = why(out) if why else "  (see the output above)"
        raise ProveItError(f"--fbk-proveit: {what} failed (exit {rc})\n{reason}")
    return out


def _why(out: str) -> str:
    """The reason `proveit.py` stopped, restated where the user will see it.

    Its own diagnosis is accurate but scrolls past behind the ic3ia
    statistics, leaving a bare "failed (exit 1)" as the last line.
    """
    if "UNSAFE" in out:
        return (
            "  ic3ia found a counterexample: the property does not hold of "
            "every reachable state.\n"
            "  This route proves `[] PROPERTY`, which is what --safety "
            "means. A property that is\n"
            "  merely *re-reached* is --buchi, and pairs with a ranking "
            "function instead:\n"
            "  `(= s0 0)` is false at step 0 for a counter starting anywhere "
            "else."
        )
    if "did not prove" in out or "unknown" in out:
        return (
            "  ic3ia could not decide it. It is IC3 with implicit predicate "
            "abstraction over\n  linear arithmetic, so a nonlinear property "
            "is out of scope by construction."
        )
    return "  (see the output above)"


def run(
    *,
    ltl_project: str | Path,
    module,
    project_dir: Path,
    project_name: str,
    property_smt: str,
    ic3ia: str | None = None,
    python: str | None = None,
) -> Path:
    """Run the whole route and return the installed certificate's path."""
    python = python or sys.executable
    root = resolve_project(ltl_project)
    # Resolve before anything expensive: a bad --ic3ia would otherwise
    # surface only after the model is written and lake has run.
    ic3ia = resolve_ic3ia(ic3ia)
    check_toolchain(python)

    ctx = LeanContext(module)
    try:
        check_na_supported(ctx)
    except NAUnsupported as e:
        raise ProveItError(f"--fbk-proveit: {e}") from e

    property_lean = property_to_bool_lean(module, property_smt, len(ctx.ctrl_next))
    model = write_na_model(project_dir, project_name, ctx, property_lean)

    targets = " ".join(_LAKE_TARGETS)
    print(f".. Building the model's imports ({targets}) in {root}")
    _run(["lake", "build", *_LAKE_TARGETS], cwd=root, what=f"lake build {targets}")

    raw_cert = model.with_name(f"{project_name}Cert.lean")
    # `proveit.py` prompts before overwriting its output and we run it with
    # no stdin, so clear a stale file rather than let it abort.
    raw_cert.unlink(missing_ok=True)

    cmd = [python, str(root / "proveit.py"), str(model), "-o", str(raw_cert)]
    if ic3ia:
        cmd += ["--ic3ia", ic3ia]
    _run(cmd, cwd=root, what="proveit.py", why=_why)

    if not raw_cert.is_file() or raw_cert.stat().st_size == 0:
        raise ProveItError(
            f"--fbk-proveit: proveit.py reported success but wrote no "
            f"certificate to {raw_cert}"
        )

    installed = project_dir / "Certificate" / "Certificate.lean"
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_cert, installed)
    print(f"Installed certificate: {installed}")
    return installed
