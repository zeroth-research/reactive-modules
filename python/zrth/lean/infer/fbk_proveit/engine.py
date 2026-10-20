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
                              ▼  processing (`leave_bool_before_generalize`)
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
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from ...codegen.context import LeanContext
from ...project.names import (
    EQUIVALENCE, PROOF, PROVEIT_DIR, cert_file,
)
from ...project.build import stream
from .na import (
    NA_IMPORTS,
    NAUnsupported,
    SlotBodies,
    atom_to_lean_na,
    check_na_supported,
    pre_over_slots,
)


def na_module_name(project_name: str) -> str:
    """Lean module name of the NA model this route writes.

    `proveit.py` compiles an out-of-tree model into an olean and imports it
    under the file stem, so the file name *is* the module name the generated
    certificate carries -- and the name the project's lakefile has to map
    back onto ``PROVEIT_DIR``.
    """
    return f"{project_name}NA"


class ProveItError(RuntimeError):
    """A step of the proveit route could not be carried out.

    `searched` is `Refused.searched`, for the one step that searches:
    `proveit.py` failing is ic3ia running and not deciding, which says
    something about the module, where every other failure here is about
    what the encoding or the checkout can take.
    """

    def __init__(self, message: str, *, searched: bool = False):
        super().__init__(message)
        self.searched = searched


class NAEncoding(NamedTuple):
    """What :func:`check_module` learned, for the files written from it.

    Both answers cost a cvc5 encoding and both are wanted twice -- by the NA
    model and by the bridge -- so the check hands them on rather than being
    asked again. `pre_lean` is `""` for a run with no `--pre`.
    """

    bodies: SlotBodies
    pre_lean: str = ""


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


def check_module(
    module,
    simplify: bool = True,
    pre_smt: str | None = None,
    encoding=None,
) -> NAEncoding:
    """The module's slot bodies and `--pre`; `ProveItError` if it has none.

    Called before anything is generated as well as inside :func:`run`: a
    module this route cannot encode would otherwise be met first by
    `create_project`, which fails about the Lean *project* -- a traceback
    out of the functional encoder, for an op that has nothing to do with
    the certificate route the user asked for.

    `--pre` is checked here for the same reason and not a weaker one: which
    preconditions this encoding can carry depends on the module's slots
    (`fbk.na.pre_over_slots`), so it is a question only the encoding can answer,
    and answering it after the project is written answers it too late.

    Answering either means encoding into cvc5, so the answers are handed
    back and carried to the two files written from them
    (:func:`write_na_model`, :func:`equivalence_lean`).  The pipeline's call,
    before the project is generated, is then the only encoding a whole run
    makes.

    `encoding` is the run's `ModuleEncoding`: the slot bodies and the
    `--pre` are read on its term manager.  A caller with none gets an
    encoding of `module` of its own.
    """
    from ...smt.encoding import ModuleEncoding

    if encoding is None:
        encoding = ModuleEncoding(module)
    ctx = LeanContext(module)
    try:
        # The bodies first: the `--pre` is parsed on the same encoding.
        bodies = check_na_supported(ctx, simplify, encoding)
        if not pre_smt:
            return NAEncoding(bodies)
        return NAEncoding(
            bodies,
            pre_over_slots(ctx, _parse(module, pre_smt, "--pre", encoding), bodies),
        )
    except NAUnsupported as e:
        raise ProveItError(f"--fbk-proveit: {e}") from e


# The step between `proveit.py`'s certificate and the project's own, which
# used to be a copy.
#
# `vmt2lean.py` proves the first validity check -- "the initial condition
# implies the invariant" -- by reverting the initial condition into the goal
# and generalising each state slot out of it. That last step cannot see
# through a `Decidable` instance. `with_reducible reduce at *`, two lines
# above, rewrites `var_k (trace 0)` to `trace 0 k` inside the proposition a
# `decide` is applied to and leaves the instance beside it still reading
# `(var_k (trace 0)).decLe`; `generalize` then abstracts the one and not the
# other and reports `result is not type correct`.
#
# It cannot arise while `INIT` is a conjunction of `var_k state == c`, which
# is every module with no external input: `BEq Int` does not take its
# operands, so there is no instance mentioning a slot. A precondition is a
# comparison, so `--pre` is what first puts a `decide` in `INIT` -- and it
# stopped all 6 of the svcomp rows that get that far.
#
# Leaving `Bool` first is the repair, and it is where the certificate's own
# `smt` call wants the goal anyway. `try`, because a goal already in `Prop`
# is most of them and `simp only` fails when it changes nothing.
#
# All of `Bool` or none of it. Normalising the `decide`s and leaving `==`
# behind is worse than doing nothing: it turns a goal `smt` was closing in
# `Bool` into a `Prop` conclusion under a `Bool` hypothesis, and that mix
# cost `fbk/m_countdown/InvBase` -- which has no precondition and was never
# the point -- its proof. So `beq_iff_eq` is in the set, and the connectives
# lead it: an `INIT` of several conjuncts is one `&&`, and what is under it
# is not reached until the `&&` is gone.
_NORM = (
    "Bool.and_eq_true, Bool.or_eq_true, "
    "beq_iff_eq, Bool.not_eq_true', decide_eq_false_iff_not, decide_eq_true_eq, "
    "Bool.not_eq_eq_eq_not, Bool.not_true, Bool.not_false"
)
_REVERT = re.compile(r"revert h; generalizeNatVar")


def leave_bool_before_generalize(cert: str) -> "tuple[str, int]":
    """`cert` with the `Bool` left before each `generalizeNatVar`, and a count.

    A count rather than a raised error when nothing matches: this rewrites
    another project's generated proof script, so a `proveit.py` that has
    changed its template should leave the certificate exactly as it came
    rather than fail a run that may well build.
    """
    text, n = _REVERT.subn(f"revert h; try simp only [{_NORM}]; generalizeNatVar", cert)
    return text, n


# `smt` reconstructs a cvc5 proof into a Lean term, and on
# `fbk/m_step2/NiS2Odd3`'s second validity check the kernel rejects the term
# it builds: `(kernel) application type mismatch`, inside a chain of
# `Smt.Reconstruct.Int.sum_ub`. That is lean-smt's reconstruction and not
# anything `vmt2lean` wrote, which is the same root as the `Failed to
# reconstruct term` on `svcomp/genady/houdini-inv` -- one fails loudly, the
# other hands the kernel a term it will not take.
#
# A validity check is linear integer arithmetic, which is what `omega`
# decides, and it builds its own proof rather than translating one. So it is
# offered first: where it applies there is no reconstruction to get wrong,
# and where it does not it fails in milliseconds and `smt` runs as before.
# Measured on `NiS2Odd3`, whose certificate builds with it and does not
# without.
#
# Only an `smt` that closes a goal -- the last tactic of a parenthesised
# group. The `try smt` further down is already allowed to fail and is left
# alone.
_SMT_CLOSER = re.compile(r"(?<![\w.])smt\b(?=\s*\))")


def try_omega_before_smt(cert: str) -> "tuple[str, int]":
    """`cert` with `omega` offered before each closing `smt`, and a count.

    A count rather than a raised error, for the reason
    :func:`leave_bool_before_generalize` gives: this rewrites another
    project's generated proof script, and one that has changed its template
    should be left as it came rather than fail a run that may well build.
    """
    return _SMT_CLOSER.subn("first | omega | smt", cert)


# What each patch hooks onto, looked for anywhere in the certificate. Both
# patches leave a certificate they do not recognise as it came, which is
# right for a template that has moved on and silent about it: a
# `generalizeNatVar` still there with no `revert h;` before it is the
# template having moved under the patch, and the certificate then fails
# where the patch was the repair. The `smt` here is a bare one, not `try
# smt` and not the end of a dotted name, as `_SMT_CLOSER` takes it.
_HOOKS = (
    (re.compile(r"\bgeneralizeNatVar\b"),
     "`generalizeNatVar` with no `revert h;` before it, so its `Bool` was "
     "not normalised"),
    (re.compile(r"(?<![\w.])(?<!try )smt\b"),
     "`smt` closing no parenthesised group, so `omega` was not offered "
     "before it"),
)


def unpatched(cert: str, *counts: int) -> "list[str]":
    """What the two patches above found nothing to change in, though the
    tactic each hooks onto is in `cert`. `counts` are theirs, in order."""
    return [what for n, (hook, what) in zip(counts, _HOOKS)
            if n == 0 and hook.search(cert)]


def _parse(module, src: str, flag: str, encoding=None):
    """`src` as a cvc5 Bool term over this module's symbols, or `ProveItError`.

    Read through `encoding`'s `PredicateParser` (an encoding of `module` of
    the caller's own when it has none): the symbols are the module's, so the
    two flags that come through here -- `--safety` and `--pre` -- read `s0`,
    `e0`, `el0` as the same things the rest of the package does.

    cvc5's parser raises a bare `RuntimeError` naming the token it stopped
    at, with no hint of which flag the text came from; the flag is this
    function's whole reason for taking one.
    """
    # lazy: keeps cvc5 off the critical path of a bare run
    from ...smt.encoding import ModuleEncoding
    from ...smt.predicates import parse_predicate

    if encoding is None:
        encoding = ModuleEncoding(module)
    try:
        term = parse_predicate(encoding.env, src)
    except RuntimeError as e:
        raise ProveItError(f"{flag}: cannot read `{src}`: {e}") from e
    if not term.getSort().isBoolean():
        raise ProveItError(f"{flag} must have sort Bool, got {term.getSort()}")
    return term


def property_to_bool_lean(module, property_smt: str, encoding=None) -> str:
    """Translate the `--safety` SMT source to Bool-valued Lean.

    The result reads state through `(var_k state)`, exactly as
    ``na.py`` binds it -- one slot per *element*, so a wire wider
    than 1x1 is read through the tuple selectors the property already uses
    (`((_ tuple.select 2) s0)`), resolved to the slot that element lives in.
    Both sides get that map from `na.slot_accessors`, which is what keeps
    the property and the transition talking about the same variables.

    The layout comes from the module's own wires. `encoding` is the run's
    `ModuleEncoding`, which the property is read on.
    """
    from ...codegen.context import LeanContext
    from .to_bool import smt_to_lean_bool
    from .na import slot_accessors

    term = _parse(module, property_smt, "--safety", encoding)
    try:
        return smt_to_lean_bool(term, slot_accessors(LeanContext(module).ctrl_next))
    except ValueError as e:
        raise ProveItError(f"--fbk-proveit: cannot encode --safety: {e}") from e


def write_na_model(
    path: Callable[[str], Path],
    project_name: str,
    ctx: LeanContext,
    property_lean: str,
    simplify: bool = True,
    bodies: SlotBodies | None = None,
    pre_lean: str = "",
) -> Path:
    """Write the NA model `proveit.py` consumes and return its path.

    The file name is the Lean module name `proveit.py` will import the
    model under (`resolve_model` uses the stem for an out-of-tree model),
    so it has to be a valid Lean identifier -- and it is the same name the
    generated lakefile's lean_lib declares, which is why both come from
    `na_module_name`.  `path` is the project handle's: the model is the one
    file this route writes itself, because `proveit.py` reads it from disk.
    """
    model = path(f"{PROVEIT_DIR}/{na_module_name(project_name)}.lean")
    model.write_text(
        atom_to_lean_na(
            ctx,
            property_lean,
            module_name=project_name,
            simplify=simplify,
            bodies=bodies,
            pre_lean=pre_lean,
        )
    )
    print(f"Wrote NA model for proveit.py: {model}")
    return model


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    what: str,
    why: Callable[[str], str] | None = None,
    searched: bool = False,
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
        raise ProveItError(f"--fbk-proveit: {what} failed (exit {rc})\n{reason}",
                           searched=searched)
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
    path: Callable[[str], Path],
    project_name: str,
    property_smt: str,
    pre_smt: str | None = None,
    ic3ia: str | None = None,
    python: str | None = None,
    simplify: bool = True,
    equivalence: bool = True,
    prechecked: NAEncoding | None = None,
    encoding=None,
    log: Callable = print,
) -> dict[str, str]:
    """Run the whole route and return the certificate it made, and the proof
    that the model is the module, as text by project-relative path.

    Only `ProveIt/` is written here, through `path` (the project handle's),
    because `proveit.py` works on files; the two Lean files that are the
    route's answer are handed back for the pipeline to install.

    `prechecked` is what `check_module` returned to a caller that has
    already made the check -- the pipeline does, before it generates the
    project.  The check is repeated here when it is left out, because `run`
    is also called on its own; `pre_smt` is what it is repeated over, so a
    direct caller gets the same refusals the pipeline gets early.

    `encoding` is the run's `ModuleEncoding` (`InferInput.smt()`): the check
    and the `--safety` property are both read on it.  A direct caller that
    passes none gets one `run` makes, so the two still share it.
    """
    python = python or sys.executable
    root = resolve_project(ltl_project)
    # Resolve before anything expensive: a bad --ic3ia would otherwise
    # surface only after the model is written and lake has run.
    ic3ia = resolve_ic3ia(ic3ia)
    check_toolchain(python)

    if encoding is None:
        from ...smt.encoding import ModuleEncoding

        encoding = ModuleEncoding(module)
    ctx = LeanContext(module)
    if prechecked is None:
        prechecked = check_module(module, simplify, pre_smt, encoding)
    bodies, pre_lean = prechecked

    property_lean = property_to_bool_lean(module, property_smt, encoding=encoding)
    model = write_na_model(
        path, project_name, ctx, property_lean, simplify, bodies, pre_lean
    )

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
    _run(cmd, cwd=root, what="proveit.py", why=_why, searched=True)

    if not raw_cert.is_file() or raw_cert.stat().st_size == 0:
        raise ProveItError(
            f"--fbk-proveit: proveit.py reported success but wrote no "
            f"certificate to {raw_cert}"
        )

    raw = raw_cert.read_text()
    text, patched = leave_bool_before_generalize(raw)
    text, omegas = try_omega_before_smt(text)
    notes = []
    if patched:
        notes.append(f"{patched} normalised out of Bool")
    if omegas:
        notes.append(f"{omegas} offered omega before smt")
    if notes:
        log(f".. proveit.py's certificate: {', '.join(notes)}")
    for what in unpatched(raw, patched, omegas):
        log(f"warning: --fbk-proveit: proveit.py's certificate has {what}; "
            f"its template may have changed under this patch")

    files = {cert_file(PROOF): text}
    if equivalence:
        files[cert_file(EQUIVALENCE)] = equivalence_lean(
            project_name, ctx,
            simplify=simplify, bodies=bodies, pre_lean=pre_lean,
        )
    return files


def equivalence_lean(
    project_name: str,
    ctx: LeanContext,
    *,
    simplify: bool = True,
    bodies: SlotBodies | None = None,
    pre_lean: str = "",
) -> str:
    """The proof that the model is the module, as the text of its file.

    The certificate `proveit.py` installs is about the NA model. This is the
    file that carries it back to the module -- see `bridge.py`.

    Lake reaches it through the `Certificate` lean_lib's glob, not through
    the root `Certificate.lean`'s import list, and that is not a detail:
    this file is stated in `Core.LTL`'s vocabulary and the certificate in
    `LTLCertifying`'s, and both libraries declare a top-level `LTLFormula`.
    A module importing the two is rejected before it is elaborated
    ("environment already contains 'LTLFormula'"), so `--build-cert` builds
    them as two modules, which is also all the bridge needs today: its
    `module_safety` takes the model-side statement as a hypothesis rather
    than reading it out of the certificate.
    """
    from .bridge import atom_to_lean_fbk_bridge

    return atom_to_lean_fbk_bridge(
        ctx,
        na_module=na_module_name(project_name),
        simplify=simplify,
        bodies=bodies,
        pre_lean=pre_lean,
    )
