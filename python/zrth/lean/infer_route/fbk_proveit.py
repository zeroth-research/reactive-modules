"""`--infer fbk-proveit`: certify through a `lean-ltl-certifying` checkout."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..common import Refused
from .api import Alias, InferInput, InferResult, InferRoute, Opt

if TYPE_CHECKING:
    from ..project import LakefileParts


def _resolve(opts: dict):
    from ..fbk_proveit import resolve_project

    return resolve_project(opts["proveit_dir"])


def _check_project_name(settings, opts) -> "str | None":
    """`-p` becomes a Lean module name on this route, so it has to be one.

    Not a cell of the table: it is a rule about another flag's *value*.
    `import 1projNA` is "unexpected token; expected identifier", four steps
    later and in generated code.
    """
    from ..fbk_proveit import na_module_name

    name = settings.project_name
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", name):
        return None
    return (
        f"--infer fbk-proveit needs a -p that is a Lean identifier: "
        f"`{na_module_name(name)}` is the module name the NA model and the "
        f"certificate are written under, and `{name}` does not start one"
    )


def _precheck(module, opts, config, claim):
    from ..fbk_proveit import ProveItError, check_module

    try:
        return check_module(
            module,
            opts["fbk_simplify"] == "cvc5",
            claim.init_pre if claim else None,
        )
    except ProveItError as e:
        # One exception type crosses the seam, so `main` catches what every
        # route raises. The message keeps naming `--fbk-proveit` itself,
        # which is what `errors_self_named` is about.
        raise Refused(str(e)) from e


def _lakefile(config, project_name: str) -> "LakefileParts":
    """What this route needs of the lakefile: the checkout required, and the
    NA model declared as a lean_lib.  Without the two, the certificate this
    route installs is a file lake reaches and cannot elaborate."""
    from ..fbk_proveit import PROVEIT_DIR, na_module_name
    from ..project import LakefileParts, LeanLib, Require

    return LakefileParts(
        libs=(
            LeanLib(
                na_module_name(project_name),
                PROVEIT_DIR,
                comment=(
                    "`--fbk-proveit`: `proveit.py` compiles the NA model out "
                    "of tree and so\n"
                    "imports it under its bare file stem, which the "
                    "certificate then names.\n"
                    "Mapping that module name back onto the file is what "
                    "lets `lake build`\n"
                    "resolve the same import the certificate was generated "
                    "against."
                ),
            ),
        ),
        requires=(
            Require(
                "LTL_Certifying",
                str(config),
                comment=(
                    "`--fbk-proveit`: the certificate imports "
                    "LTLCertifying.*. A path require\n"
                    "rather than a git one because the route already has the "
                    "checkout in hand,\n"
                    "and because that checkout pins cslib and smt to the "
                    "revisions above \u2014 the\n"
                    "two package sets resolve as one."
                ),
            ),
        ),
    )


def _run(inp: InferInput) -> InferResult:
    from ..fbk_proveit import ProveItError
    from ..fbk_proveit import run as run_proveit

    try:
        files = run_proveit(
            ltl_project=inp.config,
            module=inp.module,
            path=inp.project.path,
            project_name=inp.project.name,
            property_smt=str(inp.claim.prp),
            pre_smt=inp.claim.init_pre,
            ic3ia=inp.opts["ic3ia"],
            simplify=inp.opts["fbk_simplify"] == "cvc5",
            equivalence=inp.opts["fbk_equiv"] == "lean",
            prechecked=inp.prechecked,
            log=inp.log,
        )
    except ProveItError as e:
        raise Refused(str(e), searched=e.searched) from e
    # The certificate carries ic3ia's invariant: no predicates come back,
    # which is what this route's `returns="files"` declares.
    return InferResult(files=files)


ROUTE = InferRoute(
    name="fbk-proveit",
    summary=(
        "certify through a `lean-ltl-certifying` checkout's proveit.py "
        "(lean2vmt -> ic3ia -> vmt2lean): the project is generated bare, "
        "an NA-encoded model is written to <project>/ProveIt/, and the "
        "certificate ic3ia's invariant makes is installed as "
        "<project>/Certificate/Certificate.lean"
    ),
    kinds=frozenset({"safety"}),
    kinds_refusal=(
        "--infer fbk-proveit is incompatible with --buchi: proveit.py "
        "proves `G PROPERTY` through ic3ia, which is a safety question. "
        "Pass --safety, or infer a Buchi certificate with --infer ai-cegis."
    ),
    seeds=frozenset({"pre"}),
    seeds_refusal=(
        "the invariant comes from ic3ia and the project is generated "
        "bare. `--pre` is taken: it is an assumption about the module's "
        "inputs rather than a piece of the certificate, and the NA model "
        "carries it as an `INIT` conjunct over the slots those inputs "
        "start"
    ),
    owns=(
        "ProveIt/",
        "Certificate/Certificate.lean",
        "Certificate/Equivalence.lean",
    ),
    lakefile=_lakefile,
    resolve=_resolve,
    extra_check=_check_project_name,
    precheck=_precheck,
    run=_run,
    aliases=(
        Alias(
            "--fbk-proveit",
            "proveit_dir",
            "Older spelling of `--infer fbk-proveit --proveit-dir DIR`: "
            "it selects the route and names the checkout in one flag.",
        ),
    ),
    returns="files",
    errors_self_named=True,
    options=(
        Opt(
            ("--proveit-dir",),
            dict(
                metavar="DIR",
                help=(
                    "Path to a `lean-ltl-certifying` checkout. The "
                    "lakefile requires it, so the certificate this route "
                    "installs is buildable. `--fbk-proveit DIR` is the "
                    "older spelling and still selects this route."
                ),
            ),
            required=True,
        ),
        Opt(
            ("--ic3ia",),
            dict(
                metavar="PATH",
                help=(
                    "Path to the ic3ia executable, forwarded to "
                    "proveit.py. When omitted proveit.py falls back to "
                    "$IC3IA, then to `ic3ia` on PATH."
                ),
            ),
        ),
        Opt(
            ("--fbk-simplify",),
            dict(
                default="cvc5",
                choices=["cvc5", "none"],
                help=(
                    "Whether cvc5's rewriter runs over the transition the "
                    "NA model carries (default: cvc5). `none` leaves each "
                    "state slot the shape the module's own terms give it "
                    "-- the same transition, spelled the way the module "
                    "spells it (`x - 1` rather than `-1 + x`), which is "
                    "what to pass when reading the generated Lean against "
                    "the source. The rewriter is what makes a constant "
                    "matrix tractable, though: a 32-wide affine layer is "
                    "97 KB of term unfolded and 1.9 KB folded."
                ),
            ),
        ),
        Opt(
            ("--fbk-equiv",),
            dict(
                default="lean",
                choices=["lean", "none"],
                help=(
                    "Whether to emit the proof that the NA model is the "
                    "module (default: lean). The certificate proveit.py "
                    "installs is about the *model*; this is what carries "
                    "it back to the module, and without it a "
                    "mistranslation anywhere in the SMT path would be a "
                    "machine-checked certificate about a different "
                    "system. `none` skips it -- it is the expensive part "
                    "on a wide state (73 s at 32 slots against ~5 s for "
                    "the rest of the route)."
                ),
            ),
        ),
    ),
)
