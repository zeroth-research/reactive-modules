"""CLI entry point for generating a Lean4 project from a reactive module.

Entry point: ``uv run verith``

Quick start
-----------
A *module file* is a Python file that defines a ``module()`` function (or
class) returning a :class:`zrth.Module`.  Optionally it may also define plain
``init()`` / ``update()`` functions so that the AI inference (``--infer``) can
read the high-level logic.

Generate a bare Lean project (all certificate fields left as ``sorry``)::

    uv run verith mymodule.py -o out/ -p MyProject

Pass the property so the certificate knows what to prove.  Which flag it
goes under is the choice of proof rule, and so of what the certificate
contains::

    # `G (F x == 0)` -- holds infinitely often: invariant + ranking function
    uv run verith mymodule.py --buchi "x == 0" -o out/ -p MyProject

    # `G (x <= 100)` -- holds in every reachable state: invariant alone
    uv run verith mymodule.py --safety "x <= 100" -o out/ -p MyProject

Ask the AI to infer the certificate automatically (the ranking function too,
for a Buchi property)::

    uv run verith mymodule.py --buchi "x == 0" --infer -o out/ -p MyProject

Or infer it without an LLM at all: ``--infer nuterm`` trains a ranking function
on rollouts of the module and hands it over only once a Farkas/CEGAR decision
procedure has certified it, with the invariant inferred by Houdini.  It needs
no API key, and it reads modules whose state is scalar integers::

    uv run verith mymodule.py --buchi "(= s0 0)" --infer nuterm -o out/ -p MyProject

Use a local LLM via Ollama instead of Claude::

    uv run verith mymodule.py --buchi "x == 0" --infer \\
        --model qwen3-coder --base-url http://localhost:11434/v1 \\
        -o out/ -p MyProject

AI inference requirements
-------------------------
* **Claude (default)**: set ``ANTHROPIC_API_KEY`` and install
  ``pip install zrth[ai]``.
* **Local LLM (Ollama, vLLM, …)**: install ``pip install zrth[ai-local]``
  and provide ``--base-url`` pointing to an OpenAI-compatible endpoint.
* **Hosted OpenAI-compatible provider (OpenRouter, …)**: as above, plus set
  ``OPENROUTER_API_KEY`` (or ``OPENAI_API_KEY``), e.g.
  ``--base-url https://openrouter.ai/api/v1 --model anthropic/claude-haiku-4.5``.

Module file format
------------------
The file must expose a callable named ``module`` (override with ``-d``) that
returns a :class:`zrth.Module`::

    # mymodule.py
    from zrth import Module, Int, Var, X
    from zrth.analyzer import convert_method

    def init():
        return 0

    def update(old_x):
        x = old_x + 1
        if x == 10:
            return 0
        return x

    def module() -> Module:
        state = Var(Int([1, 1]))
        init_terms   = convert_method(init,   {},               [X(state)])
        update_terms = convert_method(update, {"old_x": state}, [X(state)])
        return Module.sequential([state], init_terms, update_terms)
"""

import argparse
import re
from pathlib import Path

from .cert import CertificateData, generate_zeroth_hammer_lean, smt_predicates_to_lean
from .common import Refused
from .smt_query import (
    DEFAULT_CALL_MS,
    DEFAULT_PHASE_MS,
    ModuleQueries,
    SmtBudget,
    pre_check,
    solver_hints,
)
from .project import (
    ENCODINGS,
    LakeBuildError,
    Layout,
    build_certificate,
    create_project,
    generate_standalone_cert_lean,
    load_module_from_file,
    na_module_name,
    write_certificate_lean,
    write_data_lean,
    write_encoding,
)
from .translate import ModuleToLean4


_EPILOG = """\
examples:
  # bare project — all certificate fields left as sorry
  uv run verith mymodule.py -o out/ -p MyProject

  # a Buchi property, `G (F (= s0 0))` (certificate stub with known P)
  uv run verith mymodule.py --buchi "(= s0 0)" -o out/ -p MyProject

  # a safety property, `G (<= s0 100)`: invariant only, no ranking function
  uv run verith mymodule.py --safety "(<= s0 100)" -o out/ -p MyProject

  # AI inference with Claude (requires ANTHROPIC_API_KEY + pip install zrth[ai])
  uv run verith mymodule.py --buchi "(= s0 0)" --infer -o out/ -p MyProject

  # ... and the same loop for a safety property (cvc5-checked route only)
  uv run verith mymodule.py --safety "(<= s0 100)" --infer ai-cegar -o out/ -p MyProject

  # no LLM: learn a ranking function and certify it before it is offered
  uv run verith mymodule.py --buchi "(= s0 0)" --infer nuterm -o out/ -p MyProject

  # AI inference with Ollama (requires pip install zrth[ai-local])
  uv run verith mymodule.py --buchi "(= s0 0)" --infer \\
      --model qwen3-coder --base-url http://localhost:11434/v1 -o out/ -p MyProject

  # ask cvc5 whether the obligations are true before spending a lake build on them
  uv run verith mymodule.py --buchi "(= s0 0)" --invariant "(<= s0 100)" --ranking "s0" \\
      --pre-check cvc5 -o out/ -p MyProject

  # let cvc5 discharge the branch conditions the invariant settles
  uv run verith mymodule.py --buchi "(= s0 0)" --invariant "(<= s0 100)" --ranking "s0" \\
      --smt-tactics cvc5 -o out/ -p MyProject

  # certify through lean-ltl-certifying's proveit.py (lean2vmt -> ic3ia -> vmt2lean)
  uv run verith mymodule.py --safety "(not (= s0 15))" -o out/ -p MyProject \\
      --fbk-proveit ~/proof-prototyping/lean-ltl-certifying --ic3ia ~/ic3ia/build/ic3ia

  # ... and build the certificate that route produces
  uv run verith mymodule.py --safety "(not (= s0 15))" -o out/ -p MyProject \\
      --fbk-proveit ~/proof-prototyping/lean-ltl-certifying --build-cert

  # build the certificate of any route that fills it: `lake build Certificate`
  uv run verith mymodule.py --buchi "(= s0 0)" --invariant "(<= s0 100)" --ranking "s0" \\
      --build-cert -o out/ -p MyProject
"""


def _build_cert(args, project_dir: Path, cert_data) -> None:
    """Honour `--build-cert`, once the certificate is written.

    `cert_data` is what the project's certificate was generated from, or
    `None` when the certificate came from elsewhere (`--fbk-proveit`
    installs ic3ia's). The predicates are re-checked here rather than only
    at parse time because `--infer` can return without one: an LLM that
    answers nothing usable leaves `inv` or `ranking` unset, and building
    that would report a proof of `sorry`.
    """
    if not args.build_cert:
        return
    if cert_data is not None:
        # A safety certificate has no ranking function to be missing.
        required = [("property", cert_data.prp), ("invariant", cert_data.inv)]
        if not cert_data.is_safety:
            required.append(("ranking", cert_data.ranking))
        missing = [name for name, value in required if not value]
        if missing:
            raise SystemExit(
                f"error: --build-cert: the certificate has no "
                f"{', '.join(missing)}, so every obligation it carries is "
                "`sorry`. With --infer this means inference produced none."
            )
    print(f".. Building the certificate in {project_dir}")
    try:
        build_certificate(project_dir)
    except LakeBuildError as e:
        raise SystemExit(f"error: --build-cert: {e}") from e


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Lean4 certificate project from a Python reactive module. "
            "The property goes under --safety (`G FORMULA`: every reachable "
            "state) or --buchi (`G (F FORMULA)`: infinitely often); with --infer "
            "the invariant is searched for -- by an LLM, or by learning a "
            "ranking function and certifying it -- along with the ranking "
            "function a Buchi certificate also needs."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "module_file",
        nargs="?",
        help=(
            "Path to a Python file defining the module (must contain a callable that "
            "returns a Module). Optional when --hammer-file is used alone."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory where the Lean project will be created (default: current directory).",
    )
    parser.add_argument(
        "-p",
        "--project-name",
        default="Rea",
        help="Name of the Lean package / library (default: Rea).",
    )
    parser.add_argument(
        "-d",
        "--module-def",
        default="module",
        help="Name of the function or class in the Python file that produces the Module (default: module).",
    )
    parser.add_argument(
        "-x",
        "--executable",
        action="store_true",
        help="Generate Main.lean and add [[lean_exe]] to lakefile for a runnable binary.",
    )
    parser.add_argument(
        "--safety",
        default=None,
        metavar="FORMULA",
        help=(
            "Safety property `G FORMULA`: an SMT-LIB 2 Bool expression over "
            "state vars `s0..sN-1` (ctrl-next components) that must hold in "
            "*every* reachable state. Certified with `rule_globally`, which "
            "needs an invariant implying FORMULA and no ranking function at "
            "all. Certifiable by --fbk-proveit (ic3ia finds the invariant) "
            "or by --infer ai-cegar. Example: '(not (= s0 15))'."
        ),
    )
    parser.add_argument(
        "--buchi",
        default=None,
        metavar="FORMULA",
        help=(
            "Buchi property `G (F FORMULA)`: an SMT-LIB 2 Bool expression "
            "over state vars `s0..sN-1` that must hold infinitely often. "
            "Certified with `rule_buchi`, which needs an invariant *and* a "
            "ranking function that decreases in every state where FORMULA "
            "is false. Inferrable by --infer. Example: '(= s0 0)'."
        ),
    )
    parser.add_argument(
        "--pre",
        default=None,
        help=(
            "Precondition as an SMT-LIB 2 Bool expression over input vars "
            "`e0..eM-1` (extl-next) and `el0..elM-1` (extl-latched). "
            "Tuple (matrix) elements use '((_ tuple.select k) eK)'. "
            "Applied to both init_pre and update_pre. "
            "Example: '(>= ((_ tuple.select 0) e0) 1.0)'."
        ),
    )
    parser.add_argument(
        "--invariant",
        default=None,
        help=(
            "Invariant as an SMT-LIB 2 Bool expression over state vars "
            "`s0..sN-1`. When combined with --infer, only a ranking "
            "function is inferred (the invariant is fixed) -- and with "
            "--safety, which infers no ranking function, the certificate is "
            "then complete and only checked. Example: '(= s0 s5)'."
        ),
    )
    parser.add_argument(
        "--ranking",
        default=None,
        help=(
            "Ranking as an SMT-LIB 2 Int expression over state vars "
            "`s0..sN-1`, for --buchi only. When combined with --infer, only "
            "an invariant is inferred. Must satisfy `ranking >= 0` under the "
            "invariant. Example: '(ite s5 (ite s4 1 2) 0)'."
        ),
    )
    parser.add_argument(
        "--infer",
        nargs="?",
        const="ai-cegar",
        default=None,
        choices=["ai", "ai-cegar", "nuterm"],
        help=(
            "Infer the certificate for --safety or --buchi: the invariant, "
            "plus the ranking function --buchi needs. `ai` uses plain LLM "
            "self-check; `ai-cegar` uses LLM + cvc5 counterexample-guided "
            "refinement (default when --infer is passed without a value); "
            "`nuterm` trains a ranking function on rollouts of the module and "
            "returns it only once a Farkas/CEGAR decision procedure has "
            "certified it, with the invariant inferred by Houdini -- no LLM "
            "and no API key, but it reads scalar-integer state only."
        ),
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="LLM model to use for inference (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL, e.g. http://localhost:11434/v1 for Ollama "
            "or https://openrouter.ai/api/v1 for OpenRouter. Hosted providers read "
            "the API key from OPENROUTER_API_KEY or OPENAI_API_KEY; local servers "
            "need no key."
        ),
    )
    parser.add_argument(
        "--cert-file",
        default=None,
        help=(
            "Write a standalone, self-contained certificate .lean file to this path "
            "instead of creating a full project.  The file inlines init/update and "
            "imports zeroth_hammer from ZerothHammer.  The other encodings of the "
            "same module are written beside it under the same stem (Rel, Scalar, "
            "ScalarRel), as they are in a project."
        ),
    )
    parser.add_argument(
        "--hammer-file",
        default=None,
        help="Write a standalone ZerothHammer.lean to this path.",
    )
    parser.add_argument(
        "--fbk-proveit",
        default=None,
        metavar="DIR",
        help=(
            "Path to a `lean-ltl-certifying` checkout. Certify through its "
            "proveit.py (lean2vmt -> ic3ia -> vmt2lean) instead of verith's "
            "own route: the project is generated bare, an NA-encoded model "
            "is written to <project>/ProveIt/, and the resulting certificate "
            "is installed as <project>/Certificate/Certificate.lean, which "
            "the project's root module imports. The lakefile requires this "
            "checkout so that certificate builds. "
            "Requires --safety -- ic3ia decides `G PROPERTY`, not `G (F "
            "PROPERTY)`; rejects --infer/--invariant/--ranking/--pre."
        ),
    )
    parser.add_argument(
        "--ic3ia",
        default=None,
        metavar="PATH",
        help=(
            "Path to the ic3ia executable, forwarded to proveit.py. "
            "When omitted proveit.py falls back to $IC3IA, then to `ic3ia` "
            "on PATH. Only meaningful with --fbk-proveit."
        ),
    )
    parser.add_argument(
        "--fbk-simplify",
        default="cvc5",
        choices=["cvc5", "none"],
        help=(
            "Whether cvc5's rewriter runs over the transition the NA model "
            "carries (default: cvc5). `none` leaves each state slot the "
            "shape the module's own terms give it -- the same transition, "
            "spelled the way the module spells it (`x - 1` rather than "
            "`-1 + x`), which is what to pass when reading the generated "
            "Lean against the source. The rewriter is what makes a constant "
            "matrix tractable, though: a 32-wide affine layer is 97 KB of "
            "term unfolded and 1.9 KB folded. Only meaningful with "
            "--fbk-proveit."
        ),
    )
    parser.add_argument(
        "--fbk-equiv",
        default="lean",
        choices=["lean", "none"],
        help=(
            "Whether to emit the proof that the NA model is the module "
            "(default: lean). The certificate proveit.py installs is about "
            "the *model*; this is what carries it back to the module, and "
            "without it a mistranslation anywhere in the SMT path would be "
            "a machine-checked certificate about a different system. "
            "`none` skips it -- it is the expensive part on a wide state "
            "(73 s at 32 slots against ~5 s for the rest of the route). "
            "Only meaningful with --fbk-proveit."
        ),
    )
    parser.add_argument(
        "--build-cert",
        action="store_true",
        help=(
            "Build the generated certificate: `lake update` then `lake "
            "build Certificate` in the project. Needs a certificate with "
            "predicates in it, so pass --invariant and --ranking, or "
            "--infer, or --fbk-proveit; an obligation left as `sorry` is an "
            "error, as is one lake cannot close. Off by default because the "
            "build resolves and compiles cslib, Mathlib, lean-smt and cvc5 "
            "-- minutes against a warm package cache, much longer against a "
            "cold one."
        ),
    )

    parser.add_argument(
        "--pre-check",
        default="none",
        choices=["none", "cvc5"],
        help=(
            "Before generating, ask cvc5 whether the certificate obligations "
            "are actually true. A refuted obligation means the invariant or "
            "ranking is wrong, which a failing `lake build` cannot tell you "
            "apart from tactics that are merely too weak. With --infer the "
            "check runs after inference, on the certificate that was "
            "inferred -- before it there is no invariant to check. Bounded "
            "by --smt-timeout / --smt-budget; anything cvc5 cannot answer is "
            "reported as unknown and changes nothing."
        ),
    )
    parser.add_argument(
        "--smt-tactics",
        default="none",
        choices=["none", "cvc5"],
        help=(
            "Let cvc5 inform the generated tactics: discharge the branch "
            "conditions the invariant already settles instead of splitting "
            "on them, hand `nlinarith` the products cvc5's own refutation "
            "needed, and drop a precondition no refutation used. Bounded by "
            "--smt-timeout / --smt-budget; strictly additive, so anything "
            "cvc5 cannot answer leaves the plan as it would have been."
        ),
    )
    parser.add_argument(
        "--smt-timeout",
        type=int,
        default=DEFAULT_CALL_MS,
        metavar="MS",
        help=(
            f"Wall-clock limit for a single SMT query, in ms "
            f"(default: {DEFAULT_CALL_MS})."
        ),
    )
    parser.add_argument(
        "--smt-budget",
        type=int,
        default=DEFAULT_PHASE_MS,
        metavar="MS",
        help=(
            f"Wall-clock limit for one phase of SMT queries, in ms "
            f"(default: {DEFAULT_PHASE_MS}). A phase that fans out over many "
            f"small queries still cannot exceed this."
        ),
    )

    args = parser.parse_args()

    # An option given an empty string -- `--fbk-proveit "$LTL"` with `LTL`
    # unset, which is how the BENCHMARKS.md invocations are written -- reads
    # as "not passed" to every truthiness test below.  The route behind the
    # flag would be dropped and the run would *succeed*, having quietly
    # generated an ordinary project instead.  No option here has a
    # meaningful empty value, so the blank itself is the error.
    blank = [
        action.option_strings[-1]
        for action in parser._actions
        if action.option_strings and getattr(args, action.dest, None) == ""
    ]
    if blank:
        verb = "was" if len(blank) == 1 else "were"
        parser.error(
            f"{', '.join(blank)} {verb} given an empty value (an unset "
            "shell variable?). Pass a real one or drop the flag."
        )

    # A certificate proves one property under one proof rule, and which rule
    # decides what the certificate even consists of -- a ranking function or
    # no ranking function. So the two flags are not a pair of properties to
    # be combined; they are the choice.
    if args.safety and args.buchi:
        parser.error(
            "--safety and --buchi are mutually exclusive: a certificate "
            "proves `G FORMULA` or `G (F FORMULA)`, under one proof rule"
        )
    # `phase_left_ms` is `max(0, phase_ms - spent)` and a phase is exhausted
    # at zero, so a non-positive budget is not "unlimited": it is a phase
    # that never runs a query and says nothing about why.
    for flag, value in (("--smt-timeout", args.smt_timeout), ("--smt-budget", args.smt_budget)):
        if value <= 0:
            parser.error(f"{flag} must be a positive number of milliseconds, got {value}")

    property_smt = args.safety or args.buchi
    kind = "safety" if args.safety else "buchi"

    # `G FORMULA` needs no ranking function -- `rule_globally` has nowhere to
    # put one. Taking it and dropping it would look like it had been used.
    if args.safety and args.ranking:
        parser.error(
            "--ranking is meaningless with --safety: `G FORMULA` is proved "
            "by an invariant that implies FORMULA, and `rule_globally` takes "
            "no ranking function. Ranking functions belong to --buchi."
        )

    # --fbk-proveit takes over the whole certification route, so anything it
    # would have to ignore is an error rather than a silent no-op.
    ltl_project = None
    if args.fbk_proveit is not None:
        if args.buchi:
            parser.error(
                "--fbk-proveit is incompatible with --buchi: proveit.py "
                "proves `G PROPERTY` through ic3ia, which is a safety "
                "question. Pass --safety, or infer a Buchi certificate with "
                "--infer ai-cegar."
            )
        if not args.safety:
            parser.error("--fbk-proveit requires --safety")
        # This route is the one that turns the project name into a Lean
        # module name -- `<Proj>NA`, which the certificate imports and the
        # lakefile declares. `import 1projNA` is "unexpected token; expected
        # identifier", four steps later and in generated code.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", args.project_name):
            parser.error(
                f"--fbk-proveit needs a -p that is a Lean identifier: "
                f"`{na_module_name(args.project_name)}` is the module name "
                f"the NA model and the certificate are written under, and "
                f"`{args.project_name}` does not start one"
            )
        conflicts = [
            name
            for name, value in (
                ("--infer", args.infer),
                ("--invariant", args.invariant),
                ("--ranking", args.ranking),
                ("--pre", args.pre),
                ("--cert-file", args.cert_file),
                ("--hammer-file", args.hammer_file),
            )
            if value
        ]
        if conflicts:
            parser.error(
                f"--fbk-proveit is incompatible with {', '.join(conflicts)}: "
                "the invariant comes from ic3ia and the project is generated "
                "bare"
            )
        # Resolved here rather than inside the route: the lakefile written by
        # `create_project` names this path, so a checkout that is not one has
        # to be rejected before any of the project exists.
        from .fbk_proveit import ProveItError, resolve_project

        try:
            ltl_project = resolve_project(args.fbk_proveit)
        except ProveItError as e:
            parser.error(str(e))
    else:
        stray = [
            name
            for name, given in (
                ("--ic3ia", bool(args.ic3ia)),
                ("--fbk-simplify", args.fbk_simplify != "cvc5"),
                ("--fbk-equiv", args.fbk_equiv != "lean"),
            )
            if given
        ]
        if stray:
            verb = "is" if len(stray) == 1 else "are"
            parser.error(
                f"{', '.join(stray)} {verb} only meaningful together with "
                "--fbk-proveit"
            )

    # --build-cert is route-independent, but it needs a certificate worth
    # building. A bare project's obligations are all `sorry`: lake compiles
    # that and proves nothing, so asking for it is a mistake, not a no-op.
    if args.build_cert:
        if args.cert_file or args.hammer_file:
            which = "--cert-file" if args.cert_file else "--hammer-file"
            parser.error(
                f"--build-cert is incompatible with {which}: that writes a "
                "file and generates no project to build"
            )
        # A safety certificate is complete without a ranking function, so
        # what counts as "worth building" follows the kind.
        supplied = property_smt and args.invariant and (args.safety or args.ranking)
        if not (args.fbk_proveit or args.infer or supplied):
            parser.error(
                "--build-cert needs a certificate to build: pass --safety "
                "with --invariant, or --buchi with --invariant and "
                "--ranking, or --infer, or --fbk-proveit. Without them "
                "every obligation is `sorry`."
            )

    # --hammer-file: generate ZerothHammer.lean and exit (no module needed)
    if args.hammer_file:
        out = Path(args.hammer_file)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(generate_zeroth_hammer_lean())
        except OSError as e:
            raise SystemExit(f"error: --hammer-file: {e}") from e
        print(f"Wrote {out}")
        return

    if not args.module_file:
        parser.error("module_file is required (unless --hammer-file is used alone)")

    if args.infer and not property_smt:
        parser.error("--infer requires --safety or --buchi")

    # `--infer ai` prompts for Lean and cross-checks with a second LLM call,
    # and both halves are written around an invariant *and* a ranking
    # function. Only the cvc5 loop and the learner state the safety
    # obligations.
    if args.safety and args.infer == "ai":
        parser.error(
            "--safety needs --infer ai-cegar or --infer nuterm: the `ai` route "
            "infers a ranking function `rule_globally` cannot take, and nothing "
            "checks that the invariant implies the property"
        )

    # The learner computes the whole certificate and proves it before handing
    # it over, so a predicate supplied alongside would be discarded -- and it
    # assumes nothing of the inputs, so a precondition would be too. Both are
    # said rather than silently dropped.
    if args.infer == "nuterm":
        discarded = [
            name
            for name, value in (
                ("--invariant", args.invariant),
                ("--ranking", args.ranking),
                ("--pre", args.pre),
            )
            if value
        ]
        if discarded:
            parser.error(
                f"--infer nuterm is incompatible with {', '.join(discarded)}: the "
                "certificate comes from the learner and its invariant holds at "
                "entry for every input, which is stronger than any precondition "
                "would make it"
            )
        if args.model != parser.get_default("model") or args.base_url:
            parser.error(
                "--model and --base-url are for the LLM routes; --infer nuterm "
                "trains its own ranking function"
            )

    # The standalone certificate is written before inference runs and then
    # returns, so the inferred predicates could never reach it -- and the
    # pre-check that waits for them would never run either.
    if args.infer and args.cert_file:
        parser.error(
            "--infer is incompatible with --cert-file: the standalone "
            "certificate is written from the predicates as supplied"
        )

    cert_data: CertificateData | None = None
    if property_smt or args.pre or args.invariant or args.ranking:
        cert_data = CertificateData(prp=property_smt, kind=kind)
        if args.pre:
            cert_data.init_pre = args.pre
            cert_data.update_pre = args.pre
        if args.invariant:
            cert_data.inv = args.invariant
        if args.ranking:
            cert_data.ranking = args.ranking

    # The loader refuses a path that is not there, a file that is not a
    # module and a `-d` the file does not define. All three are the user's
    # input, so they are reported rather than raised.
    try:
        module = load_module_from_file(args.module_file, module_def=args.module_def)
    except (OSError, AttributeError, RuntimeError) as e:
        raise SystemExit(f"error: {e}") from e
    print(module)

    # The proveit route's own shape check, before a line of Lean is written.
    # Left until `run`, a module the NA encoding cannot express is met first
    # by `create_project` -- which fails about the functional encoding, in a
    # traceback, for a route that was never going to use it.
    #
    # Making the check *is* encoding the module into cvc5, so what it
    # answered with is kept and handed to `run_proveit` below: the NA model
    # and the bridge are written from it rather than from two more
    # encodings of the same module.
    na_bodies = None
    if args.fbk_proveit:
        from .fbk_proveit import ProveItError, check_module

        try:
            na_bodies = check_module(module, args.fbk_simplify == "cvc5")
        except ProveItError as e:
            raise SystemExit(f"error: {e}") from e

    # Translate SMT-LIB predicates to Lean expression strings for codegen.
    # `cert_data` keeps the original SMT source so `magic` can parse it with
    # its own cvc5 context.
    budget = SmtBudget(per_call_ms=args.smt_timeout, phase_ms=args.smt_budget)

    # With --infer the certificate the project gets is the inferred one, so
    # the check has to wait for it -- run at this point, there is no
    # invariant yet to refute. `magic` overwrites `inv` and `ranking`
    # regardless, so anything supplied is about to be discarded anyway.
    if args.pre_check == "cvc5" and not args.infer:
        if cert_data is None:
            print(".. SMT pre-check: nothing to check (no certificate data)")
        else:
            pre_check(module, cert_data, budget)

    # The hints have to come first: a settled branch condition is spliced
    # into a tactic in expanded form, so the definition it has to match must
    # be printed unshared.
    hints = None
    if args.smt_tactics == "cvc5" and cert_data is not None:
        print(
            f".. SMT-informed tactics (cvc5): <={budget.per_call_ms} ms per "
            f"query, <={budget.phase_ms} ms total"
        )
        hints = solver_hints(ModuleQueries.build(module, cert_data), budget, log=print)

    project_cert_data = cert_data
    if cert_data is not None:
        try:
            project_cert_data = smt_predicates_to_lean(
                cert_data, module, share=not (hints and hints.determined)
            )
        except Refused as e:
            raise SystemExit(f"error: {e}") from e
        project_cert_data.hints = hints

    # --cert-file: generate a standalone, self-contained certificate file and exit
    if args.cert_file:
        out = Path(args.cert_file)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            lean_src = generate_standalone_cert_lean(module, project_cert_data)
            out.write_text(lean_src)
        except (Refused, OSError) as e:
            raise SystemExit(f"error: --cert-file: {e}") from e
        print(f"Wrote standalone certificate: {out}")
        # The same encoding table the project route walks, so these files
        # and the ones in a generated project carry the same headers and
        # the same imports. The functional encoding is already inlined in
        # the standalone certificate above, which is the base they import.
        m2l = ModuleToLean4(module)
        layout = Layout(base=out.stem, flat=True, directory=out.parent)
        for enc in ENCODINGS:
            if not enc.cert_file:
                continue
            path = write_encoding(enc, m2l, layout, out.stem)
            print(f"Wrote {enc.title.lower()}: {path}")
        return

    print(".. Generating lean code")
    try:
        project_dir = create_project(
            output_dir=Path(args.output_dir),
            module=module,
            project_name=args.project_name,
            executable=args.executable,
            cert_data=project_cert_data,
            module_file=args.module_file,
            ltl_project=ltl_project,
        )
    except (Refused, OSError) as e:
        raise SystemExit(f"error: {e}") from e

    if args.fbk_proveit is not None:
        from .fbk_proveit import ProveItError, run as run_proveit

        print(".. Certifying through lean-ltl-certifying's proveit.py")
        try:
            run_proveit(
                ltl_project=ltl_project,
                module=module,
                project_dir=project_dir,
                project_name=args.project_name,
                property_smt=property_smt,
                ic3ia=args.ic3ia,
                simplify=args.fbk_simplify == "cvc5",
                equivalence=args.fbk_equiv == "lean",
                bodies=na_bodies,
            )
        except ProveItError as e:
            raise SystemExit(f"error: {e}") from e
        # The installed certificate carries ic3ia's invariant, so there is
        # nothing of `project_cert_data` left to be complete about.
        _build_cert(args, project_dir, cert_data=None)
        print(f"\nProject ready at: {project_dir}")
        return

    lean_code = project_dir / "System" / "System.lean"

    print(".. Doing TA2Magic")
    if args.infer:
        # A missing package and a missing key are both about how --infer was
        # asked for, not about the module.
        try:
            if args.infer == "ai":
                from .magic_ai import TA2MagicAI

                magic = TA2MagicAI(
                    lean_code.read_text(), model=args.model, base_url=args.base_url
                )
            elif args.infer == "nuterm":
                from .magic_learn import TA2MagicLearn

                magic = TA2MagicLearn(lean_code.read_text(), module)
            else:  # "ai-cegar"
                from .magic_cegar import TA2MagicCEGAR

                magic = TA2MagicCEGAR(
                    lean_code.read_text(),
                    module,
                    model=args.model,
                    base_url=args.base_url,
                )
            cert_data = magic.infer(cert_data)
        except (Refused, ImportError) as e:
            raise SystemExit(f"error: --infer: {e}") from e

        if args.pre_check == "cvc5":
            pre_check(module, cert_data, budget)

        # Merge inferred inv/ranking into project_cert_data. The kind comes
        # along: it decides which proof rule the rewritten files state, and
        # `ranking` is `None` for a safety certificate by construction.
        if project_cert_data is None:
            project_cert_data = CertificateData(kind=kind)
        project_cert_data.inv = cert_data.inv
        project_cert_data.ranking = cert_data.ranking

    # After inference the predicates are new, so both files are rewritten:
    # Data.lean holds the definitions, and Certificate.lean's tactics are
    # generated from their shape.
    if args.infer:
        write_data_lean(project_dir, args.project_name, module, project_cert_data)
        write_certificate_lean(
            project_dir, args.project_name, module, project_cert_data
        )

    _build_cert(args, project_dir, cert_data=project_cert_data)

    print(f"\nProject ready at: {project_dir}")


if __name__ == "__main__":
    main()
