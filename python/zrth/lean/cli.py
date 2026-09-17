"""The `uv run verith` command line: the flags, and what combinations mean.

Split out of `main.py` so that the pipeline and the argument grammar are two
things.  What `main` gets back is :class:`Settings` -- validated, with the
property and its kind already decided and the inference route already looked
up -- rather than an `argparse.Namespace` to re-derive facts from.

Nothing here knows a route by name.  Every cross-flag rule about inference
reads a field of the route's row in :mod:`zrth.lean.infer_route`: the check
is generic, and the sentence printed when it fails comes from the row, so a
refusal still says why *that* route refuses.  Adding a route is a row there
and no edit here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import ARTIFACTS_DIR
from .infer_route import (
    DEFAULT_ROUTE,
    ROUTES,
    InferRoute,
    all_route_aliases,
    all_route_options,
    route_by_name,
    route_names,
)
from .smt_query import DEFAULT_CALL_MS, DEFAULT_PHASE_MS, SmtBudget

_EPILOG = """\
examples:
  # bare project — no property, so `P` and `ranking` are sorry
  uv run verith mymodule.py -o out/ -p MyProject

  # a Buchi property, `G (F (= s0 0))` (certificate stub with known P)
  uv run verith mymodule.py --buchi "(= s0 0)" -o out/ -p MyProject

  # a safety property, `G (<= s0 100)`: invariant only, no ranking function
  uv run verith mymodule.py --safety "(<= s0 100)" -o out/ -p MyProject

  # AI inference with Claude (requires ANTHROPIC_API_KEY + pip install zrth[ai])
  uv run verith mymodule.py --buchi "(= s0 0)" --infer -o out/ -p MyProject

  # ... and the same loop for a safety property (cvc5-checked route only)
  uv run verith mymodule.py --safety "(<= s0 100)" --infer ai-cegis -o out/ -p MyProject

  # no LLM: learn a ranking function and certify it before it is offered
  uv run verith mymodule.py --buchi "(= s0 0)" --infer nuterm -o out/ -p MyProject

  # no LLM: synthesise the invariant, congruences included (`x` even), and
  # leave it in artifacts/ for the next run to take as given
  uv run verith mymodule.py --safety "(not (= s0 1))" --infer sygus -o out/ -p MyProject

  # no LLM: one query per shape -- and a refuted shape is a *proof* that no
  # certificate of that shape exists, which --infer ai-cegis reads next run
  uv run verith mymodule.py --buchi "(= s0 0)" --infer smt-linear -o out/ -p MyProject

  # no LLM: Houdini and a ranking search, each candidate decided by cvc5 --
  # a candidate it cannot prove comes back with the counterexample that kills it
  uv run verith mymodule.py --buchi "(= s0 0)" --infer houdini -o out/ -p MyProject

  # ... the same search, refuted by the Vampire prover instead
  uv run verith mymodule.py --buchi "(= s0 0)" --infer houdini \\
      --houdini-solver vampire --vampire ~/vampire/vampire -o out/ -p MyProject

  # AI inference with Ollama (requires pip install zrth[ai-local])
  uv run verith mymodule.py --buchi "(= s0 0)" --infer \\
      --model qwen3-coder --base-url http://localhost:11434/v1 -o out/ -p MyProject

  # ask cvc5 whether the obligations are true before spending a lake build on them
  uv run verith mymodule.py --buchi "(= s0 0)" --invariant "(<= s0 100)" --ranking "s0" \\
      --pre-check cvc5 -o out/ -p MyProject

  # certify through lean-ltl-certifying's proveit.py (lean2vmt -> ic3ia -> vmt2lean)
  uv run verith mymodule.py --safety "(not (= s0 15))" -o out/ -p MyProject \\
      --infer fbk-proveit --proveit-dir ~/lean-ltl-certifying --ic3ia ~/ic3ia/build/ic3ia

  # build the certificate of any route that fills it: `lake build Certificate`
  uv run verith mymodule.py --buchi "(= s0 0)" --invariant "(<= s0 100)" --ranking "s0" \\
      --build-cert -o out/ -p MyProject
"""


# One row per predicate flag: the `CertificateData` field it fills (which is
# what a route's `seeds` set is written in), the `Settings` attribute that
# carries it, and the flag to name in a refusal. The first two differ for
# `inv`, which is why the attribute is in the table rather than patched at
# the one place that reads it.
_SEED_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("inv", "invariant", "--invariant"),
    ("ranking", "ranking", "--ranking"),
    ("pre", "pre", "--pre"),
)


@dataclass
class Settings:
    """A validated command line.

    The decisions the grammar makes are made here, once: which proof rule
    (`kind`), which route (`route`), what that route resolved at parse time
    (`route_config`), and the SMT budget.  `main` reads; it does not re-derive.
    """

    module_file: str | None
    output_dir: Path
    project_name: str
    module_def: str
    executable: bool

    prp: str | None                     # the property, whichever flag carried it
    kind: str                           # "safety" | "buchi"

    pre: str | None
    invariant: str | None
    ranking: str | None

    route: InferRoute | None = None
    route_opts: dict = field(default_factory=dict)
    route_config: object | None = None

    cert_file: str | None = None
    hammer_file: str | None = None
    build_cert: bool = False

    pre_check: str = "none"
    smt_tactics: str = "none"
    artifacts: str = "use"              # "use" | "ignore" | "reset"
    budget: SmtBudget = field(
        default_factory=lambda: SmtBudget(
            per_call_ms=DEFAULT_CALL_MS, phase_ms=DEFAULT_PHASE_MS
        )
    )

    model: str | None = None
    base_url: str | None = None

    @property
    def is_safety(self) -> bool:
        return self.kind == "safety"

    @property
    def supplied_complete(self) -> bool:
        """Whether the flags alone carry a certificate worth building.

        A safety certificate is complete without a ranking function, so what
        counts follows the kind.
        """
        return bool(self.prp and self.invariant and (self.is_safety or self.ranking))


def _infer_help() -> str:
    """`--infer`'s help, assembled from the rows."""
    lines = [
        "Infer the certificate for --safety or --buchi: the invariant, plus "
        "the ranking function --buchi needs. One of:",
    ]
    for r in ROUTES:
        kinds = " and ".join(f"--{k}" for k in sorted(r.kinds))
        lines.append(f"`{r.name}` ({kinds}) -- {r.summary}.")
    lines.append(f"Passed without a value, `{DEFAULT_ROUTE}`.")
    return " ".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verith",
        description=(
            "Generate a Lean4 certificate project from a Python reactive module. "
            "The property goes under --safety (`G FORMULA`: every reachable "
            "state) or --buchi (`G (F FORMULA)`: infinitely often); with --infer "
            "the invariant is searched for -- by an LLM, by learning a ranking "
            "function and certifying it, or by an external prover -- along with "
            "the ranking function a Buchi certificate also needs."
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
        help=(
            "Name of the function or class in the Python file that produces the "
            "Module (default: module)."
        ),
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
            "all. Example: '(not (= s0 15))'."
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
            "is false. Example: '(= s0 0)'."
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
            "`s0..sN-1`. With a route that seeds from it, only what is left "
            "of the certificate is inferred. Example: '(= s0 s5)'."
        ),
    )
    parser.add_argument(
        "--ranking",
        default=None,
        help=(
            "Ranking as an SMT-LIB 2 Int expression over state vars "
            "`s0..sN-1`, for --buchi only. With a route that seeds from it, "
            "only the invariant is inferred. Must satisfy `ranking >= 0` "
            "under the invariant. Example: '(ite s5 (ite s4 1 2) 0)'."
        ),
    )
    parser.add_argument(
        "--infer",
        nargs="?",
        const=DEFAULT_ROUTE,
        default=None,
        choices=list(route_names()),
        help=_infer_help(),
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

    # One group per route, so `--help` shows whose flag is whose.
    for route in ROUTES:
        if not route.options:
            continue
        group = parser.add_argument_group(f"--infer {route.name}")
        for opt in route.options:
            # `default=None` so that "given" is distinguishable from "not
            # given" for the stray-option check; the row's own default is
            # put back in `_check_route_options` once the route is known.
            kwargs = dict(opt.kwargs)
            kwargs["default"] = None
            group.add_argument(*opt.flags, **kwargs)

    # Deprecated spellings, from the rows that own them: each selects its
    # route and fills one of that route's options, which is why it cannot be
    # an `Opt` of its own.
    for route, alias in all_route_aliases():
        parser.add_argument(
            alias.flag, default=None, metavar="DIR", help=alias.help
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
        "--build-cert",
        action="store_true",
        help=(
            "Build the generated certificate: `lake update` then `lake "
            "build Certificate` in the project. Needs a certificate with "
            "predicates in it, so pass --invariant and --ranking, or "
            "--infer; an obligation left as `sorry` is an error, as is one "
            "lake cannot close. Off by default because the build resolves "
            "and compiles cslib, Mathlib, lean-smt and cvc5 -- minutes "
            "against a warm package cache, much longer against a cold one."
        ),
    )
    parser.add_argument(
        "--artifacts",
        default="use",
        choices=["use", "ignore", "reset"],
        help=(
            f"What to do with the project's `{ARTIFACTS_DIR}/` directory, "
            f"which is how a sequence of runs in one -o continues rather "
            f"than restarts: a predicate a previous run found but could not "
            f"prove enough of is left there, and a route that understands it "
            f"starts from it. `use` (default) lets the route read it -- an "
            f"artifact written for a different module, property or proof "
            f"rule is skipped, and what is picked up is printed, because two "
            f"identical command lines must not quietly mean different "
            f"things. `ignore` searches from scratch and still records what "
            f"this run finds. `reset` empties it first, which is what to "
            f"pass when a bad artifact is being inherited by every run."
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
    return parser


def parse_args(argv: "list[str] | None" = None) -> Settings:
    parser = build_parser()
    args = parser.parse_args(argv)

    _check_blanks(parser, args)
    _resolve_aliases(parser, args)

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
    for flag, value in (
        ("--smt-timeout", args.smt_timeout),
        ("--smt-budget", args.smt_budget),
    ):
        if value <= 0:
            parser.error(
                f"{flag} must be a positive number of milliseconds, got {value}"
            )

    # `G FORMULA` needs no ranking function -- `rule_globally` has nowhere to
    # put one. Taking it and dropping it would look like it had been used.
    if args.safety and args.ranking:
        parser.error(
            "--ranking is meaningless with --safety: `G FORMULA` is proved "
            "by an invariant that implies FORMULA, and `rule_globally` takes "
            "no ranking function. Ranking functions belong to --buchi."
        )

    route = route_by_name(args.infer) if args.infer else None
    _check_route_options(parser, args, route)

    settings = Settings(
        module_file=args.module_file,
        output_dir=Path(args.output_dir),
        project_name=args.project_name,
        module_def=args.module_def,
        executable=args.executable,
        prp=args.safety or args.buchi,
        kind="safety" if args.safety else "buchi",
        pre=args.pre,
        invariant=args.invariant,
        ranking=args.ranking,
        route=route,
        route_opts=(
            {o.dest: getattr(args, o.dest) for o in route.options} if route else {}
        ),
        cert_file=args.cert_file,
        hammer_file=args.hammer_file,
        build_cert=args.build_cert,
        pre_check=args.pre_check,
        artifacts=args.artifacts,
        smt_tactics=args.smt_tactics,
        budget=SmtBudget(per_call_ms=args.smt_timeout, phase_ms=args.smt_budget),
        model=args.model,
        base_url=args.base_url,
    )

    if route is not None:
        _check_route(parser, args, settings)

    _check_outputs(parser, settings)

    # Last, because it is the only check that touches the filesystem: a
    # route whose lakefile names a path has to reject a bad one before any
    # of the project exists, and the lakefile is written while it is created.
    if route is not None and route.resolve is not None:
        try:
            settings.route_config = route.resolve(settings.route_opts)
        except Exception as e:                      # the route's own error type
            parser.error(str(e))

    return settings


# ══════════════════════════════════════════════════════════════════════════
# The checks
# ══════════════════════════════════════════════════════════════════════════


def _check_blanks(parser: argparse.ArgumentParser, args) -> None:
    """An option given an empty string is the error, not "not passed".

    `--proveit-dir "$LTL"` with `LTL` unset -- which is how the
    BENCHMARKS.md invocations are written -- reads as "not passed" to every
    truthiness test below.  The route behind the flag would be dropped and
    the run would *succeed*, having quietly generated an ordinary project
    instead.  No option here has a meaningful empty value.
    """
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


def _resolve_aliases(parser: argparse.ArgumentParser, args) -> None:
    """Rewrite each deprecated spelling into the route and option it means.

    One loop over the rows, so a route's older flag goes away with the route
    and a second one is a row entry rather than another branch here.
    """
    for route, alias in all_route_aliases():
        given = getattr(args, alias.flag.lstrip("-").replace("-", "_"), None)
        if given is None:
            continue
        if args.infer not in (None, route.name):
            parser.error(
                f"{alias.flag} selects `--infer {route.name}`, so it cannot be "
                f"combined with --infer {args.infer}: a run takes one route"
            )
        explicit = getattr(args, alias.fills, None)
        flag = next(
            o.flags[-1] for o in route.options if o.dest == alias.fills
        )
        if explicit is not None and explicit != given:
            parser.error(
                f"{alias.flag} and {flag} name the same thing and were given "
                f"different values"
            )
        args.infer = route.name
        setattr(args, alias.fills, given)


def _check_route_options(
    parser: argparse.ArgumentParser, args, route: "InferRoute | None"
) -> None:
    """A route-scoped option is only meaningful with its route.

    Given without it, the flag would do nothing -- which is the failure mode
    worth reporting: the run would succeed having ignored what was asked.
    """
    # Every route option is parsed with `default=None`, so `None` is "not
    # given" -- including for an option whose row default is a real value.
    # An option passed the same value its row defaults to still counts as
    # given, which is what makes "only meaningful with" honest.
    stray = [
        (owner, o)
        for owner, o in all_route_options()
        if owner is not route and getattr(args, o.dest, None) is not None
    ]
    if stray:
        flags = [o.flags[-1] for _, o in stray]
        owners = sorted({f"--infer {owner.name}" for owner, _ in stray})
        verb = "is" if len(flags) == 1 else "are"
        parser.error(
            f"{', '.join(flags)} {verb} only meaningful together with "
            f"{', '.join(owners)}"
        )
    if route is None:
        return
    missing = [
        o.flags[-1]
        for o in route.options
        if o.required and getattr(args, o.dest, None) in (None, "")
    ]
    if missing:
        parser.error(
            f"--infer {route.name} requires {', '.join(missing)}"
        )
    # A route option's default has to survive the group's `default=None`
    # override, which is there so "given" is distinguishable from "not".
    for o in route.options:
        if getattr(args, o.dest, None) is None and "default" in o.kwargs:
            setattr(args, o.dest, o.kwargs["default"])


def _check_route(parser: argparse.ArgumentParser, args, settings: Settings) -> None:
    """Every rule about inference, read off the route's row."""
    route = settings.route
    if not settings.prp:
        # Naming the kinds this route has rather than both of them: a route
        # that only certifies `G P` asking for "--safety or --buchi" would
        # name a flag it goes on to refuse.
        kinds = " or ".join(f"--{k}" for k in sorted(route.kinds, reverse=True))
        parser.error(f"--infer {route.name} requires {kinds}")

    # Which proof rules this route can produce a certificate for.
    if settings.kind not in route.kinds:
        parser.error(
            route.kinds_refusal
            or (
                f"--infer {route.name} does not infer a {settings.kind} "
                f"certificate (it infers: {', '.join(sorted(route.kinds))})"
            )
        )

    # A predicate the route will not seed from would be discarded. Saying so
    # beats dropping it silently: the run would succeed, having proved
    # something about a certificate the user did not describe.
    discarded = [
        flag
        for field, attr, flag in _SEED_FLAGS
        if getattr(settings, attr) and field not in route.seeds
    ]
    if discarded:
        because = f": {route.seeds_refusal}" if route.seeds_refusal else ""
        parser.error(
            f"--infer {route.name} is incompatible with "
            f"{', '.join(discarded)}{because}"
        )

    # `--model` and `--base-url` are for the routes that call an LLM. Named
    # as the pair they are even when only one was passed: which of the two
    # was given is not the point, that this route calls no LLM is.
    if not route.uses_llm and (
        args.model != parser.get_default("model") or args.base_url
    ):
        parser.error(
            f"--model and --base-url are for the LLM routes; --infer "
            f"{route.name} calls none"
        )

    # Inference runs against the generated project, and `--cert-file` writes
    # a standalone file and returns before any of that -- so the route would
    # not run at all. Every route, not the ones that declare they read or
    # write the project: a route that declares neither still never runs.
    if settings.cert_file:
        parser.error(
            f"--infer {route.name} is incompatible with --cert-file: the "
            f"standalone certificate is written from the predicates as "
            f"supplied, before a route could infer any"
        )

    if route.extra_check is not None:
        problem = route.extra_check(settings, settings.route_opts)
        if problem:
            parser.error(problem)


def _check_outputs(parser: argparse.ArgumentParser, settings: Settings) -> None:
    # --build-cert is route-independent, but it needs a certificate worth
    # building. A bare project's unsupplied fields default to `True`
    # (`inv`, the preconditions) or to `sorry` (`P`, `ranking`), and its
    # obligations are still handed to the tactics -- which cannot close one
    # stated over a `sorry`, so `lake build` fails on it rather than
    # compiling to a proof of nothing. Asking for that is a mistake.
    if settings.build_cert:
        if settings.cert_file or settings.hammer_file:
            which = "--cert-file" if settings.cert_file else "--hammer-file"
            parser.error(
                f"--build-cert is incompatible with {which}: that writes a "
                "file and generates no project to build"
            )
        if not (settings.route or settings.supplied_complete):
            parser.error(
                "--build-cert needs a certificate to build: pass --safety "
                "with --invariant, or --buchi with --invariant and "
                "--ranking, or --infer. Without them the certificate's "
                "predicates are `True` or `sorry`, and its obligations "
                "cannot be discharged."
            )

    # `artifacts/` lives in the project, so the routes that write a file and
    # return have none. Said rather than ignored: the flag asks for
    # cross-run behaviour that this invocation cannot have.
    if settings.artifacts != "use" and (settings.cert_file or settings.hammer_file):
        which = "--cert-file" if settings.cert_file else "--hammer-file"
        parser.error(
            f"--artifacts {settings.artifacts} is incompatible with {which}: "
            f"`{ARTIFACTS_DIR}/` belongs to a generated project, and that "
            f"writes a file and generates none"
        )

    # `--hammer-file` writes one tactic file and returns, so a route asked
    # for alongside it would never run. `main.py` says this for one route;
    # it is true of every route, and of the file routes for the same reason
    # `--build-cert` is incompatible with them: there is no project.
    if settings.hammer_file and settings.route is not None:
        parser.error(
            f"--infer {settings.route.name} is incompatible with --hammer-file: that "
            f"writes ZerothHammer.lean and returns, and generates no "
            f"certificate to infer"
        )

    # `--hammer-file` alone writes one tactic file and needs no module.
    if not settings.module_file and not settings.hammer_file:
        parser.error("module_file is required (unless --hammer-file is used alone)")
