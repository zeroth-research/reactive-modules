"""The `uv run verith` command line: the flags, and what combinations mean.

Apart from the pipeline (:mod:`zrth.lean.pipeline`) so that the pipeline
and the argument grammar are two things.  What the pipeline gets back is
:class:`Settings` -- validated, with the property and its kind already
decided and the inference route already looked up -- rather than an
`argparse.Namespace` to re-derive facts from.  `parser` is the flags
(`build_parser`), `checks` the rules about which go together, `settings`
what they are read into; `parse_args` runs the three.

Nothing here knows a route by name.  Every cross-flag rule about inference
reads a field of the route's row in :mod:`zrth.lean.infer`: the check
is generic, and the sentence printed when it fails comes from the row, so a
refusal still says why *that* route refuses.  Adding a route is a row there
and no edit here.
"""

from __future__ import annotations

from pathlib import Path

from ..core.errors import Refused
from ..infer import route_by_name
from ..smt.limits import SmtBudget
from .checks import (
    check_blanks,
    resolve_aliases,
    input_adapter,
    check_input_options,
    check_route_options,
    check_route,
    check_outputs,
)
from .parser import build_parser
from .settings import SEED_FLAGS, Settings

__all__ = ["SEED_FLAGS", "Settings", "build_parser", "parse_args"]


def parse_args(argv: "list[str] | None" = None, *,
               defer_claim: bool = True) -> Settings:
    """The command line as `Settings`. With `defer_claim`, an input whose
    format can state its own property and no --safety/--buchi leaves the
    checks that need the claim to the second parse, once it is read."""
    parser = build_parser()
    args = parser.parse_args(argv)

    check_blanks(parser, args)
    resolve_aliases(parser, args)

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

    adapter = input_adapter(parser, args)
    check_input_options(parser, args, adapter)

    route = route_by_name(args.infer) if args.infer else None
    check_route_options(parser, args, route)

    settings = Settings(
        module_file=args.module_file,
        output_dir=Path(args.output_dir),
        project_name=args.project_name,
        input_format=adapter.name if adapter else None,
        input_opts=(
            {o.dest: getattr(args, o.dest) for o in adapter.options}
            if adapter else {}
        ),
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

    settings.claim_deferred = bool(
        defer_claim and adapter is not None and adapter.states_claims
        and not settings.prp
    )
    if route is not None and not settings.claim_deferred:
        check_route(parser, args, settings)

    check_outputs(parser, settings)

    # Last, because it is the only check that touches the filesystem: a
    # route whose lakefile names a path has to reject a bad one before any
    # of the project exists, and the lakefile is written while it is created.
    if route is not None and route.resolve is not None:
        try:
            settings.route_config = route.resolve(settings.route_opts)
        except Refused as e:
            parser.error(str(e))

    return settings
