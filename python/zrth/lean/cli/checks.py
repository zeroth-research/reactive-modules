"""The rules about which flags go together, each read off the rows it is
about: an input format's options, a route's options, kinds and seeds,
and the outputs a run can write.
"""

from __future__ import annotations

import argparse

from ..artifacts import ARTIFACTS_DIR
from ..core.options import given, missing, put_defaults
from ..inputs import ADAPTERS, InputAdapter, adapter_by_name, adapter_for
from ..infer import InferRoute, all_route_aliases, shared_route_options
from .settings import SEED_FLAGS, Settings


def check_blanks(parser: argparse.ArgumentParser, args) -> None:
    """An option given an empty string is the error, not "not passed".

    `--proveit-dir "$LTL"` with `LTL` unset -- which is how a copied
    benchmark command reads -- reads as "not passed" to every
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


def resolve_aliases(parser: argparse.ArgumentParser, args) -> None:
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


def input_adapter(parser: argparse.ArgumentParser,
                  args) -> InputAdapter | None:
    """The format the input is read in: `--format`, else its suffix's."""
    if args.format:
        return adapter_by_name(args.format)
    if not args.module_file:
        return None
    adapter = adapter_for(args.module_file)
    if adapter is None:
        parser.error(
            f"cannot tell the format of {args.module_file} from its suffix; "
            f"pass --format ({', '.join(a.name for a in ADAPTERS)})"
        )
    return adapter


def check_input_options(parser: argparse.ArgumentParser, args,
                        adapter: InputAdapter | None) -> None:
    """A format's option is only meaningful for input in that format, as a
    route's is only with that route; and its row default is put back once
    the format is known."""
    stray = [
        (a, o) for a in ADAPTERS for o in a.options
        if a is not adapter and given(o, args)
    ]
    if stray:
        flags = [o.flags[-1] for _, o in stray]
        formats = sorted({a.name for a, _ in stray})
        verb = "is" if len(flags) == 1 else "are"
        parser.error(
            f"{', '.join(flags)} {verb} only meaningful for "
            f"{' or '.join(formats)} input"
        )
    put_defaults(adapter.options if adapter else (), args)


def check_route_options(
    parser: argparse.ArgumentParser, args, route: "InferRoute | None"
) -> None:
    """A route-scoped option is only meaningful with its route.

    Given without it, the flag would do nothing -- which is the failure mode
    worth reporting: the run would succeed having ignored what was asked.
    """
    # An option several routes declare is stray only when *none* of them
    # was selected: `--vampire` is meaningful to two routes and a no-op
    # under the other five.
    stray = [
        (owners, o)
        for o, owners in shared_route_options()
        if route not in owners and given(o, args)
    ]
    if stray:
        flags = [o.flags[-1] for _, o in stray]
        owners = sorted({f"--infer {r.name}" for rs, _ in stray for r in rs})
        verb = "is" if len(flags) == 1 else "are"
        parser.error(
            f"{', '.join(flags)} {verb} only meaningful together with "
            f"{', '.join(owners)}"
        )
    if route is None:
        return
    absent = [o.flags[-1] for o in missing(route.options, args)]
    if absent:
        parser.error(
            f"--infer {route.name} requires {', '.join(absent)}"
        )
    put_defaults(route.options, args)


def check_route(parser: argparse.ArgumentParser, args, settings: Settings) -> None:
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
        for field, attr, flag in SEED_FLAGS
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


def check_outputs(parser: argparse.ArgumentParser, settings: Settings) -> None:
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
        if not (settings.route or settings.supplied_complete
                or settings.claim_deferred):
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
    # for alongside it would never run -- any route, the file routes too, for
    # the same reason `--build-cert` is incompatible with them: there is no
    # project.
    if settings.hammer_file and settings.route is not None:
        parser.error(
            f"--infer {settings.route.name} is incompatible with --hammer-file: that "
            f"writes ZerothHammer.lean and returns, and generates no "
            f"certificate to infer"
        )

    # `--hammer-file` alone writes one tactic file and needs no module.
    if not settings.module_file and not settings.hammer_file:
        parser.error("module_file is required (unless --hammer-file is used alone)")
