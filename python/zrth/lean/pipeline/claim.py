"""What the run is about: the module, read from its input (`load`), and
the claim -- the one the flags ask (`claim_from_flags`), or the one the
input states (`stated_claim`). `show_certificate` prints an answer back
in the input's names.
"""

from __future__ import annotations

import shlex

from ..cli import Settings, parse_args
from ..core.certificate import Certificate, Claim
from .. import inputs


def load(settings: Settings) -> "inputs.Problem":
    """The module the run is about, and the input it was read from.

    The loader refuses a path that is not there or cannot be read, a file
    that is not a module and a `-d` the file does not define, all as
    `InputError`. All of them are the user's input, so they are reported
    rather than raised. What the module file's own code raises is not a
    refusal, and keeps its traceback: an `AttributeError` from a typo in
    `module()` reads, as one `error:` line, exactly like "must define
    `module`", and an `OSError` from a file it opens like a missing input.
    """
    try:
        return inputs.load(settings.module_file, settings.input_opts,
                           format=settings.input_format)
    except (inputs.ClaimError, inputs.InputError) as e:
        raise SystemExit(f"error: {e}") from e


def claim_from_flags(
    settings: Settings, names: "inputs.Names",
) -> "tuple[Claim | None, Certificate]":
    """What the flags ask to be proved, and what of its certificate they
    supply. No claim when no predicate flag was passed.

    The flags are in the module's own terms (`inputs.names`); the claim
    is in the internal names every later stage reads, and each flag is
    printed in them once, since that is how the logs, the artifacts and
    any refusal will spell it."""
    if not (settings.prp or settings.pre or settings.invariant
            or settings.ranking):
        return None, Certificate()
    prp_flag = "--safety" if settings.is_safety else "--buchi"

    def internal(text: "str | None", flag: str) -> "str | None":
        if not text:
            return None
        try:
            spelled = inputs.to_internal(text, names, flag)
        except inputs.ClaimError as e:
            raise SystemExit(f"error: {e}") from e
        print(f".. {flag}, in internal names: {spelled}")
        return spelled

    prp = internal(settings.prp, prp_flag)
    pre = internal(settings.pre, "--pre")
    claim = Claim(prp=prp, kind=settings.kind, init_pre=pre, update_pre=pre)
    return claim, Certificate(internal(settings.invariant, "--invariant"),
                              internal(settings.ranking, "--ranking"))


def show_certificate(cert: Certificate, names: "inputs.Names") -> None:
    """The route's answer in the input's names, as `--invariant` and
    `--ranking` take it: what one run found, the next can be given."""
    shown = [(flag, text) for flag, text in (("--invariant", cert.inv),
                                             ("--ranking", cert.ranking))
             if text]
    if not shown:
        return
    print(".. The certificate, in the input's names:")
    for flag, text in shown:
        print(f"   {flag} {shlex.quote(inputs.to_surface(text, names))}")


def stated_claim(settings: Settings, problem: "inputs.Problem",
                 argv: list[str]) -> Settings:
    """The command line, with the property the input states when it gives
    none (an SMV INVARSPEC or LTLSPEC).

    The stated property is the claim exactly as if it had been passed as
    `--safety`/`--buchi`: the command line is parsed again with it added,
    so every rule about the flags applies to it unchanged. Given both, the
    run would prove one and ignore the other, so that is refused.
    """
    stated = problem.claims
    chosen = settings.input_opts.get("spec")
    listing = "; ".join(f"{i}: {c.where}" for i, c in enumerate(stated, 1))
    if settings.prp:
        if chosen is not None:
            raise SystemExit(
                "error: --spec chooses one of the input's properties, and "
                "--safety/--buchi already gives the claim"
            )
        if stated:
            raise SystemExit(
                f"error: {problem.source.path} states its own property "
                f"({listing}); pass no --safety/--buchi to prove it, or "
                "remove it to prove another"
            )
        return settings
    if not settings.claim_deferred:
        return settings
    if not stated:
        if chosen is not None:
            raise SystemExit("error: --spec: the input states no property")
        return parse_args(argv, defer_claim=False)
    if chosen is None:
        if len(stated) > 1:
            raise SystemExit(
                f"error: the input states {len(stated)} properties "
                f"({listing}); choose one with --spec N"
            )
        chosen = 1
    if not 1 <= chosen <= len(stated):
        raise SystemExit(
            f"error: --spec {chosen}: the input states {len(stated)} "
            f"propert{'y' if len(stated) == 1 else 'ies'} ({listing})"
        )
    claim = stated[chosen - 1]
    print(f".. the claim is the input's {claim.where} (--{claim.kind})")
    return parse_args(
        [*argv, f"--{claim.kind}", inputs.to_surface(claim.text, problem.names)],
        defer_claim=False,
    )
