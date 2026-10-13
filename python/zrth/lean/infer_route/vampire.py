"""`--infer vampire`: Vampire's answer literals return the certificate's coefficients."""

from __future__ import annotations

# Constants only -- the module imports cvc5 lazily, so naming its defaults
# in the row costs a bare run nothing.
from ..houdini_solver import DEFAULT_TIMEOUT
from .api import InferInput, InferResult, InferRoute, Opt
from .shared import VAMPIRE


def _resolve(opts: dict):
    from ..smt.vampire import resolve_vampire

    return resolve_vampire(opts["vampire"])


def _run(inp: InferInput) -> InferResult:
    from ..magic.vampire import TA2MagicVampire

    magic = TA2MagicVampire(
        inp.module,
        vampire=inp.config,
        timeout=inp.opts["vampire_timeout"],
        artifacts=inp.project.artifacts,
        log=inp.log,
        encoding=inp.smt,
    )
    return InferResult(magic.infer(inp.claim, inp.given))


ROUTE = InferRoute(
    name="vampire",
    summary=(
        "no LLM, and nothing proposed: the obligations are stated with "
        "the certificate itself left open -- an interval per column, "
        "and under --buchi an affine ranking function -- and Vampire's "
        "answer literals return the coefficients, which are then checked "
        "against the module's encoding with cvc5. It bounds what it can "
        "order: Int and Real state, scalar or element by element, with "
        "a Bool column carried unbounded and a bitvector refused, since "
        "Vampire's front end has none. The transition is branch-split, "
        "because a conditional defeats the search. Measured reach: two "
        "holes, which is one column under --safety; --infer houdini is "
        "what searches a wider module"
    ),
    kinds=frozenset({"safety", "buchi"}),
    kinds_refusal="",
    seeds=frozenset({"pre"}),
    seeds_refusal=(
        "the whole certificate is what this route asks Vampire for, in "
        "one question, so a supplied predicate would be half of an "
        "answer it is not able to ask for half of"
    ),
    returns="smt",
    resolve=_resolve,
    run=_run,
    options=(
        VAMPIRE,
        Opt(
            ("--vampire-timeout",),
            dict(
                type=float,
                default=DEFAULT_TIMEOUT,
                metavar="SECONDS",
                help=(
                    f"--infer vampire: wall-clock budget for all "
                    f"answer-literal calls together (default: "
                    f"{DEFAULT_TIMEOUT}). One call per template, and a "
                    f"template that does not answer inside it is not "
                    f"retried -- there is no schedule to vary, only the "
                    f"next template."
                ),
            ),
        ),
    ),
)
