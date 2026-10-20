"""`--infer ai`: one LLM call proposes the certificate, a second audits it."""

from __future__ import annotations

from ..io import InferInput, InferResult
from ..route import InferRoute


def _run(inp: InferInput) -> InferResult:
    from .engine import TA2MagicAI

    magic = TA2MagicAI(
        inp.project.encoding(""), model=inp.model, base_url=inp.base_url,
        log=inp.log,
    )
    # Lean only: it is parsed out of an LLM reply and never had an SMT term
    # behind it, which is what this route's `returns="lean"` declares. No
    # answer is an empty result, which main reports against the row.
    return InferResult(lean=magic.infer(inp.claim))


ROUTE = InferRoute(
    name="ai",
    summary=(
        "plain LLM self-check: one call proposes the invariant and the "
        "ranking function, a second is asked to find them wrong"
    ),
    kinds=frozenset({"buchi"}),
    kinds_refusal=(
        "--safety needs --infer ai-cegis or --infer nuterm: the `ai` route "
        "infers a ranking function `rule_globally` cannot take, and nothing "
        "checks that the invariant implies the property"
    ),
    # `--pre` is the one predicate this route has somewhere to put. It is
    # not something the route must hold a candidate to -- it is an
    # assumption the candidate may lean on -- and both prompts already
    # carry it: `GENERATE_SYSTEM` states the two obligations with
    # `init_pre` and `update_pre` in them, `VERIFY_SYSTEM` tells the
    # auditor to assume them, and `llm.describe_preconditions` writes the
    # predicates themselves into each user message. `Certificate/Data.lean`
    # carries them whatever the route, so `lake build` checks the reply
    # under the same assumption the reply was asked for under.
    seeds=frozenset({"pre"}),
    seeds_refusal=(
        "the `ai` route asks for the invariant and the ranking function "
        "together in one reply and nothing puts a supplied one to the "
        "obligations, so it would be overwritten rather than used -- "
        "--infer ai-cegis takes either as given and infers what is left"
    ),
    reads=("",),
    returns="lean",
    uses_llm=True,
    run=_run,
)
