"""`--infer ai-cegis`: an LLM proposes, cvc5 refutes, the counterexample returns."""

from __future__ import annotations

from ...core.options import Opt
from ..io import InferInput, InferResult
from ..route import InferRoute


def _run(inp: InferInput) -> InferResult:
    from .engine import TA2MagicCEGIS

    inp = inp.taking_inv(
        inp.resumable_inv(),
        # On `--safety` the invariant *is* the certificate, so there is
        # nothing left to infer and no LLM call to make: saying "and
        # inferring the ranking function" there would name a field
        # `rule_globally` does not have.
        "taking it as the certificate"
        if inp.claim.is_safety
        else "taking its invariant as given and inferring the ranking function",
    )

    magic = TA2MagicCEGIS(
        inp.project.encoding(""),
        inp.module,
        model=inp.model,
        base_url=inp.base_url,
        known=inp.ruled_out(),
        budget_s=inp.opts["cegis_budget"],
        encoding=inp.smt,
        budget=inp.budget,
        log=inp.log,
    )
    return InferResult(magic.infer(inp.claim, inp.given))


ROUTE = InferRoute(
    name="ai-cegis",
    summary=(
        "LLM + cvc5 counterexample-guided inductive synthesis: each "
        "candidate is put to the obligations and the model that refutes "
        "one is fed back (the default when --infer is passed without a "
        "value)"
    ),
    kinds=frozenset({"safety", "buchi"}),
    kinds_refusal="",
    seeds=frozenset({"inv", "ranking", "pre"}),
    reads=("",),
    # It already takes a fixed invariant from `--invariant` and infers
    # only what is left, so an inductive invariant a previous run left in
    # `artifacts/` is the same seed from the other source. A `note` is
    # the other direction: what `--infer smt-linear` or `--infer sygus`
    # *refuted* goes into the prompt, so the model is not asked to
    # rediscover that a shape is empty one API call at a time.
    reads_artifacts=frozenset({"inv", "note"}),
    returns="smt",
    uses_llm=True,
    run=_run,
    options=(
        Opt(
            ("--cegis-budget",),
            dict(
                type=float,
                default=None,
                metavar="SECONDS",
                help=(
                    "Keep proposing candidates for this many seconds "
                    "instead of stopping after 5 attempts. An attempt "
                    "starts only while the slowest one so far would still "
                    "finish inside the budget, so a run under an outer "
                    "time limit gives up on its own rather than being "
                    "killed mid-reply. Each attempt is an API call."
                ),
            ),
        ),
    ),
)
