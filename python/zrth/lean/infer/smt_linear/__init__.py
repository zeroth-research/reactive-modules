"""`--infer smt-linear`: one cvc5 query per linear shape.

`engine` states the shapes -- the invariant's rows, the rank and its
branches -- as templates, widening them in turn; `solve` answers each
template, by one quantified query or, where cvc5 will not state that, by a
counterexample-guided loop.
"""

from __future__ import annotations

from ...core.options import Opt
from ..io import InferInput, InferResult
from ..resume import resumable_inv, taking_inv
from ..route import InferRoute


def _run(inp: InferInput) -> InferResult:
    from .engine import SmtLinearEngine

    inp = taking_inv(
        inp, resumable_inv(inp, strengthen=inp.claim.is_safety),
        ("adding to it the conjuncts the property needs"
         if inp.claim.is_safety
         else "taking its invariant as given and ranking over it"),
    )
    engine = SmtLinearEngine(
        inp.module,
        rows=inp.opts["linear_rows"],
        budget=inp.budget,
        artifacts=inp.project.artifacts,
        log=inp.log,
        encoding=inp.smt,
    )
    return InferResult(engine.infer(inp.claim, inp.given))


ROUTE = InferRoute(
    name="smt-linear",
    summary=(
        "no LLM: one cvc5 query per shape -- a ranking function linear "
        "in the state (--buchi), an invariant that is a conjunction of "
        "linear inequalities (--safety) -- over scalar Int, Bool and "
        "bitvector components alike, where a refuted query is a *proof* "
        "that the shape is empty, written to artifacts/ for --infer "
        "ai-cegis to read into its prompt"
    ),
    kinds=frozenset({"safety", "buchi"}),
    kinds_refusal="",
    seeds=frozenset({"inv", "pre"}),
    seeds_refusal=(
        "a ranking function is what this route searches for on --buchi, "
        "so a supplied one would leave it nothing to do"
    ),
    # `--buchi` ranks over a fixed invariant and `--safety` strengthens
    # one, so both directions of the workspace matter here: what another
    # run left is either what this one ranks over or what it adds to.
    reads_artifacts=frozenset({"inv"}),
    returns="smt",
    run=_run,
    options=(
        Opt(
            ("--linear-rows",),
            dict(
                type=int,
                default=2,
                metavar="N",
                help=(
                    "How many linear inequalities the --safety invariant "
                    "may be a conjunction of (default: 2). Tried one "
                    "width at a time, smallest first, because the width "
                    "is what the search costs: a one-row invariant for "
                    "`G (s0 <= 100)` is 11 ms and two rows do not finish "
                    "in 30 s. Ignored for --buchi, whose template is one "
                    "ranking function."
                ),
            ),
        ),
    ),
)
