"""`--infer sygus`: the invariant synthesised by cvc5's SyGuS invariant track."""

from __future__ import annotations

from .api import InferInput, InferResult, InferRoute, Opt


def _run(inp: InferInput) -> InferResult:
    from ..magic.sygus import TA2MagicSygus

    magic = TA2MagicSygus(
        inp.module,
        grammar=inp.opts["sygus_grammar"],
        conjuncts=inp.opts["sygus_conjuncts"],
        budget=inp.budget,
        artifacts=inp.project.artifacts,
        log=inp.log,
        encoding=inp.smt,
    )
    return InferResult(magic.infer(inp.claim, inp.given))


ROUTE = InferRoute(
    name="sygus",
    summary=(
        "no LLM: the invariant synthesised outright by cvc5's SyGuS "
        "invariant track, over a grammar of linear facts and congruences "
        "-- the one shape `nuterm`'s Houdini lattice cannot state -- and "
        "left in artifacts/ for a later run to take as given; Int, Bool, "
        "bitvector, Real and matrix-shaped components alike, a Real kept "
        "rational rather than floored and costing only the congruences"
    ),
    kinds=frozenset({"safety"}),
    kinds_refusal=(
        "--infer sygus is incompatible with --buchi: what it synthesises "
        "is `pre -> inv`, `inv /\\ trans -> inv'`, `inv -> post`, which is "
        "what `G P` is made of, and a Buchi certificate also needs a "
        "ranking function. Synthesis loses at those -- measured, 124 s on "
        "a two-variable module for a rank `--infer nuterm` certifies in "
        "seconds. Pass --safety, or infer a Buchi certificate with "
        "--infer nuterm or --infer ai-cegis."
    ),
    seeds=frozenset({"pre"}),
    seeds_refusal=(
        "the invariant is the whole of what this route synthesises, so a "
        "supplied one leaves it nothing to do, and a ranking function "
        "belongs to --buchi, which it does not certify"
    ),
    returns="smt",
    run=_run,
    options=(
        Opt(
            ("--sygus-conjuncts",),
            dict(
                type=int,
                default=3,
                metavar="N",
                help=(
                    "A ceiling on how many atoms the synthesised "
                    "invariant may be a conjunction of (default: 3). The "
                    "widths run 1, 2, ... N and the first that answers "
                    "wins, because the search is not monotone in the "
                    "width: measured, invariants found at 1 and at 2 "
                    "time out at 3. It bounds the certificate -- every "
                    "conjunct is one more implication in the obligation "
                    "-- and it is what makes a failure informative: a "
                    "bounded conjunction is a finite space, so cvc5 can "
                    "report it *empty* rather than merely not searched, "
                    "and that proof is what lands in artifacts/."
                ),
            ),
        ),
        Opt(
            ("--sygus-grammar",),
            dict(
                default="congruence",
                choices=["congruence", "linear"],
                help=(
                    "What an atom of the synthesised invariant may be "
                    "(default: congruence). `linear` gives comparisons of "
                    "affine combinations of the state, which is roughly "
                    "what `--infer nuterm` already covers and what "
                    "`--infer smt-linear` decides in milliseconds. "
                    "`congruence` adds `(= (mod a0 + a1*s0 + ... k) 0)` "
                    "for the `k` the program mentions -- `m_step2` steps "
                    "by two, so its invariant is that `x` is even, and "
                    "that is the fact no lattice of signs and pairwise "
                    "relations can state."
                ),
            ),
        ),
    ),
)
