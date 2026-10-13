"""`--infer nuterm`: a learned ranking function, certified; Houdini's invariant."""

from __future__ import annotations

from .api import InferInput, InferResult, InferRoute


def _run(inp: InferInput) -> InferResult:
    from ..magic.learn import TA2MagicLearn

    magic = TA2MagicLearn(inp.module, log=inp.log)
    return InferResult(magic.infer(inp.claim, inp.given))


ROUTE = InferRoute(
    name="nuterm",
    summary=(
        "no LLM: a ranking function trained on rollouts of the module and "
        "returned only once a Farkas/CEGAR decision procedure has "
        "certified it, with the invariant inferred by Houdini -- scalar "
        "integer state only"
    ),
    kinds=frozenset({"safety", "buchi"}),
    kinds_refusal="",
    # `--pre` is the entry assumption Houdini checks initiation under.
    # Refusing it used to be argued from soundness -- an invariant that
    # holds at entry for *every* input is stronger than one that holds
    # only from the admitted states -- which is true of the invariants the
    # route finds and says nothing about the ones it then cannot. That is
    # the whole effect: 8 of the svcomp corpus's own Houdini invariants
    # hold only from the states the precondition admits.
    seeds=frozenset({"pre"}),
    seeds_refusal=(
        "the certificate comes from the learner, which infers both halves "
        "of it and puts a supplied one to nothing"
    ),
    returns="smt",
    run=_run,
)
