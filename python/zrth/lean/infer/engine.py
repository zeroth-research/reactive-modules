"""What an inference route's engine is: `TA2Magic`.

Each route's engine is `infer/<name>/engine.py`, built and called by the
route's row in `infer/<name>/__init__.py` (`fbk-proveit`'s is not a
`TA2Magic`: it hands back files):

* `ai` -- an LLM proposes the whole certificate.
* `ai_cegis` -- an LLM proposes, cvc5 refutes, the counterexample goes back.
* `houdini` -- candidate facts, greatest inductive subset, over either
  solver.
* `nuterm` -- Farkas/CEGAR learning from simulation traces.
* `smt_linear` -- one linear template, solved directly.
* `sygus` -- cvc5's `addSygusInvConstraint`.
* `vampire` -- Vampire derives the certificate itself, nothing proposes it.

What two engines share is in `infer.common` (`llm`, the model both LLM
routes ask; `synth`, what a searching route reads a module as) or in
`zrth.lean.smt`. No route imports another (`tests/test_layering.py`).
"""

from ..core.certificate import Certificate, Claim


class TA2Magic:
    """Base class of the engines: `infer(claim, given)` answers with a new
    `Certificate`, in SMT-LIB.

    `claim.kind == "buchi"` asks for `G (F prp)`: an invariant and a ranking
    function that decreases on every step from an `inv ∧ ¬prp` state.
    `"safety"` asks for `G prp`: an invariant that implies `prp`, and no
    ranking. Which kinds an engine takes is its route's `kinds`, checked
    before it runs. `given` is what the run already holds of the answer
    (`--invariant`, `--ranking`, or a resumed invariant); an engine reads
    only what its route's `seeds` names.

    `ai` is the exception: `infer(claim)` answers in Lean, a `LeanAnswer`.

    The route's contract is its row's `run(InferInput) -> InferResult` in
    `infer/<name>/__init__.py`, which builds the engine from what it needs and
    calls it. This class names `infer(claim, given)` and holds nothing: no
    two engines are constructed from the same arguments.
    """

    def infer(self, claim: Claim,
              given: Certificate = Certificate()) -> Certificate:
        """The certificate of `claim.prp`, under the claim's preconditions
        (`init_pre`, `update_pre`)."""
        raise NotImplementedError
