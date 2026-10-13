"""Inference of invariants and ranking functions for reactive modules.

One module here is one route's engine, a `TA2Magic` subclass that the
route's row in `infer_route/<name>.py` builds and calls
(`fbk-proveit` has none here):

* `ai` -- an LLM proposes the whole certificate (`--infer ai`).
* `cegar` -- an LLM proposes, cvc5 refutes, the counterexample goes back
  (`--infer ai-cegis`).
* `houdini` -- candidate facts, greatest inductive subset, over either
  solver (`--infer houdini`).
* `learn` -- Farkas/CEGAR learning from simulation traces (`--infer nuterm`).
* `linear` -- one linear template, solved directly (`--infer smt-linear`).
* `sygus` -- cvc5's `addSygusInvConstraint` (`--infer sygus`).
* `vampire` -- Vampire derives the certificate itself, nothing proposes it
  (`--infer vampire`).

Two are not routes: `llm`, the model both LLM routes ask, and `modes`, the
bounds `houdini` computes per mode. No route imports another
(`tests/test_layering.py`); what two share is in `zrth.lean.smt` or
`smt_synth`.

Nothing is imported here: the route rows import their own module, which is
what keeps `verith --help` cheap. Importing a route from this package's
`__init__` would undo that.
"""

from ..certificate import Certificate, Claim


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
    `infer_route/<name>.py`, which builds the engine from what it needs and
    calls it. This class names `infer(claim, given)` and holds nothing: no
    two engines are constructed from the same arguments.
    """

    def infer(self, claim: Claim,
              given: Certificate = Certificate()) -> Certificate:
        """The certificate of `claim.prp`, under the claim's preconditions
        (`init_pre`, `update_pre`)."""
        raise NotImplementedError
