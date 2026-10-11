"""Inference of invariants and ranking functions for reactive modules.

One module here is one route's engine, a `TA2Magic` subclass that the
route's `_run_*` adapter in `infer_route.py` builds and calls
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

Nothing is imported here: `ai` pulls in `anthropic` at module scope, and the
route rows import their own module, which is what keeps `verith --help`
cheap. Importing a route from this package's `__init__` would undo that.
"""

from ..cert import CertificateData


class TA2Magic:
    """Base class of the engines: `infer(cd)` fills in the certificate.

    `cd.kind == "buchi"` asks for `G (F prp)`: an invariant and a ranking
    function that decreases on every step from an `inv ∧ ¬prp` state.
    `"safety"` asks for `G prp`: an invariant that implies `prp`, and no
    ranking. Which kinds an engine takes is its route's `kinds`, checked
    before it runs.
    """

    def __init__(self, source: str):
        self.source = source

    def infer(self, cd: CertificateData) -> CertificateData:
        """Fill in `cd.inv` (and `cd.ranking` for `buchi`) for `cd.prp`,
        under the preconditions already in `cd` (`init_pre`, `update_pre`)."""
        raise NotImplementedError
