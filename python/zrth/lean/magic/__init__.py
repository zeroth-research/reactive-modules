"""Inference of invariants and ranking functions for reactive modules.

`TA2Magic` below is the contract every route implements; one module here is
one route, selected by an `infer_route` row:

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
    """
    Infers invariants and ranking functions for a reactive module
    or its source code to prove that `G (F prp)` holds about the system.
    That is, to prove that `prp` holds infinitely often.

    `cd.kind == "safety"` asks for the other property instead -- `G prp`,
    every reachable state -- and then there is no ranking function to infer:
    the invariant has to imply `prp`, and `rule_globally` does the rest.
    Only `TA2MagicCEGAR` implements that; `TA2MagicAI` is Buchi-only, and
    `main` rejects the combination rather than inferring something the
    certificate cannot use.
    """

    def __init__(self, source: str):
        self.source = source

    def infer(self, cd: CertificateData) -> CertificateData:
        """Run inference. Fills in invariant (`inv`) and
        ranking funciton (`ranking`) into `cd`. The field `prp` of `cd` is initialized
        with the property to proof to hold infinitly often. That is,
        the goal is to find invariant and ranking function on `inv \and \neg prp` states
        that proofs that the `G (F prp)` holds. For example, if the reactive module
        is generated from this code:
        ```
        def init():
            "Return initial value of x"
            return 0

        def update(old_x):
            "Returns new value of x"

            x = old_x + 1
            if x == 10:
                return 0
            return x
        ```

        Then it encodes this equivalent program:

        ```
        # init
        x = 0

        while True:
            # update
            x += 1
            if x == 10:
                x = 0
        ```

        If `prp` is `x == 0`, we can find invariant `0 <= x <= 10`
        and ranking function `10 - x` showing that if `x != 0` then
        the entity `10 - x` strictly decreases each iteration, proving
        that `x` will eventually become `0` and thus `x == 0` holds
        infinitely often.

        Some data in `cd` may be already present, like preconditions
        on inputs to init and update functions (`init_pre`, `update_pre` in `cd`,
        not shown in the example).
        """
        raise NotImplementedError
