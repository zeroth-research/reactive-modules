"""Inference of invariants and ranking functions for reactive modules."""

from .cert import CertificateData


class TA2Magic:
    """
    Infers invariants and ranking functions for a reactive module
    or its source code to prove that `G (F prp)` holds about the system.
    That is, to prove that `prp` holds infinitely often.
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
