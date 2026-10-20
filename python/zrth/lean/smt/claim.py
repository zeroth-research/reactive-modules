"""A claim read against the run's encoding of the module, for a search.

`SynthContext` is the property and preconditions of a `Claim`, parsed over
the run's `smt.encoding.ModuleEncoding`, with the module's state sorts
checked against what the searching route reads (`READING_KINDS`). The
obligations (`smt.obligations.Obligations`) are stated over one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..certificate import Claim
from ..errors import Refused
from .encoding import encoding_of


# Every kind of state component there is a reading for. A sort absent from
# here has none at all, which is a different refusal from a route declining
# a kind that exists: the first is this front end's limit, the second is the
# route's own contract.
#
# `tuple` is the odd one out, and deliberately: it is not a kind anything is
# *read* as, it is the kind that has elements. A route that takes it is
# saying it states itself over elements rather than over components -- see
# `magic.synth.component_slots` -- and what each element is read as is one
# of the other four.
READING_KINDS = ("int", "bool", "bv", "real", "tuple")


def reading_kind(sort) -> "str | None":
    """Which of :data:`READING_KINDS` a cvc5 sort is read as, or `None`."""
    for name, is_it in (("int", sort.isInteger), ("bool", sort.isBoolean),
                        ("bv", sort.isBitVector), ("real", sort.isReal),
                        ("tuple", sort.isTuple)):
        if is_it():
            return name
    return None


@dataclass
class SynthContext:
    """One module and its property in cvc5, for a search over shapes.

    Stated over the run's `smt.encoding.ModuleEncoding`, the front end the
    pre-check and the prompt routes read predicates through: the property
    arrives as SMT-LIB over ``s0..sN-1`` and that is the namespace every
    predicate this package reads or writes is in, so a certificate found
    here is in it too.
    """

    enc: object
    prp: object
    init_pre: object
    update_pre: object

    @property
    def module(self):
        return self.enc.module

    @property
    def tm(self):
        return self.enc.tm

    @property
    def msmt(self):
        return self.enc.msmt

    @property
    def env(self):
        return self.enc.env

    @property
    def state(self) -> list:
        return self.env.state_vars

    @property
    def extl_latched(self) -> list:
        return self.env.extl_latched_vars

    @property
    def extl_next(self) -> list:
        return self.env.extl_next_vars

    @property
    def names(self) -> list[str]:
        return [f"s{i}" for i in range(len(self.state))]

    @classmethod
    def build(cls, source, claim: Claim, *, route: str,
              takes: tuple[str, ...] = ("int", "bool", "bv")) -> "SynthContext":
        """The context, or the reason this module is not one `route` reads.
        `source` is the module, its encoding or a `smt.encoding.lazy` factory.

        ``takes`` is the route's contract, as :data:`READING_KINDS`, and it
        is the same tuple :func:`readings` is given -- a route that declines
        a kind here declines it there, so there is one answer to "what can
        this route read" rather than two that have to agree. What a sort
        costs beyond being readable is the route's own business:
        `--infer sygus` takes the context's Bool and then refuses it in
        `readings`, because its `synthFun` has nowhere to put a Bool column
        even though the context can weigh one.

        A kind outside ``takes`` is refused here, before a solver starts,
        and named: `--infer houdini` states its candidates in real
        arithmetic and so asks for `real`; `--infer smt-linear` reads a
        Real through a floor and asks for it too.
        """
        try:
            import cvc5                              # noqa: F401
        except ImportError:                          # pragma: no cover
            raise Refused(
                f"--infer {route} needs the `cvc5` package, which is not "
                f"importable. Install with: uv pip install cvc5"
            ) from None

        if claim.prp is None:
            raise Refused(
                f"--infer {route} needs a property: pass --safety or --buchi"
            )
        enc = encoding_of(source)
        env = enc.env

        bad = [
            f"s{i} is {sort}"
            for i, sort in enumerate(env.state_sorts)
            if reading_kind(sort) not in takes
        ]
        if bad:
            reads = ("reads a scalar state" if "real" in takes else
                     "weighs each state component as an integer")
            why = "" if "real" in takes else (
                "A Real component has no integer reading and a ranking "
                "function has to land in `Nat`. "
            )
            shaped = "" if "tuple" in takes else (
                "A matrix-shaped component would be a column per element. "
            )
            raise Refused(
                f"--infer {route} {reads}, and this module has one it "
                f"cannot: {', '.join(bad)}. {why}{shaped}"
                f"Use --infer ai-cegis."
            )
        # The transition, asked for here rather than by whichever route
        # first builds an obligation out of it. A module with an operator
        # cvc5 has no term for -- `Transpose`, say -- is a refusal either way,
        # but raised from the middle of a search it is a `ValueError`
        # traceback several steps past the fact, for a user who asked about
        # a module. Named up front instead, which is what `native.lean_gaps`
        # does for the Lean side.
        try:
            enc.init_state(enc.extl_next)
            enc.next_state(enc.state, enc.extl_latched, enc.extl_next)
        except ValueError as e:
            raise Refused(
                f"--infer {route} encodes this module's transition for cvc5, "
                f"and an operator in it has no encoding. {e}"
            ) from e
        true = enc.tm.mkBoolean(True)
        return cls(
            enc=enc,
            prp=enc.parse(claim.prp, "property", kind=claim.kind),
            init_pre=(enc.parse(claim.init_pre, "pre")
                      if claim.init_pre is not None else true),
            update_pre=(enc.parse(claim.update_pre, "pre")
                        if claim.update_pre is not None else true),
        )
