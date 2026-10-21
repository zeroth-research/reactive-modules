"""The transition without its conditionals: one `Branch` per assignment
of its `ite` conditions (`split`), each a substitution Vampire can be
asked about without an `ite` in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ...core.errors import Refused
from ...smt.terms import ite_conditions
from ...smt.claim import SynthContext

import cvc5
from cvc5 import Kind


# A branch is one assignment of the transition's `ite` conditions, so the
# case count doubles with each. Past this the question is too long to be
# worth asking, and the route says so rather than printing it.
_MAX_CONDITIONS = 5
_MAX_BRANCHES = 2 ** _MAX_CONDITIONS


@dataclass(frozen=True)
class Branch:
    """One case of a round: its guard, and the state terms under that guard.

    `guard` is the conjunction of condition literals that selects this case
    and `state` is the terms with every `ite` folded away by cvc5's rewriter
    under that assignment. The guard is a hypothesis of the obligation
    rather than a condition inside it, which is the whole point: Vampire
    answers an implication and does not answer a conditional.
    """

    guard: tuple                # cvc5 Bool terms, all of them true here
    state: tuple                # the folded terms, ite-free


def split(ctx: SynthContext, terms, what: str) -> list[Branch]:
    """`terms`, split into the ite-free cases of their own conditions.

    The round and the initial state are split *apart*, because they are not
    quantified over the same things: the initial state is a function of the
    inputs alone, so a guard taken from the round -- which tests the latched
    state -- would leave a variable in the entry obligation that nothing
    binds. Every assignment is emitted, the unsatisfiable ones included: an
    unsatisfiable guard makes its implication vacuously true, which costs
    Vampire a clause and costs this module the logic that would have worked
    out which ones those are.
    """
    tm = ctx.tm
    rewriter = cvc5.Solver(tm)
    terms = list(terms)
    conds = ite_conditions(terms)
    if len(conds) > _MAX_CONDITIONS:
        raise Refused(
            f"--infer vampire splits {what} on the {len(conds)} conditions "
            f"its `ite`s test, which is {2 ** len(conds)} cases and past the "
            f"{_MAX_BRANCHES} this route will print. `--infer houdini` "
            f"states the round once, conditionals and all."
        )
    out = []
    for bits in product((True, False), repeat=len(conds)):
        subs = [tm.mkBoolean(b) for b in bits]
        folded = tuple(rewriter.simplify(t.substitute(conds, subs))
                       if conds else t for t in terms)
        guard = tuple(c if b else tm.mkTerm(Kind.NOT, c)
                      for c, b in zip(conds, bits))
        out.append(Branch(guard, folded))
    return out
