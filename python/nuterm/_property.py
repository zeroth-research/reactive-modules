"""Claims a decision procedure can be asked to prove about a reactive module.

A claim says what is wanted, in terms of the module's wires alone: ``S[name]``
is a column's latched value, ``S.next[name]`` its value after the round, and
``W[wire]`` the value a ctrl next wire takes this round. A claim is therefore
stateable without knowing which witness will discharge it; how to prove one is a
witness's business (see :mod:`._farkas`).

Every property of a module's runs is a safety property intersected with a
liveness one (Alpern and Schneider 1985), and there is one class for each, both
taking a single predicate over a round.

:class:`Safety` holds on every round. A round carries a state and its successor,
so a predicate over one covers both state and step properties.

:class:`Liveness` names a domain no infinite stretch of consecutive rounds may
stay inside, so the run leaves it again and again. That is recurrence, and the
other liveness shapes are built from it by adding a column to the module:

  ``eventually P``          ``Liveness(not P)``
  ``eventually P forever``  ``Liveness(not P)`` with ``Safety(P -> P')``
  ``after p, eventually q`` ``Liveness(waiting)``, for a column ``p`` sets and
                            ``q`` clears
  termination               ``Liveness`` over the rounds where a column moves,
                            where leaving the domain is a fixed point

Any ω-regular liveness property reduces to recurrence the same way, by composing
its automaton into the module as columns.

This module defines no properties of its own: the general layers supply the
vocabulary and a client states the property in it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import z3


@dataclass(frozen=True)
class Safety:
    """``holds`` is true on every round.

    A round has the columns latched and next and every wire computed from them,
    so ``holds`` may be a state property (``S["i"] <= S["n"]``) or a step property
    (``S.next["i"] <= S["i"]``). The procedure treats the two alike; the proof
    layer reads off which it was given."""
    holds: object

    def domain(self, W, S):
        return z3.BoolVal(True)


@dataclass(frozen=True)
class Liveness:
    """No infinite stretch of consecutive rounds satisfies ``domain``: whenever
    the run is in it, it leaves.

    ``domain`` is ``(W, S) -> BoolRef``, the rounds a witness must show cannot go
    on forever. A rank bounded below that drops on each of them does it, and may
    rise again outside them. What is concluded is exactly that, the run leaves
    ``domain`` infinitely often, not that it stays out: staying out is a separate
    safety fact where it is wanted. A run claim, so nothing must hold on any one
    round, ``holds`` is ``None``, and the witness is the whole obligation."""
    domain: object
    holds: ClassVar[None] = None

