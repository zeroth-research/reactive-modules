"""Claims a decision procedure can be asked to prove about a reactive module.

A claim says *what you want*, in terms of the module's wires alone — ``S[name]``
a column's latched value, ``S.next[name]`` its value after the round, ``W[wire]``
the value a ctrl next wire takes this round — so it is stateable without knowing
which witness will discharge it; how to prove one is a witness's business (see
:mod:`._farkas`).

Every property of a module's runs is a safety property intersected with a
liveness one (Alpern and Schneider), and each has one shape here: one class taking
one predicate over a round. :class:`Safety` — ``holds`` is true on every round; a
round already has a state and its successor, so that is every safety property.
:class:`Liveness` — no infinite stretch of consecutive rounds satisfies ``domain``,
so the run leaves it again and again. That is recurrence, and the other liveness
shapes are it plus a safety claim plus columns: "eventually P" is
``Liveness(not P)``; "eventually P forever" is ``Liveness(not P)`` with the
stability ``Safety(P -> P')``; "after p, eventually q" is ``Liveness(waiting)`` for
a column set by ``p`` and cleared by ``q``; termination is the ``Liveness`` whose
domain is "some column moves", where leaving it is a fixed point and stability
comes free. Any ω-regular
liveness property reduces to recurrence the same way, by composing its automaton
in as columns — which is why one class is enough.
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
    on forever — a rank bounded below that drops on each of them does it, and may
    rise again outside them. What is concluded is exactly that, the run leaves
    ``domain`` infinitely often, not that it stays out: staying out is a separate
    safety fact where it is wanted. A run claim, so nothing must hold on any one
    round, ``holds`` is ``None``, and the witness is the whole obligation."""
    domain: object
    holds: ClassVar[None] = None

