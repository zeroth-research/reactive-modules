"""Properties a decision procedure can be asked to prove about a reactive module.

A property says *what must hold* and *on which steps*, in terms of the module's
wires alone. How to prove it — which certificate, which relaxation, which rows
reach an LP — belongs to the procedure, not here (see :func:`._farkas.rule_for`).
So a property is stateable without knowing which procedure will take it, and a
procedure declares which properties it has a rule for.

``domain(system)`` is the steps the property ranges over, read off the transition
rather than matched against a program shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import z3


@dataclass(frozen=True)
class Fixpoint:
    """The components ``over`` reach a state their step does not move.

    ``over`` is the ``(latched, next)`` wire pairs whose change constitutes a
    step. A reactive module's ``update`` is total — it ticks forever — so
    termination is a property of what the module *encodes*, under the convention
    that a finished program stutters: the steps that count are exactly those
    where one of these components changes. Naming them is the property's job:
    anything else composed alongside (a ranking function's own wires) may still
    move while the program is done, and is not part of what "moving" means."""
    over: tuple

    def domain(self, system):
        return z3.Or(*[system.next_of(p) != system.sym_of(p) for p in self.over])


@dataclass(frozen=True)
class Always:
    """``pred`` holds in every reachable state.

    ``pred`` is ``(W, S) -> BoolRef`` over the module: ``S[name]`` a column's
    latched value, ``W[wire]`` the value a ctrl next wire takes at that state —
    so a claim may name a computed wire, as long as its value is a function of
    the state (nothing of the next round). Its domain is *every* step — a
    stuttering step preserves any state predicate trivially, but a run may
    stutter, and the theorem quantifies over runs of whatever the domain admits."""
    pred: object

    def domain(self, system):
        return z3.BoolVal(True)
