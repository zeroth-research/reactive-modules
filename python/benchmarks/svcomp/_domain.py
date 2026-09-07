"""Derive the loop guard (the verification domain) from the program's update.

The DSL encodes a loop as ``update = ite(guard, body, self)`` per state variable,
so the guard is the condition of the top-level ``ite`` whose else-branch is the
variable itself. The transition itself is read off a
:class:`._farkas.System`, so nothing here walks the module.
"""
from __future__ import annotations

import z3



def guard_ite(sp_k, s_k):
    """If ``sp_k`` has the DSL loop shape ``ite(guard, body, self)`` (its
    else-branch is the variable ``s_k`` itself), return that ``ite`` node; else
    ``None``. The one place the guard shape is recognised — used both to derive
    the domain (the guard is ``ite.arg(0)``) and, by the Farkas verifier, to take
    the on-guard body (``ite.arg(1)``)."""
    if (z3.is_app(sp_k) and sp_k.decl().kind() == z3.Z3_OP_ITE
            and sp_k.arg(2).eq(s_k)):
        return sp_k
    return None


def guard_from_transition(s: dict, sp: dict, state) -> z3.BoolRef:
    """The loop guard from one symbolic step: for the first variable whose next
    value is ``ite(guard, body, self)``, return ``guard``."""
    for n in state:
        ite = guard_ite(sp[n], s[n])
        if ite is not None:
            return ite.arg(0)
    raise ValueError("could not extract loop guard from the update "
                     "(expected update = ite(guard, body, self))")


def domain(system):
    """A callable ``state_map -> guard`` for the loop guard, derived from
    ``system``'s transition (substituting the given state for its symbols)."""
    g = guard_from_transition(system.s_map, system.sp_map, system.names)

    def dom(state_map):
        sub = [(z3.Int(n), z3.IntVal(v) if isinstance(v, int) else v)
               for n, v in state_map.items()]
        return z3.substitute(g, *sub)

    return dom
