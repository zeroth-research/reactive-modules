"""Derive the loop guard (the verification domain) from the program's update.

The DSL encodes a loop as ``update = ite(guard, body, self)`` per state variable,
so the guard is the condition of the top-level ``ite`` whose else-branch is the variable itself. 
"""
from __future__ import annotations

import z3

from ._bench import Bench
from zrth import z3 as zz3


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


def _step(bench: Bench):
    """Symbolic one-step of the program: latched state (fresh ints) -> next."""
    prog, ctrl, _ = bench.build()
    z = {ctrl[n][0]: [z3.Int(n)] for n in bench.state}
    for atom in prog.atoms:
        for t in atom.update:
            z.update(zip(t.write, zz3.eval(t.itype, [z[w] for w in t.read])))
    s = {n: z[ctrl[n][0]][0] for n in bench.state}
    sp = {n: z[ctrl[n][1]][0] for n in bench.state}
    return s, sp


def domain(bench: Bench):
    """A callable ``state_map -> guard`` for the loop guard, derived from the
    update (substituting the given state for the canonical symbols)."""
    s, sp = _step(bench)
    g = guard_from_transition(s, sp, bench.state)

    def dom(state_map):
        sub = [(z3.Int(n), z3.IntVal(v) if isinstance(v, int) else v)
               for n, v in state_map.items()]
        return z3.substitute(g, *sub)

    return dom
