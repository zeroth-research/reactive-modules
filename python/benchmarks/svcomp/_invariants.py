"""Houdini-style loop-invariant inference to strengthen the verification domain.

Sign and relational predicates are seeded as candidates, then any that is not
**established at loop entry** (initiation) or not **preserved by the body**
(consecution) is dropped; the survivors are returned. Conjoining them with the
loop guard shrinks the verification domain to an over-approximation of the
reachable loop states.

Soundness
=========
Both filters are checked with Z3, and a ``sat`` or ``unknown`` result drops the
candidate, so a candidate is kept only when it is proved and every survivor is a
genuine inductive invariant. Initiation is checked against the ``init`` block for
*all* inputs, ignoring any precondition: stronger than required, hence sound,
though it misses invariants that hold only under a precondition.
"""
from __future__ import annotations

import z3
from z3.z3util import get_vars


# A candidate is (label, state_map -> z3.BoolRef); state_map is {var_name: expr}.
Guess = tuple


def _candidates(names) -> list[Guess]:
    """Sign predicates per variable and ±-relations between pairs."""
    cands: list[Guess] = []
    for v in names:
        cands += [
            (f"{v}>0",   (lambda st, v=v: st[v] > 0)),
            (f"{v}>=0",  (lambda st, v=v: st[v] >= 0)),
            (f"{v}<=0",  (lambda st, v=v: st[v] <= 0)),
            (f"{v}<0",   (lambda st, v=v: st[v] < 0)),
            (f"{v}>=1",  (lambda st, v=v: st[v] >= 1)),
            (f"{v}<=-1", (lambda st, v=v: st[v] <= -1)),
        ]
    vs = list(names)
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a, b = vs[i], vs[j]
            cands += [
                (f"{a}-{b}>=0", (lambda st, a=a, b=b: st[a] - st[b] >= 0)),
                (f"{a}-{b}<=0", (lambda st, a=a, b=b: st[a] - st[b] <= 0)),
                (f"{a}+{b}>=0", (lambda st, a=a, b=b: st[a] + st[b] >= 0)),
                (f"{a}+{b}<=0", (lambda st, a=a, b=b: st[a] + st[b] <= 0)),
            ]
    return cands


def _as_int_const(expr):
    """The integer value of ``expr`` if it simplifies to a constant, else None."""
    e = z3.simplify(expr)
    if z3.is_int_value(e):
        return int(e.as_long())
    if z3.is_rational_value(e):
        f = e.as_fraction()
        return int(f.numerator) if f.denominator == 1 else None
    return None


def _const_candidates(names, vals: dict) -> list[Guess]:
    """Candidates from a state's constant coordinates. ``vals`` is a symbolic state
    (the body post-state ``T(s)`` or the init state ``s0``): ``v==c`` / ``v>=c`` /
    ``v<=c`` when ``vals[v]`` is constant, and ``vi-vj==d`` / ``vi+vj==s`` when a
    pair combination is constant."""
    cands: list[Guess] = []
    for v in names:
        c = _as_int_const(vals[v])
        if c is not None:
            cands += [
                (f"{v}=={c}", (lambda st, v=v, c=c: st[v] == c)),
                (f"{v}>={c}", (lambda st, v=v, c=c: st[v] >= c)),
                (f"{v}<={c}", (lambda st, v=v, c=c: st[v] <= c)),
            ]
    vs = list(names)
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a, b = vs[i], vs[j]
            d = _as_int_const(vals[a] - vals[b])
            if d is not None:
                cands.append((f"{a}-{b}=={d}", (lambda st, a=a, b=b, d=d: st[a] - st[b] == d)))
            t = _as_int_const(vals[a] + vals[b])
            if t is not None:
                cands.append((f"{a}+{b}=={t}", (lambda st, a=a, b=b, t=t: st[a] + st[b] == t)))
    return cands


def _ite_conds(e) -> list:
    """Every ``ite`` condition occurring in ``e``."""
    out = []
    if z3.is_app(e):
        if e.decl().kind() == z3.Z3_OP_ITE:
            out.append(e.arg(0))
        for c in e.children():
            out += _ite_conds(c)
    return out


def _cond_const_candidates(names, s0: dict) -> list[Guess]:
    """``cond -> v == c``: what a state variable is on one branch of the init block.

    :func:`_const_candidates` reads a coordinate only where it is already constant,
    so ``t = ite(b >= 1, 1, -1)`` yields nothing. Pinning the condition each way
    makes both branches constant. The condition is over the init block's nondet
    inputs, so it is rewritten over the state variables latching them, and skipped
    when it mentions anything else."""
    latched = [(s0[v], v) for v in names
               if z3.is_const(s0[v]) and s0[v].decl().kind() == z3.Z3_OP_UNINTERPRETED]
    if not latched:
        return []

    def over_state(pred):
        """``pred`` over the state variables, or ``None`` if an input is left."""
        out = z3.substitute(pred, *[(e, z3.Int(v)) for e, v in latched])
        return None if {str(x) for x in get_vars(out)} - set(names) else out

    cands: list[Guess] = []
    for v in names:
        for cond in _ite_conds(s0[v]):
            for truth in (True, False):
                pred = over_state(cond if truth else z3.Not(cond))
                if pred is None:
                    continue
                k = _as_int_const(
                    z3.simplify(z3.substitute(s0[v], (cond, z3.BoolVal(truth)))))
                if k is None:
                    continue
                cands.append((f"({pred})->{v}=={k}", (
                    lambda st, pred=pred, v=v, k=k: z3.Implies(
                        z3.substitute(pred, *[(z3.Int(n), st[n]) for n in names]),
                        st[v] == k))))
    return cands


def infer_invariants(system, timeout_ms: int = 2000) -> list[Guess]:
    """Inductive invariants of ``system``: facts about every reachable state,
    holding at entry (initiation) and preserved by every step (consecution).

    The transition and the entry state are read off ``system``, which is one walk
    of the module shared with the verifier, so nothing here reads the program itself,
    and consecution ranges over every step rather than over a loop guard: an
    invariant is a fact about the module, not about any claim made of it. For a
    module that stutters when its program is done the two agree, since a stutter
    preserves any state predicate."""
    names = system.names
    s, sp, s0 = system.s_map, system.sp_map, system.entry

    # The outer if-gate precondition: assumed at loop entry (initiation) and its
    # conjuncts seeded as candidates (so precondition facts survive as invariants).
    pre = system.precondition or (lambda st: [])
    pre_init = list(pre(s0))                    # entry-gate assumptions at s0
    pre_cands = [(f"pre[{i}]", (lambda st, i=i: pre(st)[i])) for i in range(len(pre(s)))]

    # static sign/pairwise candidates, plus constants derived from the body
    # post-state T(s) and from the init state s0
    seen: set[str] = set()
    cands: list[Guess] = []
    for lbl, f in (_candidates(names)
                   + _const_candidates(names, sp)
                   + _const_candidates(names, s0)
                   + _cond_const_candidates(names, s0)
                   + pre_cands):
        if lbl not in seen:
            seen.add(lbl)
            cands.append((lbl, f))

    def unsat(goal, *assumps) -> bool:
        sol = z3.Solver(); sol.set("timeout", timeout_ms)
        for a in assumps:
            sol.add(a)
        sol.add(goal)
        return sol.check() == z3.unsat         # unknown/sat -> not proven -> drop

    # initiation: candidate holds at the initial state, under the precondition
    kept = [(lbl, f) for (lbl, f) in cands if unsat(z3.Not(f(s0)), *pre_init)]

    # consecution: Houdini fixpoint, kept(s) implies candidate(T(s)), every step
    changed = True
    while changed:
        changed = False
        assumps = [f(s) for _, f in kept]
        survivors = []
        for lbl, f in kept:
            if unsat(z3.Not(f(sp)), *assumps):
                survivors.append((lbl, f))
            else:
                changed = True
        kept = survivors
    return kept


def as_predicates(facts):
    """``facts`` as claim predicates ``(W, S) -> BoolRef``: what a Safety claim
    states and what its inductive witness carries. Houdini's own shape is
    ``state_map -> BoolRef``; the two differ only in how the columns are named."""
    return tuple((lambda W, S, f=f: f({n: S[n] for n in S.names})) for _, f in facts)
