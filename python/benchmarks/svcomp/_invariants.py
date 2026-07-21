"""Houdini-style loop-invariant inference to strengthen the verification domain.

We seed sign/relational candidate predicates, then drop any that are not **established at loop entry** (initiation) or not **preserved by
the body** (consecution), and return the survivors. Conjoining them with the loop guard shrinks the verification domain to an over-approximation of the reachable
loop states.

Soundness
=========
Both filters are checked with Z3; a ``sat`` or ``unknown`` result drops the candidate (never keeps a candidate we cannot prove), so survivors are genuine
inductive invariants. Initiation is checked against the ``init`` block for *all* inputs (ignoring any precondition) — stronger than required, hence sound, though it may miss invariants that hold only under a precondition. 
"""
from __future__ import annotations

import z3

from ._bench import Bench
from ._domain import guard_from_transition
from zrth import z3 as zz3

# A candidate is (label, state_map -> z3.BoolRef); state_map is {var_name: expr}.
Candidate = tuple


def _transition(bench: Bench):
    """Symbolic loop step: latched state (fresh ints) -> next state."""
    prog, ctrl, _extl = bench.build()
    st = {ctrl[n][0]: [z3.Int(n)] for n in bench.state}
    for atom in prog.atoms:
        for t in atom.update:
            st.update(zip(t.write, zz3.eval(t.itype, [st[w] for w in t.read])))
    s = {n: st[ctrl[n][0]][0] for n in bench.state}
    sp = {n: st[ctrl[n][1]][0] for n in bench.state}
    return s, sp


def _init_state(bench: Bench):
    """Symbolic initial state: run the init block with fresh nondet inputs."""
    prog, ctrl, extl = bench.build()
    st = {extl[name][1]: [z3.Int(f"_in_{name}")] for name in bench.inputs}
    for atom in prog.atoms:
        for t in atom.init:
            st.update(zip(t.write, zz3.eval(t.itype, [st[w] for w in t.read])))
    return {n: st[ctrl[n][1]][0] for n in bench.state}


def _candidates(bench: Bench) -> list[Candidate]:
    """Sign predicates per variable and ±-relations between pairs."""
    cands: list[Candidate] = []
    for v in bench.state:
        cands += [
            (f"{v}>0",   (lambda st, v=v: st[v] > 0)),
            (f"{v}>=0",  (lambda st, v=v: st[v] >= 0)),
            (f"{v}<=0",  (lambda st, v=v: st[v] <= 0)),
            (f"{v}<0",   (lambda st, v=v: st[v] < 0)),
            (f"{v}>=1",  (lambda st, v=v: st[v] >= 1)),
            (f"{v}<=-1", (lambda st, v=v: st[v] <= -1)),
        ]
    vs = list(bench.state)
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


def _const_candidates(bench: Bench, vals: dict) -> list[Candidate]:
    """Candidates from a state's constant coordinates (nuTerm's ``_seed_candidates``,
    applied to a cut-point segment's post-state). ``vals`` is a symbolic state
    (the body post-state ``T(s)`` or the init state ``s0``): ``v==c`` / ``v>=c`` /
    ``v<=c`` when ``vals[v]`` is constant, and ``vi-vj==d`` / ``vi+vj==s`` when a
    pair combination is constant."""
    cands: list[Candidate] = []
    for v in bench.state:
        c = _as_int_const(vals[v])
        if c is not None:
            cands += [
                (f"{v}=={c}", (lambda st, v=v, c=c: st[v] == c)),
                (f"{v}>={c}", (lambda st, v=v, c=c: st[v] >= c)),
                (f"{v}<={c}", (lambda st, v=v, c=c: st[v] <= c)),
            ]
    vs = list(bench.state)
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


def infer_invariants(bench: Bench, timeout_ms: int = 2000) -> list[Candidate]:
    """Inductive loop invariants for ``bench`` (initiation + consecution)."""
    try:
        s, sp = _transition(bench)
        s0 = _init_state(bench)
    except Exception:
        return []                              # no invariants -> plain domain
    dom = guard_from_transition(s, sp, bench.state)

    # The outer if-gate precondition: assumed at loop entry (initiation) and its
    # conjuncts seeded as candidates (so precondition facts survive as invariants).
    pre = bench.precondition or (lambda st: [])
    pre_init = list(pre(s0))                    # entry-gate assumptions at s0
    pre_cands = [(f"pre[{i}]", (lambda st, i=i: pre(st)[i])) for i in range(len(pre(s)))]

    # static sign/pairwise candidates + constants derived from the body post-state
    # (T(s)) and the init state (s0) — the cut-point segments nuTerm seeds from.
    seen: set[str] = set()
    cands: list[Candidate] = []
    for lbl, f in (_candidates(bench)
                   + _const_candidates(bench, sp)
                   + _const_candidates(bench, s0)
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

    # consecution: Houdini fixpoint — (guard & kept(s)) implies candidate(T(s))
    changed = True
    while changed:
        changed = False
        assumps = [dom] + [f(s) for _, f in kept]
        survivors = []
        for lbl, f in kept:
            if unsat(z3.Not(f(sp)), *assumps):
                survivors.append((lbl, f))
            else:
                changed = True
        kept = survivors
    return kept


def invariant_domain(invariants: list[Candidate], state_map: dict):
    """Conjoin the inferred invariants, evaluated on ``state_map``, or None."""
    if not invariants:
        return None
    return z3.And(*[f(state_map) for _, f in invariants])
