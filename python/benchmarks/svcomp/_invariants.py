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


def infer_invariants(bench: Bench, timeout_ms: int = 2000) -> list[Candidate]:
    """Inductive loop invariants for ``bench`` (initiation + consecution)."""
    try:
        s, sp = _transition(bench)
        s0 = _init_state(bench)
    except Exception:
        return []                              # no invariants -> plain domain
    dom = bench.domain(s)
    cands = _candidates(bench)

    def unsat(goal, *assumps) -> bool:
        sol = z3.Solver(); sol.set("timeout", timeout_ms)
        for a in assumps:
            sol.add(a)
        sol.add(goal)
        return sol.check() == z3.unsat         # unknown/sat -> not proven -> drop

    # initiation: candidate holds at the initial state (for all inputs)
    kept = [(lbl, f) for (lbl, f) in cands if unsat(z3.Not(f(s0)))]

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
