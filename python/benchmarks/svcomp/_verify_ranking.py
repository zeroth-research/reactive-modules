"""Verify a ranking function against a program module (termination obligation).

Here we check the *ranking* obligation: given integer NRF layers V,
that `V(s) >= 0` and `V(s) - V(s') >= delta` for every state s in the loop
domain, where s' = T(s) is the module's transition. This is the smt_oneshot
method (one Z3 query, verdict only); ReLU lowers to `If(pre >= 0, pre, 0)`.

Consumed by `_train` (round-and-rebuild picks the first V that verifies), and
usable standalone on a hand-written or externally-supplied V — no training
required.
"""

from __future__ import annotations

import z3

# torch must load before the zrth C-extension (see _bench)
from ._bench import Bench  # noqa: F401
from zrth import z3 as zz3


def V_z3(layers, x):
    """V(x) as a Z3 real expression from integer layers [(W1,b1),(W2,b2)];
    `x` is a list of Z3 arith terms (the state vector)."""
    (W1, b1), (W2, b2) = layers
    h = []
    for j in range(W1.shape[0]):
        pre = z3.RealVal(int(b1[j])) + sum(z3.RealVal(int(W1[j][k])) * x[k]
                                           for k in range(len(x)))
        h.append(z3.If(pre >= 0, pre, z3.RealVal(0)))
    return z3.RealVal(int(b2[0])) + sum(z3.RealVal(int(W2[0][j])) * h[j]
                                        for j in range(len(h)))


def verify(bench: Bench, layers, delta: float) -> bool:
    """VERIFIED iff V >= 0 and V(s) - V(s') >= delta for all s in the domain."""
    prog, ctrl, _extl = bench.build()
    state = {ctrl[n][0]: [z3.Int(n)] for n in bench.state}
    for atom in prog.atoms:
        for term in atom.update:
            state.update(zip(term.write, zz3.eval(term.itype, [state[w] for w in term.read])))
    s_syms = [state[ctrl[n][0]][0] for n in bench.state]
    sp_syms = [state[ctrl[n][1]][0] for n in bench.state]
    dom = bench.domain({n: state[ctrl[n][0]][0] for n in bench.state})
    V_s = V_z3(layers, [z3.ToReal(v) for v in s_syms])
    V_sp = V_z3(layers, [z3.ToReal(v) for v in sp_syms])

    s1 = z3.Solver(); s1.add(dom); s1.add(V_s < 0)
    if s1.check() != z3.unsat:
        return False
    s2 = z3.Solver(); s2.add(dom); s2.add(V_s - V_sp < z3.RealVal(float(delta)))
    return s2.check() == z3.unsat
