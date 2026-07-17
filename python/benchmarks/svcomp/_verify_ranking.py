"""Verify a ranking function against a program module (termination obligation).

Here we discharge the *ranking* obligation: given integer NRF layers
V, that `V(s) >= 0` and `V(s) - V(s') >= delta` for every state s in the loop
domain, where s' = T(s) is the module's transition.

Interface
=========
The obligation is packaged as an :class:`Obligation` (backend-neutral Z3 pieces)
built by :func:`build_obligation`, and a **verifier** is any callable

    Verifier = Callable[[Obligation], VerifyResult]

so different methods plug in interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import z3

# torch must load before the zrth C-extension (see _bench)
from ._bench import Bench  # noqa: F401
from zrth import z3 as zz3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Obligation:
    """The ranking obligation over the (composed) system, as Z3 terms.

    ``s_syms``/``sp_syms``: pre- and next-state (the transition). ``V_s``/``V_sp``:
    V evaluated on each. ``domain``: the loop condition over ``s_syms``. A verifier
    consumes only this — it need not know how the system was built."""
    state: tuple[str, ...]
    s_syms: list
    sp_syms: list
    V_s: object
    V_sp: object
    domain: object
    delta: float


@dataclass
class VerifyResult:
    verified: bool
    counterexample: np.ndarray | None = None   # domain state where V fails (for CEGAR)
    certificate: object | None = None           # e.g. Farkas cert (future backends)
    status: str = ""                             # VERIFIED / FAILED(...) / UNKNOWN


Verifier = Callable[[Obligation], VerifyResult]


# ---------------------------------------------------------------------------
# Building the obligation (the composition seam)
# ---------------------------------------------------------------------------

def V_z3(layers, x):
    """V(x) as a Z3 real expression from integer layers [(W1,b1),(W2,b2)];
    `x` is a list of Z3 arith terms (the state vector). ReLU -> If(pre>=0,pre,0)."""
    (W1, b1), (W2, b2) = layers
    h = []
    for j in range(W1.shape[0]):
        pre = z3.RealVal(int(b1[j])) + sum(z3.RealVal(int(W1[j][k])) * x[k]
                                           for k in range(len(x)))
        h.append(z3.If(pre >= 0, pre, z3.RealVal(0)))
    return z3.RealVal(int(b2[0])) + sum(z3.RealVal(int(W2[0][j])) * h[j]
                                        for j in range(len(h)))


def build_obligation(bench: Bench, layers, delta: float) -> Obligation:
    """Compose the program transition (s -> s') with V, as an Obligation."""
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
    return Obligation(bench.state, s_syms, sp_syms, V_s, V_sp, dom, float(delta))


# ---------------------------------------------------------------------------
# Verifiers  (Obligation -> VerifyResult)
# ---------------------------------------------------------------------------

def _model_cex(ob: Obligation, solver: z3.Solver) -> np.ndarray:
    m = solver.model()
    return np.array([m.eval(v, model_completion=True).as_long() for v in ob.s_syms],
                    dtype=np.float64)


def smt_oneshot(ob: Obligation) -> VerifyResult:
    """One-shot Z3 check: V >= 0 and V(s) - V(s') >= delta on the domain."""
    s1 = z3.Solver(); s1.add(ob.domain); s1.add(ob.V_s < 0)
    r1 = s1.check()
    if r1 == z3.sat:
        return VerifyResult(False, _model_cex(ob, s1), status="FAILED(V<0)")
    if r1 == z3.unknown:
        return VerifyResult(False, None, status="UNKNOWN(V>=0)")
    s2 = z3.Solver(); s2.add(ob.domain)
    s2.add(ob.V_s - ob.V_sp < z3.RealVal(ob.delta))
    r2 = s2.check()
    if r2 == z3.sat:
        return VerifyResult(False, _model_cex(ob, s2), status="FAILED(decrease)")
    if r2 == z3.unknown:
        return VerifyResult(False, None, status="UNKNOWN(decrease)")
    return VerifyResult(True, status="VERIFIED")


# ---------------------------------------------------------------------------
# Convenience wrappers (standalone use)
# ---------------------------------------------------------------------------

def verify(bench: Bench, layers, delta: float, verifier: Verifier = smt_oneshot) -> bool:
    """True iff `verifier` discharges the ranking obligation for V=layers."""
    return verifier(build_obligation(bench, layers, delta)).verified


def counterexample(bench: Bench, layers, delta: float,
                   verifier: Verifier = smt_oneshot) -> np.ndarray | None:
    """A domain state where V fails (for CEGAR), or None if it verifies."""
    return verifier(build_obligation(bench, layers, delta)).counterexample
