"""Verify a ranking function against a program module (termination obligation).

Here we discharge the *ranking* obligation: given integer NRF layers
V, that `V(s) >= 0` and `V(s) - V(s') >= delta` for every state s in the loop
domain, where s' = T(s) is the module's transition.

The composed system
===================
The obligation is read off the *composed* system. :func:`build_obligation`
builds `program ⊕ V(s) ⊕ V(s')` as one ``Module.parallel``. V(s) is a sequential
atom over the latched state; V(s') is a combinatorial atom awaiting the program's
next state. V(s), V(s') and s' = T(s) are all wires of that single module.

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
from ._bench import Bench, INT  # noqa: F401
from zrth import LIA, Module, Wire
from zrth import z3 as zz3
from zrth.dsl import const, dslModule, nxt, relu


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
# The neural ranking function as a reactive-module atom
# ---------------------------------------------------------------------------

def _V_term(xs, layers):
    """V(x) as an LIA term over the state wires ``xs`` (integer arithmetic,
    matching the quantized net). layers = [(W1,b1),(W2,b2)]; ReLU -> relu()."""
    (W1, b1), (W2, b2) = layers
    hid = []
    for j in range(W1.shape[0]):
        pre = const(int(b1[j]), LIA)
        for k in range(len(xs)):
            pre = pre + xs[k] * int(W1[j][k])
        hid.append(relu(pre))
    out = const(int(b2[0]), LIA)
    for j in range(len(hid)):
        out = out + hid[j] * int(W2[0][j])
    return out


def _xs(extl):
    return list(extl) if isinstance(extl, tuple) else [extl]


def _v_module(state_pairs, layers, *, read_next: bool):
    """A one-output module computing V over the program's state wires.

    ``read_next=False`` -> V(s): reads the *latched* state, so it is a **sequential**
    atom (init awaits the next state, since a sequential atom's init may not read a
    latched wire; update reads the latched state).
    ``read_next=True``  -> V(s'): only ever awaits the program's *next* state, so it
    is a **combinatorial** atom (a single ``assign`` block, no init)."""
    out = (Wire(INT), Wire(INT))
    if read_next:
        class _V(dslModule):
            def assign(self, extl):
                return _V_term([nxt(x) for x in _xs(extl)], layers)
    else:
        class _V(dslModule):
            def init(self, extl):
                return _V_term([nxt(x) for x in _xs(extl)], layers)

            def update(self, ctrl, extl):
                return _V_term(_xs(extl), layers)

    return _V(theory=LIA, ctrl=(out,), extl=tuple(state_pairs)), out


# ---------------------------------------------------------------------------
# Building the obligation (the composition seam)
# ---------------------------------------------------------------------------

def build_obligation(bench: Bench, layers, delta: float) -> Obligation:
    """Compose program ⊕ V(s) ⊕ V(s') into ONE reactive module, then read the
    ranking obligation off it.

    The program, the ranking value at the current state V(s), and the value at
    the successor V(s') are three atoms of a single ``Module.parallel`` system;
    await-ordering places V(s') after the program's transition. We symbolically
    execute one ``update`` (latched state = fresh Z3 ints) and read s' = T(s),
    V(s) and V(s') straight off the composed system's wires."""
    prog, ctrl, _extl = bench.build()
    state_pairs = [ctrl[n] for n in bench.state]
    vs_mod, vs = _v_module(state_pairs, layers, read_next=False)
    vsp_mod, vsp = _v_module(state_pairs, layers, read_next=True)
    system = Module.parallel(prog, vs_mod, vsp_mod)

    z = {ctrl[n][0]: [z3.Int(n)] for n in bench.state}
    for atom in system.atoms:
        for term in atom.update:
            z.update(zip(term.write, zz3.eval(term.itype, [z[w] for w in term.read])))
    s_syms = [z[ctrl[n][0]][0] for n in bench.state]
    sp_syms = [z[ctrl[n][1]][0] for n in bench.state]
    dom = bench.domain({n: z[ctrl[n][0]][0] for n in bench.state})
    return Obligation(bench.state, s_syms, sp_syms, z[vs[1]][0], z[vsp[1]][0],
                      dom, float(delta))


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
