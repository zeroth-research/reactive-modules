"""Verify a ranking function against a program module (termination obligation).

Here we discharge the *ranking* obligation: given integer NRF layers
V, that `V(s) >= 0` and `V(s) - V(s') >= delta` for every state s in the loop
domain, where s' = T(s) is the module's transition.

One module, a property over its wires
====================================
The program and V are composed into one module and read together: V once as a
sequential atom reading the latched state (its wire carries V(s)) and once as a
combinatorial atom awaiting the next state (V(s')). The property is
``terminates()`` over the program's columns and the witness is
``decrease(V(s) wire, V(s') wire, δ)`` — a linear predicate over two wires of the
graph. Nothing distinguishes program from rank except what the property names.

Interface
=========
The obligation is packaged as an :class:`Obligation` (backend-neutral Z3 pieces)
built by :func:`build_obligation`, and a **verifier** is any callable

    Verifier = Callable[[Obligation], VerifyResult]

so different methods plug in interchangeably.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

import numpy as np
import z3

# torch must load before the zrth C-extension (see _bench)
from ._bench import Bench, INT, pair  # noqa: F401
from ._domain import guard_from_transition
from ._farkas import System, certify, decrease, read_system, reading
from ._property import terminates
from zrth import LIA, Module, sugar
from zrth.sugar import expr, nxt, relu


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Obligation:
    """The ranking obligation over the program module, as Z3 terms.

    ``s_syms``/``sp_syms``: pre- and next-state (the transition). ``V_s``/``V_sp``:
    V evaluated on each. ``guard``: the bare loop guard over ``s_syms``.
    ``invariants``: the inferred invariant predicates over ``s_syms``, kept
    separate from the guard for certificate provenance. The verification domain
    is ``guard`` ∧ ⋀``invariants`` (see :func:`verification_domain`).

    ``net`` is V as the cell verifier reads it off V's module; ``layers`` (the
    integer NRF the trainer produced) is kept for the trainer's record only. An
    SMT verifier uses ``V_s``/``V_sp`` directly. ``system`` is what the decision
    procedure and the proof read; the entry state is theirs to read off it."""
    state: tuple[str, ...]
    s_syms: list
    sp_syms: list
    V_s: object
    V_sp: object
    delta: float
    guard: object
    invariants: tuple = ()
    layers: object = None
    net: object = None
    system: object = None    # program ⊕ V(s) ⊕ V(s'), as the verifier reads it
    prop: object = None      # terminates(), over the program's columns
    rule: object = None      # decrease(V(s) wire, V(s') wire, delta)


@dataclass
class VerifyResult:
    verified: bool
    counterexample: np.ndarray | None = None   # domain state where V fails (for CEGAR)
    certificate: object | None = None           # the Proof, when certified
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
        pre = expr(int(b1[j]), theory=LIA, sort=INT)
        for k in range(len(xs)):
            pre = pre + xs[k] * int(W1[j][k])
        hid.append(relu(pre))
    out = expr(int(b2[0]), theory=LIA, sort=INT)
    for j in range(len(hid)):
        out = out + hid[j] * int(W2[0][j])
    return out


def _xs(extl):
    return list(extl) if isinstance(extl, tuple) else [extl]


def _v_module(state_pairs, layers, *, read_next: bool):
    """A one-output module computing V over the program's state wires.

    ``read_next=False`` -> V(s): a **sequential** atom reading the *latched* state
    (its init awaits the next state, since a sequential atom's init may not read a
    latched wire). ``read_next=True`` -> V(s'): a **combinatorial** atom awaiting
    the program's *next* state. Both compute the same function; composed with the
    program they are two wires the rule can name — V at each end of a step."""
    out = pair()
    if read_next:
        class _V(sugar.Module):
            def assign(self, extl):
                return _V_term([nxt(x) for x in _xs(extl)], layers)
    else:
        class _V(sugar.Module):
            def init(self, extl):
                return _V_term([nxt(x) for x in _xs(extl)], layers)

            def update(self, ctrl, extl):
                return _V_term(_xs(extl), layers)

    return _V(theory=LIA, ctrl=(out,), extl=tuple(state_pairs)), out


# ---------------------------------------------------------------------------
# Building the obligation (the seam onto the decision procedure)
# ---------------------------------------------------------------------------

def system_of(bench: Bench) -> System:
    """``bench``'s program module, read once, with its precondition as ``assume``.

    Built per benchmark and reused for the guard, the invariants and every ranking
    candidate, none of which depend on the candidate."""
    prog, _ctrl, _extl = bench.build()
    return dataclasses.replace(read_system(prog, bench.state),
                               assume=bench.precondition)


def build_obligation(bench: Bench, layers, delta: float, invariants=None,
                     system=None) -> Obligation:
    """The ranking obligation, read off the program module and V's.

    The program is composed with two V modules — one reading the latched state,
    one awaiting the next — and read as one system. The columns are the latched
    wires some term reads (the program's); V's two wires are named by the rule.

    ``invariants`` (from :func:`._invariants.infer_invariants`) are inductive
    loop facts conjoined with the guard to shrink the verification domain to the
    reachable loop states."""
    if system is None:
        system = system_of(bench)
    inv_preds = tuple(f(system.s_map) for _, f in (invariants or []))
    vs_mod, vs = _v_module(system.pairs, layers, read_next=False)
    vsp_mod, vsp = _v_module(system.pairs, layers, read_next=True)
    composed = dataclasses.replace(
        read_system(Module.parallel(system.module, vs_mod, vsp_mod), system.names),
        assume=system.assume, invariants=inv_preds)
    z = composed.view.values
    prop = terminates()
    rule = decrease(vs[1], vsp[1], delta)
    return Obligation(composed.names, list(composed.s_syms), composed.sp_syms,
                      z[vs[1]][0], z[vsp[1]][0], float(delta),
                      guard_from_transition(composed.s_map, composed.sp_map,
                                            composed.names),
                      invariants=inv_preds, layers=layers,
                      net=reading(composed, vs[1]).net, system=composed,
                      prop=prop, rule=rule)


# ---------------------------------------------------------------------------
# Verifiers  (Obligation -> VerifyResult)
# ---------------------------------------------------------------------------

def verification_domain(ob: Obligation):
    """The states the obligation must hold on: guard ∧ invariants."""
    return z3.And(ob.guard, *ob.invariants) if ob.invariants else ob.guard


def _model_cex(ob: Obligation, solver: z3.Solver) -> np.ndarray:
    m = solver.model()
    return np.array([m.eval(v, model_completion=True).as_long() for v in ob.s_syms],
                    dtype=np.float64)


def smt_oneshot(ob: Obligation) -> VerifyResult:
    """One-shot Z3 check: V >= 0 and V(s) - V(s') >= delta on the domain."""
    dom = verification_domain(ob)
    s1 = z3.Solver(); s1.add(dom); s1.add(ob.V_s < 0)
    r1 = s1.check()
    if r1 == z3.sat:
        return VerifyResult(False, _model_cex(ob, s1), status="FAILED(V<0)")
    if r1 == z3.unknown:
        return VerifyResult(False, None, status="UNKNOWN(V>=0)")
    s2 = z3.Solver(); s2.add(dom)
    s2.add(ob.V_s - ob.V_sp < z3.RealVal(ob.delta))
    r2 = s2.check()
    if r2 == z3.sat:
        return VerifyResult(False, _model_cex(ob, s2), status="FAILED(decrease)")
    if r2 == z3.unknown:
        return VerifyResult(False, None, status="UNKNOWN(decrease)")
    return VerifyResult(True, status="VERIFIED")


def farkas_cell(ob: Obligation) -> VerifyResult:
    """Cell/CEGAR Farkas verifier: certifies ``V(s) - V(s') >= delta`` per ReLU
    cell with an exact Farkas certificate (for Lean export). Sound but incomplete
    — cells with a non-affine transition or a nonlinear/disjunctive guard atom
    cannot be certified (returns FAILED). That the rank is bounded below is the
    witness's own check (:func:`._farkas.check_ranks`)."""
    r = certify(ob.system, ob.prop, ob.rule)
    return VerifyResult(r.verified, r.counterexample, certificate=r, status=r.status)
