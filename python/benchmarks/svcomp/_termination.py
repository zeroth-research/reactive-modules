"""Verify a ranking function against a program module (termination obligation).

Here we discharge the *ranking* obligation: given integer NRF layers
V, that `V(s) >= 0` and `V(s) - V(s') >= delta` on every round the claim
counts, where s' = T(s) is the module's transition.

One module, a property over its wires
=====================================
The program and V are composed into one module and read together: V once as a
sequential atom reading the latched state (its wire carries V(s)) and once as a
combinatorial atom awaiting the next state (V(s')). The property is
``terminates()`` over the program's columns and the witness is
``decrease(V(s) wire, V(s') wire, δ)`` — a linear predicate over two wires of the
graph. Nothing distinguishes program from rank except what the property names.

Interface
=========
:func:`system_of` reads the program module once; :func:`compose` adds the rank
as two atoms and returns the composed system with the witness that names them;
:func:`terminates` is this client's claim. Certifying is the procedure's own
``certify(system, claim, witness)``, and the trainer accepts the first candidate
it certifies. Nothing here verifies anything itself.
"""

from __future__ import annotations


import z3

# torch must load before the zrth C-extension (see _bench)
from ._bench import Bench, INT, var  # noqa: F401
from ._farkas import System, certify, decrease, inductive, read_system
from ._invariants import as_predicates, infer_invariants
from ._property import Liveness, Safety
from zrth import LIA, Module, Term, X
from zrth.expr import collecting
from zrth.sugar import expr, relu


# ---------------------------------------------------------------------------
# The neural ranking function as a reactive-module atom
# ---------------------------------------------------------------------------

def _V_term(xs, layers):
    """V(x) as an LIA term over the state expressions ``xs`` (integer arithmetic,
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


def _v_module(state_vars, layers, *, read_next: bool):
    """A one-output module computing V over the program's state variables.

    ``read_next=False`` -> V(s): a **sequential** atom whose update reads the
    *latched* state, so its output wire carries V(s) (its init awaits the next
    state, since a sequential atom's init may not read a latched wire).
    ``read_next=True`` -> V(s'): a **combinatorial** atom awaiting the program's
    *next* state. Both compute the same function; composed with the program they
    are two wires the witness can name — V at each end of a step.

    The blocks are written out as ordinary terms rather than through
    :mod:`zrth.sugar`, whose methods take one parameter per variable: the state
    here is as wide as the program, which is not known until it is read."""
    out = var()
    state_vars = tuple(state_vars)

    def block(at_next: bool):
        with collecting() as terms:
            xs = [expr(X(v) if at_next else v, theory=LIA) for v in state_vars]
            terms.append(Term(LIA.Id(), [X(out)], [_V_term(xs, layers).wire]))
        return terms

    declared = (out,) + state_vars
    if read_next:
        return Module.combinatorial(declared, block(True)), out
    return Module.sequential(declared, block(True), block(False)), out


# ---------------------------------------------------------------------------
# Building the candidate (the seam onto the decision procedure)
# ---------------------------------------------------------------------------

def terminates(over=None) -> Liveness:
    """This client's claim: the columns ``over`` stop moving.

    A reactive module's update is total, it ticks forever, so termination is a
    property of what the module *encodes*, under the convention that a finished
    program stutters: the rounds that count are those where one of these columns
    changes. Leaving that domain is reaching a fixed point, which a run never
    leaves, so here "leaves infinitely often" is "leaves for good". Defaults to
    every column, which is right because the rank atoms composed alongside are
    stateless; anything stateful composed in must be left out of ``over``."""
    def moving(W, S):
        names = over if over is not None else S.names
        return z3.Or(*[S.next[n] != S[n] for n in names])
    return Liveness(moving)


def system_of(bench: Bench) -> System:
    """``bench``'s program module, read once, with its precondition as ``assume``.

    Built per benchmark and reused for the guard, the invariants and every ranking
    candidate, none of which depend on the candidate."""
    prog, _ctrl, _extl = bench.build()
    return read_system(prog, bench.state).assuming(bench.precondition)


def prove_invariants(system: System):
    """The invariants Houdini finds for ``system``, certified as one Safety claim
    with themselves as the inductive witness — the same route any safety property
    takes. ``None`` when there are none. Candidate-independent: the rank atoms add
    no column, so this is done once per program and assumed for every rank."""
    facts = infer_invariants(system)
    if not facts:
        return None
    preds = as_predicates(facts)
    return certify(system, Safety(lambda W, S: z3.And(*[p(W, S) for p in preds])),
                   inductive(preds))


def compose(system: System, layers, delta: float = 1.0):
    """``system`` with the rank composed in, and the witness that names it.

    V is written out twice as ordinary atoms — once reading the latched state, so
    its wire carries V(s), once awaiting the next, so its wire carries V(s') —
    and the whole is read as one system. The columns are unchanged, since the
    rank variables are read latched by nothing; the program's precondition and
    invariants carry over. The program comes first in the composition, so its
    init has written the next state by the time the rank's block reads it.
    Returns the composed system and ``decrease(V(s) wire, V(s') wire, delta)``."""
    vs_mod, vs = _v_module(system.vars, layers, read_next=False)
    vsp_mod, vsp = _v_module(system.vars, layers, read_next=True)
    composed = (read_system(Module.compose(system.module, vs_mod, vsp_mod), system.names)
                .assuming(system.precondition).knowing(*system.invariant_proofs))
    return composed, decrease(X(vs), X(vsp), delta)
