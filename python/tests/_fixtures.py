"""Small synthetic benchmarks, for tests that exercise the real pipeline.

A test that hands the verifier hand-written z3 terms cannot exercise anything the
verifier reads off the *module* — the guard's shape, the node view, the ranking
wire. :func:`loop_bench` builds a real module from a compact spec, so a
test goes through :func:`candidate` — the production path from a bench to a
``certify`` call — rather than a parallel one.

    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    c = candidate(bench, layers)
    proof = certify(c.system, c.claim, c.witness)
"""
from __future__ import annotations

from dataclasses import dataclass

from benchmarks.svcomp._bench import Bench, INT
from benchmarks.svcomp._termination import compose, system_of, terminates
from zrth import LIA, Wire, sugar


@dataclass
class Cand:
    """A composed system, the claim and the witness: what a test hands ``certify``."""
    system: object
    claim: object
    witness: object


def candidate(bench, layers, delta=1.0, invariants=()) -> Cand:
    """``bench``'s program with ``layers`` composed in as a rank, under
    ``terminates()`` and ``decrease``: the production path from a bench to a
    ``certify`` call. ``invariants`` are ``(label, state_map -> BoolRef)`` pairs as
    :func:`._invariants.infer_invariants` returns them."""
    system = system_of(bench)
    system = system.knowing(f(system.s_map) for _, f in invariants)
    composed, witness = compose(system, layers, delta)
    return Cand(composed, terminates(), witness)


def loop_bench(state, update, *, init=None, precondition=None, name="test"):
    """A :class:`Bench` for a closed loop over ``state``.

    ``update`` takes the latched state variables — the single one if ``state`` has
    one entry, else a tuple — and returns their next values, exactly as a DSL
    ``update`` block does. ``init`` does the same for tick 0 and defaults to zeros,
    which is all a test needs unless it exercises ``initiation``."""
    names = tuple(state)
    zeros = tuple(0 for _ in names)

    def build():
        pairs = {n: (Wire(INT), Wire(INT)) for n in names}

        class Program(sugar.Module):
            def init(self):
                return (init or (lambda: zeros))()

            def update(self, ctrl):
                return update(ctrl)

        prog = Program(theory=LIA, ctrl=tuple(pairs[n] for n in names))
        return prog, pairs, {}

    return Bench(name=name, source="", state=names, inputs=(), build=build,
                 precondition=precondition)
