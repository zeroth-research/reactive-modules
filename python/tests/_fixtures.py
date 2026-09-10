"""Small synthetic benchmarks, for tests that exercise the real pipeline.

A test that hands the verifier hand-written z3 terms cannot exercise anything the
verifier reads off the *module* — the guard's shape, the node view, the ranking
wire. :func:`loop_bench` builds a real module from a compact spec, so a
test goes through :func:`._termination.build_candidate` and the production
verifier rather than a parallel path.

    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    ob = build_candidate(bench, layers, 1.0, [])
"""
from __future__ import annotations

from benchmarks.svcomp._bench import Bench, INT
from zrth import LIA, Wire, sugar


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
