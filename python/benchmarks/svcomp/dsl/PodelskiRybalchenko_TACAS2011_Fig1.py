"""PodelskiRybalchenko-TACAS2011-Fig1 — single-variable countdown.

    int y;
    y = __VERIFIER_nondet_int();
    while (y >= 0) {
        y = y - 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        y0 = extl                               # single input, unwrapped
        return nxt(y0)                          # y nondet

    def update(self, ctrl):
        y = ctrl                                # single ctrl, unwrapped
        guard = y >= 0
        wy = y
        wy = wy - 1        # y = y - 1
        return ite(guard, wy, y)


def _build():
    y = pair()
    y0 = pair()
    prog = Program(theory=LIA, ctrl=(y,), extl=(y0,))
    return prog, {"y": y}, {"y0": y0}


BENCH = Bench(
    name="PodelskiRybalchenko-TACAS2011-Fig1",
    source="PodelskiRybalchenko-TACAS2011-Fig1.c",
    state=("y",),
    inputs=("y0",),
    build=_build,
)
