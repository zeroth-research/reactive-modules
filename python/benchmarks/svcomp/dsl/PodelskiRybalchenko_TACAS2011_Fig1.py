"""PodelskiRybalchenko-TACAS2011-Fig1 — single-variable countdown.

    int y;
    y = __VERIFIER_nondet_int();
    while (y >= 0) {
        y = y - 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, y0):  # single input, unwrapped
        return X(y0)                          # y nondet

    def update(self, y, _y0):  # single ctrl, unwrapped
        guard = y >= 0
        wy = y
        wy = wy - 1        # y = y - 1
        return ite(guard, wy, y)


def _build():
    y = var()
    y0 = var()
    prog = Program(theory=LIA, ctrl=(y,), extl=(y0,))
    return prog, {"y": y}, {"y0": y0}


BENCH = Bench(
    name="PodelskiRybalchenko-TACAS2011-Fig1",
    source="PodelskiRybalchenko-TACAS2011-Fig1.c",
    state=("y",),
    inputs=("y0",),
    build=_build,
)
