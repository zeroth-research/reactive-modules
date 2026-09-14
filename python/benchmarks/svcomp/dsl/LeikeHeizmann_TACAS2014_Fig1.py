"""LeikeHeizmann-TACAS2014-Fig1 — simple two-variable coupled loop.

    int q, y;
    q = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    while (q > 0) {
        q = q - y;
        y = y + 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, q0, y0):
        return X(q0), X(y0)                  # q, y both nondet

    def update(self, q, y, _q0, _y0):
        guard = q > 0
        wq, wy = q, y
        wq = wq - wy       # q = q - y   (old y)
        wy = wy + 1        # y = y + 1
        return ite(guard, wq, q), ite(guard, wy, y)


def _build():
    q, y = var(), var()
    q0, y0 = var(), var()
    prog = Program(theory=LIA, ctrl=(q, y), extl=(q0, y0))
    return prog, {"q": q, "y": y}, {"q0": q0, "y0": y0}


BENCH = Bench(
    name="LeikeHeizmann-TACAS2014-Fig1",
    source="LeikeHeizmann-TACAS2014-Fig1.c",
    state=("q", "y"),
    inputs=("q0", "y0"),
    build=_build,
)
