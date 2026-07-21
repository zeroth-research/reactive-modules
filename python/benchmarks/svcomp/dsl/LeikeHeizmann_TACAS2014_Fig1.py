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
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        q0, y0 = extl
        return nxt(q0), nxt(y0)                  # q, y both nondet

    def update(self, ctrl):
        q, y = ctrl
        guard = q > 0
        wq, wy = q, y
        wq = wq - wy       # q = q - y   (old y)
        wy = wy + 1        # y = y + 1
        return ite(guard, wq, q), ite(guard, wy, y)


def _build():
    q, y = pair(), pair()
    q0, y0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(q, y), extl=(q0, y0))
    return prog, {"q": q, "y": y}, {"q0": q0, "y0": y0}


BENCH = Bench(
    name="LeikeHeizmann-TACAS2014-Fig1",
    source="LeikeHeizmann-TACAS2014-Fig1.c",
    state=("q", "y"),
    inputs=("q0", "y0"),
    build=_build,
)
