"""ChenFlurMukhopadhyay-SAS2012-Ex2.10 — conjunctive guard, linear.

    int x, y;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    while (x > 0 && y < 0) {
        x = x + y;
        y = y - 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0, y0):
        return X(x0), X(y0)                  # x, y both nondet

    def update(self, x, y, _x0, _y0):
        guard = (x > 0) & (y < 0)
        wx, wy = x, y
        wx = wx + wy       # x = x + y   (old y)
        wy = wy - 1        # y = y - 1
        return ite(guard, wx, x), ite(guard, wy, y)


def _build():
    x, y = var(), var()
    x0, y0 = var(), var()
    prog = Program(theory=LIA, ctrl=(x, y), extl=(x0, y0))
    return prog, {"x": x, "y": y}, {"x0": x0, "y0": y0}


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex2.10",
    source="ChenFlurMukhopadhyay-SAS2012-Ex2.10.c",
    state=("x", "y"),
    inputs=("x0", "y0"),
    build=_build,
)
