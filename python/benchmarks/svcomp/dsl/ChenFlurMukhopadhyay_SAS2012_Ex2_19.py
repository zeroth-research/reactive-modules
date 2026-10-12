"""ChenFlurMukhopadhyay-SAS2012-Ex2.19.

    int x, y;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    while (x + y > 0) {
        x = x - 1;
        y = -2*y;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0, y0):
        return X(x0), X(y0)                 # x, y both nondet

    def update(self, x, y, _x0, _y0):
        guard = x + y > 0
        wx, wy = x, y
        wx = wx - 1        # x = x - 1
        wy = -2 * wy       # y = -2*y
        return ite(guard, wx, x), ite(guard, wy, y)


def _build():
    x, y = var(), var()
    x0, y0 = var(), var()
    prog = Program(theory=LIA, ctrl=(x, y), extl=(x0, y0))
    return prog, {"x": x, "y": y}, {"x0": x0, "y0": y0}


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex2.19",
    source="ChenFlurMukhopadhyay-SAS2012-Ex2.19.c",
    state=("x", "y"),
    inputs=("x0", "y0"),
    build=_build,
)
