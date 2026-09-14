"""AliasDarteFeautrierGonnord-SAS2010-wise.

    int x, y;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    if (x >= 0 && y >= 0) {
        while (x - y > 2 || y - x > 2) {
            if (x < y) { x = x + 1; }
            else       { y = y + 1; }
        }
    }

Precondition: x >= 0 && y >= 0.
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0, y0):
        return X(x0), X(y0)

    def update(self, x, y, _x0, _y0):
        guard = (x - y > 2) | (y - x > 2)
        wx = ite(x < y, x + 1, x)
        wy = ite(x < y, y, y + 1)
        return ite(guard, wx, x), ite(guard, wy, y)


def _build():
    x, y = var(), var()
    x0, y0 = var(), var()
    prog = Program(theory=LIA, ctrl=(x, y), extl=(x0, y0))
    return prog, {"x": x, "y": y}, {"x0": x0, "y0": y0}


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-wise",
    source="AliasDarteFeautrierGonnord-SAS2010-wise.c",
    state=("x", "y"),
    inputs=("x0", "y0"),
    build=_build,
    precondition=lambda s: [s["x"] >= 0, s["y"] >= 0],
)
