"""HeizmannHoenickeLeikePodelski-ATVA2013-Fig4.

    int x, y;
    x = __VERIFIER_nondet_int();
    y = 23;
    while (x >= y) {
        x = x - 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0):  # single input, unwrapped
        return X(x0), 23                      # x = nondet, y = 23

    def update(self, x, y, _x0):
        guard = x >= y
        wx = x - 1         # x = x - 1
        return ite(guard, wx, x), y             # y unchanged


def _build():
    x, y = var(), var()
    x0 = var()
    prog = Program(theory=LIA, ctrl=(x, y), extl=(x0,))
    return prog, {"x": x, "y": y}, {"x0": x0}


BENCH = Bench(
    name="HeizmannHoenickeLeikePodelski-ATVA2013-Fig4",
    source="HeizmannHoenickeLeikePodelski-ATVA2013-Fig4.c",
    state=("x", "y"),
    inputs=("x0",),
    build=_build,
)
