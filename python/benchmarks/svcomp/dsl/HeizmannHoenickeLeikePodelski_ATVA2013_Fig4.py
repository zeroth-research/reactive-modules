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
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0 = extl                               # single input, unwrapped
        return nxt(x0), 23                      # x = nondet, y = 23

    def update(self, ctrl):
        x, y = ctrl
        guard = x >= y
        wx = x - 1         # x = x - 1
        return ite(guard, wx, x), y             # y unchanged


def _build():
    x, y = pair(), pair()
    x0 = pair()
    prog = Program(theory=LIA, ctrl=(x, y), extl=(x0,))
    return prog, {"x": x, "y": y}, {"x0": x0}


def _domain(s):
    return s["x"] >= s["y"]


BENCH = Bench(
    name="HeizmannHoenickeLeikePodelski-ATVA2013-Fig4",
    source="HeizmannHoenickeLeikePodelski-ATVA2013-Fig4.c",
    state=("x", "y"),
    inputs=("x0",),
    build=_build,
    domain=_domain,
)
