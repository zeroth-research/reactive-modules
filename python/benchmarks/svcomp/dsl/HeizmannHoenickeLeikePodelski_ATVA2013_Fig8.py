"""HeizmannHoenickeLeikePodelski-ATVA2013-Fig8.

    int x, y;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    if (2*y >= 1) {
        while (x >= 0) {
            x = x - 2*y + 1;
        }
    }

The loop only runs under the outer `if (2*y >= 1)`; the precondition
restricts the comparison to that domain.
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0, y0 = extl
        return nxt(x0), nxt(y0)                 # x, y both nondet

    def update(self, ctrl):
        x, y = ctrl
        guard = x >= 0
        wx = x - 2 * y + 1     # x = x - 2*y + 1
        return ite(guard, wx, x), y             # y unchanged


def _build():
    x, y = pair(), pair()
    x0, y0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(x, y), extl=(x0, y0))
    return prog, {"x": x, "y": y}, {"x0": x0, "y0": y0}


BENCH = Bench(
    name="HeizmannHoenickeLeikePodelski-ATVA2013-Fig8",
    source="HeizmannHoenickeLeikePodelski-ATVA2013-Fig8.c",
    state=("x", "y"),
    inputs=("x0", "y0"),
    build=_build,
    precondition=lambda s: [2 * s["y"] >= 1],
)
