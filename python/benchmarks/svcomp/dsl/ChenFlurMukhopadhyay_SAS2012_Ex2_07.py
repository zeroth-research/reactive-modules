"""ChenFlurMukhopadhyay-SAS2012-Ex2.07 — conjunctive guard.

    int x, y;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    while (x > 0 && x < y) {
        x = 2*x;
        y = y + 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0, y0 = extl
        return nxt(x0), nxt(y0)                  # x, y both nondet

    def update(self, ctrl):
        x, y = ctrl
        guard = (x > 0) & (x < y)
        wx, wy = x, y
        wx = 2 * x         # x = 2*x
        wy = wy + 1        # y = y + 1
        return ite(guard, wx, x), ite(guard, wy, y)


def _build():
    x, y = pair(), pair()
    x0, y0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(x, y), extl=(x0, y0))
    return prog, {"x": x, "y": y}, {"x0": x0, "y0": y0}


def _domain(s):
    import z3
    return z3.And(s["x"] > 0, s["x"] < s["y"])


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex2.07",
    source="ChenFlurMukhopadhyay-SAS2012-Ex2.07.c",
    state=("x", "y"),
    inputs=("x0", "y0"),
    build=_build,
    domain=_domain,
)
