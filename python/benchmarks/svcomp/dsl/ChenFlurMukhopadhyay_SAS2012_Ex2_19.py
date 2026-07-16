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
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0, y0 = extl
        return nxt(x0), nxt(y0)                 # x, y both nondet

    def update(self, ctrl):
        x, y = ctrl
        guard = x + y > 0
        wx, wy = x, y
        wx = wx - 1        # x = x - 1
        wy = -2 * wy       # y = -2*y
        return ite(guard, wx, x), ite(guard, wy, y)


def _build():
    x, y = pair(), pair()
    x0, y0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(x, y), extl=(x0, y0))
    return prog, {"x": x, "y": y}, {"x0": x0, "y0": y0}


def _domain(s):
    return s["x"] + s["y"] > 0


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex2.19",
    source="ChenFlurMukhopadhyay-SAS2012-Ex2.19.c",
    state=("x", "y"),
    inputs=("x0", "y0"),
    build=_build,
    domain=_domain,
)
