"""ChenFlurMukhopadhyay-SAS2012-Ex2.09 — three nondet vars, param `n`.

    int x, y, n;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    n = __VERIFIER_nondet_int();
    while (x > 0 && x < n) {
        x = -x + y - 5;
        y = 2*y;
    }

Notes:
  - `n` is a nondet parameter, read in the guard, never written — held as ctrl.
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0, y0, n0 = extl
        return nxt(x0), nxt(y0), nxt(n0)        # x, y, n all nondet

    def update(self, ctrl):
        x, y, n = ctrl
        guard = (x > 0) & (x < n)
        wx, wy = x, y
        wx = wy - x - 5    # x = -x + y - 5   (old y)
        wy = 2 * wy        # y = 2*y
        return ite(guard, wx, x), ite(guard, wy, y), n   # n unchanged


def _build():
    x, y, n = pair(), pair(), pair()
    x0, y0, n0 = pair(), pair(), pair()
    prog = Program(theory=LIA, ctrl=(x, y, n), extl=(x0, y0, n0))
    return prog, {"x": x, "y": y, "n": n}, {"x0": x0, "y0": y0, "n0": n0}


def _domain(s):
    import z3
    return z3.And(s["x"] > 0, s["x"] < s["n"])


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex2.09",
    source="ChenFlurMukhopadhyay-SAS2012-Ex2.09.c",
    state=("x", "y", "n"),
    inputs=("x0", "y0", "n0"),
    build=_build,
    domain=_domain,
)
