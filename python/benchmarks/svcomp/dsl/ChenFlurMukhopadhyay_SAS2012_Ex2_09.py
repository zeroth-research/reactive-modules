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
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0, y0, n0):
        return X(x0), X(y0), X(n0)        # x, y, n all nondet

    def update(self, x, y, n, _x0, _y0, _n0):
        guard = (x > 0) & (x < n)
        wx, wy = x, y
        wx = wy - x - 5    # x = -x + y - 5   (old y)
        wy = 2 * wy        # y = 2*y
        return ite(guard, wx, x), ite(guard, wy, y), n   # n unchanged


def _build():
    x, y, n = var(), var(), var()
    x0, y0, n0 = var(), var(), var()
    prog = Program(theory=LIA, ctrl=(x, y, n), extl=(x0, y0, n0))
    return prog, {"x": x, "y": y, "n": n}, {"x0": x0, "y0": y0, "n0": n0}


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex2.09",
    source="ChenFlurMukhopadhyay-SAS2012-Ex2.09.c",
    state=("x", "y", "n"),
    inputs=("x0", "y0", "n0"),
    build=_build,
)
