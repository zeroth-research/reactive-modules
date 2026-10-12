"""ChenFlurMukhopadhyay-SAS2012-Ex3.05.

    int x, y, z;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    z = __VERIFIER_nondet_int();
    while (x >= 0 && x <= z) {
        x = 2*x + y;
        y = y + 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0, y0, z0):
        return X(x0), X(y0), X(z0)        # x, y, z all nondet

    def update(self, x, y, z, _x0, _y0, _z0):
        guard = (x >= 0) & (x <= z)
        wx, wy, wz = x, y, z
        wx = 2 * wx + wy   # x = 2*x + y
        wy = wy + 1        # y = y + 1
        return ite(guard, wx, x), ite(guard, wy, y), ite(guard, wz, z)


def _build():
    x, y, z = var(), var(), var()
    x0, y0, z0 = var(), var(), var()
    prog = Program(theory=LIA, ctrl=(x, y, z), extl=(x0, y0, z0))
    return prog, {"x": x, "y": y, "z": z}, {"x0": x0, "y0": y0, "z0": z0}


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex3.05",
    source="ChenFlurMukhopadhyay-SAS2012-Ex3.05.c",
    state=("x", "y", "z"),
    inputs=("x0", "y0", "z0"),
    build=_build,
)
