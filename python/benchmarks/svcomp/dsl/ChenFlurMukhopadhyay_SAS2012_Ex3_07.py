"""ChenFlurMukhopadhyay-SAS2012-Ex3.07.

    int x, y, z;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    z = __VERIFIER_nondet_int();
    while (x >= 0) {
        x = x + y;
        y = z;
        z = -z - 1;
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
        guard = x >= 0
        wx, wy, wz = x, y, z
        wx = wx + wy       # x = x + y
        wy = wz            # y = z
        wz = -1 * wz - 1   # z = -z - 1
        return ite(guard, wx, x), ite(guard, wy, y), ite(guard, wz, z)


def _build():
    x, y, z = var(), var(), var()
    x0, y0, z0 = var(), var(), var()
    prog = Program(theory=LIA, ctrl=(x, y, z), extl=(x0, y0, z0))
    return prog, {"x": x, "y": y, "z": z}, {"x0": x0, "y0": y0, "z0": z0}


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex3.07",
    source="ChenFlurMukhopadhyay-SAS2012-Ex3.07.c",
    state=("x", "y", "z"),
    inputs=("x0", "y0", "z0"),
    build=_build,
)
