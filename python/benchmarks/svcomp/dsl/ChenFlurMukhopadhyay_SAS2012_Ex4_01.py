"""ChenFlurMukhopadhyay-SAS2012-Ex4.01.

    int x, y, z, n;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    z = __VERIFIER_nondet_int();
    n = __VERIFIER_nondet_int();
    while (x + y >= 0 && x <= n) {
        x = 2*x + y;
        y = z;
        z = z;
        z = z + 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0, y0, z0, n0):
        return X(x0), X(y0), X(z0), X(n0)   # x, y, z, n all nondet

    def update(self, x, y, z, n, _x0, _y0, _z0, _n0):
        guard = (x + y >= 0) & (x <= n)
        wx, wy, wz, wn = x, y, z, n
        wx = 2 * wx + wy   # x = 2*x + y
        wy = wz            # y = z
        wz = wz            # z = z
        wz = wz + 1        # z = z + 1
        return (
            ite(guard, wx, x),
            ite(guard, wy, y),
            ite(guard, wz, z),
            ite(guard, wn, n),
        )


def _build():
    x, y, z, n = var(), var(), var(), var()
    x0, y0, z0, n0 = var(), var(), var(), var()
    prog = Program(theory=LIA, ctrl=(x, y, z, n), extl=(x0, y0, z0, n0))
    return (
        prog,
        {"x": x, "y": y, "z": z, "n": n},
        {"x0": x0, "y0": y0, "z0": z0, "n0": n0},
    )


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex4.01",
    source="ChenFlurMukhopadhyay-SAS2012-Ex4.01.c",
    state=("x", "y", "z", "n"),
    inputs=("x0", "y0", "z0", "n0"),
    build=_build,
)
