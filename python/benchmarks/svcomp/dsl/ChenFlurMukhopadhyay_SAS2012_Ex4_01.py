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
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0, y0, z0, n0 = extl
        return nxt(x0), nxt(y0), nxt(z0), nxt(n0)   # x, y, z, n all nondet

    def update(self, ctrl):
        x, y, z, n = ctrl
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
    x, y, z, n = pair(), pair(), pair(), pair()
    x0, y0, z0, n0 = pair(), pair(), pair(), pair()
    prog = Program(theory=LIA, ctrl=(x, y, z, n), extl=(x0, y0, z0, n0))
    return (
        prog,
        {"x": x, "y": y, "z": z, "n": n},
        {"x0": x0, "y0": y0, "z0": z0, "n0": n0},
    )


def _domain(s):
    import z3
    return z3.And(s["x"] + s["y"] >= 0, s["x"] <= s["n"])


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex4.01",
    source="ChenFlurMukhopadhyay-SAS2012-Ex4.01.c",
    state=("x", "y", "z", "n"),
    inputs=("x0", "y0", "z0", "n0"),
    build=_build,
    domain=_domain,
)
