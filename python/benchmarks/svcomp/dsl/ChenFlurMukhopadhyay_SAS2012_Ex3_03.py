"""ChenFlurMukhopadhyay-SAS2012-Ex3.03.

    int x, y, z;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    z = __VERIFIER_nondet_int();
    while (x > 0) {
        x = x + y;
        y = y + z;
        z = z - 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0, y0, z0 = extl
        return nxt(x0), nxt(y0), nxt(z0)        # x, y, z all nondet

    def update(self, ctrl):
        x, y, z = ctrl
        guard = x > 0
        wx, wy, wz = x, y, z
        wx = wx + wy       # x = x + y
        wy = wy + wz       # y = y + z
        wz = wz - 1        # z = z - 1
        return ite(guard, wx, x), ite(guard, wy, y), ite(guard, wz, z)


def _build():
    x, y, z = pair(), pair(), pair()
    x0, y0, z0 = pair(), pair(), pair()
    prog = Program(theory=LIA, ctrl=(x, y, z), extl=(x0, y0, z0))
    return prog, {"x": x, "y": y, "z": z}, {"x0": x0, "y0": y0, "z0": z0}


def _domain(s):
    return s["x"] > 0


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex3.03",
    source="ChenFlurMukhopadhyay-SAS2012-Ex3.03.c",
    state=("x", "y", "z"),
    inputs=("x0", "y0", "z0"),
    build=_build,
    domain=_domain,
)
