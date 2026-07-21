"""AliasDarteFeautrierGonnord-SAS2010-easy2-1.

    int x = 12, y = 0, z = __VERIFIER_nondet_int();
    while (z > 0) {
        x = x + 1;
        y = y - 1;
        z = z - 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        z0 = extl
        return 12, 0, nxt(z0)                    # x = 12, y = 0, z = nondet

    def update(self, ctrl):
        x, y, z = ctrl
        guard = z > 0
        wx, wy, wz = x, y, z
        wx = wx + 1
        wy = wy - 1
        wz = wz - 1
        return ite(guard, wx, x), ite(guard, wy, y), ite(guard, wz, z)


def _build():
    x, y, z = pair(), pair(), pair()
    z0 = pair()
    prog = Program(theory=LIA, ctrl=(x, y, z), extl=(z0,))
    return prog, {"x": x, "y": y, "z": z}, {"z0": z0}


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-easy2-1",
    source="AliasDarteFeautrierGonnord-SAS2010-easy2-1.c",
    state=("x", "y", "z"),
    inputs=("z0",),
    build=_build,
)
