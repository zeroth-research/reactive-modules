"""HeizmannHoenickeLeikePodelski-ATVA2013-Fig9.

    int x, y, z;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    z = __VERIFIER_nondet_int();
    if (2*y >= z) {
        while (x >= 0 && z == 1) {
            x = x - 2*y + 1;
        }
    }

The loop only runs under the outer `if (2*y >= z)`; the precondition
restricts the comparison to that domain.
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite, eq

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0, y0, z0 = extl
        return nxt(x0), nxt(y0), nxt(z0)        # x, y, z all nondet

    def update(self, ctrl):
        x, y, z = ctrl
        guard = (x >= 0) & eq(z, 1)
        wx = x - 2 * y + 1     # x = x - 2*y + 1
        return ite(guard, wx, x), y, z          # y, z unchanged


def _build():
    x, y, z = pair(), pair(), pair()
    x0, y0, z0 = pair(), pair(), pair()
    prog = Program(theory=LIA, ctrl=(x, y, z), extl=(x0, y0, z0))
    return prog, {"x": x, "y": y, "z": z}, {"x0": x0, "y0": y0, "z0": z0}


def _domain(s):
    import z3
    return z3.And(s["x"] >= 0, s["z"] == 1)


BENCH = Bench(
    name="HeizmannHoenickeLeikePodelski-ATVA2013-Fig9",
    source="HeizmannHoenickeLeikePodelski-ATVA2013-Fig9.c",
    state=("x", "y", "z"),
    inputs=("x0", "y0", "z0"),
    build=_build,
    domain=_domain,
    precondition=lambda i: 2 * i["y0"] >= i["z0"],
)
