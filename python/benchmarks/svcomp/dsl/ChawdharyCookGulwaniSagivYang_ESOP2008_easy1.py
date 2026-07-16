"""ChawdharyCookGulwaniSagivYang-ESOP2008-easy1 — if/else body, nondet param z.

    int x = 0, y = 100;
    int z = __VERIFIER_nondet_int();
    while (x < 40) {
        if (z == 0) { x = x + 1; }
        else        { x = x + 2; }
    }

Notes:
  - `y` is a constant (init 100, never written) and `z` is a nondet parameter
    (read in the branch, never written) — both held unchanged as ctrl vars.
  - `z == 0` uses `eq()` (`==` is not overloaded).
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite, eq

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        z0 = extl
        return 0, 100, nxt(z0)                  # x = 0, y = 100, z = nondet

    def update(self, ctrl):
        x, y, z = ctrl
        guard = x < 40
        wx = ite(eq(z, 0), x + 1, x + 2)        # if (z==0) x+1 else x+2
        return ite(guard, wx, x), y, z          # y, z unchanged


def _build():
    x, y, z = pair(), pair(), pair()
    z0 = pair()
    prog = Program(theory=LIA, ctrl=(x, y, z), extl=(z0,))
    return prog, {"x": x, "y": y, "z": z}, {"z0": z0}


def _domain(s):
    return s["x"] < 40


BENCH = Bench(
    name="ChawdharyCookGulwaniSagivYang-ESOP2008-easy1",
    source="ChawdharyCookGulwaniSagivYang-ESOP2008-easy1.c",
    state=("x", "y", "z"),
    inputs=("z0",),
    build=_build,
    domain=_domain,
)
