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
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, z0):
        return 0, 100, X(z0)                  # x = 0, y = 100, z = nondet

    def update(self, x, y, z, _z0):
        guard = x < 40
        wx = ite((z == 0), x + 1, x + 2)        # if (z==0) x+1 else x+2
        return ite(guard, wx, x), y, z          # y, z unchanged


def _build():
    x, y, z = var(), var(), var()
    z0 = var()
    prog = Program(theory=LIA, ctrl=(x, y, z), extl=(z0,))
    return prog, {"x": x, "y": y, "z": z}, {"z0": z0}


BENCH = Bench(
    name="ChawdharyCookGulwaniSagivYang-ESOP2008-easy1",
    source="ChawdharyCookGulwaniSagivYang-ESOP2008-easy1.c",
    state=("x", "y", "z"),
    inputs=("z0",),
    build=_build,
)
