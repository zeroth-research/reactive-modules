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
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0, y0, z0):
        return X(x0), X(y0), X(z0)        # x, y, z all nondet

    def update(self, x, y, z, _x0, _y0, _z0):
        guard = (x >= 0) & (z == 1)
        wx = x - 2 * y + 1     # x = x - 2*y + 1
        return ite(guard, wx, x), y, z          # y, z unchanged


def _build():
    x, y, z = var(), var(), var()
    x0, y0, z0 = var(), var(), var()
    prog = Program(theory=LIA, ctrl=(x, y, z), extl=(x0, y0, z0))
    return prog, {"x": x, "y": y, "z": z}, {"x0": x0, "y0": y0, "z0": z0}


BENCH = Bench(
    name="HeizmannHoenickeLeikePodelski-ATVA2013-Fig9",
    source="HeizmannHoenickeLeikePodelski-ATVA2013-Fig9.c",
    state=("x", "y", "z"),
    inputs=("x0", "y0", "z0"),
    build=_build,
    precondition=lambda s: [2 * s["y"] >= s["z"]],
)
