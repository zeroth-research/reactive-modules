"""GulavaniGulwani-CAV2008-Fig1a.

    int x, y, z, i;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    z = __VERIFIER_nondet_int();
    i = __VERIFIER_nondet_int();
    while (x < y) {
        i = i + 1;
        if (z > x) { x = x + 1; }
        else       { z = z + 1; }
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0, y0, z0, i0):
        return X(x0), X(y0), X(z0), X(i0)   # all nondet

    def update(self, x, y, z, i, _x0, _y0, _z0, _i0):
        guard = x < y
        wx, wy, wz, wi = x, y, z, i
        wi = wi + 1                             # i = i + 1
        cond = wz > wx                          # if (z > x) ...
        wx = ite(cond, wx + 1, wx)              #   x = x + 1
        wz = ite(cond, wz, wz + 1)              #   else z = z + 1
        return (ite(guard, wx, x), ite(guard, wy, y),
                ite(guard, wz, z), ite(guard, wi, i))


def _build():
    x, y, z, i = var(), var(), var(), var()
    x0, y0, z0, i0 = var(), var(), var(), var()
    prog = Program(theory=LIA, ctrl=(x, y, z, i), extl=(x0, y0, z0, i0))
    return (prog, {"x": x, "y": y, "z": z, "i": i},
            {"x0": x0, "y0": y0, "z0": z0, "i0": i0})


BENCH = Bench(
    name="GulavaniGulwani-CAV2008-Fig1a",
    source="GulavaniGulwani-CAV2008-Fig1a.c",
    state=("x", "y", "z", "i"),
    inputs=("x0", "y0", "z0", "i0"),
    build=_build,
)
