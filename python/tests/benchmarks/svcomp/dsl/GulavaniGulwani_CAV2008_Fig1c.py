"""GulavaniGulwani-CAV2008-Fig1c.

    int x, i, n;
    x = __VERIFIER_nondet_int();
    i = __VERIFIER_nondet_int();
    n = __VERIFIER_nondet_int();
    while (x < n) {
        i = i + 1;
        x = x + 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0, i0, n0):
        return X(x0), X(i0), X(n0)        # x, i, n all nondet

    def update(self, x, i, n, _x0, _i0, _n0):
        guard = x < n
        wx, wi, wn = x, i, n
        wi = wi + 1                             # i = i + 1
        wx = wx + 1                             # x = x + 1
        return ite(guard, wx, x), ite(guard, wi, i), ite(guard, wn, n)


def _build():
    x, i, n = var(), var(), var()
    x0, i0, n0 = var(), var(), var()
    prog = Program(theory=LIA, ctrl=(x, i, n), extl=(x0, i0, n0))
    return prog, {"x": x, "i": i, "n": n}, {"x0": x0, "i0": i0, "n0": n0}


BENCH = Bench(
    name="GulavaniGulwani-CAV2008-Fig1c",
    source="GulavaniGulwani-CAV2008-Fig1c.c",
    state=("x", "i", "n"),
    inputs=("x0", "i0", "n0"),
    build=_build,
)
