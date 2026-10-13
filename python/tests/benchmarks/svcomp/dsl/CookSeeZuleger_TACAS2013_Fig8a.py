"""CookSeeZuleger-TACAS2013-Fig8a.

    int x;
    x = __VERIFIER_nondet_int();
    while (x != 0) {
        if (x > 0) { x = x - 1; }
        else       { x = x + 1; }
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0):  # single input, unwrapped
        return X(x0)                          # x = nondet

    def update(self, x, _x0):
        guard = (x != 0)
        wx = ite(x > 0, x - 1, x + 1)           # if (x>0) x-1 else x+1
        return ite(guard, wx, x)


def _build():
    x = var()
    x0 = var()
    prog = Program(theory=LIA, ctrl=(x,), extl=(x0,))
    return prog, {"x": x}, {"x0": x0}


BENCH = Bench(
    name="CookSeeZuleger-TACAS2013-Fig8a",
    source="CookSeeZuleger-TACAS2013-Fig8a.c",
    state=("x",),
    inputs=("x0",),
    build=_build,
)
