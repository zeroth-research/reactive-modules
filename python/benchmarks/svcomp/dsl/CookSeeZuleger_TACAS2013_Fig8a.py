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
from zrth.dsl import dslModule, nxt, ite, ne

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0 = extl                               # single input, unwrapped
        return nxt(x0)                          # x = nondet

    def update(self, ctrl):
        x = ctrl
        guard = ne(x, 0)
        wx = ite(x > 0, x - 1, x + 1)           # if (x>0) x-1 else x+1
        return ite(guard, wx, x)


def _build():
    x = pair()
    x0 = pair()
    prog = Program(theory=LIA, ctrl=(x,), extl=(x0,))
    return prog, {"x": x}, {"x0": x0}


BENCH = Bench(
    name="CookSeeZuleger-TACAS2013-Fig8a",
    source="CookSeeZuleger-TACAS2013-Fig8a.c",
    state=("x",),
    inputs=("x0",),
    build=_build,
)
