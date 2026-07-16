"""PodelskiRybalchenko-VMCAI2004-Ex2 — single variable, affine (negative) update.

    int x;
    x = __VERIFIER_nondet_int();
    while (x >= 0) {
        x = -2*x + 10;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0 = extl                               # single input, unwrapped
        return nxt(x0)                          # x nondet

    def update(self, ctrl):
        x = ctrl                                # single ctrl, unwrapped
        guard = x >= 0
        wx = x
        wx = -2 * wx + 10      # x = -2*x + 10
        return ite(guard, wx, x)


def _build():
    x = pair()
    x0 = pair()
    prog = Program(theory=LIA, ctrl=(x,), extl=(x0,))
    return prog, {"x": x}, {"x0": x0}


def _domain(s):
    return s["x"] >= 0


BENCH = Bench(
    name="PodelskiRybalchenko-VMCAI2004-Ex2",
    source="PodelskiRybalchenko-VMCAI2004-Ex2.c",
    state=("x",),
    inputs=("x0",),
    build=_build,
    domain=_domain,
)
