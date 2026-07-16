"""Masse-VMCAI2014-Fig1a — if/else both branches negating the second variable.

    int a, b;
    a = __VERIFIER_nondet_int();
    b = __VERIFIER_nondet_int();
    while (a >= 0) {
        a = a + b;
        if (b >= 0) { b = -b - 1; }
        else        { b = -b; }
    }

Note: `-b` is encoded as `-1 * b` (unary minus is not overloaded).
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        a0, b0 = extl
        return nxt(a0), nxt(b0)                  # a, b both nondet

    def update(self, ctrl):
        a, b = ctrl
        guard = a >= 0
        wa, wb = a, b
        wa = wa + wb                                    # a = a + b   (old b)
        wb = ite(wb >= 0, -1 * wb - 1, -1 * wb)         # if (b>=0) -b-1 else -b
        return ite(guard, wa, a), ite(guard, wb, b)


def _build():
    a, b = pair(), pair()
    a0, b0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(a, b), extl=(a0, b0))
    return prog, {"a": a, "b": b}, {"a0": a0, "b0": b0}


def _domain(s):
    return s["a"] >= 0


BENCH = Bench(
    name="Masse-VMCAI2014-Fig1a",
    source="Masse-VMCAI2014-Fig1a.c",
    state=("a", "b"),
    inputs=("a0", "b0"),
    build=_build,
    domain=_domain,
)
