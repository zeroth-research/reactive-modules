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
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, a0, b0):
        return X(a0), X(b0)                  # a, b both nondet

    def update(self, a, b, _a0, _b0):
        guard = a >= 0
        wa, wb = a, b
        wa = wa + wb                                    # a = a + b   (old b)
        wb = ite(wb >= 0, -1 * wb - 1, -1 * wb)         # if (b>=0) -b-1 else -b
        return ite(guard, wa, a), ite(guard, wb, b)


def _build():
    a, b = var(), var()
    a0, b0 = var(), var()
    prog = Program(theory=LIA, ctrl=(a, b), extl=(a0, b0))
    return prog, {"a": a, "b": b}, {"a0": a0, "b0": b0}


BENCH = Bench(
    name="Masse-VMCAI2014-Fig1a",
    source="Masse-VMCAI2014-Fig1a.c",
    state=("a", "b"),
    inputs=("a0", "b0"),
    build=_build,
)
