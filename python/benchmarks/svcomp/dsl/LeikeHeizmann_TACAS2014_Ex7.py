"""LeikeHeizmann-TACAS2014-Ex7 — negation of a variable each iteration.

    int q, z;
    q = __VERIFIER_nondet_int();
    z = __VERIFIER_nondet_int();
    while (q > 0) {
        q = q + z - 1;
        z = -z;
    }

Note: `-z` is encoded as `-1 * z` (unary minus is not overloaded; `*` with a
scalar constant is allowed).
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, q0, z0):
        return X(q0), X(z0)                  # q, z both nondet

    def update(self, q, z, _q0, _z0):
        guard = q > 0
        wq, wz = q, z
        wq = wq + wz - 1       # q = q + z   (old z)
        wz = -1 * wz           # z = -z
        return ite(guard, wq, q), ite(guard, wz, z)


def _build():
    q, z = var(), var()
    q0, z0 = var(), var()
    prog = Program(theory=LIA, ctrl=(q, z), extl=(q0, z0))
    return prog, {"q": q, "z": z}, {"q0": q0, "z0": z0}


BENCH = Bench(
    name="LeikeHeizmann-TACAS2014-Ex7",
    source="LeikeHeizmann-TACAS2014-Ex7.c",
    state=("q", "z"),
    inputs=("q0", "z0"),
    build=_build,
)
