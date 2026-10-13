"""AliasDarteFeautrierGonnord-SAS2010-speedpldi2.

    int m, n, v1, v2;
    n = __VERIFIER_nondet_int();
    m = __VERIFIER_nondet_int();
    if (n >= 0 && m > 0) {
        v1 = n;
        v2 = 0;
        while (v1 > 0) {
            if (v2 < m) { v2 = v2 + 1; v1 = v1 - 1; }
            else        { v2 = 0; }
        }
    }

Precondition: n >= 0 && m > 0 (outer if also initialises v1, v2).
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, n0, m0):  # read order: n, m
        return X(m0), X(n0), X(n0), 0      # m, n, v1 = n, v2 = 0

    def update(self, m, n, v1, v2, _n0, _m0):
        guard = v1 > 0
        wv1 = ite(v2 < m, v1 - 1, v1)
        wv2 = ite(v2 < m, v2 + 1, 0)
        return m, n, ite(guard, wv1, v1), ite(guard, wv2, v2)


def _build():
    m, n, v1, v2 = var(), var(), var(), var()
    n0, m0 = var(), var()
    prog = Program(theory=LIA, ctrl=(m, n, v1, v2), extl=(n0, m0))
    return (
        prog,
        {"m": m, "n": n, "v1": v1, "v2": v2},
        {"n0": n0, "m0": m0},
    )


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-speedpldi2",
    source="AliasDarteFeautrierGonnord-SAS2010-speedpldi2.c",
    state=("m", "n", "v1", "v2"),
    inputs=("n0", "m0"),
    build=_build,
    precondition=lambda s: [s["n"] >= 0, s["m"] > 0],
)
