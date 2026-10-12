"""AliasDarteFeautrierGonnord-SAS2010-speedpldi4.

    int i, m, n;
    n = __VERIFIER_nondet_int();
    m = __VERIFIER_nondet_int();
    if (m > 0 && n > m) {
        i = n;
        while (i > 0) {
            if (i < m) { i = i - 1; }
            else       { i = i - m; }
        }
    }

Precondition: m > 0 && n > m (outer if also initialises i).
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, n0, m0):  # read order: n, m
        return X(n0), X(m0), X(n0)         # i = n, m, n

    def update(self, i, m, n, _n0, _m0):
        guard = i > 0
        wi = ite(i < m, i - 1, i - m)
        return ite(guard, wi, i), m, n


def _build():
    i, m, n = var(), var(), var()
    n0, m0 = var(), var()
    prog = Program(theory=LIA, ctrl=(i, m, n), extl=(n0, m0))
    return prog, {"i": i, "m": m, "n": n}, {"n0": n0, "m0": m0}


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-speedpldi4",
    source="AliasDarteFeautrierGonnord-SAS2010-speedpldi4.c",
    state=("i", "m", "n"),
    inputs=("n0", "m0"),
    build=_build,
    precondition=lambda s: [s["m"] > 0, s["n"] > s["m"]],
)
