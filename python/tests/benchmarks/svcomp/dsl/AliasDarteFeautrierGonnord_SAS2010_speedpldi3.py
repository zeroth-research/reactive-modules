"""AliasDarteFeautrierGonnord-SAS2010-speedpldi3.

    int i, j, m, n;
    n = __VERIFIER_nondet_int();
    m = __VERIFIER_nondet_int();
    if (m > 0 && n > m) {
        i = 0;
        j = 0;
        while (i < n) {
            if (j < m) { j = j + 1; }
            else       { j = 0; i = i + 1; }
        }
    }

Precondition: m > 0 && n > m (outer if also initialises i, j).
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, n0, m0):  # read order: n, m
        return 0, 0, X(m0), X(n0)            # i = 0, j = 0, m, n

    def update(self, i, j, m, n, _n0, _m0):
        guard = i < n
        wi = ite(j < m, i, i + 1)
        wj = ite(j < m, j + 1, 0)
        return ite(guard, wi, i), ite(guard, wj, j), m, n


def _build():
    i, j, m, n = var(), var(), var(), var()
    n0, m0 = var(), var()
    prog = Program(theory=LIA, ctrl=(i, j, m, n), extl=(n0, m0))
    return (
        prog,
        {"i": i, "j": j, "m": m, "n": n},
        {"n0": n0, "m0": m0},
    )


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-speedpldi3",
    source="AliasDarteFeautrierGonnord-SAS2010-speedpldi3.c",
    state=("i", "j", "m", "n"),
    inputs=("n0", "m0"),
    build=_build,
    precondition=lambda s: [s["m"] > 0, s["n"] > s["m"]],
)
