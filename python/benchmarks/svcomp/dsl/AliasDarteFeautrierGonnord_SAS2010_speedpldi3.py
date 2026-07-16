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
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        n0, m0 = extl                            # read order: n, m
        return 0, 0, nxt(m0), nxt(n0)            # i = 0, j = 0, m, n

    def update(self, ctrl):
        i, j, m, n = ctrl
        guard = i < n
        wi = ite(j < m, i, i + 1)
        wj = ite(j < m, j + 1, 0)
        return ite(guard, wi, i), ite(guard, wj, j), m, n


def _build():
    i, j, m, n = pair(), pair(), pair(), pair()
    n0, m0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(i, j, m, n), extl=(n0, m0))
    return (
        prog,
        {"i": i, "j": j, "m": m, "n": n},
        {"n0": n0, "m0": m0},
    )


def _domain(s):
    import z3
    return z3.And(s["i"] < s["n"], s["m"] > 0, s["n"] > s["m"])


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-speedpldi3",
    source="AliasDarteFeautrierGonnord-SAS2010-speedpldi3.c",
    state=("i", "j", "m", "n"),
    inputs=("n0", "m0"),
    build=_build,
    domain=_domain,
    precondition=lambda i: i["m0"] > 0 and i["n0"] > i["m0"],
)
