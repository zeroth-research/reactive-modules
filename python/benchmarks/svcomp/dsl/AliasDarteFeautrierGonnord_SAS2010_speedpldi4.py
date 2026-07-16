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
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        n0, m0 = extl                            # read order: n, m
        return nxt(n0), nxt(m0), nxt(n0)         # i = n, m, n

    def update(self, ctrl):
        i, m, n = ctrl
        guard = i > 0
        wi = ite(i < m, i - 1, i - m)
        return ite(guard, wi, i), m, n


def _build():
    i, m, n = pair(), pair(), pair()
    n0, m0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(i, m, n), extl=(n0, m0))
    return prog, {"i": i, "m": m, "n": n}, {"n0": n0, "m0": m0}


def _domain(s):
    import z3
    return z3.And(s["i"] > 0, s["m"] > 0, s["n"] > s["m"])


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-speedpldi4",
    source="AliasDarteFeautrierGonnord-SAS2010-speedpldi4.c",
    state=("i", "m", "n"),
    inputs=("n0", "m0"),
    build=_build,
    domain=_domain,
    precondition=lambda i: i["m0"] > 0 and i["n0"] > i["m0"],
)
