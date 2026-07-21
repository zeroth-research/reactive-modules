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
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        n0, m0 = extl                            # read order: n, m
        return nxt(m0), nxt(n0), nxt(n0), 0      # m, n, v1 = n, v2 = 0

    def update(self, ctrl):
        m, n, v1, v2 = ctrl
        guard = v1 > 0
        wv1 = ite(v2 < m, v1 - 1, v1)
        wv2 = ite(v2 < m, v2 + 1, 0)
        return m, n, ite(guard, wv1, v1), ite(guard, wv2, v2)


def _build():
    m, n, v1, v2 = pair(), pair(), pair(), pair()
    n0, m0 = pair(), pair()
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
