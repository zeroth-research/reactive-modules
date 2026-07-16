"""AliasDarteFeautrierGonnord-SAS2010-terminate.

    int i, j, k, ell;
    i = __VERIFIER_nondet_int();
    j = __VERIFIER_nondet_int();
    k = __VERIFIER_nondet_int();
    while (i <= 100 && j <= k) {
        ell = i;
        i = j;
        j = ell + 1;
        k = k - 1;
    }

`ell` is a per-iteration temporary (plain local, not a ctrl var).
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        i0, j0, k0 = extl
        return nxt(i0), nxt(j0), nxt(k0)

    def update(self, ctrl):
        i, j, k = ctrl
        guard = (i <= 100) & (j <= k)
        wi, wj, wk = i, j, k
        ell = wi          # ell = i    (old i)
        wi = wj           # i = j
        wj = ell + 1      # j = ell + 1
        wk = wk - 1       # k = k - 1
        return ite(guard, wi, i), ite(guard, wj, j), ite(guard, wk, k)


def _build():
    i, j, k = pair(), pair(), pair()
    i0, j0, k0 = pair(), pair(), pair()
    prog = Program(theory=LIA, ctrl=(i, j, k), extl=(i0, j0, k0))
    return prog, {"i": i, "j": j, "k": k}, {"i0": i0, "j0": j0, "k0": k0}


def _domain(s):
    import z3
    return z3.And(s["i"] <= 100, s["j"] <= s["k"])


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-terminate",
    source="AliasDarteFeautrierGonnord-SAS2010-terminate.c",
    state=("i", "j", "k"),
    inputs=("i0", "j0", "k0"),
    build=_build,
    domain=_domain,
)
