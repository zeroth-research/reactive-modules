"""ColonSipma-TACAS2001-Fig1.

    int k, i, j, tmp;
    k = __VERIFIER_nondet_int();
    i = __VERIFIER_nondet_int();
    j = __VERIFIER_nondet_int();
    while (i <= 100 && j <= k) {
        tmp = i;
        i = j;
        j = tmp + 1;
        k = k - 1;
    }

`tmp` is a pure per-iteration temporary (a plain local, not ctrl).
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        k0, i0, j0 = extl
        return nxt(k0), nxt(i0), nxt(j0)        # k, i, j all nondet

    def update(self, ctrl):
        k, i, j = ctrl
        guard = (i <= 100) & (j <= k)
        wk, wi, wj = k, i, j
        tmp = wi          # tmp = i        (old i)
        wi = wj           # i   = j
        wj = tmp + 1      # j   = tmp + 1  (old i + 1)
        wk = wk - 1       # k   = k - 1
        return ite(guard, wk, k), ite(guard, wi, i), ite(guard, wj, j)


def _build():
    k, i, j = pair(), pair(), pair()
    k0, i0, j0 = pair(), pair(), pair()
    prog = Program(theory=LIA, ctrl=(k, i, j), extl=(k0, i0, j0))
    return prog, {"k": k, "i": i, "j": j}, {"k0": k0, "i0": i0, "j0": j0}


def _domain(s):
    import z3
    return z3.And(s["i"] <= 100, s["j"] <= s["k"])


BENCH = Bench(
    name="ColonSipma-TACAS2001-Fig1",
    source="ColonSipma-TACAS2001-Fig1.c",
    state=("k", "i", "j"),
    inputs=("k0", "i0", "j0"),
    build=_build,
    domain=_domain,
)
