"""AliasDarteFeautrierGonnord-SAS2010-cousot9.

    int i, j, N;
    j = __VERIFIER_nondet_int();
    N = __VERIFIER_nondet_int();
    i = N;
    while (i > 0) {
        if (j > 0) {
            j = j - 1;
        } else {
            j = N;
            i = i - 1;
        }
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        j0, N0 = extl                            # read order: j, N
        return nxt(N0), nxt(j0), nxt(N0)         # i = N, j = nondet, N = nondet

    def update(self, ctrl):
        i, j, N = ctrl
        guard = i > 0
        wi, wj, wN = i, j, N
        wj = ite(j > 0, j - 1, N)                # if j>0: j-1 else j=N
        wi = ite(j > 0, i, i - 1)                # else branch also does i=i-1
        return ite(guard, wi, i), ite(guard, wj, j), N


def _build():
    i, j, N = pair(), pair(), pair()
    j0, N0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(i, j, N), extl=(j0, N0))
    return prog, {"i": i, "j": j, "N": N}, {"j0": j0, "N0": N0}


def _domain(s):
    return s["i"] > 0


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-cousot9",
    source="AliasDarteFeautrierGonnord-SAS2010-cousot9.c",
    state=("i", "j", "N"),
    inputs=("j0", "N0"),
    build=_build,
    domain=_domain,
)
