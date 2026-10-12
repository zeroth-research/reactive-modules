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
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, j0, N0):  # read order: j, N
        return X(N0), X(j0), X(N0)         # i = N, j = nondet, N = nondet

    def update(self, i, j, N, _j0, _N0):
        guard = i > 0
        wi, wj, wN = i, j, N
        wj = ite(j > 0, j - 1, N)                # if j>0: j-1 else j=N
        wi = ite(j > 0, i, i - 1)                # else branch also does i=i-1
        return ite(guard, wi, i), ite(guard, wj, j), N


def _build():
    i, j, N = var(), var(), var()
    j0, N0 = var(), var()
    prog = Program(theory=LIA, ctrl=(i, j, N), extl=(j0, N0))
    return prog, {"i": i, "j": j, "N": N}, {"j0": j0, "N0": N0}


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-cousot9",
    source="AliasDarteFeautrierGonnord-SAS2010-cousot9.c",
    state=("i", "j", "N"),
    inputs=("j0", "N0"),
    build=_build,
)
