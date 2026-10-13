"""AliasDarteFeautrierGonnord-SAS2010-ndecr.

    int i, n;
    n = __VERIFIER_nondet_int();
    i = n - 1;
    while (i > 1) {
        i = i - 1;
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, n0):
        return X(n0) - 1, X(n0)              # i = n - 1, n = nondet

    def update(self, i, n, _n0):
        guard = i > 1
        wi = i - 1
        return ite(guard, wi, i), n


def _build():
    i, n = var(), var()
    n0 = var()
    prog = Program(theory=LIA, ctrl=(i, n), extl=(n0,))
    return prog, {"i": i, "n": n}, {"n0": n0}


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-ndecr",
    source="AliasDarteFeautrierGonnord-SAS2010-ndecr.c",
    state=("i", "n"),
    inputs=("n0",),
    build=_build,
)
