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
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        n0 = extl
        return nxt(n0) - 1, nxt(n0)              # i = n - 1, n = nondet

    def update(self, ctrl):
        i, n = ctrl
        guard = i > 1
        wi = i - 1
        return ite(guard, wi, i), n


def _build():
    i, n = pair(), pair()
    n0 = pair()
    prog = Program(theory=LIA, ctrl=(i, n), extl=(n0,))
    return prog, {"i": i, "n": n}, {"n0": n0}


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-ndecr",
    source="AliasDarteFeautrierGonnord-SAS2010-ndecr.c",
    state=("i", "n"),
    inputs=("n0",),
    build=_build,
)
