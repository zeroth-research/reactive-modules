"""NoriSharma-FSE2013-Fig7 — two-disjunct guard, ghost/parameter variables.

    int a, b, c, i, j, M, N;
    i = __VERIFIER_nondet_int();
    j = __VERIFIER_nondet_int();
    M = __VERIFIER_nondet_int();
    N = __VERIFIER_nondet_int();
    a = i;
    b = j;
    c = 0;
    while (i < M || j < N) {
        i = i + 1;
        j = j + 1;
        c = c + 1;
    }

Notes:
  - Declaration order `a, b, c, i, j, M, N` (ctrl order); C reads nondet as
    `i, j, M, N` (inputs order).
  - `a`, `b` are copies of `i`, `j` at entry and are never written; `M`, `N`
    are nondet parameters, never written. All are held unchanged.
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        i0, j0, M0, N0 = extl
        # ctrl order: a, b, c, i, j, M, N
        return nxt(i0), nxt(j0), 0, nxt(i0), nxt(j0), nxt(M0), nxt(N0)

    def update(self, ctrl):
        a, b, c, i, j, M, N = ctrl
        guard = (i < M) | (j < N)
        wc, wi, wj = c, i, j
        wi = wi + 1        # i = i + 1
        wj = wj + 1        # j = j + 1
        wc = wc + 1        # c = c + 1
        return (a, b, ite(guard, wc, c),
                ite(guard, wi, i), ite(guard, wj, j), M, N)


def _build():
    a, b, c, i, j, M, N = pair(), pair(), pair(), pair(), pair(), pair(), pair()
    i0, j0, M0, N0 = pair(), pair(), pair(), pair()
    prog = Program(theory=LIA, ctrl=(a, b, c, i, j, M, N),
                   extl=(i0, j0, M0, N0))
    return (prog,
            {"a": a, "b": b, "c": c, "i": i, "j": j, "M": M, "N": N},
            {"i0": i0, "j0": j0, "M0": M0, "N0": N0})


BENCH = Bench(
    name="NoriSharma-FSE2013-Fig7",
    source="NoriSharma-FSE2013-Fig7.c",
    state=("a", "b", "c", "i", "j", "M", "N"),
    inputs=("i0", "j0", "M0", "N0"),
    build=_build,
)
