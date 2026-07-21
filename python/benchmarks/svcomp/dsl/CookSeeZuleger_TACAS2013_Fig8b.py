"""CookSeeZuleger-TACAS2013-Fig8b.

    int x, M;
    x = __VERIFIER_nondet_int();
    M = __VERIFIER_nondet_int();
    if (M > 0) {
        while (x != M) {
            if (x > M) { x = 0; }
            else       { x = x + 1; }
        }
    }

The loop only runs under the outer `if (M > 0)`; the precondition restricts
the comparison to that domain.
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite, ne

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0, M0 = extl
        return nxt(x0), nxt(M0)                 # x, M both nondet

    def update(self, ctrl):
        x, M = ctrl
        guard = ne(x, M)
        wx = ite(x > M, 0, x + 1)               # if (x>M) 0 else x+1
        return ite(guard, wx, x), M             # M unchanged


def _build():
    x, M = pair(), pair()
    x0, M0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(x, M), extl=(x0, M0))
    return prog, {"x": x, "M": M}, {"x0": x0, "M0": M0}


BENCH = Bench(
    name="CookSeeZuleger-TACAS2013-Fig8b",
    source="CookSeeZuleger-TACAS2013-Fig8b.c",
    state=("x", "M"),
    inputs=("x0", "M0"),
    build=_build,
    precondition=lambda s: [s["M"] > 0],
)
