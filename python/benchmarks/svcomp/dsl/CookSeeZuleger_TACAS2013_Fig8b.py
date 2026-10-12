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
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0, M0):
        return X(x0), X(M0)                 # x, M both nondet

    def update(self, x, M, _x0, _M0):
        guard = (x != M)
        wx = ite(x > M, 0, x + 1)               # if (x>M) 0 else x+1
        return ite(guard, wx, x), M             # M unchanged


def _build():
    x, M = var(), var()
    x0, M0 = var(), var()
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
