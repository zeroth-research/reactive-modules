"""AliasDarteFeautrierGonnord-SAS2010-speedFails4.

    int i, x, n, b, t;
    i = __VERIFIER_nondet_int();
    x = __VERIFIER_nondet_int();
    n = __VERIFIER_nondet_int();
    b = __VERIFIER_nondet_int();
    if (b >= 1) { t = 1; } else { t = -1; }
    while (x <= n) {
        if (b >= 1) { x = x + t; }
        else        { x = x - t; }
    }
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, expr, ite

from .._bench import Bench, INT, var


class Program(Module):
    def init(self, i0, x0, n0, b0):
        # t = 1 if b>=1 else -1  (b set once before the loop); `ite` needs one
        # branch as an Expr to pin the theory and sort
        one = expr(1, theory=LIA, sort=INT)
        return X(i0), X(x0), X(n0), X(b0), ite(X(b0) >= 1, one, -1)

    def update(self, i, x, n, b, t, _i0, _x0, _n0, _b0):
        guard = x <= n
        wx = ite(b >= 1, x + t, x - t)
        return i, ite(guard, wx, x), n, b, t


def _build():
    i, x, n, b, t = var(), var(), var(), var(), var()
    i0, x0, n0, b0 = var(), var(), var(), var()
    prog = Program(theory=LIA, ctrl=(i, x, n, b, t), extl=(i0, x0, n0, b0))
    return (
        prog,
        {"i": i, "x": x, "n": n, "b": b, "t": t},
        {"i0": i0, "x0": x0, "n0": n0, "b0": b0},
    )


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-speedFails4",
    source="AliasDarteFeautrierGonnord-SAS2010-speedFails4.c",
    state=("i", "x", "n", "b", "t"),
    inputs=("i0", "x0", "n0", "b0"),
    build=_build,
)
