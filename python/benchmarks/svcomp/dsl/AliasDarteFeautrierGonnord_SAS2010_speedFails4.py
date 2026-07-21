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
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        i0, x0, n0, b0 = extl
        # t = 1 if b>=1 else -1  (b set once before the loop)
        return nxt(i0), nxt(x0), nxt(n0), nxt(b0), ite(nxt(b0) >= 1, 1, -1)

    def update(self, ctrl):
        i, x, n, b, t = ctrl
        guard = x <= n
        wx = ite(b >= 1, x + t, x - t)
        return i, ite(guard, wx, x), n, b, t


def _build():
    i, x, n, b, t = pair(), pair(), pair(), pair(), pair()
    i0, x0, n0, b0 = pair(), pair(), pair(), pair()
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
