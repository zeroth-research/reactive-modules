"""ChenFlurMukhopadhyay-SAS2012-Ex2.06 — per-iteration temp `oldx`.

    int x, y, oldx;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    while (4*x + y > 0) {
        oldx = x;
        x = -2*oldx + 4*y;
        y = 4*oldx;
    }

Notes:
  - `oldx` is a per-iteration temporary (assigned from x, used only that
    iteration) — a plain local, NOT ctrl.
  - NONTERMINATING for some inputs (e.g. x=1, y=1), so expect many
    inconclusive trials; decisive trials must agree.
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0, y0 = extl
        return nxt(x0), nxt(y0)                  # x, y both nondet

    def update(self, ctrl):
        x, y = ctrl
        guard = 4 * x + y > 0
        wx, wy = x, y
        oldx = wx              # oldx = x        (old x)
        wx = -2 * oldx + 4 * wy   # x = -2*oldx + 4*y
        wy = 4 * oldx          # y = 4*oldx
        return ite(guard, wx, x), ite(guard, wy, y)


def _build():
    x, y = pair(), pair()
    x0, y0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(x, y), extl=(x0, y0))
    return prog, {"x": x, "y": y}, {"x0": x0, "y0": y0}


def _domain(s):
    return 4 * s["x"] + s["y"] > 0


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex2.06",
    source="ChenFlurMukhopadhyay-SAS2012-Ex2.06.c",
    state=("x", "y"),
    inputs=("x0", "y0"),
    build=_build,
    domain=_domain,
)
