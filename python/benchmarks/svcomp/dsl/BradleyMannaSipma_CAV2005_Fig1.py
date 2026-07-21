"""BradleyMannaSipma-CAV2005-Fig1.

    int y1, y2;
    y1 = __VERIFIER_nondet_int();
    y2 = __VERIFIER_nondet_int();
    if (y1 > 0 && y2 > 0) {
        while (y1 != y2) {
            if (y1 > y2) { y1 = y1 - y2; }
            else         { y2 = y2 - y1; }
        }
    }

Precondition: y1 > 0 && y2 > 0.
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite, ne

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        y1_0, y2_0 = extl
        return nxt(y1_0), nxt(y2_0)

    def update(self, ctrl):
        y1, y2 = ctrl
        guard = ne(y1, y2)
        wy1 = ite(y1 > y2, y1 - y2, y1)
        wy2 = ite(y1 > y2, y2, y2 - y1)
        return ite(guard, wy1, y1), ite(guard, wy2, y2)


def _build():
    y1, y2 = pair(), pair()
    y1_0, y2_0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(y1, y2), extl=(y1_0, y2_0))
    return prog, {"y1": y1, "y2": y2}, {"y1_0": y1_0, "y2_0": y2_0}


BENCH = Bench(
    name="BradleyMannaSipma-CAV2005-Fig1",
    source="BradleyMannaSipma-CAV2005-Fig1.c",
    state=("y1", "y2"),
    inputs=("y1_0", "y2_0"),
    build=_build,
    precondition=lambda s: [s["y1"] > 0, s["y2"] > 0],
)
