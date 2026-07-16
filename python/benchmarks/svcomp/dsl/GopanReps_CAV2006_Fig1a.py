"""GopanReps-CAV2006-Fig1a.

    int x, y;
    x = 0;
    y = 0;
    while (y >= 0) {
        if (x <= 50) { y = y + 1; }
        else         { y = y - 1; }
        x = x + 1;
    }

No nondet inputs: both variables start from constants.
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        return 0, 0                             # x = 0, y = 0

    def update(self, ctrl):
        x, y = ctrl
        guard = y >= 0
        wx, wy = x, y
        wy = ite(wx <= 50, wy + 1, wy - 1)      # if (x<=50) y+1 else y-1
        wx = wx + 1                             # x = x + 1
        return ite(guard, wx, x), ite(guard, wy, y)


def _build():
    x, y = pair(), pair()
    prog = Program(theory=LIA, ctrl=(x, y), extl=())
    return prog, {"x": x, "y": y}, {}


def _domain(s):
    return s["y"] >= 0


BENCH = Bench(
    name="GopanReps-CAV2006-Fig1a",
    source="GopanReps-CAV2006-Fig1a.c",
    state=("x", "y"),
    inputs=(),
    build=_build,
    domain=_domain,
)
