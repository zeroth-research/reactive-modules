"""NoriSharma-FSE2013-Fig8 — if/else with sequential update inside a branch.

    int c, u, v, w, x, y, z;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    z = __VERIFIER_nondet_int();
    u = x;
    v = y;
    w = z;
    c = 0;
    while (x >= y) {
        c = c + 1;
        if (z > 1) {
            z = z - 1;
            x = x + z;      // uses the just-decremented z
        } else {
            y = y + 1;
        }
    }

Notes:
  - Declaration order `c, u, v, w, x, y, z` (ctrl order); C reads nondet as
    `x, y, z` (inputs order).
  - `u, v, w` are entry copies of `x, y, z`, never written; held unchanged.
  - In the then-branch `x = x + z` reads the decremented `z`, so the new `x`
    is `x + (z - 1)`.
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        x0, y0, z0 = extl
        # ctrl order: c, u, v, w, x, y, z
        return 0, nxt(x0), nxt(y0), nxt(z0), nxt(x0), nxt(y0), nxt(z0)

    def update(self, ctrl):
        c, u, v, w, x, y, z = ctrl
        guard = x >= y
        wc, wx, wy, wz = c, x, y, z
        wc = wc + 1                          # c = c + 1
        zcond = wz > 1
        new_x = ite(zcond, wx + (wz - 1), wx)   # then: x = x + (z-1)
        new_y = ite(zcond, wy, wy + 1)          # else: y = y + 1
        new_z = ite(zcond, wz - 1, wz)          # then: z = z - 1
        wx, wy, wz = new_x, new_y, new_z
        return (ite(guard, wc, c), u, v, w,
                ite(guard, wx, x), ite(guard, wy, y), ite(guard, wz, z))


def _build():
    c, u, v, w, x, y, z = pair(), pair(), pair(), pair(), pair(), pair(), pair()
    x0, y0, z0 = pair(), pair(), pair()
    prog = Program(theory=LIA, ctrl=(c, u, v, w, x, y, z),
                   extl=(x0, y0, z0))
    return (prog,
            {"c": c, "u": u, "v": v, "w": w, "x": x, "y": y, "z": z},
            {"x0": x0, "y0": y0, "z0": z0})


BENCH = Bench(
    name="NoriSharma-FSE2013-Fig8",
    source="NoriSharma-FSE2013-Fig8.c",
    state=("c", "u", "v", "w", "x", "y", "z"),
    inputs=("x0", "y0", "z0"),
    build=_build,
)
