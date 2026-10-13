"""AliasDarteFeautrierGonnord-SAS2010-easy2-2.

    int x = 0, y = 0, z = __VERIFIER_nondet_int();
    while (z > 0) {
        x = x + 1;
        y = y - 1;
        z = z - 1;
    }

Original header comment:

    Program used in the experimental evaluation of the following paper.
    2010SAS - Alias,Darte,Feautrier,Gonnord, Multi-dimensional Rankings, Program Termination, and Complexity Bounds of Flowchart Programs

    Date: 2014
    Author: Caterina Urban
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, z0):
        return 0, 0, X(z0)                     # x = 0, y = 0, z = nondet

    def update(self, x, y, z, _z0):
        guard = z > 0
        wx, wy, wz = x, y, z
        wx = wx + 1
        wy = wy - 1
        wz = wz - 1
        return ite(guard, wx, x), ite(guard, wy, y), ite(guard, wz, z)


def _build():
    x, y, z = var(), var(), var()
    z0 = var()
    prog = Program(theory=LIA, ctrl=(x, y, z), extl=(z0,))
    return prog, {"x": x, "y": y, "z": z}, {"z0": z0}


BENCH = Bench(
    name="AliasDarteFeautrierGonnord-SAS2010-easy2-2",
    source="AliasDarteFeautrierGonnord-SAS2010-easy2-2.c",
    state=("x", "y", "z"),
    inputs=("z0",),
    build=_build,
)
