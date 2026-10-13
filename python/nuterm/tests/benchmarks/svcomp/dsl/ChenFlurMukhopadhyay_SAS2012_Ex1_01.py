"""ChenFlurMukhopadhyay-SAS2012-Ex1.01: single variable, scalar-mul update.

    int x;
    x = __VERIFIER_nondet_int();
    while (x > 0) {
        x = -2*x + 10;
    }

Original header comment:

    Date: 2014-06-08
    Author: leike@informatik.uni-freiburg.de


    This is Example 1.1 from the test suit used in

    Termination Proofs for Linear Simple Loops.
    Hong Yi Chen, Shaked Flur, and Supratik Mukhopadhyay.
    SAS 2012.

    The test suite is available at the following URL.
    https://tigerbytes2.lsu.edu/users/hchen11/lsl/LSL_benchmark.txt

    Comment: terminating, non-linear
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, x0):
        return X(x0),                          # x nondet

    def update(self, x, _x0):
        guard = x > 0
        wx = -2 * x + 10   # x = -2*x + 10
        return ite(guard, wx, x),


def _build():
    x = var()
    x0 = var()
    prog = Program(theory=LIA, ctrl=(x,), extl=(x0,))
    return prog, {"x": x}, {"x0": x0}


BENCH = Bench(
    name="ChenFlurMukhopadhyay-SAS2012-Ex1.01",
    source="ChenFlurMukhopadhyay-SAS2012-Ex1.01.c",
    state=("x",),
    inputs=("x0",),
    build=_build,
)
