"""genady — constant-initialised, no nondet inputs (closed program).

    int i, j;
    j = 1;
    i = 10000;
    while (i - j >= 1) {
        j = j + 1;
        i = i - 1;
    }

Notes:
  - Declaration order `i, j` (ctrl order); the C initialises `j` then `i`.
  - No `__VERIFIER_nondet_int()` calls, so there are no `extl` inputs.
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, ite

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        return 10000, 1                          # ctrl order i, j: i=10000, j=1

    def update(self, ctrl):
        i, j = ctrl
        guard = (i - j) >= 1
        wi, wj = i, j
        wj = wj + 1        # j = j + 1
        wi = wi - 1        # i = i - 1
        return ite(guard, wi, i), ite(guard, wj, j)


def _build():
    i, j = pair(), pair()
    prog = Program(theory=LIA, ctrl=(i, j), extl=())
    return prog, {"i": i, "j": j}, {}


def _domain(s):
    return (s["i"] - s["j"]) >= 1


BENCH = Bench(
    name="genady",
    source="genady.c",
    state=("i", "j"),
    inputs=(),
    build=_build,
    domain=_domain,
)
