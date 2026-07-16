"""TEMPLATE — copy this file to encode one SV-COMP benchmark.

Steps:
  1. Copy to ``<Author><Venue><Year>_<Fig/Ex>.py``.
  2. Paste the C loop into the docstring for reference.
  3. Fill in ``state`` / ``inputs``, the ``init`` / ``update`` blocks, the
     ``_build`` wiring, and ``_domain``.

Key rules:
  - state var -> ctrl wire pair (C declaration order); nondet -> extl input
    (list ``inputs`` in the order the C reads nondet); constant init -> the
    literal; guard computed once, wrapped as ``ite(guard, new, old)`` per var.
  - `==`/`!=` are not overloaded: use ``eq(a, b)`` / ``ne(a, b)``. Boolean
    ``& | ~``; `*` only with a scalar constant.
  - WORKING-COPY body rule (below): unpack ctrl to originals, copy to working
    vars, transcribe the C body line-for-line reassigning the copies (each RHS
    reads the current copies == C sequential semantics), then return
    ``ite(guard, <copy>, <original>)`` per state var.

The worked example below encodes:

    // ColonSipma-TACAS2001-Fig1.c  (not the actual file — just an illustration)
    int k, i, j, tmp;
    k = nondet(); i = nondet(); j = nondet();
    while (i <= 100 && j <= k) {
        tmp = i;
        i   = j;
        j   = tmp + 1;
        k   = k - 1;
    }

Delete this module-level explanation and keep only your program's C in the
real encoding.
"""

from __future__ import annotations

from zrth import LIA, Wire
from zrth.dsl import dslModule, nxt, ite, eq, ne  # noqa: F401  (eq/ne used by many encodings)

from .._bench import Bench, INT, pair


class Program(dslModule):
    # while (i <= 100 && j <= k) { tmp = i; i = j; j = tmp + 1; k = k - 1; }
    def init(self, extl):
        # `extl` is the tuple of nondet inputs, in the order passed to `extl=`.
        # A single input is unwrapped (so `x0 = extl`); here we have three.
        k0, i0, j0 = extl
        # each ctrl var's initial value, aligned to `ctrl=(k, i, j)`:
        return nxt(k0), nxt(i0), nxt(j0)      # all nondet-initialised

    def update(self, ctrl):
        # `ctrl` gives the LATCHED (pre-state) values, in declaration order.
        k, i, j = ctrl

        # loop condition — computed once; also mirrored in `_domain` below.
        guard = (i <= 100) & (j <= k)

        # --- body, line-for-line, on WORKING COPIES (see "Key rules" above) ---
        wk, wi, wj = k, i, j
        tmp = wi          # tmp = i        (old i)
        wi  = wj          # i   = j
        wj  = tmp + 1     # j   = tmp + 1  (old i + 1)
        wk  = wk - 1      # k   = k - 1

        # step the body while in-loop, otherwise hold (self-loop at exit):
        return ite(guard, wk, k), ite(guard, wi, i), ite(guard, wj, j)


def _build():
    k, i, j = pair(), pair(), pair()          # ctrl (state)
    k0, i0, j0 = pair(), pair(), pair()        # extl (nondet inputs)
    prog = Program(theory=LIA, ctrl=(k, i, j), extl=(k0, i0, j0))
    return prog, {"k": k, "i": i, "j": j}, {"k0": k0, "i0": i0, "j0": j0}


def _domain(s):
    # Loop condition as a Z3 predicate over the LATCHED symbols (mirrors `guard`).
    import z3
    return z3.And(s["i"] <= 100, s["j"] <= s["k"])


BENCH = Bench(
    name="TEMPLATE-do-not-run",
    source="<file>.c",   # filename in this package's c/ directory
    state=("k", "i", "j"),
    inputs=("k0", "i0", "j0"),
    build=_build,
    domain=_domain,
)
