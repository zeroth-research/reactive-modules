"""LeikeHeizmann-TACAS2014-Ex9 — nested if/else, three-conjunct guard.

    int p, q;
    q = __VERIFIER_nondet_int();
    p = __VERIFIER_nondet_int();
    while (q > 0 && p > 0 && p != q) {
        if (q < p) { q = q - 1; }
        else { if (p < q) { p = p - 1; } }
    }

Notes:
  - Declaration order is `p, q` (ctrl order), but the C reads nondet as
    `q` then `p` (inputs order `q0, p0`).
"""

from __future__ import annotations

from zrth import LIA
from zrth.sugar import Module, X, ite

from .._bench import Bench, var


class Program(Module):
    def init(self, q0, p0):  # C reads q then p
        return X(p0), X(q0)                  # ctrl order: p, q

    def update(self, p, q, _q0, _p0):
        guard = (q > 0) & (p > 0) & (p != q)
        wp, wq = p, q
        # if (q<p) q=q-1; else { if (p<q) p=p-1; }
        wq = ite(q < p, q - 1, q)
        wp = ite(q < p, p, ite(p < q, p - 1, p))
        return ite(guard, wp, p), ite(guard, wq, q)


def _build():
    p, q = var(), var()
    q0, p0 = var(), var()
    prog = Program(theory=LIA, ctrl=(p, q), extl=(q0, p0))
    return prog, {"p": p, "q": q}, {"q0": q0, "p0": p0}


BENCH = Bench(
    name="LeikeHeizmann-TACAS2014-Ex9",
    source="LeikeHeizmann-TACAS2014-Ex9.c",
    state=("p", "q"),
    inputs=("q0", "p0"),
    build=_build,
)
