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
  - `p != q` uses `ne()`.
"""

from __future__ import annotations

from zrth import LIA
from zrth.dsl import dslModule, nxt, ite, ne

from .._bench import Bench, pair


class Program(dslModule):
    def init(self, extl):
        q0, p0 = extl                            # C reads q then p
        return nxt(p0), nxt(q0)                  # ctrl order: p, q

    def update(self, ctrl):
        p, q = ctrl
        guard = (q > 0) & (p > 0) & ne(p, q)
        wp, wq = p, q
        # if (q<p) q=q-1; else { if (p<q) p=p-1; }
        wq = ite(q < p, q - 1, q)
        wp = ite(q < p, p, ite(p < q, p - 1, p))
        return ite(guard, wp, p), ite(guard, wq, q)


def _build():
    p, q = pair(), pair()
    q0, p0 = pair(), pair()
    prog = Program(theory=LIA, ctrl=(p, q), extl=(q0, p0))
    return prog, {"p": p, "q": q}, {"q0": q0, "p0": p0}


def _domain(s):
    import z3
    return z3.And(s["q"] > 0, s["p"] > 0, s["p"] != s["q"])


BENCH = Bench(
    name="LeikeHeizmann-TACAS2014-Ex9",
    source="LeikeHeizmann-TACAS2014-Ex9.c",
    state=("p", "q"),
    inputs=("q0", "p0"),
    build=_build,
    domain=_domain,
)
