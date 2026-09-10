#!/usr/bin/env python3
"""Numerically verify the three obligations for every nn2 case, by exhaustion.

`verith` proves; this script only says whether there is anything to prove.
A case that fails here is a broken test, not a tool limit.
"""
import math
from fractions import Fraction as F


def relu(v):
    return v if v >= 0 else (F(0) if isinstance(v, F) else 0)


def eval_net(layers, xs):
    """Numeric twin of cases.net / cases.net_py."""
    cur = list(xs)
    for i, (W, b) in enumerate(layers):
        out = [sum(F(w) * x for w, x in zip(row, cur)) + F(bias)
               for row, bias in zip(W, b)]
        cur = out if i == len(layers) - 1 else [relu(o) for o in out]
    return cur[0] if len(cur) == 1 else cur


def to_nat(v):
    """Lean `Int.toNat` (already an Int) — for Real, `to_int` floors first."""
    return max(0, v)


def floor_(v):
    return F(math.floor(v))


# ── module semantics ────────────────────────────────────────────────
MODULES = {
    "m_countdown": dict(
        init=(100,), dom=[range(-200, 301)],
        step=lambda s: ((100,) if s[0] == 0 else (s[0] - 1,)),
    ),
    "m_twovars": dict(
        init=(0, 10), dom=[range(-50, 51), range(-50, 51)],
        step=lambda s: ((s[0] + 1, s[1]) if s[0] < s[1] else (0, 10)),
    ),
    "m_relu_vec": dict(
        init=(3, 2, 1), dom=[range(-4, 9)] * 3,
        step=lambda s: tuple(relu(v - 1) for v in s),
    ),
    "m_relu_net": dict(
        init=(4, 0), dom=[range(-6, 13), range(-6, 13)],
        step=lambda s: (relu(s[0] - 1), relu(s[1]) + 1),
    ),
    "m_relu_net4": dict(
        init=(4, 0, 0, 0), dom=[range(-3, 8)] * 4,
        step=lambda s: (relu(s[0] - 1), relu(s[1]) + relu(s[0]),
                        relu(s[2]) + 1, relu(s[3])),
    ),
    "m_toward5": dict(
        init=(10,), dom=[range(-30, 41)],
        step=lambda s: ((s[0] + 1,) if s[0] < 5 else
                        ((s[0] - 1,) if s[0] > 5 else (5,))),
    ),
    "m_lra_lin": dict(
        init=(F(5),),
        dom=[[F(k, 2) for k in range(-20, 41)]],
        step=lambda s: ((s[0] - 1,) if s[0] > 0 else (F(5),)),
    ),
    "m_relu_lra": dict(
        init=(F(5),),
        dom=[[F(k, 2) for k in range(-20, 41)]],
        step=lambda s: (relu(s[0] - 1),),
    ),
}


def check(name, mod, inv, P, rank, verbose=True):
    """inv/P/rank are python callables on the state tuple. Returns list of bugs."""
    m = MODULES[mod]
    bad = []
    if not inv(m["init"]):
        bad.append(f"init_inv FALSE: inv{m['init']} does not hold")

    import itertools
    n_inv = 0
    for s in itertools.product(*m["dom"]):
        if not inv(s):
            continue
        n_inv += 1
        sp = m["step"](s)
        if not inv(sp):
            bad.append(f"step_inv FALSE: inv{s} but not inv{sp}")
            if len(bad) > 4:
                break
        if not P(s):
            r, rp = rank(s), rank(sp)
            if not (rp < r):
                bad.append(f"hrank FALSE: s={s} s'={sp} rank {r} -> {rp}")
                if len(bad) > 4:
                    break
    # sanity: the invariant must actually be bounded inside the domain
    edges = [tuple(d[0] if not isinstance(d, range) else d.start for d in m["dom"]),
             tuple(d[-1] if not isinstance(d, range) else d.stop - 1 for d in m["dom"])]
    for e in edges:
        if inv(e):
            bad.append(f"WARNING: inv holds at domain edge {e}; widen the domain")
    if verbose:
        tag = "OK  " if not bad else "BAD "
        print(f"{tag}{name:<18} ({n_inv} states satisfy inv)")
        for b in bad[:5]:
            print(f"      {b}")
    return bad
