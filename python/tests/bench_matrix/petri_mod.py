"""One Petri net, as a module `verith` can load.

`petri/nets.py` holds every net in the `petri` suite as a marking and a flow
relation; this file turns any one of them into a reactive module. Which one is
named by `$PETRI_NET`, so a single checked-in adapter serves the whole suite
and the command that measures a case is still a command that can be pasted:

    PETRI_NET=mutex uv run verith tests/bench_matrix/petri_mod.py \
        --safety '(<= (+ s2 s5) 1)' --pre '(and (>= e0 0) (<= e0 5))' ...

The module is built with `zrth.expr`'s `collecting()` rather than the
`sugar.Module` DSL, because the DSL binds variables to *named* parameters of
`update` and a net's arity is not known until the table is read.

## The step

One tick is one attempted firing.

    marking    one 1x1 variable per place, in `Net.places` order -- so a
               property over `s0..sN-1` reads off the place list, and the
               round-robin counter (when there is one) is the last component,
               after every place, so the place indices do not move between a
               net and its `-rr` variant.
    selection  `e0`: which transition to try. Integer markings select by
               equality (`e0 == k`), the continuous net by interval
               (`k <= e0 < k+1`), so that every real the precondition admits
               selects a transition rather than stalling on a non-integer.
    amount     `e1`, continuous nets only: how much of the transition to fire.
               `--pre` keeps it in (0, 1]; a transition is enabled only for an
               amount its input places can pay.
    dead       a selection whose transition is not enabled leaves the marking
               alone. That adds no reachable marking and makes the transition
               relation total, which `--safety` over `X(state)` needs.
    effect     a *sum* of guarded deltas, not a chain of overrides. Exactly
               one transition fires, so the two are equivalent -- and only
               the first one compiles at this size. `_step` says why.

Arcs beyond the classical ones -- inhibitor, reset, transfer -- are each a
field on `Trans` and a clause here; `petri/nets.py` says what each buys.
"""
from __future__ import annotations

import os

from zrth import Int, LIA, LRA, Module, Real, Term, Var
from zrth.expr import collecting, expr, ite
from zrth.sugar import X          # the `next` wire of a Var *or* of an Expr


def _num(v, theory, sort):
    return expr(v, theory=theory, sort=sort)


def _enabled(tr, m, sel, k, ntrans, lam):
    """When transition `k` fires: it was selected, and its arcs admit it."""
    if lam is None:
        g = sel == k
    elif k == ntrans - 1:
        # The continuous net's selector is a Real, so it selects by interval:
        # no value `--pre` admits fails to select a transition. The top
        # interval is closed, which is what the precondition's upper bound
        # lands in.
        g = sel >= float(k)
    else:
        g = (sel >= float(k)) & (sel < float(k + 1))
    for p, w in tr.pre.items():
        # A continuous transition has to be paid for at the amount it fires.
        g = g & (m[p] >= (w * lam if lam is not None else w))
    for p, w in tr.inh.items():
        g = g & (m[p] < (float(w) if lam is not None else w))
    return g


def _fold(parts):
    """`parts` summed as a balanced tree, so the sum is log-deep."""
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i]
                 for i in range(0, len(parts), 2)]
    return parts[0]


def _step(net, m, sel, lam, theory, sort):
    """The marking after the selected transition, or `m` where it is dead.

    Exactly one transition fires per tick, so an ordinary place's next value
    is its *sum* of guarded deltas rather than a chain of overrides, and that
    choice is not cosmetic. `System/Circ.lean` is laid out by walking back
    from the outputs without merging a shared subterm, and each layer is then
    bubble-sorted, so what the file costs is the number of distinct
    output-to-input *paths*, squared. Chaining -- `ite(g, running + d,
    running)` -- reads the running value twice and so doubles that count per
    transition: the seven-place mutual-exclusion net came to a 131-wide layer
    and 800 KB of `Circ.lean`, which Lean's `simp` does not finish. Summing
    `ite(g, d, 0)` instead puts constants in both branches, and a constant is
    where the walk stops, so the count is merely additive.

    A place a reset or transfer arc touches cannot be written additively --
    those arcs *replace* a marking rather than add to it -- so it keeps the
    chain. Neither arc appears in a net with more than three transitions
    here, which is what keeps that path affordable.
    """
    zero = _num(0 if lam is None else 0.0, theory, sort)
    guards = [_enabled(tr, m, sel, k, len(net.trans), lam)
              for k, tr in enumerate(net.trans)]
    out = []
    for p in net.places:
        replaced = [k for k, tr in enumerate(net.trans)
                    if p in tr.reset or p in tr.xfer or p in set(tr.xfer.values())]
        if replaced:
            val = m[p]
            for k, tr in enumerate(net.trans):
                if p in tr.reset or p in tr.xfer:
                    val = ite(guards[k], zero, val)       # cleared, or emptied
                elif p in tr.xfer.values():
                    src = next(a for a, b in tr.xfer.items() if b == p)
                    val = ite(guards[k], val + m[src], val)
                else:
                    delta = tr.post.get(p, 0) - tr.pre.get(p, 0)
                    if delta:
                        val = ite(guards[k], val + delta, val)
            out.append(val)
            continue
        parts = [m[p]]
        for k, tr in enumerate(net.trans):
            delta = tr.post.get(p, 0) - tr.pre.get(p, 0)
            if not delta:
                continue
            paid = (delta * lam if lam is not None
                    else _num(delta, theory, sort))
            parts.append(ite(guards[k], paid, zero))
        out.append(_fold(parts))
    return out


def build(net) -> Module:
    """`net` as a sequential reactive module."""
    if net.sched == "rr" and net.real:
        raise ValueError("petri_mod: a continuous net has no round-robin variant")
    theory = LRA if net.real else LIA
    sort = Real([1, 1]) if net.real else Int([1, 1])
    ntrans = len(net.trans)

    places = [Var(sort) for _ in net.places]
    rr = net.sched == "rr"
    # The counter goes after every place, so `s0..s_{P-1}` mean the same thing
    # in a net and in its `-rr` variant.
    ctrl = places + ([Var(sort)] if rr else [])
    extl = [] if rr else [Var(sort)] + ([Var(Real([1, 1]))] if net.real else [])

    with collecting() as init:
        start = list(net.init) + ([0] if rr else [])
        for var, k in zip(ctrl, start):
            e = _num(float(k) if net.real else k, theory, sort)
            init.append(Term(theory.Id(), [X(var)], [e.wire]))

    with collecting() as update:
        cur = [expr(v, theory=theory) for v in ctrl]
        m = dict(zip(net.places, cur))
        if rr:
            counter = cur[-1]
            sel, lam = counter, None
        else:
            ins = [expr(v, theory=theory) for v in extl]
            sel = X(ins[0])
            lam = X(ins[1]) if net.real else None
        vals = _step(net, m, sel, lam, theory, sort)
        if rr:
            vals.append(ite(counter >= ntrans - 1, _num(0, theory, sort), counter + 1))
        for var, e in zip(ctrl, vals):
            update.append(Term(theory.Id(), [X(var)], [e.wire]))

    return Module.sequential(ctrl + extl, init, update)


def module() -> Module:
    name = os.environ.get("PETRI_NET")
    if not name:
        raise SystemExit(
            "PETRI_NET is not set: it names the net this adapter should "
            "build, e.g. PETRI_NET=mutex (see tests/bench_matrix/petri/nets.py)"
        )
    from tests.bench_matrix.petri.nets import NETS

    if name not in NETS:
        raise SystemExit(f"PETRI_NET={name!r}: no such net; have {', '.join(NETS)}")
    return build(NETS[name])
