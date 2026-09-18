"""Hybrid Petri net: two discrete machines around one continuous place.

Alla and David's hybrid Petri nets mix discrete places, which hold whole
tokens, with continuous ones, which hold a real level that the enabled
transitions move at a rate. Here a feeder and a drain sit either side of one
fluid buffer: the machines are 1-safe discrete places (so, Bool), the buffer
is the continuous place, and the drain's rate is an *input* -- a continuous
transition whose speed the environment sets, within the bounds `--pre` gives
it.

    m1' = if m1 then v <= 7   else v <= 4      the feeder, on when low
    m2' = if m2 then v > 0    else v >= 15/2   the drain, on when nearly full
    in    = [m1] * 3/4
    out   = [m2] * min(r, v + in)              r in [7/8, 1], the input
    v'    = (v + in) - out

Two things here are not hybrid-automaton dynamics but Petri net semantics
written as arithmetic:

  * `min(r, v + in)` -- a continuous place cannot go below zero any more than
    a discrete one can hold -1 token, so the drain moves what is there rather
    than what it was asked for. That clamp is live: the drain runs until the
    buffer is empty, so the reachable set touches v = 0 exactly.
  * both thresholds are read off the level *before* the flow, the one-tick lag
    every sampled model in this suite has -- and here it compounds: the feeder
    keeps running for a tick past 7 and the drain starts a tick after 15/2, so
    the peak is 33/4 = 8.25, well above either threshold. A safe band read off
    the guards alone is wrong by two ticks of flow.

Every admissible rate beats the feeder (7/8 > 3/4), so the buffer both fills
and empties whatever the environment does: the two recurrences hold for all
of `--pre`, and it is the interval, not one rate, that has to be in the
invariant.

state:  s0 = v (Real), s1 = m1, s2 = m2 (Bool)
inputs: e0 = r (Real), the drain's rate this tick
"""
from zrth import Bool, LRA, Module, Real, Var, sugar
from zrth.sugar import X, expr, ite

REAL = Real([1, 1])


def _n(v):
    """`v` as a Real constant: an `ite` branch has to be an expression."""
    return expr(v, theory=LRA, sort=REAL)


class Fluid(sugar.Module):
    def init(self, _r):
        return 0.0, False, False

    def update(self, v, m1, m2, r):
        avail = v + ite(m1, _n(0.75), _n(0.0))
        drain = ite(m2, ite(avail >= X(r), X(r), avail), _n(0.0))
        return (avail - drain,
                ite(m1, v <= 7.0, v <= 4.0),
                ite(m2, v > 0.0, v >= 7.5))


def module() -> Module:
    return Fluid(theory=LRA, ctrl=(Var(REAL), Var(Bool([1, 1])), Var(Bool([1, 1]))),
                 extl=(Var(REAL),))
