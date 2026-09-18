"""Reactor rod control: two rods, each with a refractory timer.

Alur's reactor (`Principles of Cyber-Physical Systems`, ch. 7): the core heats
itself, and either of two control rods can be dropped in to cool it. A rod
that has just been withdrawn cannot go back for `C` ticks, so the two timers
are the state that decides whether the next overheat can be answered at all.
The reactor must be shut down if the core passes 545 with neither rod
available -- `down` latches that, and the property the benchmark is about is
that it never latches.

    T'  = 17/16 T - 25     no rod        (the core heats itself)
        = 17/16 T - 60     rod 1 in
        = 17/16 T - 56     rod 2 in      (the weaker rod)
    c_i' = 0 while rod i is in, else min(c_i + 1, 64)
    rod i goes in  at T >= 545 if no rod is in and c_i >= 12, rod 1 preferred
    rod i comes out at T <= 510
    down' = down or (T >= 545 and no rod is in and neither timer has reached 12)

Whether `down` can latch is a question about *two* interleaved timers against
one cycle length, which is why the module is here: the cycle is about nine
ticks and the refractory window twelve, so the rods have to alternate, and an
invariant that does not relate c1 and c2 to the rod states cannot see it.

state: s0 = T, s1 = c1, s2 = c2 (Real), s3 = r1, s4 = r2, s5 = down (Bool)
"""
from zrth import Bool, LRA, Module, Real, Var, sugar
from zrth.sugar import expr, ite

REAL = Real([1, 1])
BOOL = Bool([1, 1])


def _n(v):
    """`v` as a Real constant: an `ite` branch has to be an expression."""
    return expr(v, theory=LRA, sort=REAL)


class Reactor(sugar.Module):
    def init(self):
        return 510.0, 12.0, 12.0, False, False, False

    def update(self, t, c1, c2, r1, r2, down):
        hot, idle = t >= 545.0, (~r1) & (~r2)
        return (
            ite(r1, 1.0625 * t - 60.0,
                ite(r2, 1.0625 * t - 56.0, 1.0625 * t - 25.0)),
            ite(r1, _n(0.0), ite(c1 >= 64.0, _n(64.0), c1 + 1.0)),
            ite(r2, _n(0.0), ite(c2 >= 64.0, _n(64.0), c2 + 1.0)),
            ite(r1, t > 510.0, idle & hot & (c1 >= 12.0)),
            ite(r2, t > 510.0, idle & hot & (c1 < 12.0) & (c2 >= 12.0)),
            down | (idle & hot & (c1 < 12.0) & (c2 < 12.0)),
        )


def module() -> Module:
    return Reactor(theory=LRA, ctrl=(Var(REAL), Var(REAL), Var(REAL),
                                     Var(BOOL), Var(BOOL), Var(BOOL)))
