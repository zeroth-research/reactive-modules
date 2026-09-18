"""Bouncing ball: the textbook hybrid automaton, forward-Euler sampled.

Free fall with a discrete reset at the floor, restitution 1/2. The step is
dt = 1/4 with g = 4, so gravity costs exactly one unit of velocity per tick
and every constant stays dyadic:

    h_next = h + v/4
    bounce = h_next <= 0
    h' = if bounce then 0 else h_next
    v' = if bounce then -v/2 + 1/2 else v - 1        i.e. -(v - 1)/2

Euler's error is the interesting part: the reset fires on the *sampled*
crossing, so the ball never goes below the floor but each bounce is taken
slightly late, and the apex falls away faster than the continuous model's
1/4 per bounce. The velocity at the floor converges to a fixed point of
v = -(v-1)/2, i.e. 1/3 -- a value no dyadic constant in the module can
name, so the run never becomes eventually periodic.

state: s0 = h, s1 = v (Real)
"""
from zrth import LRA, Module, Real, Var, sugar
from zrth.sugar import expr, ite

REAL = Real([1, 1])


def _n(v):
    """`v` as a Real constant: an `ite` branch has to be an expression."""
    return expr(v, theory=LRA, sort=REAL)


class Ball(sugar.Module):
    def init(self):
        return 4.0, 0.0

    def update(self, h, v):
        nh, nv = h + 0.25 * v, v - 1.0
        bounce = nh <= 0.0
        # `-(v - 1)/2` is written out as `-v/2 + 1/2`, not as a coefficient on
        # `nv`: every scaling here multiplies a *state variable*. `System/`'s
        # scalar-to-matrix equivalence proof closes for a `Linear` term that
        # reads a ctrl component and does not for one that reads a derived
        # wire, so a module that scales an intermediate value reaches the
        # matrix as BUILD-FAIL on every route and measures none of them.
        return ite(bounce, _n(0.0), nh), ite(bounce, -0.5 * v + 0.5, nv)


def module() -> Module:
    return Ball(theory=LRA, ctrl=(Var(REAL), Var(REAL)))
