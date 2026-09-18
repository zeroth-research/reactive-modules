"""Adaptive cruise control: a saturated linear controller closing a gap.

A follower tracks a leader at a set distance of 5. The state is the gap and
the relative velocity (leader minus follower); the follower's acceleration is
a proportional-derivative command on both, clipped to the actuator's range:

    cmd = -1/2 dv - 1/4 g + 5/4          the unclipped command
    g'  = g + dv/2
    dv' = dv + 1/2 max(-1, min(1, cmd))
        = dv + 1/2   if cmd >= 1
          dv - 1/2   if cmd <= -1
          3/4 dv - 1/8 g + 5/8   otherwise

Unsaturated the closed loop is [[1, 1/2], [-1/8, 3/4]], whose eigenvalues are
complex with modulus sqrt(13/16) ~ 0.901: the gap converges to 5 while
oscillating about it, so the safety bound is decided by the *transient*, not
the limit. The run starts at a gap of 10, which saturates the actuator for the
first several ticks -- the piece a purely linear Lyapunov argument does not
see.

The clip is distributed into the three branches rather than applied to `cmd`
and then scaled, so that every scaling multiplies a *state variable*.
`System/`'s scalar-to-matrix equivalence proof closes for a `Linear` term
that reads a ctrl component and does not for one that reads a derived wire,
and a module that scales an intermediate value reaches the matrix as
BUILD-FAIL on every route, measuring none of them.

state: s0 = g, s1 = dv (Real)
"""
from zrth import LRA, Module, Real, Var, sugar
from zrth.sugar import ite

REAL = Real([1, 1])


class Cruise(sugar.Module):
    def init(self):
        return 10.0, 0.0

    def update(self, g, dv):
        cmd = -0.5 * dv - 0.25 * g + 1.25
        half = ite(cmd >= 1.0, dv + 0.5,
                   ite(cmd <= -1.0, dv - 0.5,
                       0.75 * dv - 0.125 * g + 0.625))
        return g + 0.5 * dv, half


def module() -> Module:
    return Cruise(theory=LRA, ctrl=(Var(REAL), Var(REAL)))
