"""`m_watertank` with an adversarial consumption disturbance.

The stochastic version of the tank benchmark makes the outflow a disturbance
rather than a constant. Here the disturbance is an *external input* -- a
reactive module's inputs are resampled every tick and constrained only by
`--pre` -- so the property has to hold for every consumption profile the
precondition admits, which is the verification question rather than a
simulation of one profile:

    h' = h + 3/2 - c   while filling     h' = h - 1/2 - c   while not
    c in [0, 1/4]                        (the precondition)

Without the precondition `c` is unbounded and nothing is safe, which is the
same lesson `m_relu_input` carries in the limit matrix.

state:  s0 = h (Real), s1 = fill (Bool)
inputs: e0 = c (Real), the consumption this tick
"""
from zrth import Bool, LRA, Module, Real, Var, sugar
from zrth.sugar import X, ite


class Tank(sugar.Module):
    def init(self, _c):
        return 9.0, False

    def update(self, h, fill, c):
        return (ite(fill, h + 1.5 - X(c), h - 0.5 - X(c)),
                ite(h <= 6.0, True, ite(h >= 12.0, False, fill)))


def module() -> Module:
    return Tank(theory=LRA, ctrl=(Var(Real([1, 1])), Var(Bool([1, 1]))),
                extl=(Var(Real([1, 1])),))
