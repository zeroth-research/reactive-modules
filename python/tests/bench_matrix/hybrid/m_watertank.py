"""Leaking water tank under a sampled PLC controller.

The ARCH-COMP "leaking tank" benchmark: the tank leaks at a constant rate and
is refilled from an unlimited source at a larger constant rate; the PLC starts
refilling when a sensor reads `h <= 6` and stops when it reads `h >= 12`.

    h' = h + 3/2   while filling        h' = h - 1/2   while not
    fill' = h <= 6 or (fill and h < 12)

As in `m_thermostat` the controller samples, so it overshoots: the run covers
roughly [5.5, 14.5], not [6, 12]. That gap is the whole point of the
benchmark -- the safe band has to absorb one tick of actuation lag.

state: s0 = h (Real), s1 = fill (Bool)
"""
from zrth import Bool, LRA, Module, Real, Var, sugar
from zrth.sugar import ite


class Tank(sugar.Module):
    def init(self):
        return 9.0, False

    def update(self, h, fill):
        return (ite(fill, h + 1.5, h - 0.5),
                ite(h <= 6.0, True, ite(h >= 12.0, False, fill)))


def module() -> Module:
    return Tank(theory=LRA, ctrl=(Var(Real([1, 1])), Var(Bool([1, 1]))))
