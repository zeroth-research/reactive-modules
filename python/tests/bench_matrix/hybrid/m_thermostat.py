"""Thermostat with hysteresis: the canonical hybrid automaton, sampled.

Henzinger's two-mode thermostat (`The Theory of Hybrid Automata`, LICS'96)
under a first-order Euler step, which is what makes it a reactive module:
Real state, discrete time.

    off:  T' = T + 1/8 (10 - T)   = 0.875 T + 1.25    (decays toward 10)
    on:   T' = T + 1/8 (30 - T)   = 0.875 T + 3.75    (decays toward 30)
    on'   = T <= 18 or (on and T < 22)

The controller is a Moore machine -- `on'` is read off the *current* T -- so
it overshoots each setpoint by one tick: the run settles into roughly
[16.7, 23.2] rather than [18, 22], which is why the safe band is wider than
the hysteresis band. Every coefficient is a dyadic rational, so the float32
tensor holding it *is* the constant the certificate reasons about.

state: s0 = T (Real), s1 = on (Bool)
"""
from zrth import Bool, LRA, Module, Real, Var, sugar
from zrth.sugar import ite


class Thermostat(sugar.Module):
    def init(self):
        return 20.0, False

    def update(self, t, on):
        return (ite(on, 0.875 * t + 3.75, 0.875 * t + 1.25),
                ite(t <= 18.0, True, ite(t >= 22.0, False, on)))


def module() -> Module:
    return Thermostat(theory=LRA, ctrl=(Var(Real([1, 1])), Var(Bool([1, 1]))))
