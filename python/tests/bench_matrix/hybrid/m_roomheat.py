"""Room heating: two rooms, one heater that moves between them.

The Fehnker-Ivancic room heating benchmark at its smallest interesting size
(RHB 2/1): each room loses heat to the outside and exchanges heat with its
neighbour, and a single heater is moved to whichever room has fallen below
the threshold and is the colder of the two by at least a degree.

    T1' = T1 + 1/8 (10 - T1) + 1/16 (T2 - T1) + [heater in 1] * 3
    T2' = T2 + 1/4 (10 - T2) + 1/16 (T1 - T2) + [heater in 2] * 3
    in2' = (T2 <= 18 and T2 <= T1 - 1) or (in2 and not (T1 <= 18 and T1 <= T2 - 1))

Room 2 is the draughtier one -- it loses heat to the outside twice as fast --
so the two rooms are not interchangeable and the heater does not simply
alternate. Parking the heater in room 1 forever would settle the rooms at
(28, 16), and 16 is below the threshold: it is that gap that makes the heater
move, and it is why neither room's recurrence can be proved without the
other room's temperature in the invariant.

state: s0 = T1, s1 = T2 (Real), s2 = in2 (Bool)
"""
from zrth import Bool, LRA, Module, Real, Var, sugar
from zrth.sugar import expr, ite

REAL = Real([1, 1])


def _n(v):
    """`v` as a Real constant: an `ite` branch has to be an expression."""
    return expr(v, theory=LRA, sort=REAL)


class RoomHeat(sugar.Module):
    def init(self):
        return 17.0, 17.0, False

    def update(self, t1, t2, in2):
        heat = ite(in2, _n(0.0), _n(3.0))
        return (0.8125 * t1 + 0.0625 * t2 + 1.25 + heat,
                0.6875 * t2 + 0.0625 * t1 + 2.5 + (_n(3.0) - heat),
                ite((t2 <= 18.0) & (t2 <= t1 - 1.0), True,
                    ite((t1 <= 18.0) & (t1 <= t2 - 1.0), False, in2)))


def module() -> Module:
    return RoomHeat(theory=LRA,
                    ctrl=(Var(REAL), Var(REAL), Var(Bool([1, 1]))))
