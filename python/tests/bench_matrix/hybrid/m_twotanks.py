"""Two tanks: the ARCH-COMP/Fehnker-Ivancic two-tank benchmark, sampled.

Two tanks in a closed system, connected by a pipe; a pump on each pipe
enables or disables its flow. Tank 1 takes a constant inflow, spills into
tank 2 through pump `v1`, and tank 2 drains through pump `v2`. Each pump is
bang-bang with hysteresis on its own tank, so neither controller sees the
other's level -- the coupling is only through the water.

    x1' = x1 + 1/2 - [v1]           x2' = x2 + [v1] - 3/4 [v2]
    v1' = x1 >= 6 or (v1 and x1 > 2)
    v2' = x2 >= 6 or (v2 and x2 > 2)

The two loops have different periods (tank 2's outflow is 3/4 of its
inflow), so the run is not periodic in any short window and the reachable
band is wider than either controller's own hysteresis band.

state: s0 = x1, s1 = x2 (Real), s2 = v1, s3 = v2 (Bool)
"""
from zrth import Bool, LRA, Module, Real, Var, sugar
from zrth.sugar import expr, ite

REAL = Real([1, 1])


def _n(v):
    """`v` as a Real constant: an `ite` branch has to be an expression."""
    return expr(v, theory=LRA, sort=REAL)


class TwoTanks(sugar.Module):
    def init(self):
        return 4.0, 4.0, False, False

    def update(self, x1, x2, v1, v2):
        into2 = ite(v1, _n(1.0), _n(0.0))
        return (x1 + 0.5 - into2,
                x2 + into2 - ite(v2, _n(0.75), _n(0.0)),
                ite(x1 >= 6.0, True, ite(x1 <= 2.0, False, v1)),
                ite(x2 >= 6.0, True, ite(x2 <= 2.0, False, v2)))


def module() -> Module:
    return TwoTanks(theory=LRA, ctrl=(Var(REAL), Var(REAL),
                                      Var(Bool([1, 1])), Var(Bool([1, 1]))))
