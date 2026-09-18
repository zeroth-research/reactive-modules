"""Time Petri net: a transition with a firing window, and its clock.

Merlin's time Petri nets give each transition an interval [a, b] and a clock
that runs while the transition is enabled: it may not fire before `a`, and
under *strong* semantics it must fire by `b` -- time is simply not allowed to
pass beyond it. A clock is a real number that grows by an amount nobody in
the net chooses, so a time Petri net is a hybrid model, and this is it as a
reactive module: one net place (`working`), one clock, and the two things a
tick can be given from outside.

    dt    how much time passes this tick, in (0, 1] -- the environment's, not
          the net's, so the clock takes values no constant here names
    go    whether the scheduler fires `done` when the window allows it

    step  = if c + dt >= 5 then 5 - c else dt      time stops at the deadline
    fire  = working and c >= 2 and (go or c >= 5)  earliest 2, urgent at 5
    c'    = if working then (if fire then 0 else c + step) else 0
    w'    = if working then not fire else true

The truncation in `step` *is* strong semantics: the deadline is enforced by
denying the time step, not by an error transition, which is why the clock
bound in `cases.py` holds for every environment. The recurrence row is the
one that separates the semantics -- drop `c >= 5` from `fire` and a scheduler
that says `go = false` forever keeps `working` set for ever, so `G F not
working` holds only because the firing is urgent.

state:  s0 = c (Real), s1 = working (Bool)
inputs: e0 = dt (Real), e1 = go (Bool)
"""
from zrth import Bool, LRA, Module, Real, Var, sugar
from zrth.sugar import X, expr, ite

REAL = Real([1, 1])
EARLIEST, LATEST = 2.0, 5.0


def _n(v):
    """`v` as a Real constant: an `ite` branch has to be an expression."""
    return expr(v, theory=LRA, sort=REAL)


class Timed(sugar.Module):
    def init(self, _dt, _go):
        return 0.0, False

    def update(self, c, working, dt, go):
        step = ite(c + X(dt) >= LATEST, LATEST - c, X(dt))
        fire = working & (c >= EARLIEST) & (X(go) | (c >= LATEST))
        return (ite(working, ite(fire, _n(0.0), c + step), _n(0.0)),
                ite(working, ~fire, True))


def module() -> Module:
    return Timed(theory=LRA, ctrl=(Var(REAL), Var(Bool([1, 1]))),
                 extl=(Var(REAL), Var(Bool([1, 1]))))
