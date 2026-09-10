"""LIA 1x1: x goes 0,2,4,6,8,10 then back to 0 — an even-parity invariant."""
from zrth import Module, Int, LIA, Var, X
from zrth.analyzer import convert_method


def init():
    return 0


def update(old_x):
    if old_x == 10:
        return 0
    return old_x + 2


def module() -> Module:
    s = Var(Int([1, 1]))
    return Module.sequential(
        [s],
        convert_method(init, {}, [X(s)], theory=LIA),
        convert_method(update, {"old_x": s}, [X(s)], theory=LIA),
    )
