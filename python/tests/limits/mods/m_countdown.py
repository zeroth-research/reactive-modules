"""LIA 1x1: x = 100, then x-1 each step, reset to 100 at 0. Baseline control."""
from zrth import Module, Int, LIA, Var, X
from zrth.analyzer import convert_method


def init():
    return 100


def update(old_x):
    if old_x == 0:
        return 100
    return old_x - 1


def module() -> Module:
    s = Var(Int([1, 1]))
    return Module.sequential(
        [s],
        convert_method(init, {}, [X(s)], theory=LIA),
        convert_method(update, {"old_x": s}, [X(s)], theory=LIA),
    )
