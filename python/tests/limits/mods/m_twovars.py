"""LIA, two 1x1 wires: x counts up to y (=10), then both reset."""
from zrth import Module, Int, LIA, Var, X
from zrth.analyzer import convert_method


def init():
    return 0, 10


def update(old_x, old_y):
    if old_x < old_y:
        return old_x + 1, old_y
    return 0, 10


def module() -> Module:
    x = Var(Int([1, 1]))
    y = Var(Int([1, 1]))
    return Module.sequential(
        [x, y],
        convert_method(init, {}, [X(x), X(y)], theory=LIA),
        convert_method(update, {"old_x": x, "old_y": y}, [X(x), X(y)], theory=LIA),
    )
