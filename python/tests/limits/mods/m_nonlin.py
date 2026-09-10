"""Two counters that both shrink — a product ranking is the natural (and
nonlinear) choice: x in 0..3 cycles, y drops when x wraps."""
from zrth import Module, Int, LIA, Var, X
from zrth.analyzer import convert_method


def init():
    return 3, 3


def update(old_x, old_y):
    if old_x > 0:
        return old_x - 1, old_y
    if old_y > 0:
        return 3, old_y - 1
    return 3, 3


def module() -> Module:
    x = Var(Int([1, 1]))
    y = Var(Int([1, 1]))
    return Module.sequential(
        [x, y],
        convert_method(init, {}, [X(x), X(y)], theory=LIA),
        convert_method(update, {"old_x": x, "old_y": y}, [X(x), X(y)], theory=LIA),
    )
