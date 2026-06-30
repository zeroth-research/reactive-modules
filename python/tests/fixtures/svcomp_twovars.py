"""SV-COMP style: two variables with conditional increment/decrement.

Python semantics:
    x starts at 0, y starts at 10.
    Each step: if x < y then x += 1, else x = 0 and y = 10.
    Property: x == y holds infinitely often.

Invariant: 0 <= x <= y and y == 10
Ranking: y - x (when x != y)
"""
from zrth import Wire, Module, Sort as dt, LIA
from zrth.analyzer import convert_method


def init():
    return 0, 10


def update(old_x, old_y):
    if old_x < old_y:
        return old_x + 1, old_y
    return 0, 10


def module() -> Module:
    x = (Wire(dt.Int([1, 1])), Wire(dt.Int([1, 1])))
    y = (Wire(dt.Int([1, 1])), Wire(dt.Int([1, 1])))
    init_terms = convert_method(init, {}, [x[1], y[1]], theory=LIA)
    update_terms = convert_method(
        update, {"old_x": x, "old_y": y}, [x[1], y[1]], theory=LIA
    )
    return Module.sequential(init_terms, update_terms, obs=[x, y])
