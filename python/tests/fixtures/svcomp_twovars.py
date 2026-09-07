"""SV-COMP style: two variables with conditional increment/decrement.

Python semantics:
    x starts at 0, y starts at 10.
    Each step: if x < y then x += 1, else x = 0 and y = 10.
    Property: x == y holds infinitely often.

Invariant: 0 <= x <= y and y == 10
Ranking: y - x (when x != y)
"""
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
    init_terms = convert_method(init, {}, [X(x), X(y)], theory=LIA)
    update_terms = convert_method(
        update, {"old_x": x, "old_y": y}, [X(x), X(y)], theory=LIA
    )
    return Module.sequential([x, y], init_terms, update_terms)
