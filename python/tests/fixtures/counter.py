"""Simple counter reactive module fixture for CLI tests.

Python semantics:
    x starts at 0, increments each step, resets to 0 when it reaches 10.
    Property: x == 0 holds infinitely often.
"""
from zrth import Module, Int, LIA, Var, X
from zrth.analyzer import convert_method


def init():
    return 0


def update(old_x):
    x = old_x + 1
    if x == 10:
        return 0
    return x


def module() -> Module:
    state = Var(Int([1, 1]))
    init_terms = convert_method(init, {}, [X(state)], theory=LIA)
    update_terms = convert_method(update, {"old_x": state}, [X(state)], theory=LIA)
    return Module.sequential([state], init_terms, update_terms)
