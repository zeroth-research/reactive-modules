"""Simple counter reactive module fixture for CLI tests.

Python semantics:
    x starts at 0, increments each step, resets to 0 when it reaches 10.
    Property: x == 0 holds infinitely often.
"""
from zrth import Wire, Module, DType as dt
from zrth.analyzer import convert_method


def init():
    return 0


def update(old_x):
    x = old_x + 1
    if x == 10:
        return 0
    return x


def module() -> Module:
    state = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    init_terms = convert_method(init, {}, [state[1]])
    update_terms = convert_method(update, {"old_x": state}, [state[1]])
    return Module.sequential(init_terms, update_terms, obs=[state])
