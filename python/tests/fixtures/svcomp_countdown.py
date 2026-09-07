"""SV-COMP style: countdown from N to 0.

Python semantics:
    x starts at 100, decrements each step, resets to 100 when it reaches 0.
    Property: x == 0 holds infinitely often.

Invariant: 0 <= x <= 100
Ranking: x (when x != 0)
"""
from zrth import Module, Int, LIA, Var, X
from zrth.analyzer import convert_method


def init():
    return 100


def update(old_x):
    if old_x == 0:
        return 100
    return old_x - 1


def module() -> Module:
    state = Var(Int([1, 1]))
    init_terms = convert_method(init, {}, [X(state)], theory=LIA)
    update_terms = convert_method(update, {"old_x": state}, [X(state)], theory=LIA)
    return Module.sequential([state], init_terms, update_terms)
