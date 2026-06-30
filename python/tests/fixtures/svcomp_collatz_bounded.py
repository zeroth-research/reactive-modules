"""SV-COMP style: bounded Collatz-like iteration.

Python semantics:
    x starts at 7.
    Each step: if x == 1 then reset to 7,
               elif x is even then x = x // 2,
               else x = x + 1 (bounded variant — always terminates).
    Property: x == 1 holds infinitely often.

Note: this is NOT real Collatz (which uses 3n+1). We use n+1 for odd
numbers, which guarantees termination because x always decreases on the
even branch and x+1 is always even.

Invariant: 1 <= x <= 8
Ranking: x - 1 (when x != 1)
"""
from zrth import Wire, Module, Sort as dt, LIA
from zrth.analyzer import convert_method


def init():
    return 7


def update(old_x):
    if old_x == 1:
        return 7
    if old_x > 4:
        return old_x - 3   # big steps down
    if old_x > 1:
        return old_x - 1   # small steps down
    return old_x


def module() -> Module:
    state = (Wire(dt.Int([1, 1])), Wire(dt.Int([1, 1])))
    init_terms = convert_method(init, {}, [state[1]], theory=LIA)
    update_terms = convert_method(update, {"old_x": state}, [state[1]], theory=LIA)
    return Module.sequential(init_terms, update_terms, obs=[state])
