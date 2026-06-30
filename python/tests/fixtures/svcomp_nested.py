"""SV-COMP style: nested loops (outer + inner counter).

Python semantics:
    i starts at 0, j starts at 0.
    Inner loop: j increments until j == 3, then j resets and i increments.
    Outer loop: i increments until i == 3, then both reset to 0.
    Property: i == 0 and j == 0 holds infinitely often.

Invariant: 0 <= i <= 3 and 0 <= j <= 3
Ranking: (3 - i) * 4 + (3 - j) (when not at (0,0))
"""
from zrth import Wire, Module, Sort as dt, LIA
from zrth.analyzer import convert_method


def init():
    return 0, 0


def update(old_i, old_j):
    if old_j < 3:
        return old_i, old_j + 1
    # j reached 3, reset j and advance i
    if old_i < 3:
        return old_i + 1, 0
    # both at max, reset
    return 0, 0


def module() -> Module:
    i = (Wire(dt.Int([1, 1])), Wire(dt.Int([1, 1])))
    j = (Wire(dt.Int([1, 1])), Wire(dt.Int([1, 1])))
    init_terms = convert_method(init, {}, [i[1], j[1]], theory=LIA)
    update_terms = convert_method(
        update, {"old_i": i, "old_j": j}, [i[1], j[1]], theory=LIA
    )
    return Module.sequential(init_terms, update_terms, obs=[i, j])
