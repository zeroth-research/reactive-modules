"""SV-COMP style: Euclidean GCD iteration.

Python semantics:
    a starts at 12, b starts at 8.
    Each step: if a > b then a = a - b, elif b > a then b = b - a,
               else (a == b, GCD found) reset to (12, 8).
    Property: a == b holds infinitely often (GCD is reached).

Invariant: a >= 1 and b >= 1
Ranking: a + b - 2 (when a != b; decreases because we subtract the smaller)
"""
from zrth import Module, Int, LIA, Var, X
from zrth.analyzer import convert_method


def init():
    return 12, 8


def update(old_a, old_b):
    if old_a > old_b:
        return old_a - old_b, old_b
    if old_b > old_a:
        return old_a, old_b - old_a
    # a == b: GCD found, restart
    return 12, 8


def module() -> Module:
    a = Var(Int([1, 1]))
    b = Var(Int([1, 1]))
    init_terms = convert_method(init, {}, [X(a), X(b)], theory=LIA)
    update_terms = convert_method(
        update, {"old_a": a, "old_b": b}, [X(a), X(b)], theory=LIA
    )
    return Module.sequential([a, b], init_terms, update_terms)
