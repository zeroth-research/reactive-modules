"""SV-COMP style: a countdown whose starting point is an input.

Python semantics:
    x starts at the external input n, decrements while it is positive, and
    stays once it reaches 0.
    Property: 0 <= x.

The property holds only of runs that start at 0 <= n, so `0 <= x` fails
initiation from every state the init block can produce and holds from the
ones `--pre "(>= e0 0)"` admits. That is the whole difference a precondition
makes, in the smallest module that can show it.

`n` is never written, so it is an external input; `x` starts at it *bare*,
which is what lets a constraint on the input be read back as one on `x`.
"""
from zrth import Int, LIA, Module, Term, Var, X
from zrth.analyzer import convert_method


def update(old_x):
    if old_x > 0:
        return old_x - 1
    return old_x


def module() -> Module:
    x = Var(Int([1, 1]))
    n = Var(Int([1, 1]))
    init_terms = [Term(LIA.Id(), [X(x)], [X(n)])]
    update_terms = convert_method(update, {"old_x": x}, [X(x)], theory=LIA)
    return Module.sequential([x, n], init_terms, update_terms)
