from zrth.expr import Expr, Var
from random import randrange

from typing import Callable, Any


def nxt(var: Var) -> Var:
    """
    Get next variable for `var`.
    """
    assert isinstance(var, Var), var
    return Var(f"nxt({var.name})")


class IfThen(Expr):
    """
    An expression representing `IfThen` term.

    This expression is used inside `Choose` terms.
    """

    def __init__(self, cond: Expr | bool, expr: Any):
        super().__init__("ifthen", [cond, expr])

    def cond(self):
        return self.get_children()[0]

    def expr(self):
        return self.get_children()[1]

    def is_concrete(self) -> bool:
        return not any(isinstance(c, Expr) for c in self.get_children())


def _choose(alist):
    assert all(isinstance(a, IfThen) for a in alist), alist
    if all(a.is_concrete() for a in alist):
        # execute choose concretely
        sat_args = [arg for arg in alist if arg.cond()]
        if sat_args:
            return sat_args[randrange(len(sat_args))].expr()

        # return None
        raise RuntimeError("No satisfiable branch in a choose statement")

    return Expr("choose", alist)


def _choose_or(alist):
    choices = alist[:-1]
    last = alist[-1]

    # the last argument may not be `IfThen`, in which case
    # it is the default (unconditional) argument
    assert not isinstance(last, IfThen), alist
    assert all(isinstance(a, IfThen) for a in choices), alist

    if not isinstance(last, Expr) and all(a.is_concrete() for a in choices):
        # execute choose_or concretely
        sat_args = [arg for arg in choices if arg.cond()]
        if sat_args:
            return sat_args[randrange(len(sat_args))].expr()
        return last

    return Expr("choose_or", alist)


def choose(*args):
    """
    Implementation of choose construct.
    """
    alist = list(args)
    if isinstance(alist[-1], IfThen):
        return _choose(alist)
    else:
        return _choose_or(alist)
