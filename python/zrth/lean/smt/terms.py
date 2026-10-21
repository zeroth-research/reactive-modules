"""Walks over cvc5 terms.

A term is a DAG: the encoder shares subterms, and a net shares most of
them. Every walk here visits each distinct node once, by `getId()`; walked
as its tree, a deep term is exponential.

`Kind` is imported where it is used, so importing this loads no cvc5.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator

ORDERS = ("pre", "lifo")


def nodes(roots: Iterable, *, order: str = "pre") -> Iterator:
    """Each distinct node reachable from `roots`, once.

    `pre` is depth-first preorder, roots and children left to right. `lifo`
    is a stack seeded with `roots` and pushed each node's children in order,
    so the last root and the last child come first. Both reach the same
    nodes; a caller whose output order is visible says which it needs.
    """
    if order not in ORDERS:
        raise ValueError(f"order is one of {ORDERS}, not {order!r}")
    stack = list(roots)
    if order == "pre":
        stack.reverse()
    seen: set = set()
    while stack:
        node = stack.pop()
        key = node.getId()
        if key in seen:
            continue
        seen.add(key)
        yield node
        kids = list(node)
        if order == "pre":
            kids.reverse()
        stack.extend(kids)


def free_constants(roots: Iterable, *, order: str = "pre") -> dict:
    """Every free constant under `roots`, by symbol, first occurrence kept.

    cvc5 has no "the declarations this term needs", so the term is walked. A
    constant is `Kind.CONSTANT`; an operator, a bound variable and a literal
    are none of them declared.
    """
    from cvc5 import Kind

    out: dict = {}
    for node in nodes(roots, order=order):
        if node.getKind() == Kind.CONSTANT:
            out.setdefault(node.getSymbol(), node)
    return out


def state_dependent(term) -> bool:
    """Whether `term` reads a free constant at all, or is ground."""
    from cvc5 import Kind

    return any(node.getKind() == Kind.CONSTANT
               for node in nodes([term], order="lifo"))


def literals(roots: Iterable) -> list:
    """Every Int and Real literal node under `roots`."""
    from cvc5 import Kind

    kinds = (Kind.CONST_INTEGER, Kind.CONST_RATIONAL)
    return [node for node in nodes(roots) if node.getKind() in kinds]


def ite_conditions(roots: Iterable, *, order: str = "pre",
                   key: Callable = str,
                   where: "Callable | None" = None) -> list:
    """The condition of every `ite` under `roots`, one per `key`, in `order`.

    `key` is what makes two conditions one: the printed form by default, so
    a transition testing `s0 = 0` in three components is one condition.
    `where`, given an `ite`, says whether its condition is wanted.
    """
    from cvc5 import Kind

    out: dict = {}
    for node in nodes(roots, order=order):
        if node.getKind() == Kind.ITE and (where is None or where(node)):
            cond = node[0]
            out.setdefault(key(cond), cond)
    return list(out.values())


@functools.cache
def _min_max_table() -> dict:
    """`ite (a ≥ b) a b` is `max a b`, `ite (a ≥ b) b a` is `min a b`, and
    so on for each strict/non-strict comparison. The flag says whether the
    branches are in the same order as the comparison's operands. Built on
    first use: the keys are cvc5 kinds."""
    from cvc5 import Kind

    return {
        (Kind.GEQ, True): "max", (Kind.GT, True): "max",
        (Kind.LEQ, True): "min", (Kind.LT, True): "min",
        (Kind.GEQ, False): "min", (Kind.GT, False): "min",
        (Kind.LEQ, False): "max", (Kind.LT, False): "max",
    }


def min_max_of(t) -> str | None:
    """`"min"` / `"max"` if this `ite` is really one, else None.

    The printer (`proof.from_smt`) folds such an `ite` into `min`/`max`, and
    feature detection (`proof.facts`) asks the same question here rather
    than counting `max` in the printer's output and thereby depending on it.
    """
    from cvc5 import Kind

    if t.getKind() != Kind.ITE or not t.getSort().isInteger():
        return None
    cond = t[0]
    kind = cond.getKind()
    if kind not in (Kind.GEQ, Kind.GT, Kind.LEQ, Kind.LT):
        return None
    lhs, rhs = cond[0], cond[1]
    then_, else_ = t[1], t[2]
    if lhs == then_ and rhs == else_:
        in_order = True
    elif lhs == else_ and rhs == then_:
        in_order = False
    else:
        return None
    return _min_max_table()[(kind, in_order)]


def selector_index(t) -> int:
    """Which element `t`, an `APPLY_SELECTOR` on a tuple, reads: its flat
    index.

    By identity against the datatype's own selectors rather than by reading
    the index out of the printed name: the name is cvc5's to spell.
    """
    dt = t[1].getSort().getDatatype()[0]
    for j in range(dt.getNumSelectors()):
        if t[0] == dt[j].getTerm():
            return j
    raise ValueError(f"no element of {t[1].getSort()} is read by {t}")
