"""``zrth.sugar``: author a reactive Module by subclassing (à la ``torch.nn.Module``).

Subclass ``sugar.Module``, pass a ``theory`` and the ``ctrl`` (and optional ``extl``)
variables — each a ``(latched, next)`` **wire pair** — and override ``init`` / ``update``
to return the next-state values as a tuple aligned with ``ctrl``. **Instantiating the
subclass *is* a base Module** — ``init`` / ``update`` run and the sequential module is
built in the constructor.

    from zrth import LIA, Int, Wire
    from zrth.sugar import Module

    class Counter(Module):
        def init(self):            return 0
        def update(self, ctrl):    return ctrl + 1

    INT = Int([1, 1])
    x = (Wire(INT), Wire(INT))                 # a (latched, next) wire pair
    m = Counter(theory=LIA, ctrl=(x,))         # a base Module with theory LIA (closed)

``ctrl`` / ``extl`` are tuples of wire pairs (a single var is unwrapped, so you can write
``def update(self, ctrl): return ctrl + 1``); a var reads as its latched value and
``nxt(v)`` gives its next wire. ``init`` / ``update`` return a tuple aligned with ``ctrl``
— entry *i* is ``ctrl[i]``'s next value (return the var itself to keep it). ``extl`` vars
are external inputs: they are declared but their next value is *not* driven here (the base
Module classifies undriven wires as ``extl``, so a module with ``extl`` is open).

Only **sequential** modules are built here; the base Module also has ``combinatorial`` /
``parallel``, which this front-end does not expose yet (combinatorial logic may instead
come from the torch front-end).

Config comes from **constructor kwargs** (``theory=``, ``ctrl=``, ``extl=``). Two base-class
constraints force this (both flagged for design review):
  * it cannot flow through ``super().__init__`` — the base Module is a frozen pyo3 class
    whose constructor runs at ``__new__`` (before any ``__init__``), so this builds there;
    and
  * config *class attributes* named ``ctrl``/``extl`` would shadow the base Module's own
    ``ctrl``/``extl`` getters, so those names can't be reused declaratively.
"""

import inspect

from .zrth import Module as _Module, Term as _Term, Var as _Var, X as _X, d as _d

from .expr import expr, cast, ite, relu, argmax, collecting, Expr, X, d  # re-exported for authoring

# Public authoring surface: `from zrth.sugar import Module, expr, nxt, ite, cast, ...`
__all__ = ["Module", "expr", "cast", "ite", "relu", "argmax", "Expr", "X", "d"]


def _as_tuple(r) -> tuple:
    if r is None:
        return ()
    if isinstance(r, tuple):
        return r
    if isinstance(r, list):
        return tuple(r)
    return (r,)


def _build_init_block(cls, ctrl, extl, theory) -> list:
    fn = getattr(cls, "init", None)
    if fn is None:
        return None
    nparams = len(inspect.signature(fn).parameters)

    # we assume all arguments are taken , otherwise it's too ambiguous
    args = tuple(expr(v, theory=theory) for v in extl)
    if nparams != len(args) + 1:
        raise ValueError(f"init expects len(extl) == {len(args)} params, got {nparams - 1}")

    with collecting() as terms:
        vals = _as_tuple(fn(None, *args))
        if len(vals) != len(ctrl):
            raise ValueError(f"init expects {len(ctrl)} return values, got {len(vals)}")
        for var, val in zip(ctrl, vals):
            e = val if isinstance(val, Expr) else expr(val, theory=theory, sort=var.dtype)
            terms.append(_Term(theory.Id(), [_X(var)], [e.wire]))
        return terms


def _build_update_block(cls, ctrl, extl, theory) -> list:
    fn = getattr(cls, "update", None)
    if fn is None:
        return None
    nparams = len(inspect.signature(fn).parameters)

    # we assume all arguments are taken, otherwise it's too ambiguous
    args = tuple(expr(v, theory=theory) for v in ctrl + extl)
    if nparams != len(args) + 1:
        raise ValueError(f"update expects len(ctrl + extl) == {len(args)} params, got {nparams - 1}")

    with collecting() as terms:
        vals = _as_tuple(fn(None, *args))
        if len(vals) != len(ctrl):
            raise ValueError(f"update expects {len(ctrl)} return values, got {len(vals)}")
        for var, val in zip(ctrl, vals):
            e = val if isinstance(val, Expr) else expr(val, theory=theory, sort=var.dtype)
            terms.append(_Term(theory.Id(), [_X(var)], [e.wire]))
        return terms


def _build_delay_block(cls, ctrl, extl, theory) -> list:
    fn = getattr(cls, "delay", None)
    if fn is None:
        return None
    nparams = len(inspect.signature(fn).parameters)

    # we assume all arguments are taken, otherwise it's too ambiguous
    args = tuple(expr(v, theory=theory) for v in ctrl + extl)
    if nparams != len(args) + 1:
        raise ValueError(f"delay expects len(ctrl + extl) == {len(args)} params, got {nparams - 1}")

    with collecting() as terms:
        vals = _as_tuple(fn(None, *args))
        if len(vals) != len(ctrl):
            raise ValueError(f"update expects {len(ctrl)} return values, got {len(vals)}")
        for var, val in zip(ctrl, vals):
            e = val if isinstance(val, Expr) else expr(val, theory=theory, sort=var.dtype)
            terms.append(_Term(theory.Id(), [_d(var)], [e.wire]))
        return terms


class Module(_Module):
    def __new__(cls, ctrl=(), extl=(), theory=None):
        if theory is None or ctrl is None:
            raise TypeError(f"{cls.__name__}: `theory` is a required constructor arg")

        ctrl = tuple(v for v in ctrl)
        extl = tuple(v for v in extl) if extl is not None else ()

        init = _build_init_block(cls, ctrl, extl, theory)
        update = _build_update_block(cls, ctrl, extl, theory)
        delay = _build_delay_block(cls, ctrl, extl, theory)

        self = super().__new__(cls, init=init, update=update, delay=delay, obs=ctrl + extl)
        self._theory = theory
        return self
