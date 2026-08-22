"""``zrth.sugar``: author a reactive Module by subclassing (à la ``torch.nn.Module``).

Subclass ``sugar.Module``, pass a ``theory`` and the ``ctrl`` (and optional ``extl``)
variables — ``zrth.Var`` objects — and override any of ``init`` / ``update`` / ``delay``.
**Instantiating the subclass *is* a base Module** — the methods run symbolically and the
module is built in the constructor:

    from zrth import LIA, Int, Var
    from zrth.sugar import Module

    class Counter(Module):
        def init(self):         return 0
        def update(self, x):    return x + 1

    x = Var(Int([1, 1]))
    m = Counter(theory=LIA, ctrl=(x,))      # a closed sequential base Module

The methods receive the variables **unpacked, as positional parameters**: ``init`` takes
the ``extl`` variables, ``update`` and ``delay`` take ``ctrl`` followed by ``extl`` (all
of them — arities are checked). A parameter reads as the variable's latched value;
``X(v)`` is its next and ``d(v)`` its derivative expression. Each method returns a tuple
aligned with ``ctrl`` (a single value for a single variable): entry *i* drives
``ctrl[i]``'s next wire (``init`` / ``update``) or its derivative wire (``delay``).

The **combination of overridden methods selects the module kind**, mirroring the base
constructors: ``init``+``update`` is sequential, ``init``+``delay`` differential, all
three hybrid, ``update`` alone jump, ``update``+``delay`` uninitialized, ``delay`` alone
flow, ``init`` alone constant, and none hold. For example, an open hybrid clock:

    class Clock(Module):
        def init(self, t):        return 0
        def update(self, x, t):   return ite(X(t) == 1, 0, x)   # discrete reset
        def delay(self, x, t):    return 1 * d(t)               # continuous drift

    x, t = Var(Real([1, 1])), Var(Real([1, 1]))
    m = Clock(theory=LRA, ctrl=(x,), extl=(t,))

``ctrl`` and ``extl`` must be **ordered tuples**: their order is meaningful — it
determines how the variables bind to the positional parameters of ``init`` / ``update``
/ ``delay`` and how the returned values align with ``ctrl``. ``hide`` is only tested
for membership, so it can be any collection — preferably a ``set``. ``extl`` variables
are external inputs: declared but not driven here, so a module with ``extl`` is open;
``prvt`` hides the given variables in the built module (partially observable).

Config comes from **constructor kwargs** (``theory=``, ``ctrl=``, ``extl=``, ``hide=``).
Two base-class constraints force this (both flagged for design review):
  * it cannot flow through ``super().__init__`` — the base Module is a frozen pyo3 class
    whose constructor runs at ``__new__`` (before any ``__init__``), so this builds there;
    and
  * config *class attributes* named ``ctrl``/``extl`` would shadow the base Module's own
    ``ctrl``/``extl`` getters, so those names can't be reused declaratively.
"""

import inspect

from .zrth import Module as _Module, Term as _Term, X as base_X, d as base_d, Var

from .expr import expr, cast, ite, relu, argmax, collecting, Expr, X as expr_X, d as expr_d  # re-exported for authoring

# Public authoring surface: `from zrth.sugar import Module, expr, X, d, ite, cast, ...`
__all__ = ["Module", "expr", "cast", "ite", "relu", "argmax", "Expr", "X", "d", "Var"]


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
            terms.append(_Term(theory.Id(), [base_X(var)], [e.wire]))
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
            terms.append(_Term(theory.Id(), [base_X(var)], [e.wire]))
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
            raise ValueError(f"delay expects {len(ctrl)} return values, got {len(vals)}")
        for var, val in zip(ctrl, vals):
            e = val if isinstance(val, Expr) else expr(val, theory=theory, sort=var.dtype)
            terms.append(_Term(theory.Id(), [base_d(var)], [e.wire]))
        return terms


def X(var):
    if isinstance(var, Expr):
        return expr_X(var)
    elif isinstance(var, Var):
        return base_X(var)
    else:
        raise ValueError(f"sugar.X expects Expr or Var, got {var}")


def d(var):
    if isinstance(var, Expr):
        return expr_d(var)
    elif isinstance(var, Var):
        return base_d(var)
    else:
        raise ValueError(f"sugar.d expects Expr or Var, got {var}")


class Module(_Module):
    def __new__(cls, ctrl=(), extl=(), hide=frozenset(), theory=None):
        if theory is None:
            raise TypeError(f"{cls.__name__}: `theory` is a required constructor arg")

        ctrl = tuple(v for v in ctrl)
        extl = tuple(v for v in extl)

        init = _build_init_block(cls, ctrl, extl, theory)
        update = _build_update_block(cls, ctrl, extl, theory)
        delay = _build_delay_block(cls, ctrl, extl, theory)

        self = super().__new__(
            cls, init=init, update=update, delay=delay, vars=ctrl + extl, hide=hide
        )
        self._theory = theory
        return self
