"""dslModule: a reactive Module you write by subclassing (à la ``torch.nn.Module``).

Subclass ``dslModule``, pass a ``theory`` and the ``ctrl`` (and optional ``extl``)
variables — each a ``(latched, next)`` **wire pair** — and override the block methods
to return the next-state values as a tuple aligned with ``ctrl``. **Instantiating the
subclass *is* a base Module** — the block methods run and the module is built in the
constructor.

    from zrth import LIA, Sort, Wire
    from zrth.dsl import dslModule

    class Counter(dslModule):                   # sequential: has latched state
        def init(self):            return 0
        def update(self, ctrl):    return ctrl + 1

    INT = Sort.Int([1, 1])
    x = (Wire(INT), Wire(INT))                 # a (latched, next) wire pair
    m = Counter(theory=LIA, ctrl=(x,))         # a base Module with theory LIA (closed)

``ctrl`` / ``extl`` are tuples of wire pairs (a single var is unwrapped, so you can write
``def update(self, ctrl): return ctrl + 1``); a var reads as its latched value and
``nxt(v)`` gives its next wire. The block methods return a tuple aligned with ``ctrl``
— entry *i* is ``ctrl[i]``'s next value (return the var itself to keep it). ``extl`` vars
are external inputs: they are declared but their next value is *not* driven here (the base
Module classifies undriven wires as ``extl``, so a module with ``extl`` is open).

Two kinds of module are built here, chosen by which methods the subclass defines:

  * **sequential** — ``init`` (tick 0) + ``update`` (tick > 0). Has latched state:
    ``update`` may read the latched ``ctrl``; ``init`` may not (there is no previous
    tick to latch), so ``init`` reads only the awaited inputs.
  * **combinatorial** — ``assign`` only. Memoryless: it reads only the awaited inputs
    (``extl`` and their ``nxt`` wires) and has no ``init`` (no tick-0 state to set).

Config comes from **constructor kwargs** (``theory=``, ``ctrl=``, ``extl=``). Two base-class
constraints force this (both flagged for design review):
  * it cannot flow through ``super().__init__`` — the base Module is a frozen pyo3 class
    whose constructor runs at ``__new__`` (before any ``__init__``), so ``dslModule`` builds
    there; and
  * config *class attributes* named ``ctrl``/``extl`` would shadow ``Module``'s own
    ``ctrl``/``extl`` getters, so those names can't be reused declaratively.
"""

import inspect

from .zrth import Module, Term
from . import expr as E
from .expr import nxt, ite, eq, ne, const, relu, argmax, as_expr, Expr  # re-exported for authoring

# Public authoring surface: `from zrth.dsl import dslModule, nxt, ite, ...`
__all__ = ["dslModule", "nxt", "ite", "eq", "ne", "const", "relu", "argmax", "as_expr", "Expr"]


def _as_tuple(r) -> tuple:
    if r is None:
        return ()
    if isinstance(r, tuple):
        return r
    if isinstance(r, list):
        return tuple(r)
    return (r,)


def _invoke(fn, ctrl_arg, extl_arg, mode: str):
    """Call an init/update/assign method, passing args per its arity (self is unused
    at construction time, so it is passed as None). ``update`` may read the latched
    ``ctrl`` and the ``extl`` inputs; ``init`` and ``assign`` read only the awaited
    inputs (``extl``), since neither may read a latched wire."""
    nparams = len(inspect.signature(fn).parameters)  # includes `self`
    if mode == "update":
        args = (ctrl_arg, extl_arg) if nparams >= 3 else (ctrl_arg,)
    else:  # "init" or "assign": awaited inputs only
        args = (extl_arg,) if nparams >= 2 else ()
    return _as_tuple(fn(None, *args))


def _block_terms(theory, ctrl_vars: tuple, values: tuple) -> list:
    """Terms for one block: every value's terms (deps first, and sub-expressions shared
    across values emitted once), then an Id driving each ctrl var's next wire."""
    if len(values) != len(ctrl_vars):
        raise ValueError(f"expected {len(ctrl_vars)} return value(s), got {len(values)}")
    exprs = [as_expr(v, theory) for v in values]
    terms = E.collect_terms(*exprs)          # one shared pass -> a reused subterm appears once
    for var, e in zip(ctrl_vars, exprs):
        terms.append(Term(theory.Id(), [nxt(var).wire], [e.wire]))
    return terms


class dslModule(Module):
    def __new__(cls, *, theory=None, ctrl=None, extl=None, name=None):
        if theory is None or ctrl is None:
            raise TypeError(
                f"{cls.__name__}: `theory` and `ctrl` are required constructor kwargs"
            )

        ctrl_pairs = [tuple(p) for p in ctrl]
        extl_pairs = [tuple(p) for p in extl] if extl else []
        ctrl_vars = tuple(E.var(p, theory) for p in ctrl_pairs)
        extl_vars = tuple(E.var(p, theory) for p in extl_pairs)

        init_fn = getattr(cls, "init", None)
        update_fn = getattr(cls, "update", None)
        assign_fn = getattr(cls, "assign", None)

        # a single variable is unwrapped, so `x = ctrl` works as well as `x, y = ctrl`
        ctrl_arg = ctrl_vars[0] if len(ctrl_vars) == 1 else ctrl_vars
        extl_arg = extl_vars[0] if len(extl_vars) == 1 else extl_vars
        obs = [list(p) for p in (ctrl_pairs + extl_pairs)]

        if assign_fn is not None:
            # combinatorial (memoryless): a single `assign` block, no init/update
            if init_fn is not None or update_fn is not None:
                raise TypeError(
                    f"{cls.__name__}: define `assign` (combinatorial) OR `init`+`update` "
                    f"(sequential), not both")
            assign_terms = _block_terms(theory, ctrl_vars,
                                        _invoke(assign_fn, ctrl_arg, extl_arg, "assign"))
            self = super().__new__(cls, assign=assign_terms, obs=obs)
        else:
            # sequential: init (tick 0) + update (tick > 0)
            if init_fn is None:
                raise TypeError(
                    f"{cls.__name__} must define an `init` method (every ctrl variable needs "
                    f"an initial value), or an `assign` method for a combinatorial module")
            if update_fn is None:
                raise TypeError(f"{cls.__name__} must define an `update` method")
            init_terms = _block_terms(theory, ctrl_vars, _invoke(init_fn, ctrl_arg, extl_arg, "init"))
            update_terms = _block_terms(theory, ctrl_vars, _invoke(update_fn, ctrl_arg, extl_arg, "update"))
            self = super().__new__(cls, init=init_terms, update=update_terms, obs=obs)
        self._theory = theory
        self._ctrl_vars = ctrl_vars
        self._extl_vars = extl_vars
        return self
