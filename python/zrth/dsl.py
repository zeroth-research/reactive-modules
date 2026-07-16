"""dslModule: a reactive Module you write by subclassing (à la ``torch.nn.Module``).

Subclass ``dslModule``, pass a ``theory`` and the ``ctrl`` (and optional ``extl``)
variables — each a ``(latched, next)`` **wire pair** — and override ``init`` / ``update``
to return the next-state values as a tuple aligned with ``ctrl``. **Instantiating the
subclass *is* a base Module** — ``init`` / ``update`` run and the sequential module is
built in the constructor.

    from zrth import LIA, Sort, Wire
    from zrth.dsl import dslModule

    class Counter(dslModule):
        def init(self):            return 0
        def update(self, ctrl):    return ctrl + 1

    INT = Sort.Int([1, 1])
    x = (Wire(INT), Wire(INT))                 # a (latched, next) wire pair
    m = Counter(theory=LIA, ctrl=(x,))         # a base Module with theory LIA (closed)

``ctrl`` / ``extl`` are tuples of wire pairs (a single var is unwrapped, so you can write
``def update(self, ctrl): return ctrl + 1``); a var reads as its latched value and
``nxt(v)`` gives its next wire. ``init`` / ``update`` return a tuple aligned with ``ctrl``
— entry *i* is ``ctrl[i]``'s next value (return the var itself to keep it). ``extl`` vars
are external inputs: they are declared but their next value is *not* driven here (the base
Module classifies undriven wires as ``extl``, so a module with ``extl`` is open).

Only **sequential** modules are built here; the base ``Module`` also has ``combinatorial`` /
``parallel``, which the DSL does not expose yet (combinatorial logic may instead come from the
torch front-end).

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
from .expr import expr, cast, nxt, ite, eq, ne, relu, argmax, collecting, Expr  # re-exported for authoring

# Public authoring surface: `from zrth.dsl import dslModule, expr, nxt, ite, cast, ...`
__all__ = ["dslModule", "expr", "cast", "nxt", "ite", "eq", "ne", "relu", "argmax", "Expr"]


def _as_tuple(r) -> tuple:
    if r is None:
        return ()
    if isinstance(r, tuple):
        return r
    if isinstance(r, list):
        return tuple(r)
    return (r,)


def _invoke(fn, ctrl_arg, extl_arg, is_init: bool):
    """Call an init/update method, passing ctrl/extl per its arity (self is unused
    at construction time, so it is passed as None)."""
    nparams = len(inspect.signature(fn).parameters)  # includes `self`
    if is_init:
        args = (extl_arg,) if nparams >= 2 else ()
    else:
        args = (ctrl_arg, extl_arg) if nparams >= 3 else (ctrl_arg,)
    return _as_tuple(fn(None, *args))


def _build_block(theory, ctrl_vars, fn, ctrl_arg, extl_arg, is_init) -> list:
    """Run one block (init/update) inside a collector: every Term built while the user
    method runs is captured in dependency order (a shared sub-expression once); then an
    Id drives each ctrl var's next wire from its returned value."""
    with collecting() as terms:
        values = _invoke(fn, ctrl_arg, extl_arg, is_init)
        if len(values) != len(ctrl_vars):
            raise ValueError(f"expected {len(ctrl_vars)} return value(s), got {len(values)}")
        for var, v in zip(ctrl_vars, values):
            e = v if isinstance(v, Expr) else expr(v, theory=theory, sort=var.dtype)
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
        ctrl_vars = tuple(expr(p, theory=theory) for p in ctrl_pairs)
        extl_vars = tuple(expr(p, theory=theory) for p in extl_pairs)

        init_fn = getattr(cls, "init", None)
        update_fn = getattr(cls, "update", None)
        if init_fn is None:
            raise TypeError(f"{cls.__name__} must define an `init` method (every ctrl variable needs an initial value)")
        if update_fn is None:
            raise TypeError(f"{cls.__name__} must define an `update` method")

        # a single variable is unwrapped, so `x = ctrl` works as well as `x, y = ctrl`
        ctrl_arg = ctrl_vars[0] if len(ctrl_vars) == 1 else ctrl_vars
        extl_arg = extl_vars[0] if len(extl_vars) == 1 else extl_vars

        init_terms = _build_block(theory, ctrl_vars, init_fn, ctrl_arg, extl_arg, True)
        update_terms = _build_block(theory, ctrl_vars, update_fn, ctrl_arg, extl_arg, False)

        obs = [list(p) for p in (ctrl_pairs + extl_pairs)]
        self = super().__new__(cls, init=init_terms, update=update_terms, obs=obs)
        self._theory = theory
        self._ctrl_vars = ctrl_vars
        self._extl_vars = extl_vars
        return self
