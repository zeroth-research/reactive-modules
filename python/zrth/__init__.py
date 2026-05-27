from .zrth import (
    Wire,
    DType,
    IType as _IType,
    Term,
    Module,
)


#####################################################################
# IType façade with an explicit "current theory"
#####################################################################

# The current theory MUST be set via `set_theory(...)` before any
# theory-implicit access like `IType.Add` is performed.
_current_theory = None

_AVAILABLE_THEORIES = (_IType.LIA, _IType.LRA, _IType.BV)


def set_theory(theory):
    """Set the implicit theory used to resolve top-level `IType.<op>` access.

    `theory` must be one of `IType.LIA`, `IType.LRA`, `IType.BV`.

    After `set_theory(IType.LRA)`, `IType.Add` resolves to `IType.LRA.Add`,
    `IType.Lt` to `IType.LRA.Lt`, etc. The theory-specific spellings
    (`IType.LIA.X`, `IType.LRA.X`, `IType.BV.X`) always remain available
    regardless of the current theory.
    """
    assert theory in _AVAILABLE_THEORIES, (
        f"set_theory: {theory!r} is not one of "
        f"{[t.__name__ for t in _AVAILABLE_THEORIES]}"
    )
    global _current_theory
    _current_theory = theory


def get_theory():
    return _current_theory


def _require_theory(name):
    if _current_theory is None:
        raise RuntimeError(
            f"IType.{name}: no theory is set — call `set_theory(IType.LIA"
            f" | IType.LRA | IType.BV)` first"
        )


_ALIASES = {"Neq": "Ne"}


def _resolve_tensor(t):
    """Build a Const op for an arbitrary torch.Tensor under the current theory.

    Bool tensors always go to the theory's `ConstBool`. Other tensors are
    mapped to the theory's data-carrying constant: `ConstInt` for LIA,
    `ConstReal` for LRA, `Const` for BV.
    """
    import torch
    _require_theory("Tensor")
    if t.dtype == torch.bool:
        if _current_theory is _IType.BV:
            return _IType.BV.Const(t)
        return _current_theory.ConstBool(t)
    if _current_theory is _IType.LIA:
        return _IType.LIA.ConstInt(t)
    if _current_theory is _IType.LRA:
        return _IType.LRA.ConstReal(t)
    if _current_theory is _IType.BV:
        return _IType.BV.Const(t)
    raise RuntimeError(f"unsupported current theory: {_current_theory!r}")


class _ITypeMeta(type):
    # Make `isinstance(x, IType)` work for instances minted by the Rust class.
    def __instancecheck__(cls, instance):
        return isinstance(instance, _IType)

    def __getattr__(cls, name):
        if name == "Tensor":
            return _resolve_tensor
        name = _ALIASES.get(name, name)
        _require_theory(name)
        try:
            return getattr(_current_theory, name)
        except AttributeError as e:
            raise AttributeError(
                f"IType has no attribute {name!r} (current theory: {_current_theory})"
            ) from e


class IType(metaclass=_ITypeMeta):
    """Public IType façade.

    - `IType.LIA`, `IType.LRA`, `IType.BV` are theory-specific namespaces.
    - Anything else (`IType.Add`, `IType.Ite`, `IType.Lt`, ...) is delegated to
      the theory set via `set_theory(...)`. Without `set_theory` the access
      raises.
    - `isinstance(x, IType)` returns True for any instance produced by the
      underlying Rust class (i.e. by any of the namespace-qualified spellings).
    """
    LIA = _IType.LIA
    LRA = _IType.LRA
    BV = _IType.BV


#####################################################################
# DType convenience constructors
#####################################################################


def Bool(*shape):
    return DType.Bool([*shape] if shape else [1])


def Int(*shape):
    return DType.Int([*shape])


def Real(*shape):
    return DType.Real([*shape])


def Float(*args):
    return DType.Float([*args])


#####################################################################
# Derived-operation helpers
#
# These helpers lower a missing op to a sequence of primitive terms,
# using the IType facade (so they pick up the current theory). Each
# helper takes a caller-allocated output Wire plus input Wires and
# returns a list[Term] whose last write is the output.
#####################################################################


def xnor(out: Wire, a: Wire, b: Wire) -> list[Term]:
    """out = Xnor(a, b) ≡ Not(Xor(a, b))."""
    tmp = Wire(out.dtype)
    return [
        Term(IType.Xor, [tmp], [a, b]),
        Term(IType.Not, [out], [tmp]),
    ]


def implies(out: Wire, a: Wire, b: Wire) -> list[Term]:
    """out = Implies(a, b) ≡ Or(Not(a), b)."""
    nota = Wire(a.dtype)
    return [
        Term(IType.Not, [nota], [a]),
        Term(IType.Or, [out], [nota, b]),
    ]


# --- BV-only: two's-complement-derived ops -----------------------------------


def bv_neg(out: Wire, x: Wire) -> list[Term]:
    """out = -x via two's complement ≡ Add(Not(x), 1).

    Bit-width of the result is inferred from `out.dtype`. The intermediate
    constant 1 carries the same width via BV's type inference.
    """
    notx = Wire(x.dtype)
    one = Wire(out.dtype)
    return [
        Term(_IType.BV.Const(1), [one]),
        Term(_IType.BV.Not, [notx], [x]),
        Term(_IType.BV.Add, [out], [notx, one]),
    ]


def bv_sub(out: Wire, a: Wire, b: Wire) -> list[Term]:
    """out = a - b ≡ Add(a, bv_neg(b))."""
    negb = Wire(b.dtype)
    terms = bv_neg(negb, b)
    terms.append(Term(_IType.BV.Add, [out], [a, negb]))
    return terms


def bv_mod(out: Wire, a: Wire, b: Wire, *, signed: bool = False) -> list[Term]:
    """out = a mod b ≡ Sub(a, Mul(Div(a, b), b)).

    `signed=False` uses `UDiv`; `signed=True` uses `SDiv`.
    """
    div = Wire(out.dtype)
    prod = Wire(out.dtype)
    div_op = _IType.BV.SDiv if signed else _IType.BV.UDiv
    terms = [
        Term(div_op, [div], [a, b]),
        Term(_IType.BV.Mul, [prod], [div, b]),
    ]
    terms.extend(bv_sub(out, a, prod))
    return terms


from .gym import Wrapper, Env
from .smv import parse_smv
from .smt import z3

# Submodule access: from zrth.gym import Env
#                   from zrth.torch import Module
from . import gym as gym
from . import torch as torch


__all__ = [
    "Wire",
    "DType",
    "IType",
    "Term",
    "Module",
    "Env",
    "set_theory",
    "get_theory",
    "xnor",
    "implies",
    "bv_neg",
    "bv_sub",
    "bv_mod",
]
