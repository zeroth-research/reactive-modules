from typing import override

from .zrth import DType, IType, Term, Wire
from .builder import TermBuilder, builder_for
import torch

# types that we can convert to [Expr]
type ToExpr = Expr | int | bool | float | torch.Tensor


# ---------------------------------------------------------------------------
# Expr class hierarchy
# ---------------------------------------------------------------------------


class Expr:
    """Base class for symbolic expressions. All instances are created via _expr_from_term."""

    @property
    def builder(self) -> TermBuilder:
        return self._builder

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def wire(self):
        return self._wire

    @property
    def shape(self):
        return self._shape

    @property
    def args(self):
        return self._args

    @property
    def itype(self):
        return self._itype

    @override
    def __str__(self) -> str:
        return (
            f"{self._itype}({', '.join(map(str, self._args))})"
            if self._args
            else f"{self._itype}"
        )


class BExpr(Expr):
    def __and__(self, rhs: "BExpr") -> "BExpr":
        return conj(self, rhs)

    def __or__(self, rhs: "BExpr") -> "BExpr":
        return disj(self, rhs)

    def __invert__(self) -> "BExpr":
        return neg(self)


class AExpr(Expr):
    def __add__(self, other: "AExpr") -> "AExpr":
        return add(self, other)

    def __mul__(self, other: "AExpr") -> "AExpr":
        return mul(self, other)

    def __sub__(self, other: "AExpr") -> "AExpr":
        return sub(self, other)

    def __truediv__(self, other: "AExpr") -> "AExpr":
        return div(self, other)

    def __lt__(self, other: "AExpr") -> BExpr:
        return lt(self, other)

    def __gt__(self, other: "AExpr") -> BExpr:
        return gt(self, other)

    def __le__(self, other: "AExpr") -> BExpr:
        return le(self, other)

    def __ge__(self, other: "AExpr") -> BExpr:
        return ge(self, other)

    def __eq__(self, other: "AExpr") -> BExpr:
        return eq(self, other)

    def __ne__(self, other: "AExpr") -> BExpr:
        return neq(self, other)

    def __matmul__(self, other):
        return matmul(self, other)


class WExpr(Expr):
    """Expression class for word-level (bitvector) types."""

    def __add__(self, other: "WExpr") -> "WExpr":
        return w_add(self, other)

    def __sub__(self, other: "WExpr") -> "WExpr":
        return w_sub(self, other)

    def __mul__(self, other: "WExpr") -> "WExpr":
        return w_mul(self, other)

    def __truediv__(self, other: "WExpr") -> "WExpr":
        return w_div(self, other)

    def __mod__(self, other: "WExpr") -> "WExpr":
        return w_mod(self, other)

    def __neg__(self) -> "WExpr":
        return w_neg(self)

    def __abs__(self) -> "WExpr":
        return w_abs(self)

    def __and__(self, other: "WExpr") -> "WExpr":
        return w_and(self, other)

    def __or__(self, other: "WExpr") -> "WExpr":
        return w_or(self, other)

    def __xor__(self, other: "WExpr") -> "WExpr":
        return w_xor(self, other)

    def __invert__(self) -> "WExpr":
        return w_not(self)

    def __lt__(self, other: "WExpr") -> BExpr:
        return w_lt(self, other)

    def __gt__(self, other: "WExpr") -> BExpr:
        return w_gt(self, other)

    def __le__(self, other: "WExpr") -> BExpr:
        return w_le(self, other)

    def __ge__(self, other: "WExpr") -> BExpr:
        return w_ge(self, other)

    def __eq__(self, other: "WExpr") -> BExpr:
        return w_eq(self, other)

    def __ne__(self, other: "WExpr") -> BExpr:
        return w_neq(self, other)


# ---------------------------------------------------------------------------
# Core factory
# ---------------------------------------------------------------------------


def _expr_from_term(term: Term, builder: TermBuilder, *args: Expr) -> Expr:
    """Create the right Expr subclass from an already-constructed Term."""
    dtype = term.write[0].dtype
    if dtype.is_bool():
        cls = BExpr
    elif dtype.is_real() or dtype.is_int():
        cls = AExpr
    else:
        cls = WExpr
    e = cls.__new__(cls)
    e._term = term
    e._builder = builder
    e._itype = term.itype
    e._wire = term.write[0]
    e._dtype = dtype
    e._shape = dtype.shape
    e._args = list(args)
    return e


# ---------------------------------------------------------------------------
# Boolean operators
# ---------------------------------------------------------------------------


def conj(first: BExpr, *others) -> BExpr:
    b = first._builder
    acc = first
    for other in others:
        acc = _expr_from_term(b.and_(acc.wire, other.wire), b, acc, other)
    return acc


def disj(first: BExpr, *others) -> BExpr:
    b = first._builder
    acc = first
    for other in others:
        acc = _expr_from_term(b.or_(acc.wire, other.wire), b, acc, other)
    return acc


def neg(e: BExpr) -> BExpr:
    return _expr_from_term(e._builder.not_(e.wire), e._builder, e)


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------


def add(first: AExpr, *others) -> AExpr:
    b = first._builder
    acc = first
    for other in others:
        acc = _expr_from_term(b.add(acc.wire, other.wire), b, acc, other)
    return acc


def sub(lhs: AExpr, rhs: AExpr) -> AExpr:
    b = lhs._builder
    return _expr_from_term(b.sub(lhs.wire, rhs.wire), b, lhs, rhs)


def mul(first: AExpr, *others) -> AExpr:
    b = first._builder
    acc = first
    for other in others:
        acc = _expr_from_term(b.mul(acc, other.wire), b, acc, other)
    return acc


def div(num: AExpr, den: AExpr) -> AExpr:
    b = num._builder
    return _expr_from_term(b.div(num.wire, den.wire), b, num, den)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def lt(lhs: AExpr, rhs: AExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.lt(lhs.wire, rhs.wire), b, lhs, rhs)


def gt(lhs: AExpr, rhs: AExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.gt(lhs.wire, rhs.wire), b, lhs, rhs)


def ge(lhs: AExpr, rhs: AExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.ge(lhs.wire, rhs.wire), b, lhs, rhs)


def le(lhs: AExpr, rhs: AExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.le(lhs.wire, rhs.wire), b, lhs, rhs)


def eq(lhs: AExpr, rhs: AExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.eq(lhs.wire, rhs.wire), b, lhs, rhs)


def neq(lhs: AExpr, rhs: AExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.ne(lhs.wire, rhs.wire), b, lhs, rhs)


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


def ite(cond: BExpr, iftrue: Expr, iffalse: Expr) -> Expr:
    if not isinstance(iftrue, (BExpr, AExpr)):
        raise ValueError(f"ite: expected AExpr or BExpr, got {type(iftrue).__name__}")
    if type(iftrue) is not type(iffalse):
        raise ValueError(f"ite: branch types must match, got {type(iftrue).__name__} and {type(iffalse).__name__}")
    b = iftrue._builder
    return _expr_from_term(b.ite(cond.wire, iftrue.wire, iffalse.wire), b, cond, iftrue, iffalse)


# ---------------------------------------------------------------------------
# Tensor operations
# ---------------------------------------------------------------------------


def argmax(arg: Expr) -> Expr:
    return _expr_from_term(arg._builder.argmax(arg.wire), arg._builder, arg)


def matmul(lhs: Expr, rhs: Expr) -> Expr:
    term = lhs._builder.matmul(lhs._term, rhs.wire)
    return _expr_from_term(term, lhs._builder, lhs, rhs)


# ---------------------------------------------------------------------------
# Word-level (BV) operators
# ---------------------------------------------------------------------------


def w_add(first: WExpr, *others) -> WExpr:
    b = first._builder
    acc = first
    for other in others:
        acc = _expr_from_term(b.add(acc.wire, other.wire), b, acc, other)
    return acc


def w_sub(first: WExpr, *others) -> WExpr:
    b = first._builder
    acc = first
    for other in others:
        acc = _expr_from_term(b.sub(acc.wire, other.wire), b, acc, other)
    return acc


def w_mul(first: WExpr, *others) -> WExpr:
    b = first._builder
    acc = first
    for other in others:
        acc = _expr_from_term(b.mul(acc.wire, other.wire), b, acc, other)
    return acc


def w_div(num: WExpr, den: WExpr) -> WExpr:
    b = num._builder
    return _expr_from_term(b.div(num.wire, den.wire), b, num, den)


def w_mod(num: WExpr, den: WExpr) -> WExpr:
    b = num._builder
    return _expr_from_term(b.div(num.wire, den.wire), b, num, den)  # TODO: BV.Mod


def w_neg(e: WExpr) -> WExpr:
    return _expr_from_term(e._builder.neg(e.wire), e._builder, e)


def w_abs(e: WExpr) -> WExpr:
    return _expr_from_term(e._builder.abs_(e.wire), e._builder, e)


def w_and(first: WExpr, *others) -> WExpr:
    b = first._builder
    acc = first
    for other in others:
        acc = _expr_from_term(b.and_(acc.wire, other.wire), b, acc, other)
    return acc


def w_or(first: WExpr, *others) -> WExpr:
    b = first._builder
    acc = first
    for other in others:
        acc = _expr_from_term(b.or_(acc.wire, other.wire), b, acc, other)
    return acc


def w_xor(first: WExpr, *others) -> WExpr:
    b = first._builder
    acc = first
    for other in others:
        acc = _expr_from_term(b.xor_(acc.wire, other.wire), b, acc, other)
    return acc


def w_not(e: WExpr) -> WExpr:
    return _expr_from_term(e._builder.not_(e.wire), e._builder, e)


def w_lt(lhs: WExpr, rhs: WExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.lt(lhs.wire, rhs.wire), b, lhs, rhs)


def w_gt(lhs: WExpr, rhs: WExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.gt(lhs.wire, rhs.wire), b, lhs, rhs)


def w_le(lhs: WExpr, rhs: WExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.le(lhs.wire, rhs.wire), b, lhs, rhs)


def w_ge(lhs: WExpr, rhs: WExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.ge(lhs.wire, rhs.wire), b, lhs, rhs)


def w_eq(lhs: WExpr, rhs: WExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.eq(lhs.wire, rhs.wire), b, lhs, rhs)


def w_neq(lhs: WExpr, rhs: WExpr) -> BExpr:
    b = lhs._builder
    return _expr_from_term(b.ne(lhs.wire, rhs.wire), b, lhs, rhs)


# ---------------------------------------------------------------------------
# Terminals — theory is required
# ---------------------------------------------------------------------------


def Bool(x: bool | str | torch.Tensor, theory, shape=None) -> BExpr:
    b = builder_for(theory)
    if isinstance(x, bool):
        return _expr_from_term(b.const_bool(x), b)
    elif isinstance(x, torch.Tensor):
        return _expr_from_term(b.const(x), b)
    elif isinstance(x, str):
        return _expr_from_term(b.uninterpreted(x, DType.Bool(shape or [1])), b)
    raise ValueError(f"Invalid argument to Bool: {type(x).__name__}")


def Real(x: float | str | torch.Tensor, theory, shape=None) -> AExpr:
    b = builder_for(theory)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return _expr_from_term(b.const_for_value(float(x)), b)
    elif isinstance(x, torch.Tensor):
        return _expr_from_term(b.const(x), b)
    elif isinstance(x, str):
        return _expr_from_term(b.uninterpreted(x, DType.Float(shape or [1])), b)
    raise ValueError(f"Invalid argument to Real: {type(x).__name__}")
