from typing import Any, override
import torch
from torch.utils.checkpoint import checkpoint

from .zrth import DType, IType, Term, Wire
from .context import get_ctx, Context


# types that we can convert to [Expr]
type ToExpr = Expr | int | bool | float | torch.Tensor


class Expr:
    def __init__(self, itype: IType, dtype: DType, *args, ctx):
        assert all(isinstance(arg, Expr) for arg in args)
        # super().__init__(self, itype, [ctx.tmp_wire(self._dtype)], [a.wire() for a in args])
        self._term = Term(itype, [ctx.tmp_wire(dtype)], [a.wire for a in args])

        self._ctx = ctx

        self._itype = itype

        self._wire = self._term.write()[0]
        self._dtype = self._wire.dtype()
        self._shape = self._dtype.shape

        self._args = [*args]

        return self

    @property
    def ctx(self) -> Context:
        return self._ctx

    @property
    def dtype(self):
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

    # def argmax(self) -> "Expr":
    #     return Argmax(self)

    @property
    def itype(self):
        return self._itype

    def __and__(self, rhs: "Expr") -> "Expr":
        return conj(self, rhs)

    def __or__(self, rhs: "Expr") -> "Expr":
        return disj(self, rhs)

    def __invert__(self) -> "Expr":
        return neg(self)

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


def elementwise_op(itype: IType, wtype: DType, rtype: DType, first, *others):
    if not isinstance(first, Expr):
        raise Exception("type coercion unsupported")
    ctx = first.ctx
    shape = first.shape
    for arg in others:
        if not isinstance(arg, Expr):
            raise Exception("type coercion unsupported")
        if arg.ctx != ctx:
            raise Exception("ctx mismatch")
        if shape != arg.shape:
            raise Exception("size mismatch")
        if rtype != arg.dtype:
            raise Exception("dtype mismatch")

    match wtype:
        case DType.Bool():
            return BExpr(itype, wtype, first, *others, ctx=ctx)
        case DType.Real():
            return AExpr(itype, wtype, first, *others, ctx=ctx)
        case DType.Int():
            return AExpr(itype, wtype, first, *others, ctx=ctx)
        case _:
            raise NotImplementedError


# ========================================
# Elementwise logical operators
# ========================================


def elementwise_bool_op(itype, first: BExpr, *others):
    if not isinstance(first.dtype, DType.Bool):
        raise Exception("invalid dtype")

    return elementwise_op(itype, first.dtype, first.dtype, first, *others)


# Logical or (we have to avoid clash with Python keyword 'or')
def disj(first: BExpr, *others) -> BExpr:
    return elementwise_bool_op(IType.Or(), first, *others)


# Logical and
def conj(first: BExpr, *others) -> BExpr:
    return elementwise_bool_op(IType.And(), first, *others)


# Logical not
def neg(e: BExpr) -> BExpr:
    if not isinstance(e.dtype, DType.Bool):
        raise Exception("invalid dtype")

    return Expr(IType.Not, e.dtype, e, ctx=e.ctx())


# ========================================
# Elementwise arithmetics operators
# ========================================


def elementwise_arith_op(itype, first: Expr, *others):
    if not isinstance(first.dtype, (DType.Real, DType.Int)):
        raise Exception("invalid dtype")

    return elementwise_op(itype, first.dtype, first.dtype, first, *others)


def add(first: AExpr, *others) -> AExpr:
    return elementwise_arith_op(IType.Add(), first, *others)


def mul(first: AExpr, *others) -> AExpr:
    return elementwise_arith_op(IType.Mul(), first, *others)


def div(num: AExpr, den: AExpr) -> AExpr:
    return elementwise_arith_op(IType.Div(), num, den)


def sub(min: AExpr, sub: AExpr) -> AExpr:
    return elementwise_arith_op(IType.Sub(), min, sub)


# ========================================
# Elementwise predicates
# ========================================

def elementwise_predicate(itype, lhs: AExpr, rhs: AExpr):
    if not isinstance(lhs.dtype, (DType.Real, DType.Int)):
        raise Exception("invalid dtype")
    if not isinstance(rhs.dtype, (DType.Real, DType.Int)):
        raise Exception("invalid dtype")

    return elementwise_op(itype, DType.Bool(lhs.shape), lhs.dtype, lhs, rhs)


def lt(lhs: AExpr, rhs: AExpr) -> BExpr:
    return elementwise_predicate(IType.Lt(), lhs, rhs)


def gt(lhs: AExpr, rhs: AExpr) -> BExpr:
    return elementwise_predicate(IType.Gt(), lhs, rhs)


def ge(lhs: AExpr, rhs: AExpr) -> BExpr:
    return elementwise_predicate(IType.Ge(), lhs, rhs)


def le(lhs: AExpr, rhs: AExpr) -> BExpr:
    return elementwise_predicate(IType.Le(), lhs, rhs)


def eq(lhs: AExpr, rhs: AExpr) -> BExpr:
    return elementwise_predicate(IType.Eq(), lhs, rhs)


def neq(lhs: AExpr, rhs: AExpr) -> BExpr:
    return elementwise_predicate(IType.Neq(), lhs, rhs)


# ========================================
# Terminals
# ========================================


def Bool(x: bool | str | torch.Tensor, shape=None, ctx=_global_context) -> BExpr:
    if isinstance(x, bool):
        assert shape is None
        dtype = DType.Bool([1])
        t = torch.Tensor([x])
        return BExpr(itype=IType.Tensor(t), dtype=dtype, ctx=ctx)
    elif isinstance(x, torch.Tensor):
        assert shape is None or shape == x.shape
        assert x.dtype == torch.bool
        dtype = DType.Bool(x.shape)
        return BExpr(itype=IType.Tensor(x), dtype=dtype, ctx=ctx)
    elif isinstance(x, str):
        # register symbol into context
        dtype = DType.Bool(shape if shape is not None else [1])
        ctx.declare_const(x, dtype)
        return BExpr(itype=IType.Uninterpreted(x), dtype=dtype, ctx=ctx)


def Real(x: float | str | torch.Tensor, shape=None, ctx=_global_context) -> AExpr:
    if isinstance(x, float):
        assert shape is None
        dtype = DType.Real([1])
        t = torch.Tensor([x])
        return AExpr(itype=IType.Tensor(t), dtype=dtype, ctx=ctx)
    elif isinstance(x, torch.Tensor):
        assert shape is None or shape == x.shape
        dtype = DType.Real(x.shape)
        return AExpr(itype=IType.Tensor(x), dtype=dtype, ctx=ctx)
    elif isinstance(x, str):
        # register symbol into context
        dtype = DType.Real(shape if shape is not None else [1])
        (ctx or get_ctx()).declare_const(x, dtype)
        return AExpr(itype=IType.Uninterpreted(x), dtype=dtype, ctx=ctx)
