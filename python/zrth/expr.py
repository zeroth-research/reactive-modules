from typing import Any, override
import torch
from torch.utils.checkpoint import checkpoint

from .zrth import DType, IType, Term, Wire
from .context import get_ctx, Context


# types that we can convert to [Expr]
type ToExpr = Expr | int | bool | float | torch.Tensor


class Expr:
    def __init__(self, itype: IType, dtype: DType, *args, ctx):
        # super().__init__(self, itype, [ctx.tmp_wire(self._dtype)], [a.wire() for a in args])
        self._term = Term(itype, [ctx.tmp_wire(dtype)], [a.wire for a in args])
        self._ctx = ctx

        self._itype = itype

        self._wire = self._term.write()[0]
        self._dtype = self._wire.dtype()
        self._shape = self._dtype.shape

        self._args = [*args]

    @property
    def ctx(self):
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


def elementwise_op(itype, first, *others):
    if not isinstance(first, Expr):
        raise Exception("type coercion unsupported")
    dtype = first.dtype
    ctx = first.ctx
    shape = first.shape
    for arg in others:
        if not isinstance(arg, Expr):
            raise Exception("type coercion unsupported")

        if arg.ctx != ctx:
            raise Exception("ctx mismatch")

        if shape != arg.shape:
            raise Exception("size mismatch")
        elif dtype != arg.dtype:
            raise Exception("dtype mismatch")

    return Expr(itype, dtype, first, *others, ctx=ctx)


# ========================================
# Elementwise logical operators
# ========================================


def elementwise_bool_op(itype, first: Expr, *others):
    if not isinstance(first.dtype, DType.Bool):
        raise Exception("invalid dtype")

    return elementwise_op(itype, first, *others)


# Logical or (we have to avoid clash with Python keyword 'or')
def disj(first: Expr, *others) -> Expr:
    return elementwise_bool_op(IType.Or(), first, *others)


# Logical and
def conj(first: Expr, *others) -> Expr:
    return elementwise_bool_op(IType.And(), first, *others)


# Logical not
def neg(e: Expr) -> Expr:
    if not isinstance(e.dtype(), DType.TensorBool):
        raise Exception("invalid dtype")

    return Expr(IType.Not, e.dtype(), e, ctx=e.ctx())


# ========================================
# Terminals
# ========================================


def Bool(x: bool | str | torch.Tensor, shape=None, ctx=_global_context):
    if isinstance(x, bool):
        assert shape is None
        dtype = DType.Bool([1])
        t = torch.Tensor([x])
        return Expr(itype=IType.Tensor(t), dtype=dtype, ctx=ctx)
    elif isinstance(x, torch.Tensor):
        assert shape is None or shape == x.shape
        assert x.dtype == torch.bool
        dtype = DType.Bool(x.shape)
        return Expr(itype=IType.Tensor(x), dtype=dtype, ctx=ctx)
    elif isinstance(x, str):
        # register symbol into context
        dtype = DType.Bool(shape if shape is not None else [1])
        ctx.declare_const(x, dtype)
        return Expr(itype=IType.Uninterpreted(x), dtype=dtype, ctx=ctx)


def Real(x: float | str | torch.Tensor, shape=None, ctx=_global_context):
    if isinstance(x, float):
        assert shape is None
        dtype = DType.Real([1])
        t = torch.Tensor([x])
        return Expr(itype=IType.Tensor(t), dtype=dtype, ctx=ctx)
    elif isinstance(x, torch.Tensor):
        assert shape is None or shape == x.shape
        dtype = DType.Real(x.shape)
        return Expr(itype=IType.Tensor(x), dtype=dtype, ctx=ctx)
    elif isinstance(x, str):
        # register symbol into context
        dtype = DType.Real(shape if shape is not None else [1])
        (ctx or get_ctx()).declare_const(x, dtype)
        return Expr(itype=IType.Uninterpreted(x), dtype=dtype, ctx=ctx)
