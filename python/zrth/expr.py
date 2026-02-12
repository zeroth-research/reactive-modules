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
        self._term = Term(itype, [ctx.tmp_wire(dtype)], [a.wire() for a in args])
        self._ctx = ctx

        self._itype = itype

        self._wire = self._term.write()[0]
        self._dtype = self._wire.dtype()
        self._dims = self._dtype.dims()

        self._args = [*args]

    def ctx(self) -> Context:
        return self._ctx

    def dtype(self):
        return self._dtype

    def itype(self):
        return self._itype

    def wire(self):
        return self._wire

    def dims(self):
        return self._dims

    def args(self):
        return self._args

    # def argmax(self) -> "Expr":
    #     return Argmax(self)

    def __and__(self, rhs: "Expr") -> "Expr":
        return conj(self, rhs)

    def __or__(self, rhs: "Expr") -> "Expr":
        return disj(self, rhs)

    def __invert__(self) -> "Expr":
        return neg(self)

    @override
    def __str__(self) -> str:
        return (
            f"{self.itype()}({', '.join(map(str, self._args))})"
            if self._args
            else f"{self.itype()}"
        )


def elementwise_op(itype, first, *others):
    if not isinstance(first, Expr):
        raise Exception("type coercion unsupported")
    dtype = first.dtype()
    ctx = first.ctx()
    dims = dtype.dims()
    for arg in others:
        if not isinstance(arg, Expr):
            raise Exception("type coercion unsupported")

        if arg.ctx() != ctx:
            raise Exception("ctx mismatch")

        if dims != arg.dims():
            raise Exception("size mismatch")
        elif dtype != arg.dtype():
            raise Exception("dtype mismatch")

    return Expr(itype, dtype, first, *others, ctx=ctx)


# ========================================
# Elementwise logical operators
# ========================================


def elementwise_bool_op(itype, first: Expr, *others):
    if not isinstance(first.dtype(), DType.Bool):
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


def Bool(x: bool | str | torch.Tensor, ctx=_global_context):
    if isinstance(x, bool):
        dtype = DType.Bool([1])
        t = torch.Tensor([x])
        return Expr(itype=IType.Tensor(t), dtype=dtype, ctx=ctx)
    elif isinstance(x, torch.Tensor):
        dtype = DType.Bool(x.size())
        return Expr(itype=IType.Tensor(x), dtype=dtype, ctx=ctx)
    elif isinstance(x, str):
        # register symbol into context
        dtype = DType.Bool([1])
        ctx.declare_const(x, dtype)
        return Expr(itype=IType.Uninterpreted(x), dtype=dtype, ctx=ctx)


def Real(x: float | str | torch.Tensor, ctx=None):
    if isinstance(x, float):
        dtype = DType.TensorReal([1])
        t = torch.Tensor([x])
        return Expr(itype=IType.Tensor(t), dtype=dtype, ctx=ctx)
    elif isinstance(x, torch.Tensor):
        dtype = DType.TensorReal(x.size())
        return Expr(itype=IType.Tensor(x), dtype=dtype, ctx=ctx)
    elif isinstance(x, str):
        # register symbol into context
        dtype = DType.TensorReal([1])
        (ctx or get_ctx()).declare_const(x, dtype)
        return Expr(itype=IType.Uninterpreted(x), dtype=dtype, ctx=ctx)
