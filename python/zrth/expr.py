from typing import Any, override

from .zrth import DType, IType, Term, Wire
import torch


# types that we can convert to [Expr]
type ToExpr = Expr | int | bool | float | torch.Tensor


class Expr:
    def __init__(self, itype: IType, dtype: DType, *args):
        assert all(isinstance(arg, Expr) for arg in args)
        self._term = Term(itype, [Wire(dtype)], [a.wire for a in args])

        self._itype = itype

        self._wire = self._term.write[0]
        self._dtype = self._wire.dtype()
        self._shape = self._dtype.shape

        self._args = [*args]

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

    # def __and__(self, rhs: "Expr") -> "Expr":
    #    return conj(self, rhs)
    #
    # def __or__(self, rhs: "Expr") -> "Expr":
    #    return disj(self, rhs)
    #
    # def __invert__(self) -> "Expr":
    #    return neg(self)

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

    def __eq__(self, other: "AExpr") -> BExpr:  # ty: ignore
        return eq(self, other)

    def __ne__(self, other: "AExpr") -> BExpr:  # ty: ignore
        return neq(self, other)

    def __matmul__(self, other):
        return matmul(self, other)


def _elementwise_op(itype: IType, wtype: DType, rtype: DType, first, *others):
    if not isinstance(first, Expr):
        raise Exception("type coercion unsupported")
    shape = first.shape
    for arg in others:
        if not isinstance(arg, Expr):
            raise Exception("type coercion unsupported")
        if shape != arg.shape:
            raise Exception("size mismatch")
        if rtype != arg.dtype:
            raise Exception("dtype mismatch")

    match wtype:
        case DType.Bool(_):
            return BExpr(itype, wtype, first, *others)
        case DType.Real(_):
            return AExpr(itype, wtype, first, *others)
        case DType.Int(_):
            return AExpr(itype, wtype, first, *others)
        case _:
            raise NotImplementedError


# ========================================
# Elementwise logical operators
# ========================================


def _elementwise_bool_op(itype, first: BExpr, *others):
    if first.dtype.kind() != "Bool":
        raise ValueError(f"invalid dtype, expected Bool(...), got: `{first.dtype}`")

    return _elementwise_op(itype, first.dtype, first.dtype, first, *others)


# Logical or (we have to avoid clash with Python keyword 'or')
def disj(first: BExpr, *others) -> BExpr:
    return _elementwise_bool_op(IType.Or(), first, *others)


# Logical and
def conj(first: BExpr, *others) -> BExpr:
    return _elementwise_bool_op(IType.And(), first, *others)


# Logical not
def neg(e: BExpr) -> BExpr:
    if e.dtype.kind() != "Bool":
        raise ValueError(f"invalid dtype, expected Bool(...), got: `{first.dtype}`")

    return BExpr(IType.Not(), e.dtype, e)


# ========================================
# Elementwise arithmetics operators
# ========================================


def _elementwise_arith_op(itype, first: Expr, *others):
    if not isinstance(first.dtype, (DType.Real, DType.Int)):
        raise Exception("invalid dtype")

    return _elementwise_op(itype, first.dtype, first.dtype, first, *others)


def add(first: AExpr, *others) -> AExpr:
    return _elementwise_arith_op(IType.Add(), first, *others)


def mul(first: AExpr, *others) -> AExpr:
    return _elementwise_arith_op(IType.Mul(), first, *others)


def div(num: AExpr, den: AExpr) -> AExpr:
    return _elementwise_arith_op(IType.Div(), num, den)


def sub(min: AExpr, sub: AExpr) -> AExpr:
    return _elementwise_arith_op(IType.Sub(), min, sub)


# ========================================
# Elementwise predicates
# ========================================


def _elementwise_predicate(itype, lhs: AExpr, rhs: AExpr):
    if not isinstance(lhs.dtype, (DType.Real, DType.Int)):
        raise Exception("invalid dtype")
    if not isinstance(rhs.dtype, (DType.Real, DType.Int)):
        raise Exception("invalid dtype")

    return _elementwise_op(itype, DType.Bool(lhs.shape), lhs.dtype, lhs, rhs)


def lt(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Lt(), lhs, rhs)


def gt(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Gt(), lhs, rhs)


def ge(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Ge(), lhs, rhs)


def le(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Le(), lhs, rhs)


def eq(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Eq(), lhs, rhs)


def neq(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Neq(), lhs, rhs)


# ========================================
# Control Flow
# ========================================


def ite(cond: BExpr, iftrue: Expr, iffalse: Expr):
    if not cond.shape == [1]:
        raise Exception("invalid Boolean condition")
    if iftrue.dtype != iffalse.dtype:
        raise Exception("dtype mismatch")

    assert isinstance(iftrue, (BExpr, AExpr))
    assert type(iftrue) is type(iffalse)
    return type(iftrue)(IType.Ite(), iftrue.dtype, cond, iftrue, iffalse)


# ========================================
# Tensor operations
# ========================================


def argmax(arg: Expr):
    if len(arg.shape) > 1:
        raise NotImplementedError(
            "argmax not supported on matrices or higher-dimensional tensors"
        )
    return AExpr(IType.Argmax(), DType.Int([1]), arg)


def matmul(lhs: Expr, rhs: Expr):
    assert type(lhs) is type(rhs)

    if len(lhs.shape) == 2 and len(rhs.shape) == 1:
        # matrix @ vector
        if lhs.shape[-1] != rhs.shape[0]:
            raise RuntimeError("size mismatch")

        wtype = type(lhs.dtype)(lhs.shape[:-1])

        # TODO: differentiate itype to eliminate ambiguity, or parameterise it
        return type(lhs)(IType.MatMul(), wtype, lhs, rhs)

    elif len(lhs.shape) == len(rhs.shape) == 2:
        # matrix @ matrix
        if lhs.shape[-1] != rhs.shape[0]:
            raise RuntimeError("size mismatch")

        wtype = type(lhs.dtype)([lhs.shape[0], rhs.shape[1]])

        return type(lhs)(IType.MatMul(), wtype, lhs, rhs)

    raise RuntimeError(f"Unsupported matrix multiplication {lhs.shape} x {rhs.shape}")

    # TODO: allow broadcasting


# ========================================
# Terminals
# ========================================


def Bool(x: bool | str | torch.Tensor, shape=None) -> BExpr:
    if isinstance(x, bool):
        assert shape is None
        dtype = DType.Bool([1])
        t = torch.Tensor([x])
        return BExpr(itype=IType.Tensor(t), dtype=dtype)
    elif isinstance(x, torch.Tensor):
        assert shape is None or shape == x.shape
        assert x.dtype == torch.bool
        dtype = DType.Bool(x.shape)
        return BExpr(itype=IType.Tensor(x), dtype=dtype)
    elif isinstance(x, str):
        # register symbol into context
        dtype = DType.Bool(shape if shape is not None else [1])
        return BExpr(itype=IType.Uninterpreted(x), dtype=dtype)

    raise ValueError("Invalid argument to `Bool`")


def Real(x: float | str | torch.Tensor, shape=None) -> AExpr:
    if isinstance(x, float):
        assert shape is None
        dtype = DType.Real([1])
        t = torch.Tensor([x])
        return AExpr(itype=IType.Tensor(t), dtype=dtype)
    elif isinstance(x, torch.Tensor):
        assert shape is None or shape == x.shape
        dtype = DType.Real(x.shape)
        return AExpr(itype=IType.Tensor(x), dtype=dtype)
    elif isinstance(x, str):
        # register symbol into context
        dtype = DType.Real(shape if shape is not None else [1])
        return AExpr(itype=IType.Uninterpreted(x), dtype=dtype)

    raise ValueError("Invalid argument to `Real`")
