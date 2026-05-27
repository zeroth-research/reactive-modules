from typing import Any, override

from .zrth import DType, Term, Wire
from . import IType
import torch

# types that we can convert to [Expr]
type ToExpr = Expr | int | bool | float | torch.Tensor


class Expr:
    def __init__(self, itype: IType, dtype: DType, *args):
        assert all(isinstance(arg, Expr) for arg in args)
        self._term = Term(itype, [Wire(dtype)], [a.wire for a in args])

        self._itype = itype

        self._wire = self._term.write[0]
        self._dtype = self._wire.dtype
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


def _make_expr(itype, wtype: DType, *args):
    if wtype.is_bool():
        return BExpr(itype, wtype, *args)
    if wtype.is_real() or wtype.is_int():
        return AExpr(itype, wtype, *args)
    if wtype.is_bv():
        return WExpr(itype, wtype, *args)
    raise NotImplementedError(f"unsupported dtype: {wtype}")


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

    # Fold left so binary ops get exactly two reads each
    acc = first
    for arg in others:
        acc = _make_expr(itype, wtype, acc, arg)
    return acc


# ========================================
# Elementwise logical operators
# ========================================


def _elementwise_bool_op(itype, first: BExpr, *others):
    if not first.dtype.is_bool():
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
    if not e.dtype.is_bool():
        raise ValueError(f"invalid dtype, expected Bool(...), got: `{e.dtype}`")

    return BExpr(IType.Not(), e.dtype, e)


# ========================================
# Elementwise arithmetics operators
# ========================================


def _elementwise_arith_op(itype, first: Expr, *others):
    if not (first.dtype.is_real() or first.dtype.is_int()):
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
    if not (lhs.dtype.is_real() or lhs.dtype.is_int()):
        raise Exception("invalid dtype")
    if not (rhs.dtype.is_real() or rhs.dtype.is_int()):
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
# Word-level operators
# ========================================


def _word_arith_op(itype, first: WExpr, *others):
    if not first.dtype.is_bv():
        raise Exception("invalid dtype for word-level op")
    return _elementwise_op(itype, first.dtype, first.dtype, first, *others)


def _word_predicate(itype, lhs: WExpr, rhs: WExpr):
    if not lhs.dtype.is_bv():
        raise Exception("invalid dtype for word-level op")
    return _elementwise_op(itype, DType.Bool([1]), lhs.dtype, lhs, rhs)


def w_add(first: WExpr, *others) -> WExpr:
    return _word_arith_op(IType.Add(), first, *others)


def w_sub(first: WExpr, *others) -> WExpr:
    return _word_arith_op(IType.Sub(), first, *others)


def w_mul(first: WExpr, *others) -> WExpr:
    return _word_arith_op(IType.Mul(), first, *others)


def w_div(num: WExpr, den: WExpr) -> WExpr:
    return _word_arith_op(IType.Div(), num, den)


def w_mod(num: WExpr, den: WExpr) -> WExpr:
    return _word_arith_op(IType.Mod(), num, den)


def w_neg(e: WExpr) -> WExpr:
    if not e.dtype.is_bv():
        raise Exception("invalid dtype for word-level op")
    return WExpr(IType.Neg(), e.dtype, e)


def w_abs(e: WExpr) -> WExpr:
    if not e.dtype.is_bv():
        raise Exception("invalid dtype for word-level op")
    return WExpr(IType.Abs(), e.dtype, e)


def w_and(first: WExpr, *others) -> WExpr:
    return _word_arith_op(IType.And(), first, *others)


def w_or(first: WExpr, *others) -> WExpr:
    return _word_arith_op(IType.Or(), first, *others)


def w_xor(first: WExpr, *others) -> WExpr:
    return _word_arith_op(IType.Xor(), first, *others)


def w_not(e: WExpr) -> WExpr:
    if not e.dtype.is_bv():
        raise Exception("invalid dtype for word-level op")
    return WExpr(IType.Not(), e.dtype, e)


def w_lt(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Lt(), lhs, rhs)


def w_gt(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Gt(), lhs, rhs)


def w_le(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Le(), lhs, rhs)


def w_ge(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Ge(), lhs, rhs)


def w_eq(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Eq(), lhs, rhs)


def w_neq(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Neq(), lhs, rhs)


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
        return BExpr(itype=IType.Tensor(torch.tensor([x])), dtype=dtype)
    elif isinstance(x, torch.Tensor):
        assert shape is None or shape == x.shape
        assert x.dtype == torch.bool
        dtype = DType.Bool(x.shape)
        return BExpr(itype=IType.Tensor(x), dtype=dtype)
    elif isinstance(x, str):

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

        dtype = DType.Real(shape if shape is not None else [1])
        return AExpr(itype=IType.Uninterpreted(x), dtype=dtype)

    raise ValueError("Invalid argument to `Real`")
