from typing import Any, override

from .zrth import DType, IType, Term, Wire

Arith = IType.Arith
import torch

# types that we can convert to [Expr]
type ToExpr = Expr | int | bool | float | torch.Tensor



class _Uninterpreted:
    """Python-only itype for symbolic/free variables (no Rust Term backing)."""
    def __init__(self, name: str):
        self.name = name

    def __call__(self):
        return self

    def __repr__(self):
        return f"Uninterpreted({self.name!r})"


class Expr:
    def __init__(self, itype, dtype: DType, *args):
        assert all(isinstance(arg, Expr) for arg in args)
        self._itype = itype

        if isinstance(itype, _Uninterpreted):
            # Free/symbolic variable — no Term, just a bare wire
            self._term = None
            self._wire = Wire(dtype)
        else:
            self._term = Term(itype, [Wire(dtype)], [a.wire for a in args])
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


def _make_expr(itype: IType, wtype: DType, *args):
    if wtype.is_bool():
        return BExpr(itype, wtype, *args)
    elif wtype.is_real() or wtype.is_int() or wtype.is_float():
        return AExpr(itype, wtype, *args)
    elif wtype.is_bv():
        return WExpr(itype, wtype, *args)
    else:
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

    # Fold left: chain binary applications for variadic calls
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
    return _elementwise_bool_op(IType.Bool.Or, first, *others)


# Logical and
def conj(first: BExpr, *others) -> BExpr:
    return _elementwise_bool_op(IType.Bool.And, first, *others)


# Logical not
def neg(e: BExpr) -> BExpr:
    if not e.dtype.is_bool():
        raise ValueError(f"invalid dtype, expected Bool(...), got: `{e.dtype}`")

    return BExpr(IType.Bool.Not, e.dtype, e)


# ========================================
# Elementwise arithmetics operators
# ========================================


def _elementwise_arith_op(itype, first: Expr, *others):
    if not (first.dtype.is_real() or first.dtype.is_int() or first.dtype.is_float()):
        raise Exception("invalid dtype")

    return _elementwise_op(itype, first.dtype, first.dtype, first, *others)


def add(first: AExpr, *others) -> AExpr:
    return _elementwise_arith_op(IType.mk(Arith.Add, first.dtype), first, *others)


def mul(first: AExpr, *others) -> AExpr:
    return _elementwise_arith_op(IType.mk(Arith.Mul, first.dtype), first, *others)


def div(num: AExpr, den: AExpr) -> AExpr:
    return _elementwise_arith_op(IType.mk(Arith.Div, num.dtype), num, den)


def sub(min: AExpr, sub: AExpr) -> AExpr:
    return _elementwise_arith_op(IType.mk(Arith.Sub, min.dtype), min, sub)


# ========================================
# Elementwise predicates
# ========================================


def _elementwise_predicate(itype, lhs: AExpr, rhs: AExpr):
    if not (lhs.dtype.is_real() or lhs.dtype.is_int() or lhs.dtype.is_float()):
        raise Exception("invalid dtype")
    if not (rhs.dtype.is_real() or rhs.dtype.is_int() or rhs.dtype.is_float()):
        raise Exception("invalid dtype")

    return _elementwise_op(itype, DType.Bool(lhs.shape), lhs.dtype, lhs, rhs)


def lt(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Cmp.Lt, lhs, rhs)


def gt(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Cmp.Gt, lhs, rhs)


def ge(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Cmp.Ge, lhs, rhs)


def le(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Cmp.Le, lhs, rhs)


def eq(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Cmp.Eq, lhs, rhs)


def neq(lhs: AExpr, rhs: AExpr) -> BExpr:
    return _elementwise_predicate(IType.Cmp.Ne, lhs, rhs)


# ========================================
# Control Flow
# ========================================


def ite(cond: BExpr, iftrue: Expr, iffalse: Expr):
    if not (cond.shape == [1] or cond.shape == [1, 1]):
        raise Exception("invalid Boolean condition")
    if iftrue.dtype != iffalse.dtype:
        raise Exception("dtype mismatch")

    assert isinstance(iftrue, (BExpr, AExpr))
    assert type(iftrue) is type(iffalse)
    return type(iftrue)(IType.Ite, iftrue.dtype, cond, iftrue, iffalse)


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
    return _word_arith_op(IType.BV.Add, first, *others)


def w_sub(first: WExpr, *others) -> WExpr:
    return _word_arith_op(IType.mk(Arith.Sub, first.dtype), first, *others)


def w_mul(first: WExpr, *others) -> WExpr:
    return _word_arith_op(IType.BV.Mul, first, *others)


def w_div(num: WExpr, den: WExpr) -> WExpr:
    return _word_arith_op(IType.mk(Arith.Div, num.dtype), num, den)


def w_mod(num: WExpr, den: WExpr) -> WExpr:
    return _word_arith_op(IType.mk(Arith.Mod, num.dtype), num, den)


def w_neg(e: WExpr) -> WExpr:
    if not e.dtype.is_bv():
        raise Exception("invalid dtype for word-level op")
    return WExpr(IType.mk(Arith.Neg, e.dtype), e.dtype, e)


def w_abs(e: WExpr) -> WExpr:
    if not e.dtype.is_bv():
        raise Exception("invalid dtype for word-level op")
    return WExpr(IType.mk(Arith.Abs, e.dtype), e.dtype, e)


def w_and(first: WExpr, *others) -> WExpr:
    return _word_arith_op(IType.BV.And, first, *others)


def w_or(first: WExpr, *others) -> WExpr:
    return _word_arith_op(IType.BV.Or, first, *others)


def w_xor(first: WExpr, *others) -> WExpr:
    return _word_arith_op(IType.BV.Xor, first, *others)


def w_not(e: WExpr) -> WExpr:
    if not e.dtype.is_bv():
        raise Exception("invalid dtype for word-level op")
    return WExpr(IType.BV.Not, e.dtype, e)


def w_lt(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Cmp.Lt, lhs, rhs)


def w_gt(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Cmp.Gt, lhs, rhs)


def w_le(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Cmp.Le, lhs, rhs)


def w_ge(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Cmp.Ge, lhs, rhs)


def w_eq(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Cmp.Eq, lhs, rhs)


def w_neq(lhs: WExpr, rhs: WExpr) -> BExpr:
    return _word_predicate(IType.Cmp.Ne, lhs, rhs)


# ========================================
# Tensor operations
# ========================================


def argmax(arg: Expr):
    if arg.shape[0] > 1:
        raise NotImplementedError(
            "argmax not supported on matrices or higher-dimensional tensors"
        )
    return AExpr(IType.Tensor.Argmax, DType.Int([1]), arg)


def matmul(lhs: Expr, rhs: Expr):
    assert type(lhs) is type(rhs)

    # shape is always [rows, cols]; a vector has rows==1
    lhs_rows, lhs_cols = lhs.shape
    rhs_rows, rhs_cols = rhs.shape

    if rhs_rows == 1:
        # row-vector [1, k] — transpose to [k, 1] so standard matmul applies
        if lhs_cols != rhs_cols:
            raise RuntimeError("size mismatch")
        col_dtype = rhs.dtype.reshape([rhs_cols, 1])
        rhs = type(rhs)(IType.mk(Arith.Transpose, rhs.dtype), col_dtype, rhs)
        rhs_rows, rhs_cols = rhs.shape

    # matrix @ matrix: [m, k] @ [k, n] -> [m, n]
    if lhs_cols != rhs_rows:
        raise RuntimeError("size mismatch")
    wtype = lhs.dtype.reshape([lhs_rows, rhs_cols])
    return type(lhs)(IType.mk(Arith.MatMul, lhs.dtype), wtype, lhs, rhs)


# ========================================
# Terminals
# ========================================


def Bool(x: bool | str | torch.Tensor, shape=None) -> BExpr:
    if isinstance(x, bool):
        assert shape is None
        dtype = DType.Bool([1])
        t = torch.tensor([x])
        return BExpr(itype=IType.from_tensor(t), dtype=dtype)
    elif isinstance(x, torch.Tensor):
        assert shape is None or shape == x.shape
        assert x.dtype == torch.bool
        dtype = DType.Bool(x.shape)
        return BExpr(itype=IType.from_tensor(x), dtype=dtype)
    elif isinstance(x, str):
        # register symbol into context
        dtype = DType.Bool(shape if shape is not None else [1])
        return BExpr(itype=_Uninterpreted(x), dtype=dtype)

    raise ValueError("Invalid argument to `Bool`")


def _tensor_to_real_const(t: torch.Tensor) -> "IType":
    """Build IType.Real.Const from a float tensor."""
    if t.dim() == 1:
        t = t.unsqueeze(0)
    rows, cols = t.shape[0], t.shape[1]
    data = [[float(t[i, j]) for j in range(cols)] for i in range(rows)]
    return IType.Real.Const(data)


def Real(x: float | str | torch.Tensor, shape=None) -> AExpr:
    if isinstance(x, float):
        assert shape is None
        dtype = DType.Real([1])
        return AExpr(itype=IType.Real.Const([[x]]), dtype=dtype)
    elif isinstance(x, torch.Tensor):
        assert shape is None or list(x.shape) == list(shape) if shape is not None else True
        t_float = x.to(torch.float64)
        dtype = DType.Real(list(x.shape) if x.dim() > 0 else [1])
        return AExpr(itype=_tensor_to_real_const(t_float), dtype=dtype)
    elif isinstance(x, str):
        # register symbol into context
        dtype = DType.Real(shape if shape is not None else [1])
        return AExpr(itype=_Uninterpreted(x), dtype=dtype)

    raise ValueError("Invalid argument to `Real`")
