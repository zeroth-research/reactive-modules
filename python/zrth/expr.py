"""Expression: eager, theory-baked symbolic expressions for the DSL.

An ``Expr`` is a thin, *eager* wrapper: composing one immediately builds the underlying
``Term``. Exprs are split into subclasses by *operation family* (the sort determines the
class):

    AExpr  — Arithmetic   Int / Real     +  -  *  @   and comparisons (-> BExpr)
    BExpr  — Boolean       Bool           &  |  ~  ^
    WExpr  — Word (BV)     BitVec         AExpr + bitwise + signed/unsigned + width

Build/leaf construction and coercion go through one function, ``expr()``:

    expr((latched, next), theory=LRA)          # a state variable (nxt(v) gives its next)
    expr(Wire(...), theory=LRA)                # a bare wire
    expr(True, theory=LRA)                     # a Bool constant
    expr(3, theory=LRA, sort=Sort.Real)        # a numeric constant (sort is REQUIRED)
    expr(5, theory=BV, sort=Sort.BitVec(32, [1, 1]), signed=True)

``expr()`` is **not idempotent** — passing an existing ``Expr`` is an error. Operators
coerce a *raw* operand through it (using the sibling's sort), but never convert between
sorts: mixing sorts raises, and an explicit conversion is ``cast()``.

Each operator builds its own ``Term`` in place (no dispatch layer, no ``builder``).
``collect_terms`` walks a tree and returns every term it created, dependencies first —
that's what ``dslModule`` uses to populate a base Module.

Equality: ``==`` / ``!=`` are Python object identity; use ``eq()`` / ``ne()`` for the
comparison predicates.
"""

from typing import override

import torch

from .zrth import Sort, Term, Wire, LRA, LIA, BV
from .builder import NonLinearError


# --- sort plumbing (Sorts and ops expose no accessors, by theory_pyo3 design) ---


def _shape(sort: Sort) -> list:
    match sort:
        case Sort.Bool(s) | Sort.Int(s) | Sort.Real(s):
            return list(s)
        case Sort.BitVec(_, s):
            return list(s)
    raise TypeError(f"sort has no shape: {sort}")


def _with_shape(sort: Sort, shape: list) -> Sort:
    match sort:
        case Sort.Bool(_):
            return Sort.Bool(shape)
        case Sort.Int(_):
            return Sort.Int(shape)
        case Sort.Real(_):
            return Sort.Real(shape)
        case Sort.BitVec(bw, _):
            return Sort.BitVec(bw, shape)
    raise TypeError(f"unknown sort: {sort}")


def _bv_width(sort: Sort) -> int:
    match sort:
        case Sort.BitVec(bw, _):
            return bw
    raise TypeError(f"not a BitVec sort: {sort}")


def _family(sort: Sort) -> str:
    """A key identifying a sort's family (ignoring shape); two exprs may combine only
    if their families match."""
    match sort:
        case Sort.Bool(_):        return "Bool"
        case Sort.Int(_):         return "Int"
        case Sort.Real(_):        return "Real"
        case Sort.BitVec(bw, _):  return f"BitVec{bw}"
    raise TypeError(f"unknown sort: {sort}")


def _normalize_shape(shape: list) -> list:
    if len(shape) == 0:
        return [1, 1]
    if len(shape) == 1:
        return [1, shape[0]]
    return shape


def _const_tensor(op):
    match op:
        case LRA.ConstReal(t) | LRA.ConstBool(t) | LIA.ConstInt(t) | LIA.ConstBool(t) | BV.Const(t):
            return t
    return None


def _is_wire_pair(v) -> bool:
    return isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(w, Wire) for w in v)


# --- result-sort -> subclass (the ONE place mapping sort to class) ----------


def _wrap(wire, theory, *, term=None, args=(), next=None, signed=False) -> "Expr":
    match wire.dtype:
        case Sort.Bool(_):
            return BExpr(wire, theory, term=term, args=args, next=next)
        case Sort.Int(_) | Sort.Real(_):
            return AExpr(wire, theory, term=term, args=args, next=next)
        case Sort.BitVec(_, _):
            return WExpr(wire, theory, term=term, args=args, next=next, signed=signed)
    raise TypeError(f"no Expr class for sort {wire.dtype}")


# ---------------------------------------------------------------------------
# Base Expr
# ---------------------------------------------------------------------------


class Expr:
    """Shared plumbing; instances are always one of AExpr / BExpr / WExpr (via ``expr()``)."""

    def __init__(self, wire, theory, *, term=None, args=(), next=None):
        self._wire = wire
        self._theory = theory
        self._term = term          # Term that produced _wire (None for a leaf)
        self._args = tuple(args)   # child exprs (for collect_terms' tree walk)
        self._next = next          # next wire, for a variable leaf (nxt(v))

    # --- accessors ---
    @property
    def wire(self) -> Wire:
        return self._wire

    @property
    def dtype(self) -> Sort:
        return self._wire.dtype

    @property
    def shape(self) -> list:
        return _shape(self._wire.dtype)

    @property
    def term(self) -> Term | None:
        return self._term

    @property
    def args(self) -> tuple["Expr", ...]:
        return self._args

    @property
    def theory(self):
        return self._theory

    # --- coercion (raw literal -> Expr of my sort; never converts an Expr) ---
    def _coerce(self, o) -> "Expr":
        return o if isinstance(o, Expr) else expr(o, theory=self._theory, sort=self.dtype)

    def _coerce_same(self, o) -> "Expr":
        o = self._coerce(o)
        if _family(self.dtype) != _family(o.dtype):
            raise TypeError(f"cannot combine {self.dtype} and {o.dtype} implicitly; use cast()")
        return o

    def _result(self, term: Term, *operands: "Expr") -> "Expr":
        return _wrap(term.write[0], self._theory, term=term, args=(self, *operands),
                     signed=getattr(self, "_signed", False))

    def _binop(self, op, out: Sort, o) -> "Expr":
        """Coerce `o` to my sort, then build a binary Term with `op` and output sort `out`."""
        o = self._coerce_same(o)
        return self._result(Term(op, [Wire(out)], [self._wire, o._wire]), o)

    def _unop(self, op, out: Sort | None = None) -> "Expr":
        return self._result(Term(op, [Wire(out if out is not None else self.dtype)], [self._wire]))

    @override
    def __repr__(self) -> str:
        if self._term is None:
            return f"Wire#{self._wire.id}"
        head = str(self._term.itype)
        return f"{head}({', '.join(map(repr, self._args))})" if self._args else head


# ---------------------------------------------------------------------------
# AExpr — arithmetic (Int/Real, theory LRA or LIA)
# ---------------------------------------------------------------------------


class AExpr(Expr):
    def __add__(self, o):   return self._binop(self._theory.Add(), self.dtype, o)
    def __sub__(self, o):   return self._binop(self._theory.Sub(), self.dtype, o)
    def __radd__(self, o):  return self._coerce(o).__add__(self)
    def __rsub__(self, o):  return self._coerce(o).__sub__(self)

    def __mul__(self, o):                      # const*var folds to Linear (LRA/LIA are linear)
        o = self._coerce_same(o)
        return self._result(_mul_as_linear(self._theory, self, o), o)

    def __rmul__(self, o):  return self._coerce(o).__mul__(self)

    def __matmul__(self, o):                   # a constant left operand folds to Linear
        o = self._coerce_same(o)
        return self._result(_matmul(self._theory, self, o), o)

    # comparisons -> Bool -> BExpr
    def __lt__(self, o):  return self._binop(self._theory.Lt(), Sort.Bool(self.shape), o)
    def __le__(self, o):  return self._binop(self._theory.Le(), Sort.Bool(self.shape), o)
    def __gt__(self, o):  return self._binop(self._theory.Gt(), Sort.Bool(self.shape), o)
    def __ge__(self, o):  return self._binop(self._theory.Ge(), Sort.Bool(self.shape), o)


def _mul_as_linear(theory, a: Expr, b: Expr) -> Term:
    """LIA/LRA are linear: `const * var` folds to a Linear op; `var * var` cannot."""
    const_cls = LRA.ConstReal if theory is LRA else LIA.ConstInt
    linear = LRA.Linear if theory is LRA else LIA.Linear
    scalar_dtype = torch.float32 if theory is LRA else torch.int64
    for c, v in ((a, b), (b, a)):
        if c._term is None or not isinstance(c._term.itype, const_cls):
            continue
        data = _const_tensor(c._term.itype)
        if data is None or data.numel() != 1 or len(v.shape) < 2 or v.shape[-1] != 1:
            continue
        A = torch.tensor([[data.item()]], dtype=scalar_dtype)
        bias = torch.zeros(1, 1, dtype=scalar_dtype)
        return Term(linear(A, bias), [Wire(v.dtype)], [v._wire])
    raise NonLinearError(theory.__name__)


def _matmul(theory, a: Expr, b: Expr) -> Term:
    out_shape = [a.shape[0], b.shape[1]]
    const_cls = LRA.ConstReal if theory is LRA else LIA.ConstInt
    if a._term is not None and isinstance(a._term.itype, const_cls):
        linear = LRA.Linear if theory is LRA else LIA.Linear
        out_sort = Sort.Real(out_shape) if theory is LRA else Sort.Int(out_shape)
        return Term(linear(_const_tensor(a._term.itype), torch.empty(0, 0)), [Wire(out_sort)], [b._wire])
    raise RuntimeError(f"{theory.__name__} matmul requires a constant left operand; use a Linear instead")


# ---------------------------------------------------------------------------
# BExpr — boolean (Bool, theory LRA or LIA)
# ---------------------------------------------------------------------------


class BExpr(Expr):
    def __and__(self, o):  return self._binop(self._theory.And(), self.dtype, o)
    def __or__(self, o):   return self._binop(self._theory.Or(), self.dtype, o)
    def __xor__(self, o):  return self._binop(self._theory.Xor(), self.dtype, o)
    def __invert__(self):  return self._unop(self._theory.Not())


# ---------------------------------------------------------------------------
# WExpr — word / bit-vector (BitVec, theory BV). Inherits +,- from AExpr.
# ---------------------------------------------------------------------------


class WExpr(AExpr):
    def __init__(self, wire, theory, *, term=None, args=(), next=None, signed=False):
        super().__init__(wire, theory, term=term, args=args, next=next)
        self._signed = signed

    @property
    def signed(self) -> bool:
        return self._signed

    def _bv1(self) -> Sort:
        return Sort.BitVec(1, self.shape)

    # arithmetic: inherits +,- from AExpr (BV.Add/Sub); mul is a real BV multiply (no fold)
    def __mul__(self, o):        return self._binop(BV.Mul(), self.dtype, o)
    def __floordiv__(self, o):   return self._binop(BV.SDiv() if self._signed else BV.UDiv(), self.dtype, o)
    def __mod__(self, o):        return self._binop(BV.SMod() if self._signed else BV.UMod(), self.dtype, o)

    def __matmul__(self, o):
        o = self._coerce_same(o)
        out = Wire(Sort.BitVec(_bv_width(self.dtype), [self.shape[0], o.shape[1]]))
        return self._result(Term(BV.MatMul(), [out], [self._wire, o._wire]), o)

    # comparisons pick signed/unsigned; result is BitVec(1)
    def __lt__(self, o):  return self._binop(BV.SLt() if self._signed else BV.ULt(), self._bv1(), o)
    def __le__(self, o):  return self._binop(BV.SLe() if self._signed else BV.ULe(), self._bv1(), o)
    def __gt__(self, o):  return self._binop(BV.SGt() if self._signed else BV.UGt(), self._bv1(), o)
    def __ge__(self, o):  return self._binop(BV.SGe() if self._signed else BV.UGe(), self._bv1(), o)

    # bitwise, width-preserving
    def __and__(self, o):  return self._binop(BV.And(), self.dtype, o)
    def __or__(self, o):   return self._binop(BV.Or(), self.dtype, o)
    def __xor__(self, o):  return self._binop(BV.Xor(), self.dtype, o)
    def __invert__(self):  return self._unop(BV.Not())


# ---------------------------------------------------------------------------
# expr() — single construction / coercion entry point
# ---------------------------------------------------------------------------


def expr(value, *, theory=None, sort=None, signed=False) -> Expr:
    if isinstance(value, Expr):
        raise TypeError("expr() builds from a raw value; got an Expr already")
    if theory is None:
        raise TypeError("expr() requires theory=")
    if _is_wire_pair(value):                   # (latched, next) -> state variable
        return _wrap(value[0], theory, next=value[1], signed=signed)
    if isinstance(value, Wire):                # single wire -> bare expr (no nxt)
        return _wrap(value, theory, signed=signed)
    return _const(value, theory, sort, signed)


def _resolve_sort(sort, shape) -> Sort:
    """`sort` may be a concrete Sort (e.g. from coercion) or a family (Sort.Real/Int/Bool)."""
    if isinstance(sort, Sort):
        return _with_shape(sort, shape)
    if sort is Sort.Bool:
        return Sort.Bool(shape)
    if sort is Sort.Int:
        return Sort.Int(shape)
    if sort is Sort.Real:
        return Sort.Real(shape)
    raise TypeError("a bit-vector literal needs a width: pass sort=Sort.BitVec(width, [...])")


def _wants_float(sort) -> bool:
    return isinstance(sort, Sort.Real) or sort is Sort.Real


def _const(value, theory, sort, signed) -> Expr:
    if isinstance(value, bool):                # Bool is unambiguous -> no sort= needed
        tensor, family = torch.tensor([[value]], dtype=torch.bool), Sort.Bool
    else:
        if sort is None:
            raise TypeError(
                "expr(): a numeric literal needs an explicit sort= "
                "(e.g. sort=Sort.Real, or sort=Sort.BitVec(32, [1, 1]))"
            )
        tensor = value if isinstance(value, torch.Tensor) else \
            torch.tensor(value, dtype=torch.float32 if _wants_float(sort) else torch.int64)
        family = sort
    shape = _normalize_shape(list(tensor.size()))
    tensor = tensor.reshape(shape)             # theory const ops require a 2-D initializer
    w = Wire(_resolve_sort(family, shape))
    return _wrap(w, theory, term=Term.constant(_const_op(theory, tensor), [w]), signed=signed)


def _const_op(theory, tensor):
    # NOTE (single-Const track): this collapses to `theory.Const(tensor)` (sort from the
    # write wire); today it is redundantly split by sort.
    is_bool = tensor.dtype == torch.bool
    if theory is LRA:  return LRA.ConstBool(tensor) if is_bool else LRA.ConstReal(tensor)
    if theory is LIA:  return LIA.ConstBool(tensor) if is_bool else LIA.ConstInt(tensor)
    return BV.Const(tensor)


def cast(e: Expr, sort) -> Expr:
    """Explicit sort conversion. Only trivial (same-sort) casts are supported for now;
    a genuine conversion op is not yet in the IR."""
    target = _resolve_sort(sort, e.shape)
    if e.dtype == target:
        return e
    raise NotImplementedError(f"cast {e.dtype} -> {target} is not supported yet")


# ---------------------------------------------------------------------------
# Non-operator ops
# ---------------------------------------------------------------------------


def nxt(v: Expr) -> Expr:
    """The `next` wire of a variable expr (as built from a wire pair)."""
    if v._next is None:
        raise ValueError("nxt() expects a variable (built from a wire pair)")
    return _wrap(v._next, v._theory, signed=getattr(v, "_signed", False))


def ite(cond: Expr, iftrue, iffalse) -> Expr:
    a, b = iftrue, iffalse
    if not isinstance(a, Expr) and not isinstance(b, Expr):
        raise TypeError("ite(): at least one branch must be an Expr")
    if not isinstance(a, Expr):
        a = expr(a, theory=b._theory, sort=b.dtype)
    if not isinstance(b, Expr):
        b = expr(b, theory=a._theory, sort=a.dtype)
    if _family(a.dtype) != _family(b.dtype):
        raise TypeError(f"ite branches have different sorts {a.dtype}, {b.dtype}; use cast()")
    term = Term(cond._theory.Ite(), [Wire(a.dtype)], [cond._wire, a._wire, b._wire])
    return _wrap(term.write[0], cond._theory, term=term, args=(cond, a, b),
                 signed=getattr(a, "_signed", False))


def _cmp_out(a: Expr) -> Sort:
    return Sort.BitVec(1, a.shape) if a._theory is BV else Sort.Bool(a.shape)


def eq(a: Expr, b) -> Expr:
    b = a._coerce_same(b)
    return a._result(Term(a._theory.Eq(), [Wire(_cmp_out(a))], [a._wire, b._wire]), b)


def ne(a: Expr, b) -> Expr:
    b = a._coerce_same(b)
    return a._result(Term(a._theory.Ne(), [Wire(_cmp_out(a))], [a._wire, b._wire]), b)


def relu(e: Expr) -> Expr:
    return e._result(Term(e._theory.ReLU(), [Wire(e.dtype)], [e._wire]))


def argmax(e: Expr) -> Expr:
    return e._result(Term(e._theory.Argmax(), [Wire(_with_shape(e.dtype, [1, 1]))], [e._wire]))


# ---------------------------------------------------------------------------
# Term collection (used by dslModule to populate the base Module)
# ---------------------------------------------------------------------------


def collect_terms(*roots: Expr) -> list[Term]:
    """Every term created under the given expr trees, dependencies first, de-duplicated."""
    seen: set[int] = set()
    out: list[Term] = []

    def visit(e: Expr) -> None:
        for child in e._args:
            visit(child)
        if e._term is not None and id(e._term) not in seen:
            seen.add(id(e._term))
            out.append(e._term)

    for r in roots:
        visit(r)
    return out
