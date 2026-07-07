"""Expression: eager, theory-baked symbolic expressions for the DSL.

An ``Expr`` is a thin, *eager* wrapper: composing one (operators, ``ite``, ``nxt``)
immediately builds the underlying ``Term``. Each node caches

- ``wire``   — its output wire,
- ``term``   — the term that produced ``wire`` (``None`` for a bare leaf wire),
- ``args``   — its child exprs (so a module can walk the tree),
- ``pair``   — for a *variable* leaf, the ``(latched, next)`` wire pair.

A variable reads as its **latched** value; ``nxt(v)`` gives an expr over its **next**
wire. **Each operation builds its own Term in place** — the op choice, output sort, and
any per-theory switch live directly in the operator (``__add__``, ``__lt__``, ``ite``, …).
There is no dispatch layer and no ``builder``.

``collect_terms`` walks a tree and returns every term it created, dependencies first —
that's what ``dslModule`` uses to populate a base Module.

Equality: ``==`` / ``!=`` are Python object identity here; use ``eq()`` / ``ne()`` for
the comparison predicates. Ordering (``< <= > >=``) and ``+ - * @ & | ~`` are overloaded.
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


# ---------------------------------------------------------------------------
# Expr
# ---------------------------------------------------------------------------


class Expr:
    def __init__(self, wire, theory, term=None, args=(), pair=None, name=None):
        self._wire = wire
        self._theory = theory
        self._term = term
        self._args = tuple(args)
        self._pair = pair
        self._name = name

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

    def _coerce(self, o) -> "Expr":
        return as_expr(o, self._theory)

    def _result(self, term: Term, *operands: "Expr") -> "Expr":
        """Wrap a freshly-built Term as an Expr node, recording the child operands."""
        return Expr(term.write[0], self._theory, term=term, args=(self, *operands))

    # --- arithmetic ---
    def __add__(self, o):
        o = self._coerce(o)
        return self._result(Term(self._theory.Add(), [Wire(self.dtype)], [self._wire, o._wire]), o)

    def __radd__(self, o):
        return self._coerce(o).__add__(self)

    def __sub__(self, o):
        o = self._coerce(o)
        return self._result(Term(self._theory.Sub(), [Wire(self.dtype)], [self._wire, o._wire]), o)

    def __rsub__(self, o):
        return self._coerce(o).__sub__(self)

    def __mul__(self, o):
        o = self._coerce(o)
        if self._theory is BV:
            return self._result(Term(BV.Mul(), [Wire(self.dtype)], [self._wire, o._wire]), o)
        # LIA/LRA are linear: `const * var` folds to a Linear op; `var * var` cannot.
        const_cls = LRA.ConstReal if self._theory is LRA else LIA.ConstInt
        linear = LRA.Linear if self._theory is LRA else LIA.Linear
        scalar_dtype = torch.float32 if self._theory is LRA else torch.int64
        for c, v in ((self, o), (o, self)):
            if c._term is None or not isinstance(c._term.itype, const_cls):
                continue
            data = _const_tensor(c._term.itype)
            if data is None or data.numel() != 1 or len(v.shape) < 2 or v.shape[-1] != 1:
                continue
            A = torch.tensor([[data.item()]], dtype=scalar_dtype)
            bias = torch.zeros(1, 1, dtype=scalar_dtype)
            return self._result(Term(linear(A, bias), [Wire(v.dtype)], [v._wire]), o)
        raise NonLinearError(self._theory.__name__)

    def __rmul__(self, o):
        return self._coerce(o).__mul__(self)

    def __matmul__(self, o):
        o = self._coerce(o)
        out_shape = [self.shape[0], o.shape[1]]
        if self._theory is BV:
            out = Wire(Sort.BitVec(_bv_width(self.dtype), out_shape))
            return self._result(Term(BV.MatMul(), [out], [self._wire, o._wire]), o)
        # LIA/LRA: only a constant left operand is expressible (as a Linear).
        const_cls = LRA.ConstReal if self._theory is LRA else LIA.ConstInt
        if self._term is not None and isinstance(self._term.itype, const_cls):
            linear = LRA.Linear if self._theory is LRA else LIA.Linear
            out_sort = Sort.Real(out_shape) if self._theory is LRA else Sort.Int(out_shape)
            term = Term(linear(_const_tensor(self._term.itype), torch.empty(0, 0)), [Wire(out_sort)], [o._wire])
            return self._result(term, o)
        raise RuntimeError(f"{self._theory.__name__} matmul requires a constant left operand; use a Linear instead")

    # --- comparisons (== / != stay identity; use eq() / ne()) ---
    # TODO: BV comparisons are always unsigned (U*). `Sort.BitVec` carries no signedness,
    # so the signed ops (SLt/SLe/…) are unreachable here; add a signed path if BV gains it.
    def __lt__(self, o):
        o = self._coerce(o)
        op = BV.ULt() if self._theory is BV else self._theory.Lt()
        out = Sort.BitVec(1, self.shape) if self._theory is BV else Sort.Bool(self.shape)
        return self._result(Term(op, [Wire(out)], [self._wire, o._wire]), o)

    def __le__(self, o):
        o = self._coerce(o)
        op = BV.ULe() if self._theory is BV else self._theory.Le()
        out = Sort.BitVec(1, self.shape) if self._theory is BV else Sort.Bool(self.shape)
        return self._result(Term(op, [Wire(out)], [self._wire, o._wire]), o)

    def __gt__(self, o):
        o = self._coerce(o)
        op = BV.UGt() if self._theory is BV else self._theory.Gt()
        out = Sort.BitVec(1, self.shape) if self._theory is BV else Sort.Bool(self.shape)
        return self._result(Term(op, [Wire(out)], [self._wire, o._wire]), o)

    def __ge__(self, o):
        o = self._coerce(o)
        op = BV.UGe() if self._theory is BV else self._theory.Ge()
        out = Sort.BitVec(1, self.shape) if self._theory is BV else Sort.Bool(self.shape)
        return self._result(Term(op, [Wire(out)], [self._wire, o._wire]), o)

    # --- logical / bitwise (BV preserves the operand width; LIA/LRA yield Bool) ---
    def __and__(self, o):
        o = self._coerce(o)
        out = self.dtype if self._theory is BV else Sort.Bool(self.shape)
        return self._result(Term(self._theory.And(), [Wire(out)], [self._wire, o._wire]), o)

    def __or__(self, o):
        o = self._coerce(o)
        out = self.dtype if self._theory is BV else Sort.Bool(self.shape)
        return self._result(Term(self._theory.Or(), [Wire(out)], [self._wire, o._wire]), o)

    def __xor__(self, o):
        o = self._coerce(o)
        out = self.dtype if self._theory is BV else Sort.Bool(self.shape)
        return self._result(Term(self._theory.Xor(), [Wire(out)], [self._wire, o._wire]), o)

    def __invert__(self):
        out = self.dtype if self._theory is BV else Sort.Bool(self.shape)
        return self._result(Term(self._theory.Not(), [Wire(out)], [self._wire]))

    @override
    def __repr__(self) -> str:
        if self._term is None:
            return f"Var({self._name!r})" if self._name else f"Wire#{self._wire.id}"
        head = str(self._term.itype)
        return f"{head}({', '.join(map(repr, self._args))})" if self._args else head


# ---------------------------------------------------------------------------
# Leaves & non-operator ops
# ---------------------------------------------------------------------------


def var(pair: tuple[Wire, Wire], theory, name: str | None = None) -> Expr:
    """A state/interface variable leaf; reads as its *latched* wire. `nxt(v)` -> next."""
    return Expr(pair[0], theory, term=None, args=(), pair=pair, name=name)


def nxt(v: Expr) -> Expr:
    """The `next` wire of a variable expr (as returned by `var`)."""
    if v._pair is None:
        raise ValueError("nxt() expects a variable (wire-pair) expression")
    return Expr(v._pair[1], v._theory, term=None, args=())


def const(value, theory) -> Expr:
    """A constant leaf in `theory`, from a Python scalar/bool or a tensor."""
    if isinstance(value, bool):
        tensor = torch.tensor([[value]], dtype=torch.bool)
    elif isinstance(value, torch.Tensor):
        tensor = value
    elif theory is LRA:
        tensor = torch.tensor([float(value)], dtype=torch.float32)
    else:
        tensor = torch.tensor([int(value)], dtype=torch.int64)

    shape = _normalize_shape(list(tensor.size()))
    tensor = tensor.reshape(shape)  # theory const ops require a 2-D initializer
    is_bool = tensor.dtype == torch.bool
    if theory is LRA:
        op, sort = (LRA.ConstBool(tensor), Sort.Bool(shape)) if is_bool else (LRA.ConstReal(tensor), Sort.Real(shape))
    elif theory is LIA:
        op, sort = (LIA.ConstBool(tensor), Sort.Bool(shape)) if is_bool else (LIA.ConstInt(tensor), Sort.Int(shape))
    else:
        op, sort = BV.Const(tensor), (Sort.BitVec(1, shape) if is_bool else Sort.BitVec(32, shape))
    w = Wire(sort)
    return Expr(w, theory, term=Term.constant(op, [w]))


def as_expr(value, theory) -> Expr:
    """Coerce a bare Python value to a const expr; pass Exprs through unchanged."""
    return value if isinstance(value, Expr) else const(value, theory)


def ite(cond: Expr, iftrue, iffalse) -> Expr:
    iftrue, iffalse = cond._coerce(iftrue), cond._coerce(iffalse)
    term = Term(cond._theory.Ite(), [Wire(iftrue.dtype)], [cond._wire, iftrue._wire, iffalse._wire])
    return cond._result(term, iftrue, iffalse)


def eq(a: Expr, b) -> Expr:
    b = a._coerce(b)
    out = Sort.BitVec(1, a.shape) if a._theory is BV else Sort.Bool(a.shape)
    return a._result(Term(a._theory.Eq(), [Wire(out)], [a._wire, b._wire]), b)


def ne(a: Expr, b) -> Expr:
    b = a._coerce(b)
    out = Sort.BitVec(1, a.shape) if a._theory is BV else Sort.Bool(a.shape)
    return a._result(Term(a._theory.Ne(), [Wire(out)], [a._wire, b._wire]), b)


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
