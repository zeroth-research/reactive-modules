"""SMT encoding utilities for the CEGAR magic driver.

Maps Python IR types (`Sort`) to cvc5 sorts, and provides element access
helpers for matrix-shaped state components.

Matrix representation
---------------------
`Mat t m n` is encoded as a flat cvc5 tuple of `m*n` values, row-major
(element `[i][j]` at index `i*n + j`). When `m = n = 1`, the tuple
collapses to the scalar sort `t` — this matches the common Lean use of
`Mat t 1 1` for scalar values.

What this module owns is that representation. *Which* term each op builds
is the `smt` column of the op table in `ops.py`, one cell per variant;
`translate_terms` walks the IR and hands each op an `SmtOp` to build with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

import cvc5
from cvc5 import Kind

from zrth import Wire, Sort, BitVec, Bool, Int, Real, Term
from .common import itype_name, dtype_shape
from .ops import smt_emitter


@dataclass(frozen=True)
class MatShape:
    """Resolved (m, n) matrix dimensions for a wire."""

    m: int
    n: int

    @property
    def total(self) -> int:
        return self.m * self.n

    @property
    def is_scalar(self) -> bool:
        return self.total == 1


def wire_shape(wire: Wire) -> MatShape:
    """Resolve a `Wire`'s Python shape to (m, n) for the Mat encoding.

    Scalars (`shape == [] or [1]`) → (1, 1).
    Vectors (`shape == [n]`) → (1, n).
    Matrices (`shape == [m, n]`) → (m, n).
    Anything else raises.
    """
    shape = dtype_shape(wire.dtype)
    if shape in ([], [1]):
        return MatShape(1, 1)
    if len(shape) == 1:
        return MatShape(1, shape[0])
    if len(shape) == 2:
        return MatShape(shape[0], shape[1])
    raise ValueError(f"Unsupported Sort shape for SMT encoding: {shape}")


def elem_sort(tm: cvc5.TermManager, dt: Sort) -> cvc5.Sort:
    """cvc5 sort for the element type of a Sort (ignoring shape)."""
    if isinstance(dt, Bool):
        return tm.getBooleanSort()
    if isinstance(dt, Int):
        return tm.getIntegerSort()
    if isinstance(dt, Real):
        return tm.getRealSort()
    if isinstance(dt, BitVec):
        return tm.mkBitVectorSort(dt._0)
    raise ValueError(f"Unsupported Sort for SMT encoding: {dt}")


def wire_sort(tm: cvc5.TermManager, wire: Wire) -> cvc5.Sort:
    """cvc5 sort for a wire: scalar or flat tuple of `m*n` elements."""
    elem = elem_sort(tm, wire.dtype)
    shape = wire_shape(wire)
    if shape.is_scalar:
        return elem
    return tm.mkTupleSort(*([elem] * shape.total))


def mat_select(
    tm: cvc5.TermManager,
    term: cvc5.Term,
    shape: MatShape,
    i: int,
    j: int,
) -> cvc5.Term:
    """Return element `[i][j]` of a matrix-typed cvc5 term.

    For scalars (1×1) the term is returned as-is. Otherwise emits a
    tuple-projection at flat index `i*n + j`.
    """
    if shape.is_scalar:
        if i != 0 or j != 0:
            raise IndexError(f"scalar wire has no element ({i}, {j})")
        return term
    idx = i * shape.n + j
    ctor = term.getSort().getDatatype()[0]
    return tm.mkTerm(Kind.APPLY_SELECTOR, ctor[idx].getTerm(), term)


def mat_pack(
    tm: cvc5.TermManager,
    shape: MatShape,
    elems: list[cvc5.Term],
) -> cvc5.Term:
    """Build a matrix cvc5 term from row-major element list."""
    if len(elems) != shape.total:
        raise ValueError(f"expected {shape.total} elements, got {len(elems)}")
    if shape.is_scalar:
        return elems[0]
    return tm.mkTuple(elems)


# -----------------------------------------------------------------
#   Term → cvc5.Term translation
# -----------------------------------------------------------------


def _scalar_const(tm: cvc5.TermManager, dt: Sort, raw) -> cvc5.Term:
    """cvc5 literal for a scalar Python value of the given Sort."""
    if isinstance(dt, Bool):
        return tm.mkBoolean(bool(raw))
    if isinstance(dt, Int):
        return tm.mkInteger(int(raw))
    if isinstance(dt, Real):
        return tm.mkReal(float(raw))
    if isinstance(dt, BitVec):
        return tm.mkBitVector(dt._0, int(raw))
    raise ValueError(f"Unsupported scalar Sort: {dt}")


def _tensor_const(tm: cvc5.TermManager, wire: Wire, tensor) -> cvc5.Term:
    """Materialize a torch tensor as a matrix-shaped cvc5 term."""
    shape = wire_shape(wire)
    dt = wire.dtype
    if shape.is_scalar:
        return _scalar_const(tm, dt, tensor.item())
    data = tensor.reshape(shape.m, shape.n)
    elems = [
        _scalar_const(tm, dt, data[i, j].item())
        for i in range(shape.m)
        for j in range(shape.n)
    ]
    return mat_pack(tm, shape, elems)


def _elementwise(
    tm: cvc5.TermManager,
    shape: MatShape,
    op: Callable[..., cvc5.Term],
    *mats: cvc5.Term,
) -> cvc5.Term:
    """Apply `op` element-wise to equally-shaped matrix terms."""
    if shape.is_scalar:
        return op(*mats)
    elems = [
        op(*(mat_select(tm, m, shape, i, j) for m in mats))
        for i in range(shape.m)
        for j in range(shape.n)
    ]
    return mat_pack(tm, shape, elems)


def _matmul(
    tm: cvc5.TermManager,
    a: cvc5.Term,
    b: cvc5.Term,
    a_shape: MatShape,
    b_shape: MatShape,
) -> cvc5.Term:
    """Unrolled matrix product. Returns a term of shape (a.m, b.n)."""
    if a_shape.n != b_shape.m:
        raise ValueError(
            f"MatMul dim mismatch: {a_shape.m}x{a_shape.n} * {b_shape.m}x{b_shape.n}"
        )
    out_shape = MatShape(a_shape.m, b_shape.n)
    elems: list[cvc5.Term] = []
    for i in range(a_shape.m):
        for j in range(b_shape.n):
            prods = [
                tm.mkTerm(
                    Kind.MULT,
                    mat_select(tm, a, a_shape, i, k),
                    mat_select(tm, b, b_shape, k, j),
                )
                for k in range(a_shape.n)
            ]
            elems.append(prods[0] if len(prods) == 1 else tm.mkTerm(Kind.ADD, *prods))
    return mat_pack(tm, out_shape, elems)


def _argmax_flat(
    tm: cvc5.TermManager,
    x: cvc5.Term,
    shape: MatShape,
) -> cvc5.Term:
    """Scalar Int: row-major flat index of the max element of a `Mat t m n`.

    Matches `torch.argmax`, which flattens before searching and yields the
    single index `i * n + j` rather than an `[i, j]` pair. The comparison is
    strict against the incumbent, so the lowest flat index wins a tie, and
    the scan starts from element 0 rather than a neutral value. For a single
    row the flat index is the column index, which is what
    `Core.Mat.argmax_1d` computes; `Core.Mat.argmax` covers the general case.
    """
    xs = [
        mat_select(tm, x, shape, i, j)
        for i in range(shape.m)
        for j in range(shape.n)
    ]
    if not xs:
        raise ValueError(f"argmax needs a non-empty matrix, got {shape}")
    best_idx = tm.mkInteger(0)
    best_val = xs[0]
    for k in range(1, len(xs)):
        cond = tm.mkTerm(Kind.GT, xs[k], best_val)
        best_idx = tm.mkTerm(Kind.ITE, cond, tm.mkInteger(k), best_idx)
        best_val = tm.mkTerm(Kind.ITE, cond, xs[k], best_val)
    return best_idx


def _reduce_flat(
    tm: cvc5.TermManager,
    x: cvc5.Term,
    shape: MatShape,
    take_min: bool,
) -> cvc5.Term:
    """Scalar: the min/max element of a `Mat t m n`, folded row-major.

    `Min`/`Max` are *unary reductions* in the theory -- one read wire of any
    shape, a `Mat t 1 1` written -- exactly like `Argmax`, and `Core.Mat`
    folds them row-major from element `(0,0)`. They were wired to
    `_elementwise` with a binary lambda instead, so a real `Max` term raised
    `TypeError: <lambda>() missing 1 required positional argument`, taking
    down every SMT query about a module that uses one.
    """
    xs = [
        mat_select(tm, x, shape, i, j)
        for i in range(shape.m)
        for j in range(shape.n)
    ]
    if not xs:
        raise ValueError(f"Min/Max needs a non-empty matrix, got {shape}")
    kind = Kind.LEQ if take_min else Kind.GEQ
    best = xs[0]
    for nxt in xs[1:]:
        # `best` first, matching `foldl (fun best p => Min.min best (x p))`.
        best = tm.mkTerm(Kind.ITE, tm.mkTerm(kind, best, nxt), best, nxt)
    return best


def _bool_or_bv1(tm: cvc5.TermManager, pred: cvc5.Term, want_bv1: bool) -> cvc5.Term:
    """cvc5 comparison Kinds (EQUAL, BITVECTOR_ULT, ...) always yield a native
    Bool term. LIA/LRA comparisons want that Bool as-is, but BV comparisons are
    typed to write a `BV<1>` wire — reify the predicate as a 0/1 bit."""
    if not want_bv1:
        return pred
    return tm.mkTerm(Kind.ITE, pred, tm.mkBitVector(1, 1), tm.mkBitVector(1, 0))


@dataclass(frozen=True)
class SmtOp:
    """One term being translated, and every handle its emitter needs.

    The emitters live in `ops.py`, one cell of the op table, and that module
    must not import cvc5 — the Lean path has to keep working where cvc5 is
    absent. So everything cvc5 arrives here: the term manager, the `Kind`
    enum as `K`, and thin wrappers over this module's matrix primitives.
    """

    tm: cvc5.TermManager
    term: Term
    name: str
    args: list[cvc5.Term]
    in_shapes: list[MatShape]
    out_shape: MatShape

    #: `cvc5.Kind`, reached through the context rather than imported.
    K: ClassVar = Kind

    @property
    def write(self) -> Wire:
        """The single wire this term writes."""
        return self.term.write[0]

    @property
    def is_bv(self) -> bool:
        """True when the written wire is a BitVec, which is what decides
        between a BitVec Kind and its Bool/Int/Real counterpart."""
        return isinstance(self.write.dtype, BitVec)

    # -- builders ---------------------------------------------------------
    def mk(self, kind, *args: cvc5.Term) -> cvc5.Term:
        return self.tm.mkTerm(kind, *args)

    def unop(self, kind) -> Callable[[cvc5.Term], cvc5.Term]:
        return lambda a: self.tm.mkTerm(kind, a)

    def binop(self, kind) -> Callable[[cvc5.Term, cvc5.Term], cvc5.Term]:
        return lambda a, b: self.tm.mkTerm(kind, a, b)

    def integer(self, value: int) -> cvc5.Term:
        return self.tm.mkInteger(value)

    def bitvector(self, width: int, value: int) -> cvc5.Term:
        return self.tm.mkBitVector(width, value)

    def const(self, dt: Sort, raw) -> cvc5.Term:
        return _scalar_const(self.tm, dt, raw)

    def tensor(self, wire: Wire, data) -> cvc5.Term:
        return _tensor_const(self.tm, wire, data)

    # -- matrices ---------------------------------------------------------
    def shape(self, m: int, n: int) -> MatShape:
        return MatShape(m, n)

    def scalar(self, index: int, i: int = 0, j: int = 0) -> cvc5.Term:
        """Element `[i][j]` of read operand `index`."""
        return mat_select(self.tm, self.args[index], self.in_shapes[index], i, j)

    def pack(self, elems: list[cvc5.Term]) -> cvc5.Term:
        """Build the written wire's matrix from its row-major elements."""
        return mat_pack(self.tm, self.out_shape, elems)

    def pack_at(self, shape: MatShape, elems: list[cvc5.Term]) -> cvc5.Term:
        return mat_pack(self.tm, shape, elems)

    def elementwise(self, op: Callable[..., cvc5.Term], *mats: cvc5.Term) -> cvc5.Term:
        return _elementwise(self.tm, self.out_shape, op, *mats)

    def matmul(
        self, a: cvc5.Term, b: cvc5.Term, a_shape: MatShape, b_shape: MatShape
    ) -> cvc5.Term:
        return _matmul(self.tm, a, b, a_shape, b_shape)

    def argmax_flat(self, index: int) -> cvc5.Term:
        return _argmax_flat(self.tm, self.args[index], self.in_shapes[index])

    def reduce_flat(self, index: int, take_min: bool) -> cvc5.Term:
        return _reduce_flat(self.tm, self.args[index], self.in_shapes[index], take_min)

    def bool_or_bv1(self, pred: cvc5.Term, want_bv1: bool) -> cvc5.Term:
        return _bool_or_bv1(self.tm, pred, want_bv1)


def translate_terms(
    tm: cvc5.TermManager,
    terms: list[Term],
    input_bindings: dict[int, cvc5.Term],
) -> dict[int, cvc5.Term]:
    """Walk Python IR `terms`, producing wire_id → cvc5.Term for every wire.

    `input_bindings` maps block-input wire IDs to their already-built
    cvc5 terms. The returned dict is a superset including every computed
    wire.

    Every variant is dispatched through `ops.OPS` — constants included — so
    what this encoder covers is one column of that table rather than the
    shape of an if/elif chain, and a variant it does not cover is refused
    with the reason the table records.
    """
    wt: dict[int, cvc5.Term] = dict(input_bindings)

    for term in terms:
        emit = smt_emitter(term.itype)
        write = term.write[0]
        wt[write.id] = emit(
            SmtOp(
                tm=tm,
                term=term,
                name=itype_name(term.itype),
                args=[wt[w.id] for w in term.read],
                in_shapes=[wire_shape(w) for w in term.read],
                out_shape=wire_shape(write),
            )
        )

    return wt
