"""SMT encoding utilities for the CEGAR magic driver.

Maps Python IR types (`Sort`) to cvc5 sorts, and provides element access
helpers for matrix-shaped state components.

Matrix representation
---------------------
`Mat t m n` is encoded as a flat cvc5 tuple of `m*n` values, row-major
(element `[i][j]` at index `i*n + j`). When `m = n = 1`, the tuple
collapses to the scalar sort `t` — this matches the common Lean use of
`Mat t 1 1` for scalar values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cvc5
from cvc5 import Kind

from zrth import Wire, Sort, Term
from .common import itype_name, dtype_shape


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
    if isinstance(dt, Sort.Bool):
        return tm.getBooleanSort()
    if isinstance(dt, Sort.Int):
        return tm.getIntegerSort()
    if isinstance(dt, Sort.Real):
        return tm.getRealSort()
    if isinstance(dt, Sort.BitVec):
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
    if isinstance(dt, Sort.Bool):
        return tm.mkBoolean(bool(raw))
    if isinstance(dt, Sort.Int):
        return tm.mkInteger(int(raw))
    if isinstance(dt, Sort.Real):
        return tm.mkReal(float(raw))
    if isinstance(dt, Sort.BitVec):
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


def _argmax_1d(
    tm: cvc5.TermManager,
    x: cvc5.Term,
    shape: MatShape,
) -> cvc5.Term:
    """Scalar Int: index of max element in a `Mat t 1 n` term."""
    if shape.m != 1:
        raise ValueError(f"argmax_1d needs a 1-row matrix, got {shape}")
    n = shape.n
    xs = [mat_select(tm, x, shape, 0, j) for j in range(n)]
    best_idx = tm.mkInteger(0)
    best_val = xs[0]
    for j in range(1, n):
        cond = tm.mkTerm(Kind.GT, xs[j], best_val)
        best_idx = tm.mkTerm(Kind.ITE, cond, tm.mkInteger(j), best_idx)
        best_val = tm.mkTerm(Kind.ITE, cond, xs[j], best_val)
    return best_idx


def _unop_scalar(tm, kind):
    return lambda a: tm.mkTerm(kind, a)


def _binop_scalar(tm, kind):
    return lambda a, b: tm.mkTerm(kind, a, b)


def _bool_or_bv1(tm: cvc5.TermManager, pred: cvc5.Term, want_bv1: bool) -> cvc5.Term:
    """cvc5 comparison Kinds (EQUAL, BITVECTOR_ULT, ...) always yield a native
    Bool term. LIA/LRA comparisons want that Bool as-is, but BV comparisons are
    typed to write a `BV<1>` wire — reify the predicate as a 0/1 bit."""
    if not want_bv1:
        return pred
    return tm.mkTerm(Kind.ITE, pred, tm.mkBitVector(1, 1), tm.mkBitVector(1, 0))


def translate_terms(
    tm: cvc5.TermManager,
    terms: list[Term],
    input_bindings: dict[int, cvc5.Term],
) -> dict[int, cvc5.Term]:
    """Walk Python IR `terms`, producing wire_id → cvc5.Term for every wire.

    `input_bindings` maps block-input wire IDs to their already-built
    cvc5 terms. The returned dict is a superset including every computed
    wire.
    """
    wt: dict[int, cvc5.Term] = dict(input_bindings)

    for term in terms:
        name = itype_name(term.itype)
        write = term.write[0]
        out_shape = wire_shape(write)

        if name == "Tensor":
            wt[write.id] = _tensor_const(tm, write, term.itype._0)
            continue
        if name in ("ConstBool", "ConstInt", "ConstReal", "Const"):
            payload = term.itype._0
            numel = payload.numel() if hasattr(payload, "numel") else 1
            if numel > 1:
                # matrix constant: materialize per element (payload matches shape)
                wt[write.id] = _tensor_const(tm, write, payload)
            else:
                # BV.Const's element type isn't encoded in the name (unlike
                # ConstBool/ConstInt/ConstReal) — read it off the write wire.
                elem = (
                    write.dtype
                    if name == "Const"
                    else {
                        "ConstBool": Sort.Bool([1, 1]),
                        "ConstInt": Sort.Int([1, 1]),
                        "ConstReal": Sort.Real([1, 1]),
                    }[name]
                )
                v = _scalar_const(tm, elem, payload)
                wt[write.id] = mat_pack(tm, out_shape, [v] * out_shape.total)
            continue

        args = [wt[w.id] for w in term.read]
        in_shapes = [wire_shape(w) for w in term.read]

        is_bv = isinstance(write.dtype, Sort.BitVec)

        if name == "Id":
            wt[write.id] = args[0]
        elif name == "Not":
            kind = Kind.BITVECTOR_NOT if is_bv else Kind.NOT
            wt[write.id] = _elementwise(tm, out_shape, _unop_scalar(tm, kind), args[0])
        elif name == "And":
            kind = Kind.BITVECTOR_AND if is_bv else Kind.AND
            wt[write.id] = _elementwise(tm, out_shape, _binop_scalar(tm, kind), *args)
        elif name == "Or":
            kind = Kind.BITVECTOR_OR if is_bv else Kind.OR
            wt[write.id] = _elementwise(tm, out_shape, _binop_scalar(tm, kind), *args)
        elif name == "Xor":
            kind = Kind.BITVECTOR_XOR if is_bv else Kind.XOR
            wt[write.id] = _elementwise(tm, out_shape, _binop_scalar(tm, kind), *args)
        elif name == "Ite":
            # cond is Mat Bool 1 1 (LIA/LRA) or Mat BV<1> 1 1 (BV); extract its
            # scalar, coercing a BV<1> condition to a genuine Bool via `!= 0`.
            cond = mat_select(tm, args[0], in_shapes[0], 0, 0)
            cond_dtype = term.read[0].dtype
            if isinstance(cond_dtype, Sort.BitVec):
                cond = tm.mkTerm(
                    Kind.DISTINCT, cond, tm.mkBitVector(cond_dtype._0, 0)
                )
            wt[write.id] = tm.mkTerm(Kind.ITE, cond, args[1], args[2])
        elif name == "Add":
            kind = Kind.BITVECTOR_ADD if is_bv else Kind.ADD
            wt[write.id] = _elementwise(tm, out_shape, _binop_scalar(tm, kind), *args)
        elif name == "Sub":
            kind = Kind.BITVECTOR_SUB if is_bv else Kind.SUB
            wt[write.id] = _elementwise(tm, out_shape, _binop_scalar(tm, kind), *args)
        elif name == "Mul":
            kind = Kind.BITVECTOR_MULT if is_bv else Kind.MULT
            wt[write.id] = _elementwise(tm, out_shape, _binop_scalar(tm, kind), *args)
        elif name == "Neg":
            kind = Kind.BITVECTOR_NEG if is_bv else Kind.NEG
            wt[write.id] = _elementwise(
                tm, out_shape, lambda a: tm.mkTerm(kind, a), args[0]
            )
        elif name in ("UDiv", "SDiv", "UMod", "SMod"):
            kind = {
                "UDiv": Kind.BITVECTOR_UDIV,
                "SDiv": Kind.BITVECTOR_SDIV,
                "UMod": Kind.BITVECTOR_UREM,
                "SMod": Kind.BITVECTOR_SMOD,
            }[name]
            wt[write.id] = _elementwise(tm, out_shape, _binop_scalar(tm, kind), *args)
        elif name == "BVToBool":
            # Despite the name, output is BV<1> (non-zero test), not Bool.
            a = mat_select(tm, args[0], in_shapes[0], 0, 0)
            bw = term.read[0].dtype._0
            pred = tm.mkTerm(Kind.DISTINCT, a, tm.mkBitVector(bw, 0))
            wt[write.id] = mat_pack(tm, out_shape, [_bool_or_bv1(tm, pred, True)])
        elif name == "Mod":
            a = mat_select(tm, args[0], in_shapes[0], 0, 0)
            b = mat_select(tm, args[1], in_shapes[1], 0, 0)
            wt[write.id] = mat_pack(tm, out_shape, [tm.mkTerm(Kind.INTS_MODULUS, a, b)])
        elif name in ("Lt", "Le", "Gt", "Ge", "Eq", "Ne"):
            kind = {
                "Lt": Kind.LT,
                "Le": Kind.LEQ,
                "Gt": Kind.GT,
                "Ge": Kind.GEQ,
                "Eq": Kind.EQUAL,
                "Ne": Kind.DISTINCT,
            }[name]
            a = mat_select(tm, args[0], in_shapes[0], 0, 0)
            b = mat_select(tm, args[1], in_shapes[1], 0, 0)
            # LIA/LRA Eq/Ne write Bool; BV.Eq/BV.Ne write BV<1> (bv.rs).
            pred = tm.mkTerm(kind, a, b)
            wt[write.id] = mat_pack(tm, out_shape, [_bool_or_bv1(tm, pred, is_bv)])
        elif name in ("ULe", "ULt", "UGe", "UGt", "SLe", "SLt", "SGe", "SGt"):
            kind = {
                "ULe": Kind.BITVECTOR_ULE,
                "ULt": Kind.BITVECTOR_ULT,
                "UGe": Kind.BITVECTOR_UGE,
                "UGt": Kind.BITVECTOR_UGT,
                "SLe": Kind.BITVECTOR_SLE,
                "SLt": Kind.BITVECTOR_SLT,
                "SGe": Kind.BITVECTOR_SGE,
                "SGt": Kind.BITVECTOR_SGT,
            }[name]
            a = mat_select(tm, args[0], in_shapes[0], 0, 0)
            b = mat_select(tm, args[1], in_shapes[1], 0, 0)
            # These are BV-only ops; output is BV<1>, not the native Bool cvc5
            # gives back from BITVECTOR_U*/S* comparison Kinds.
            pred = tm.mkTerm(kind, a, b)
            wt[write.id] = mat_pack(tm, out_shape, [_bool_or_bv1(tm, pred, True)])
        elif name == "Min":
            wt[write.id] = _elementwise(
                tm,
                out_shape,
                lambda x, y: tm.mkTerm(Kind.ITE, tm.mkTerm(Kind.LEQ, x, y), x, y),
                *args,
            )
        elif name == "Max":
            wt[write.id] = _elementwise(
                tm,
                out_shape,
                lambda x, y: tm.mkTerm(Kind.ITE, tm.mkTerm(Kind.GEQ, x, y), x, y),
                *args,
            )
        elif name == "MatMul":
            wt[write.id] = _matmul(tm, args[0], args[1], in_shapes[0], in_shapes[1])
        elif name == "Linear":
            # Convention Y = A·X + B: A ([out,in]) and B ([out,1] or empty) are
            # baked into the op; args[0] is the single read wire X ([in,batch]).
            A_tensor = term.itype._0
            B_tensor = term.itype._1
            a_rows, a_cols = int(A_tensor.shape[0]), int(A_tensor.shape[1])
            A_shape = MatShape(a_rows, a_cols)
            A_data = A_tensor.reshape(a_rows, a_cols)
            A_term = mat_pack(
                tm,
                A_shape,
                [
                    _scalar_const(tm, write.dtype, A_data[i, j].item())
                    for i in range(a_rows)
                    for j in range(a_cols)
                ],
            )
            ax = _matmul(tm, A_term, args[0], A_shape, in_shapes[0])
            if B_tensor.numel() == 0:
                wt[write.id] = ax
            else:
                B_data = B_tensor.reshape(int(B_tensor.shape[0]), int(B_tensor.shape[1]))
                B_term = mat_pack(
                    tm,
                    out_shape,
                    [
                        _scalar_const(tm, write.dtype, B_data[i, 0].item())
                        for i in range(out_shape.m)
                        for _ in range(out_shape.n)
                    ],
                )
                wt[write.id] = _elementwise(
                    tm, out_shape, _binop_scalar(tm, Kind.ADD), ax, B_term
                )
        elif name == "ReLU":
            zero = _scalar_const(tm, write.dtype, 0)
            wt[write.id] = _elementwise(
                tm,
                out_shape,
                lambda a: tm.mkTerm(Kind.ITE, tm.mkTerm(Kind.GEQ, a, zero), a, zero),
                args[0],
            )
        elif name == "TensorGet":
            wt[write.id] = mat_pack(
                tm, out_shape, [mat_select(tm, args[0], in_shapes[0], 0, 0)]
            )
        elif name == "ToUnsigned":
            a = mat_select(tm, args[0], in_shapes[0], 0, 0)
            zero = tm.mkInteger(0)
            cast = tm.mkTerm(Kind.ITE, tm.mkTerm(Kind.GEQ, a, zero), a, zero)
            wt[write.id] = mat_pack(tm, out_shape, [cast])
        elif name == "Argmax":
            idx = _argmax_1d(tm, args[0], in_shapes[0])
            # _argmax_1d builds an Int index; LRA modules carry it on a Real wire
            if isinstance(write.dtype, Sort.Real):
                idx = tm.mkTerm(Kind.TO_REAL, idx)
            wt[write.id] = mat_pack(tm, out_shape, [idx])
        else:
            raise ValueError(f"SMT translator: unsupported IType {name}")

    return wt
