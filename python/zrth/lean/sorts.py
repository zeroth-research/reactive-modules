"""Sorts and shapes of wires, and the Lean types they print as."""

from __future__ import annotations

from zrth import BitVec, Bool, Int, Real, Sort, Wire


def dtype_shape(dt) -> list:
    """Shape of a Sort, across all element types.

    Bool/Int/Real carry the shape in field ``_0``; BitVec carries the bit-width
    in ``_0`` and the shape in ``_1``. main always uses 2-D shapes (a scalar is
    ``[1, 1]``).
    """
    if isinstance(dt, BitVec):
        return list(dt._1)
    return list(dt._0)


def _is_scalar_shape(shape: list) -> bool:
    """True if the shape denotes a single element ([], [1], [1, 1], ...)."""
    return all(d == 1 for d in shape)


def lean_elem_type(sort: Sort) -> str:
    """The Lean type of one element of `sort`, ignoring its shape: `Bool`,
    `Int`, `Real` (proofs read the module's floats as reals) or
    `(BitVec w)`."""
    if isinstance(sort, Bool):
        return "Bool"
    if isinstance(sort, Int):
        return "Int"
    if isinstance(sort, Real):
        return "Real"
    if isinstance(sort, BitVec):
        return f"(BitVec {sort._0})"
    raise ValueError(f"Unsupported Sort for Lean conversion: {sort}")


def dtype_to_lean_type(wire: Wire, simple_types=False) -> str:
    """Map a Wire's Sort to a native Lean type (Bool, Int, Fin m → Fin n → Int)."""

    dt = wire.dtype
    shape = dtype_shape(dt)
    ty = lean_elem_type(dt)

    if _is_scalar_shape(shape):
        return ty if simple_types else f"(Mat {ty} 1 1)"
    if len(shape) == 1:
        return f"(Mat {ty} 1 {shape[0]})"
    if len(shape) == 2:
        return f"(Mat {ty} {shape[0]} {shape[1]})"
    raise ValueError(f"Unsupported Sort shape: {shape}")


def itype_name(itype) -> str:
    """Get the variant name of an op, e.g. LIA.Add() -> 'Add'."""
    name = type(itype).__name__
    # PyO3 exports theory ops as LIA_Add, LRA_Real, BV_MatMul, etc.
    for prefix in ("LIA_", "LRA_", "BV_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# Constant op variants across theories (LIA.Int/Bool, LRA.Real/Bool,
# BV.Const). The element type is encoded in the variant name;
# scalar-vs-matrix is decided by the wire's shape.
_CONST_VARIANTS = frozenset({"Int", "Real", "Bool", "Const"})


def is_constant_name(name: str) -> bool:
    """True if the variant name denotes a constant in any theory."""
    return name in _CONST_VARIANTS


def _is_scalar_tensor(wire: Wire) -> bool:
    """True if the wire carries a scalar Bool, Int or BitVec (not a matrix).

    Bool used to answer `True` for any shape, skipping the check the other
    two get, so a Bool *matrix* constant was never interned and reached
    `_tensor_to_lean_inline`, which calls `.item()` on it.
    """
    dt = wire.dtype
    if isinstance(dt, (Bool, Int, BitVec)):
        return _is_scalar_shape(dtype_shape(dt))
    return False


def _is_scalar_wire(wire: Wire) -> bool:
    """True if the wire is a scalar (single element, any scalar dtype)."""
    return _is_scalar_shape(dtype_shape(wire.dtype))


# ---------------------------------------------------------------------------
# Flat-scalar helpers — support multi-element (Mat T 1 n) wires in the
# scalar / relational encoding by expanding each wire into n scalars.
# ---------------------------------------------------------------------------

def _flat_element_type(wire: Wire) -> str:
    """Base Lean scalar type for one element of the wire (no Mat wrapper)."""
    return lean_elem_type(wire.dtype)
