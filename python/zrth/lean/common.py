from zrth.lean.constants import _tensor_to_lean_inline
from zrth import Term, Wire, DType


def _accessor(pos: int, total: int) -> str:
    """Accessor for position `pos` in a product (tuple) of `total` elements.

    total=1: '' (value itself)
    total=2: '.1', '.2'
    total=3: '.1', '.2.1', '.2.2'
    total=4: '.1', '.2.1', '.2.2.1', '.2.2.2'
    """
    if total == 1:
        return ""
    if pos == total - 1:
        return ".2" * pos
    return ".2" * pos + ".1"


def dtype_to_lean_type(wire: Wire) -> str:
    """Map a Wire's DType to a native Lean type (Bool, Int, Fin m → Fin n → Int)."""
    dt = wire.dtype
    shape = dt.shape

    if isinstance(dt, DType.Bool):
        ty = "Bool"
    elif isinstance(dt, DType.Int):
        ty = "Int"
    elif isinstance(dt, DType.Float):
        # TODO: Float is *NOT* Real, but we stick to that for proofs atm
        ty = "Real"
    else:
        raise ValueError(f"Unsupported DType for Lean conversion: {dt}")

    if shape == [1] or shape == []:
        return ty
    elif len(shape) == 1:
        return f"(Mat {ty} 1 {shape[0]})"
    elif len(shape) == 2:
        return f"(Mat {ty} {shape[0]} {shape[1]})"
    else:
        raise ValueError(f"Unsupported Bool shape: {shape}")

    raise ValueError(f"Unsupported DType for Lean conversion: {dt}")


def itype_name(itype) -> str:
    """Get the variant name of an IType, e.g. IType.Add() -> 'Add'."""
    name = type(itype).__name__
    # PyO3 exports variants as IType_Add, IType_Tensor, etc.
    if name.startswith("IType_"):
        return name[6:]
    return name


def _constant_expr(
    const_name: str, term: Term, w: Wire, constants: dict[int, str]
) -> str:
    if const_name == "Tensor":
        wire_id = w.id
        if wire_id in constants:
            return constants[wire_id]
        else:
            return _tensor_to_lean_inline(term.itype._0, w)
    elif const_name == "ConstBool":
        val = bool(term.itype._0)
        return "true" if val else "false"
    else:
        return str(int(term.itype._0))

    raise NotImplementedError
