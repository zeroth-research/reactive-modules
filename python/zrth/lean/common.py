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


def dtype_to_lean_type(wire: Wire, simple_types=False) -> str:
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
        return ty if simple_types else f"(Mat {ty} 1 1)"
    elif len(shape) == 1:
        return f"(Mat {ty} 1 {shape[0]})"
    elif len(shape) == 2:
        return f"(Mat {ty} {shape[0]} {shape[1]})"
    else:
        raise ValueError(f"Unsupported DType shape: {shape}")

    raise ValueError(f"Unsupported DType for Lean conversion: {dt}")


def itype_name(itype) -> str:
    """Get the variant name of an IType, e.g. IType.Add() -> 'Add'."""
    name = type(itype).__name__
    # PyO3 exports variants as IType_Add, IType_Tensor, etc.
    if name.startswith("IType_"):
        return name[6:]
    return name


# ======================================================================
#  Constants
# ======================================================================
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


def _get_dtype_item(dtype: DType, item) -> str:
    if isinstance(dtype, DType.Bool):
        return "true" if bool(item) else "false"
    if isinstance(dtype, DType.Int):
        return str(int(item))
    if isinstance(dtype, DType.Float):
        return str(float(item))
    raise NotImplementedError(f"Unhnadled type: {dtype}")


def _tensor_to_lean_def(name: str, tensor, wire: Wire) -> str:
    """
    Generate a top-level Lean definition for a constant tensor.

    E.g.:
        @[simp] def A : Fin 3 → Fin 2 → Int := fun i j =>
          match i, j with
          | 0, 0 => 0 | 0, 1 => 1
          ...
    """
    shape = wire.dtype.shape

    # Scalar bool constant
    # if isinstance(wire.dtype, DType.Bool):
    #     val = bool(tensor.item())
    #     lean_val = "true" if val else "false"
    #     return f"@[simp] def {name} : Bool := {lean_val}\n"
    #
    # # Scalar int constant
    # if isinstance(wire.dtype, DType.Int) and (shape == [1] or shape == []):
    #     val = int(tensor.item())
    #     return f"@[simp] def {name} : Int := {val}\n"
    if shape == [1] or shape == []:
        val = _get_dtype_item(wire.dtype, tensor.item())
        ty = dtype_to_lean_type(wire)
        assert "Mat" in ty and " 1 1" in ty, ty
        return f"@[simp] def {name} : {ty} := fun _ _ => {val}\n"

    # Matrix constant
    if len(shape) >= 1:
        if len(shape) == 1:
            m, n = 1, shape[0]
        else:
            m, n = shape[0], shape[1]

        lines = [f"@[simp] def {name} : {dtype_to_lean_type(wire)} := fun i j =>"]
        lines.append("  match i, j with")

        data = tensor.reshape(m, n)
        for i in range(m):
            row_entries = []
            for j in range(n):
                val = _get_dtype_item(wire.dtype, data[i, j].item())
                row_entries.append(f"| {i}, {j} => {val}")
            lines.append("  " + " ".join(row_entries))

        return "\n".join(lines) + "\n"

    raise ValueError(
        f"Cannot generate Lean constant for dtype={wire.dtype}, shape={shape}"
    )


def _is_scalar_tensor(wire: Wire) -> bool:
    """True if the wire carries a scalar Bool or Int (not a matrix)."""
    dt = wire.dtype
    if isinstance(dt, DType.Bool):
        return True
    if isinstance(dt, DType.Int):
        shape = dt.shape
        return shape == [] or shape == [1]
    return False


def _tensor_to_lean_inline(tensor, wire: Wire) -> str:
    """Return an inline Lean literal for a scalar tensor."""
    if isinstance(wire.dtype, DType.Bool):
        return "true" if bool(tensor.item()) else "false"
    if isinstance(wire.dtype, DType.Int):
        return str(int(tensor.item()))
    raise ValueError(f"Cannot inline tensor with dtype={wire.dtype}")
