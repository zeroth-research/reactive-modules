from zrth import Wire, DType


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
    if isinstance(wire.dtype, DType.Bool):
        val = bool(tensor.item())
        lean_val = "true" if val else "false"
        return f"@[simp] def {name} : Bool := {lean_val}\n"

    # Scalar int constant
    if isinstance(wire.dtype, DType.Int) and (shape == [1] or shape == []):
        val = int(tensor.item())
        return f"@[simp] def {name} : Int := {val}\n"

    # Matrix constant
    if isinstance(wire.dtype, DType.Int) and len(shape) >= 1:
        if len(shape) == 1:
            m, n = shape[0], 1
        else:
            m, n = shape[0], shape[1]

        lines = [f"@[simp] def {name} : Fin {m} → Fin {n} → Int := fun i j =>"]
        lines.append("  match i, j with")

        data = tensor.reshape(m, n)
        for i in range(m):
            row_entries = []
            for j in range(n):
                val = int(data[i, j].item())
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
