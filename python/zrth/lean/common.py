from zrth import Term, Wire, DType, Module


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
    if len(shape) == 1:
        return f"(Mat {ty} 1 {shape[0]})"
    if len(shape) == 2:
        return f"(Mat {ty} {shape[0]} {shape[1]})"
    raise ValueError(f"Unsupported DType shape: {shape}")


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
class ConstantRegistry:
    """Registry of matrix constants (wire_id -> Lean name + top-level def).

    Scalar Bool/Int tensors are inlined at their use site and not registered.
    """

    def __init__(self):
        self._by_id: dict[int, str] = {}
        self._defs: list[str] = []
        self._counter = 0

    def intern(self, term: Term) -> None:
        """If term is a matrix Tensor, mint a name and top-level def for it.

        Idempotent: calling again for the same output wire is a no-op.
        """
        if itype_name(term.itype) != "Tensor":
            return
        out_wire = term.write[0]
        if _is_scalar_tensor(out_wire) or out_wire.id in self._by_id:
            return
        name = f"c{self._counter}"
        self._counter += 1
        self._by_id[out_wire.id] = name
        self._defs.append(_tensor_to_lean_def(name, term.itype._0, out_wire))

    def lookup(self, wire_id: int) -> str | None:
        return self._by_id.get(wire_id)

    def names(self) -> list[str]:
        return list(self._by_id.values())

    def defs(self) -> list[str]:
        return list(self._defs)


def _constant_expr(
    const_name: str, term: Term, w: Wire, constants: "ConstantRegistry"
) -> str:
    if const_name == "Tensor":
        name = constants.lookup(w.id)
        if name is not None:
            return name
        return _tensor_to_lean_inline(term.itype._0, w)
    if const_name == "ConstBool":
        val = "true" if bool(term.itype._0) else "false"
        return f"(fun _ _ => {val})"
    return f"(fun _ _ => ({int(term.itype._0)} : Int))"


def _any_float_wire(*wire_lists: list[Wire]) -> bool:
    """True if any wire across the given lists has a Float dtype."""
    for wires in wire_lists:
        for w in wires:
            if isinstance(w.dtype, DType.Float):
                return True
    return False


def _bind_wires(params: list[tuple[str, list[Wire]]]) -> dict[int, str]:
    """Map each input wire id to its Lean accessor expression.

    E.g. [("ctrl", [w0, w1]), ("extl_n", [w2])] ->
        {w0.id: "ctrl.1", w1.id: "ctrl.2", w2.id: "extl_n"}
    """
    out: dict[int, str] = {}
    for name, wires in params:
        n = len(wires)
        for i, w in enumerate(wires):
            out[w.id] = f"{name}{_accessor(i, n)}"
    return out


class LeanContext:
    """Pre-computed artifacts shared by module and certificate codegen.

    Runs a one-time discovery pass over the module's init/update terms (and
    optionally the certificate terms) to populate the constant registry and
    the wire-name bindings for each block. Read-only after construction;
    codegen consumers do not mutate it.
    """

    def __init__(self, module: Module, cert_terms: "list | None" = None):
        atoms = list(module.atoms)
        if len(atoms) != 1:
            raise ValueError(
                f"LeanContext currently supports single-atom modules, got {len(atoms)}"
            )

        self.module = module
        self.atom = atoms[0]

        self.extl_latched: list[Wire] = [p[0] for p in module.extl]
        self.extl_next: list[Wire] = [p[1] for p in module.extl]
        self.ctrl_latched: list[Wire] = [p[0] for p in module.ctrl]
        self.ctrl_next: list[Wire] = [p[1] for p in module.ctrl]

        self.constants = ConstantRegistry()
        for term in self.atom.init:
            self.constants.intern(term)
        for term in self.atom.update:
            self.constants.intern(term)
        for term in cert_terms or []:
            self.constants.intern(term)

        # Modules that use Float/Real require `noncomputable` on defs that
        # compare or branch on real-typed values (Real lacks decidable equality).
        self.uses_real = _any_float_wire(
            self.extl_latched,
            self.extl_next,
            self.ctrl_latched,
            self.ctrl_next,
            [t.write[0] for t in self.atom.init],
            [t.write[0] for t in self.atom.update],
        )

        self.init_wire_names = _bind_wires([("extl_n", self.extl_next)])
        self.update_wire_names = _bind_wires(
            [
                ("ctrl", self.ctrl_latched),
                ("extl_l", self.extl_latched),
                ("extl_n", self.extl_next),
            ]
        )


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
    """Return an inline `Mat _ 1 1` literal for a scalar tensor."""
    if isinstance(wire.dtype, DType.Bool):
        val = "true" if bool(tensor.item()) else "false"
        return f"(fun _ _ => {val})"
    if isinstance(wire.dtype, DType.Int):
        return f"(fun _ _ => ({int(tensor.item())} : Int))"
    raise ValueError(f"Cannot inline tensor with dtype={wire.dtype}")
