from zrth import Term, Wire, Sort, Module


def dtype_shape(dt) -> list:
    """Shape of a Sort, across all element types.

    Bool/Int/Real carry the shape in field ``_0``; BitVec carries the bit-width
    in ``_0`` and the shape in ``_1``. main always uses 2-D shapes (a scalar is
    ``[1, 1]``).
    """
    if isinstance(dt, Sort.BitVec):
        return list(dt._1)
    return list(dt._0)


def _is_scalar_shape(shape: list) -> bool:
    """True if the shape denotes a single element ([], [1], [1, 1], ...)."""
    return all(d == 1 for d in shape)


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
    """Map a Wire's Sort to a native Lean type (Bool, Int, Fin m → Fin n → Int)."""

    dt = wire.dtype
    shape = dtype_shape(dt)

    if isinstance(dt, Sort.Bool):
        ty = "Bool"
    elif isinstance(dt, Sort.Int):
        ty = "Int"
    elif isinstance(dt, Sort.Real):
        # TODO: Float is *NOT* Real, but we stick to that for proofs atm
        ty = "Real"
    else:
        raise ValueError(f"Unsupported Sort for Lean conversion: {dt}")

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
    # PyO3 exports theory ops as LIA_Add, LRA_ConstReal, BV_MatMul, etc.
    for prefix in ("LIA_", "LRA_", "BV_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# Constant op variants across theories (LIA.ConstInt/ConstBool,
# LRA.ConstReal/ConstBool, BV.Const). The element type is encoded in the
# variant name; scalar-vs-matrix is decided by the wire's shape.
_CONST_VARIANTS = frozenset({"ConstInt", "ConstReal", "ConstBool", "Const"})


def is_constant_name(name: str) -> bool:
    """True if the variant name denotes a constant in any theory."""
    return name in _CONST_VARIANTS


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
        if not is_constant_name(itype_name(term.itype)):
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
    # Matrix constants are interned as top-level defs; scalars are inlined as
    # a 1x1 matrix literal driven by the wire's element type (Bool/Int/Real).
    name = constants.lookup(w.id)
    if name is not None:
        return name
    return _tensor_to_lean_inline(term.itype._0, w)


def _any_float_wire(*wire_lists: list[Wire]) -> bool:
    """True if any wire across the given lists has a Float dtype."""
    for wires in wire_lists:
        for w in wires:
            if isinstance(w.dtype, Sort.Real):
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


def _get_dtype_item(dtype: Sort, item) -> str:
    if isinstance(dtype, Sort.Bool):
        return "true" if bool(item) else "false"
    if isinstance(dtype, Sort.Int):
        return str(int(item))
    if isinstance(dtype, Sort.Real):
        return _float_literal(float(item))
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
    shape = dtype_shape(wire.dtype)

    if _is_scalar_shape(shape):
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
    if isinstance(dt, Sort.Bool):
        return True
    if isinstance(dt, Sort.Int):
        return _is_scalar_shape(dtype_shape(dt))
    return False


def _is_scalar_wire(wire: Wire) -> bool:
    """True if the wire is a scalar (single element, any scalar dtype)."""
    return _is_scalar_shape(dtype_shape(wire.dtype))


def _float_literal(v: float) -> str:
    """Lean Real literal for a float value."""
    if v == int(v):
        return f"({int(v)} : Real)"
    return f"({v} : Real)"


def _tensor_to_lean_inline(tensor, wire: Wire) -> str:
    """Return an inline `Mat _ 1 1` literal for a scalar tensor."""
    if isinstance(wire.dtype, Sort.Bool):
        val = "true" if bool(tensor.item()) else "false"
        return f"(fun _ _ => {val})"
    if isinstance(wire.dtype, Sort.Int):
        return f"(fun _ _ => ({int(tensor.item())} : Int))"
    if isinstance(wire.dtype, Sort.Real):
        return f"(fun _ _ => {_float_literal(float(tensor.item()))})"
    raise ValueError(f"Cannot inline tensor with dtype={wire.dtype}")


def _tensor_to_lean_scalar(tensor, wire: Wire) -> str:
    """Return a bare scalar literal (no Mat wrapper) for a scalar tensor."""
    if isinstance(wire.dtype, Sort.Bool):
        return "true" if bool(tensor.item()) else "false"
    if isinstance(wire.dtype, Sort.Int):
        return f"({int(tensor.item())} : Int)"
    if isinstance(wire.dtype, Sort.Real):
        return _float_literal(float(tensor.item()))
    raise ValueError(f"Cannot inline scalar for dtype={wire.dtype}")


def _bind_wires_scalar(params: list[tuple[str, list[Wire]]]) -> dict[int, str]:
    """Like _bind_wires but appends ' 0 0' for scalar (Mat 1 1) input wires."""
    out: dict[int, str] = {}
    for name, wires in params:
        n = len(wires)
        for i, w in enumerate(wires):
            base = f"{name}{_accessor(i, n)}"
            out[w.id] = f"{base} 0 0" if _is_scalar_wire(w) else base
    return out


# ---------------------------------------------------------------------------
# Flat-scalar helpers — support multi-element (Mat T 1 n) wires in the
# scalar / relational encoding by expanding each wire into n scalars.
# ---------------------------------------------------------------------------

def _flat_element_type(wire: Wire) -> str:
    """Base Lean scalar type for one element of the wire (no Mat wrapper)."""
    dt = wire.dtype
    if isinstance(dt, Sort.Bool):
        return "Bool"
    if isinstance(dt, Sort.Int):
        return "Int"
    if isinstance(dt, Sort.Real):
        return "Real"
    raise ValueError(f"Unsupported Sort for scalar element: {dt}")


def _flat_indices(wire: Wire) -> list[tuple[int, int]]:
    """Row-major (row, col) indices for all elements of the wire.

    shape [] or [1] → [(0, 0)]
    shape [n]       → [(0, 0), (0, 1), ..., (0, n-1)]
    shape [m, n]    → [(i, j) for i in range(m) for j in range(n)]
    """
    shape = dtype_shape(wire.dtype)
    if _is_scalar_shape(shape):
        return [(0, 0)]
    if len(shape) == 1:
        return [(0, j) for j in range(shape[0])]
    if len(shape) == 2:
        return [(i, j) for i in range(shape[0]) for j in range(shape[1])]
    raise ValueError(f"Shape {shape} not supported for scalar encoding")


def _flat_size(wire: Wire) -> int:
    """Total number of scalar elements in the wire."""
    return len(_flat_indices(wire))


def _vec_from_scalars(scalars: list[str], elem_ty: str) -> str:
    """Build a ``Fin n → T`` expression using a typed ``Fin.cons`` chain.

    Annotates the whole chain with ``: Fin n → T`` so Lean knows the
    application type before elaborating the ``j`` argument — without this,
    the application site has type ``?m j`` which Lean can't unify with ``T``.
    """
    n = len(scalars)
    result = f"(Fin.elim0 : Fin 0 → {elem_ty})"
    for s in reversed(scalars):
        result = f"(Fin.cons {s} {result})"
    return f"({result} : Fin {n} → {elem_ty})"


def _sort_elem_ty(sort: Sort) -> str:
    """Lean scalar element type ("Bool"/"Int"/"Real") for a Sort, ignoring shape."""
    if isinstance(sort, Sort.Bool):
        return "Bool"
    if isinstance(sort, Sort.Int):
        return "Int"
    if isinstance(sort, Sort.Real):
        return "Real"
    raise ValueError(f"Unsupported Sort for element type: {sort}")


def tensor_to_mat_expr(tensor, elem_sort: Sort, shape: list[int]) -> str:
    """Inline `Mat` literal for a baked-in constant tensor, as a `match` on the
    `Fin` indices.

    A `match i, j with | r, c => v` is opaque to `split_ifs`, so an unreduced
    constant matrix (e.g. a large one whose surrounding `Fin.sum_univ_n` did not
    fire) degrades to an unsolved goal rather than a combinatorial `split_ifs`
    blow-up. Concrete indices still reduce by matcher iota under `simp`.
    `elem_sort` selects element formatting (Int/Real/Bool); `shape` is the
    tensor's `[m, n]` (or `[n]` / scalar).
    """
    if _is_scalar_shape(shape):
        m, n = 1, 1
    elif len(shape) == 1:
        m, n = 1, shape[0]
    else:
        m, n = shape[0], shape[1]
    ty = _sort_elem_ty(elem_sort)
    data = tensor.reshape(m, n)
    arms = " ".join(
        f"| {i}, {j} => {_get_dtype_item(elem_sort, data[i, j].item())}"
        for i in range(m)
        for j in range(n)
    )
    return f"(fun (i : Fin {m}) (j : Fin {n}) => ((match i, j with {arms}) : {ty}))"


def _mat_from_scalars(slots: list[str], shape: list[int], elem_ty: str) -> str:
    """Build a Lean ``Mat T m n`` expression from flat scalar slot strings.

    shape [] or [1]: ``(fun _ _ => slots[0])``
    shape [n]:       ``(fun _ j => Fin.cons s0 (Fin.cons s1 ...) j)``
    shape [m, n]:    ``(fun i j => Fin.cons row0 (Fin.cons row1 ...) i j)``
    """
    if _is_scalar_shape(shape):
        assert len(slots) == 1
        return f"(fun _ _ => {slots[0]})"
    if len(shape) == 1:
        n = shape[0]
        assert len(slots) == n, f"expected {n} slots, got {len(slots)}"
        return f"(fun _ j => {_vec_from_scalars(slots, elem_ty)} j)"
    if len(shape) == 2:
        r, c = shape
        assert len(slots) == r * c
        row_exprs = [_vec_from_scalars(slots[ri * c : (ri + 1) * c], elem_ty) for ri in range(r)]
        return f"(fun i j => {_vec_from_scalars(row_exprs, f'Fin {c} → {elem_ty}')} i j)"
    raise ValueError(f"Shape {shape} not supported for _mat_from_scalars")
