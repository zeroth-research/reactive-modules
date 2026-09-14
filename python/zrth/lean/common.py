from dataclasses import dataclass
from typing import NamedTuple

from zrth import Term, Wire, Var, Sort, BitVec, Bool, Int, Real, Module, X


class Refused(ValueError):
    """This run cannot be carried out, and the message says why.

    A refusal is a decision the generator makes about its input -- an op with
    no Lean form, a state element the executable cannot print, a name the
    property does not have. It is not a defect, so `main` prints it as
    `error: ...` and exits; everything else keeps its traceback, which is
    what tells the two apart.

    `ValueError` is the base because that is what these sites raised before
    the type existed, and callers that catch it still do.
    """


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

    if isinstance(dt, Bool):
        ty = "Bool"
    elif isinstance(dt, Int):
        ty = "Int"
    elif isinstance(dt, Real):
        # TODO: Float is *NOT* Real, but we stick to that for proofs atm
        ty = "Real"
    elif isinstance(dt, BitVec):
        ty = f"(BitVec {dt._0})"
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


def _constant_expr(term: Term, w: Wire, constants: "ConstantRegistry") -> str:
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
            if isinstance(w.dtype, Real):
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

        self.extl_latched: list[Var] = list(module.extl)
        self.extl_next: list[Wire] = [X(v) for v in module.extl]
        self.ctrl_latched: list[Var] = list(module.ctrl)
        self.ctrl_next: list[Wire] = [X(v) for v in module.ctrl]

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
    if isinstance(dtype, Bool):
        return "true" if bool(item) else "false"
    if isinstance(dtype, Int):
        return str(int(item))
    if isinstance(dtype, Real):
        return _float_literal(float(item))
    if isinstance(dtype, BitVec):
        # `ofInt`, not `ofNat`: a negative entry has no `Neg ℕ` instance,
        # so `BitVec.ofNat w -3` does not elaborate. `ofInt` wraps to
        # two's complement and handles non-negative values identically.
        return f"(BitVec.ofInt {dtype._0} ({int(item)}))"
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
        # Catch-all: enumerated arms already cover all indices, but Lean can't
        # verify `0..k-1` exhausts `Fin k` for large k (see tensor_to_mat_expr).
        lines.append(f"  | _, _ => {_get_dtype_item(wire.dtype, 0)}")

        return "\n".join(lines) + "\n"

    raise ValueError(
        f"Cannot generate Lean constant for dtype={wire.dtype}, shape={shape}"
    )


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


def _float_literal(v: float) -> str:
    """Lean Real literal for a float value."""
    if v == int(v):
        return f"({int(v)} : Real)"
    return f"({v} : Real)"


def _tensor_to_lean_inline(tensor, wire: Wire) -> str:
    """Return an inline `Mat _ 1 1` literal for a scalar tensor."""
    if isinstance(wire.dtype, Bool):
        val = "true" if bool(tensor.item()) else "false"
        return f"(fun _ _ => {val})"
    if isinstance(wire.dtype, Int):
        return f"(fun _ _ => ({int(tensor.item())} : Int))"
    if isinstance(wire.dtype, Real):
        return f"(fun _ _ => {_float_literal(float(tensor.item()))})"
    if isinstance(wire.dtype, BitVec):
        return f"(fun _ _ => {_get_dtype_item(wire.dtype, tensor.item())})"
    raise ValueError(f"Cannot inline tensor with dtype={wire.dtype}")


def _tensor_to_lean_scalar(tensor, wire: Wire) -> str:
    """Return a bare scalar literal (no Mat wrapper) for a scalar tensor."""
    if isinstance(wire.dtype, Bool):
        return "true" if bool(tensor.item()) else "false"
    if isinstance(wire.dtype, Int):
        return f"({int(tensor.item())} : Int)"
    if isinstance(wire.dtype, Real):
        return _float_literal(float(tensor.item()))
    if isinstance(wire.dtype, BitVec):
        return _get_dtype_item(wire.dtype, tensor.item())
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
    if isinstance(dt, Bool):
        return "Bool"
    if isinstance(dt, Int):
        return "Int"
    if isinstance(dt, Real):
        return "Real"
    if isinstance(dt, BitVec):
        return f"(BitVec {dt._0})"
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


class Slot(NamedTuple):
    """One element of a flattened wire list: which wire, and where in it.

    `wire` is a position in the list the layout was built from, *not* a wire
    id. A slot's own position in the flat tuple is its index in
    :attr:`FlatLayout.slots`, and that is the only place the flat order is
    defined.
    """

    wire: int
    row: int
    col: int


@dataclass(frozen=True)
class FlatLayout:
    """The row-major flattening of a wire list into scalar elements.

    One walk over the wires, in the views its consumers need: `slots` per
    element, :meth:`span` per wire, `total` for the width, and the two string
    views (:meth:`flat_accessors`, :meth:`wire_accessor`) that the encodings
    used to rebuild beside each other.

    Every view was being derived independently -- seven walks at the last
    count, two of them in the same file -- and `KNOWN_ISSUES` #20 and #22 are
    both two of those drifting apart: a per-wire index used where a
    per-element one was meant. Both are `int`, so the one thing this can do
    about that class of mistake is refuse to hand out the per-wire view by
    position: :meth:`span` takes the wire itself.

    The row-major order is not a free choice. `smt_encode` packs `Mat t m n`
    into a cvc5 tuple at index `i*n + j`, and `smt_to_lean_bool` resolves the
    model's slot reads through it, so a column-major walk here would describe
    a transposed module. Only the NA route has a fixture wide enough to
    notice (`test_lean_fbk.py`'s `_wide_matrix`);
    `tests/test_lean_flat_layout.py` pins the rest.
    """

    wires: "tuple[Wire, ...]"
    slots: "tuple[Slot, ...]"
    # Per wire, where its elements start in the flat tuple. Private: reach it
    # through `span`, which needs the wire rather than a position.
    _offsets: "tuple[int, ...]"

    @property
    def total(self) -> int:
        """How many scalar elements the whole list flattens to."""
        return len(self.slots)

    def _position(self, wire: Wire) -> int:
        for i, w in enumerate(self.wires):
            if w.id == wire.id:
                return i
        raise KeyError(f"wire {wire.id} is not in this layout")

    def span(self, wire: Wire) -> "tuple[int, int]":
        """`(offset, size)`: where `wire`'s elements sit in the flat tuple."""
        i = self._position(wire)
        return self._offsets[i], _flat_size(wire)

    def slots_of(self, wire: Wire) -> "tuple[Slot, ...]":
        """`wire`'s own slots, in flat order."""
        offset, size = self.span(wire)
        return self.slots[offset : offset + size]

    def flat_accessors(self, base: str) -> "list[str]":
        """Read every slot out of a flat product named `base`: `base.2.1`, …"""
        return [f"{base}{_accessor(k, self.total)}" for k in range(self.total)]

    def wire_accessor(self, base: str, slot: Slot) -> str:
        """Read one slot out of a per-*wire* product of `Mat`s named `base`.

        The other direction from :meth:`flat_accessors`: `base` here is the
        functional state, so the element is a projection to the wire followed
        by its `row col` application.
        """
        acc = _accessor(slot.wire, len(self.wires))
        return f"{base}{acc} {slot.row} {slot.col}"

    def element_types(self) -> "list[str]":
        """The Lean element type of each slot, in flat order."""
        return [_flat_element_type(self.wires[s.wire]) for s in self.slots]


def flat_layout(wires: "list[Wire]") -> FlatLayout:
    """Flatten `wires` to scalar elements, row-major within each wire.

    The single owner of that order; see :class:`FlatLayout`.
    """
    slots: list[Slot] = []
    offsets: list[int] = []
    for i, w in enumerate(wires):
        offsets.append(len(slots))
        slots.extend(Slot(i, r, c) for r, c in _flat_indices(w))
    return FlatLayout(tuple(wires), tuple(slots), tuple(offsets))


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


def linear_list_literals(term: Term) -> "tuple[int, str, str, str]":
    """List-literal operands for a baked-constant `Linear` op (`Y = A·X + B`).

    Returns ``(out_rows, A_literal, b_literal, elem_ty)`` where ``A_literal`` is a
    ``List (List t)`` and ``b_literal`` a ``List t`` (empty bias → zeros), both
    type-ascribed. Shared by the native (`matVecAffine`) and circuit (`Box.linear`)
    emitters so the reflected form is generated in one place.
    """
    out_wire = term.write[0]
    A = term.itype._0
    B = term.itype._1
    out_m = dtype_shape(out_wire.dtype)[0]
    in_dim = int(A.shape[1])
    ty = _sort_elem_ty(out_wire.dtype)
    rows = [
        "[" + ", ".join(_get_dtype_item(out_wire.dtype, A[i][l].item()) for l in range(in_dim)) + "]"
        for i in range(out_m)
    ]
    a_lit = f"([{', '.join(rows)}] : List (List {ty}))"
    if B.numel() == 0:
        bias = [_get_dtype_item(out_wire.dtype, 0) for _ in range(out_m)]
    else:
        bias = [_get_dtype_item(out_wire.dtype, B[i][0].item()) for i in range(out_m)]
    b_lit = f"([{', '.join(bias)}] : List {ty})"
    return out_m, a_lit, b_lit, ty


def _sort_elem_ty(sort: Sort) -> str:
    """Lean scalar element type ("Bool"/"Int"/"Real") for a Sort, ignoring shape."""
    if isinstance(sort, Bool):
        return "Bool"
    if isinstance(sort, Int):
        return "Int"
    if isinstance(sort, Real):
        return "Real"
    if isinstance(sort, BitVec):
        return f"(BitVec {sort._0})"
    raise ValueError(f"Unsupported Sort for element type: {sort}")


def _mat_from_scalars(slots: list[str], shape: list[int], elem_ty: str) -> str:
    """Build a Lean ``Mat T m n`` expression from flat scalar slot strings.

    shape [] or [1]: ``(fun _ _ => slots[0])``
    otherwise:       ``(fun i j => match i, j with | 0, 0 => s0 | ...)``

    Matched on the indices rather than built from nested ``Fin.cons``. The
    equivalence theorems in the scalar encoding prove `pack (unpack x) = x`
    elementwise, which means discharging the goal at each concrete index
    after `fin_cases`; a `Fin.cons` chain does not reduce there (a numeral
    index is not syntactically `Fin.succ`, and neither `Fin.cons_succ`,
    `Fin.cons` unfolding nor the `Matrix.cons_val_*` set rewrites it), while
    a `match` does. It is also the form `_tensor_to_lean_def` already emits
    for constants, whose side of those goals always reduced.
    """
    if _is_scalar_shape(shape):
        assert len(slots) == 1
        return f"(fun _ _ => {slots[0]})"
    if len(shape) == 1:
        m, n = 1, shape[0]
    elif len(shape) == 2:
        m, n = shape
    else:
        raise ValueError(f"Shape {shape} not supported for _mat_from_scalars")
    assert len(slots) == m * n, f"expected {m * n} slots, got {len(slots)}"
    arms = " ".join(
        f"| {i}, {j} => {slots[i * n + j]}" for i in range(m) for j in range(n)
    )
    # Catch-all for the same reason as `_tensor_to_lean_def`: Lean cannot see
    # that `0..k-1` exhausts `Fin k` for larger k.
    return f"(fun i j => match i, j with {arms} | _, _ => {slots[0]})"
