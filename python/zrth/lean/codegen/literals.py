"""Constants as Lean text: inlined literals, and the matrix constants a
module's terms define (`ConstantRegistry`)."""

from __future__ import annotations


from zrth import BitVec, Bool, Int, Real, Sort, Term, Wire
from ..lowering.sorts import is_scalar_tensor, dtype_to_lean_type, is_constant_name, itype_name, lean_elem_type, wire_shape
from ..lowering.values import decimal_fraction


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
        if is_scalar_tensor(out_wire) or out_wire.id in self._by_id:
            return
        name = f"c{self._counter}"
        self._counter += 1
        self._by_id[out_wire.id] = name
        self._defs.append(tensor_to_lean_def(name, term.itype._0, out_wire))

    def lookup(self, wire_id: int) -> str | None:
        return self._by_id.get(wire_id)

    def names(self) -> list[str]:
        return list(self._by_id.values())

    def defs(self) -> list[str]:
        return list(self._defs)


def constant_expr(term: Term, w: Wire, constants: "ConstantRegistry") -> str:
    # Matrix constants are interned as top-level defs; scalars are inlined as
    # a 1x1 matrix literal driven by the wire's element type (Bool/Int/Real).
    name = constants.lookup(w.id)
    if name is not None:
        return name
    return tensor_to_lean_inline(term.itype._0, w)


def lean_int(value) -> str:
    """An Int literal, ascribed whatever its sign, so it reads the same in
    every position."""
    return f"({int(value)} : Int)"


def lean_real(value) -> str:
    """A Real literal, exact: `(2 : Real)` or `((p : Real) / q)`."""
    q = decimal_fraction(value)
    if q.denominator == 1:
        return f"({q.numerator} : Real)"
    return f"(({q.numerator} : Real) / {q.denominator})"


def lean_bitvec(width: int, value) -> str:
    """A BitVec literal. `ofInt`, not `ofNat`: a negative entry has no
    `Neg ℕ` instance, so `BitVec.ofNat w -3` does not elaborate; `ofInt`
    wraps to two's complement and reads non-negative values identically."""
    return f"(BitVec.ofInt {width} ({int(value)}))"


def lean_literal(sort: Sort, value) -> str:
    """`value` as a Lean literal of one element of `sort`: every constant
    the encodings and the certificate print goes through here."""
    if isinstance(sort, Bool):
        return "true" if bool(value) else "false"
    if isinstance(sort, Int):
        return lean_int(value)
    if isinstance(sort, Real):
        return lean_real(value)
    if isinstance(sort, BitVec):
        return lean_bitvec(sort._0, value)
    raise ValueError(f"no Lean literal for a value of sort {sort}")


def tensor_to_lean_def(name: str, tensor, wire: Wire) -> str:
    """
    Generate a top-level Lean definition for a constant tensor.

    E.g.:
        @[simp] def A : Fin 3 → Fin 2 → Int := fun i j =>
          match i, j with
          | 0, 0 => 0 | 0, 1 => 1
          ...
    """
    shape = wire_shape(wire)

    if shape.is_scalar:
        val = lean_literal(wire.dtype, tensor.item())
        ty = dtype_to_lean_type(wire)
        assert "Mat" in ty and " 1 1" in ty, ty
        return f"@[simp] def {name} : {ty} := fun _ _ => {val}\n"

    # Matrix constant
    lines = [f"@[simp] def {name} : {dtype_to_lean_type(wire)} := fun i j =>"]
    lines.append("  match i, j with")

    data = tensor.reshape(shape.m, shape.n)
    for i in range(shape.m):
        row_entries = []
        for j in range(shape.n):
            val = lean_literal(wire.dtype, data[i, j].item())
            row_entries.append(f"| {i}, {j} => {val}")
        lines.append("  " + " ".join(row_entries))
    # Catch-all: enumerated arms already cover all indices, but Lean can't
    # verify `0..k-1` exhausts `Fin k` for large k (see tensor_to_mat_expr).
    lines.append(f"  | _, _ => {lean_literal(wire.dtype, 0)}")

    return "\n".join(lines) + "\n"


def tensor_to_lean_inline(tensor, wire: Wire) -> str:
    """Return an inline `Mat _ 1 1` literal for a scalar tensor."""
    return f"(fun _ _ => {lean_literal(wire.dtype, tensor.item())})"


def tensor_to_lean_scalar(tensor, wire: Wire) -> str:
    """Return a bare scalar literal (no Mat wrapper) for a scalar tensor."""
    return lean_literal(wire.dtype, tensor.item())


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
    # Rows of `A`, which is what `Y` has whatever its wire's shape says.
    out_m = int(A.shape[0])
    in_dim = int(A.shape[1])
    ty = lean_elem_type(out_wire.dtype)
    rows = [
        "[" + ", ".join(lean_literal(out_wire.dtype, A[i][l].item()) for l in range(in_dim)) + "]"
        for i in range(out_m)
    ]
    a_lit = f"([{', '.join(rows)}] : List (List {ty}))"
    if B.numel() == 0:
        bias = [lean_literal(out_wire.dtype, 0) for _ in range(out_m)]
    else:
        bias = [lean_literal(out_wire.dtype, B[i][0].item()) for i in range(out_m)]
    b_lit = f"([{', '.join(bias)}] : List {ty})"
    return out_m, a_lit, b_lit, ty
