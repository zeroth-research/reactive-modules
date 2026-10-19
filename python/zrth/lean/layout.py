"""Where a wire's elements sit: tuple accessors, the flat element order
every scalar encoding shares (`FlatLayout`), and the matrix rebuilt from it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from zrth import Wire
from .sorts import _flat_element_type, MatShape, wire_shape


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


def _flat_indices(wire: Wire) -> list[tuple[int, int]]:
    """Row-major (row, col) indices for all elements of the wire: for an
    m x n wire, `(i, j)` for `i` in `range(m)`, `j` in `range(n)`."""
    shape = wire_shape(wire)
    return [(i, j) for i in range(shape.m) for j in range(shape.n)]


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
    count, two of them in the same file -- and the defects `d49ebdb9` and
    `fe8600da` fixed were both two of those drifting apart: a per-wire index
    used where a per-element one was meant. Both are `int`, so the one thing
    this can do about that class of mistake is refuse to hand out the
    per-wire view by position: :meth:`span` takes the wire itself.

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


def _mat_from_scalars(slots: list[str], shape: MatShape, elem_ty: str) -> str:
    """Build a Lean ``Mat T m n`` expression from flat scalar slot strings.

    1x1:       ``(fun _ _ => slots[0])``
    otherwise: ``(fun i j => match i, j with | 0, 0 => s0 | ...)``

    Matched on the indices rather than built from nested ``Fin.cons``. The
    equivalence theorems in the scalar encoding prove `pack (unpack x) = x`
    elementwise, which means discharging the goal at each concrete index
    after `fin_cases`; a `Fin.cons` chain does not reduce there (a numeral
    index is not syntactically `Fin.succ`, and neither `Fin.cons_succ`,
    `Fin.cons` unfolding nor the `Matrix.cons_val_*` set rewrites it), while
    a `match` does. It is also the form `_tensor_to_lean_def` already emits
    for constants, whose side of those goals always reduced.
    """
    if shape.is_scalar:
        assert len(slots) == 1
        return f"(fun _ _ => {slots[0]})"
    m, n = shape.m, shape.n
    assert len(slots) == m * n, f"expected {m * n} slots, got {len(slots)}"
    arms = " ".join(
        f"| {i}, {j} => {slots[i * n + j]}" for i in range(m) for j in range(n)
    )
    # Catch-all for the same reason as `_tensor_to_lean_def`: Lean cannot see
    # that `0..k-1` exhausts `Fin k` for larger k.
    return f"(fun i j => match i, j with {arms} | _, _ => {slots[0]})"
