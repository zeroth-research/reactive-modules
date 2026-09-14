"""One flattening, one owner — and the shapes the fixtures barely reach.

Everything below `Mat` turns a list of wires into a flat tuple of scalar
elements and then reads it back: `Scalar.unpack_ctrl` and `Scalar.pack`, the
`ScalarRel` slices, the NA model's `var_k`, the bridge's `toSlots`, and
cvc5's own tuple in `smt_encode`. Those were seven independent walks over the
same row-major order, agreeing by construction and by comment;
`KNOWN_ISSUES` #20 and #22 are what that was worth. They now all go through
`common.flat_layout`, and this is the test that says so.

The shapes here are synthetic on purpose. Mutating `_flat_indices` to
column-major and running the fast suite fails exactly one existing test:
`test_lean_fbk.py::test_the_emitted_transition_agrees_with_the_module`, whose
`_wide_matrix` fixture carries a 2x2 state for precisely this reason and
covers the NA route end to end. The other 522 pass. `m_mixed` is the only
module where the per-wire and per-element index spaces differ at all, and no
*scalarized* fixture holds a wire wider than a vector — where row-major and
column-major coincide — so `Scalar.pack`/`unpack_ctrl`, the `ScalarRel`
slices, `_mat_from_scalars`' placement and the flat product type are all
blind to a transposition. Hence `[2, 3]` and `[3, 2]` below: rectangular, so
a swapped `m`/`n` cannot hide either.
"""

import re

import pytest
import torch

from zrth import Bool, Int, LIA, Module, Real, Term, Var, Wire, X
from zrth.lean.common import (
    LeanContext,
    _accessor,
    _flat_element_type,
    _flat_indices,
    _flat_size,
    _mat_from_scalars,
    dtype_shape,
    flat_layout,
)
from zrth.lean.native import _product_type_scalar
from zrth.lean.smt_encode import wire_shape
from zrth.lean.translate._shared import _scalar_bindings_with_recon
from zrth.lean.translate.fbk import _slot_accessors
from zrth.lean.translate.scalar import _pack_body, _unpack_body

# One wire list per case. The last three are the ones no fixture has: a
# genuinely 2-D wire, and a mixed list where the wire index and the slot
# index disagree in both directions.
WIRE_LISTS = {
    "one scalar": [Int([1, 1])],
    "two scalars": [Int([1, 1]), Bool([1, 1])],
    "row vector": [Int([1, 3])],
    "column vector": [Int([3, 1])],
    "two vectors": [Int([2, 1]), Real([3, 1])],
    "2x3": [Int([2, 3])],
    "3x2": [Int([3, 2])],
    "scalar then 2x3": [Bool([1, 1]), Int([2, 3])],
    "2x3 then scalar": [Int([2, 3]), Bool([1, 1])],
    "2x3 then 3x2": [Int([2, 3]), Int([3, 2])],
}

CASES = list(WIRE_LISTS.items())
IDS = [name for name, _ in CASES]


@pytest.fixture(params=[wires for _, wires in CASES], ids=IDS)
def wires(request):
    """Fresh `Var`s each time: a layout is keyed by wire identity."""
    return [Var(dt) for dt in request.param]


def _expected_slots(wires):
    """(wire index, row, col) per element, spelled out independently here."""
    out = []
    for i, w in enumerate(wires):
        shape = dtype_shape(w.dtype)
        m, n = (1, 1) if all(d == 1 for d in shape) else (
            (1, shape[0]) if len(shape) == 1 else tuple(shape)
        )
        out += [(i, r, c) for r in range(m) for c in range(n)]
    return out


# ── the order itself ────────────────────────────────────────────────────────


def test_slots_are_row_major_within_each_wire(wires):
    layout = flat_layout(wires)
    assert [tuple(s) for s in layout.slots] == _expected_slots(wires)


def test_spans_tile_the_flat_tuple(wires):
    """Every slot belongs to exactly one wire, and the wires are in order."""
    layout = flat_layout(wires)
    covered = []
    for w in wires:
        offset, size = layout.span(w)
        assert size == _flat_size(w)
        covered += list(range(offset, offset + size))
        assert layout.slots_of(w) == layout.slots[offset : offset + size]
    assert covered == list(range(layout.total))
    assert layout.total == sum(_flat_size(w) for w in wires)


def test_span_needs_the_wire_not_a_position(wires):
    """The per-wire view is unreachable by index — that is the point of it."""
    layout = flat_layout(wires)
    with pytest.raises(KeyError):
        layout.span(Var(Int([1, 1])))


# ── the coupling to cvc5's tuple, which no fixture can break ────────────────


def test_slot_order_is_the_smt_tuple_order(wires):
    """Slot `k` of a wire is element `i*n + j` of `smt_encode`'s cvc5 tuple.

    `smt_to_lean_bool` resolves the NA model's reads through `_slot_accessors`
    while the term it prints selects with `mat_select`, so these two orders
    being one order is what makes the model describe *this* module. Nothing
    else checks it: for a vector (`m == 1` or `n == 1`) row-major and
    column-major agree, and no fixture has a wider ctrl wire.
    """
    layout = flat_layout(wires)
    for w in wires:
        shape = wire_shape(w)
        assert shape.total == _flat_size(w)
        for k, slot in enumerate(layout.slots_of(w)):
            assert (slot.row, slot.col) == (k // shape.n, k % shape.n)
            assert slot.row * shape.n + slot.col == k


# ── the emitters that read the flat tuple ───────────────────────────────────


def _flat_positions(exprs, base, total):
    """Which flat component each `base.2.1`-style accessor reads."""
    by_acc = {f"{base}{_accessor(k, total)}": k for k in range(total)}
    return [by_acc[e] for e in exprs]


def _tuple_parts(body: str, n: int) -> "list[str]":
    """Split `_build_tuple`'s output back into its `n` components.

    `n` has to be told, not inferred: `_build_tuple` of one element is that
    element, so a single `Mat` reconstruction is indistinguishable from a
    tuple by its parentheses alone. The split is depth-aware for the same
    reason — a reconstruction's `match i, j with` arms are full of commas.
    """
    if n == 1:
        return [body]
    depth, cur, parts = 0, "", []
    for ch in body[1:-1]:
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        depth += (ch == "(") - (ch == ")")
        cur += ch
    parts.append(cur.strip())
    assert len(parts) == n, f"split {body!r} into {len(parts)}, expected {n}"
    return parts


def test_unpack_reads_every_slot_once_in_order(wires):
    """`Scalar.unpack_ctrl` is the flat order, spelled as matrix reads."""
    layout = flat_layout(wires)
    parts = _tuple_parts(_unpack_body("ctrl", wires), layout.total)
    assert parts == [layout.wire_accessor("ctrl", s) for s in layout.slots]
    assert len(parts) == layout.total


def test_pack_rebuilds_each_wire_from_its_own_span(wires):
    """`Scalar.pack` puts flat slot `k` back where `unpack` took it from."""
    layout = flat_layout(wires)
    parts = _tuple_parts(_pack_body("s", wires), len(wires))
    assert len(parts) == len(wires)
    for w, part in zip(wires, parts):
        offset, size = layout.span(w)
        read = _flat_positions(re.findall(r"s(?:\.[12])*", part), "s", layout.total)
        # A `Mat` arm per element, in the layout's order, over this wire's
        # span alone -- plus `_mat_from_scalars`' catch-all, which repeats the
        # first slot.
        assert read[:size] == list(range(offset, offset + size))
        assert set(read) <= set(range(offset, offset + size))


def test_pack_places_each_slot_at_its_own_row_and_column(wires):
    """The `| i, j => s_k` arms agree with the slot `unpack` read at `(i, j)`.

    `_mat_from_scalars` indexes its slots `i*n + j` while `_flat_indices`
    enumerates them; this is the one place those two meet.
    """
    layout = flat_layout(wires)
    for w in wires:
        offset, _ = layout.span(w)
        slots = [f"s{k}" for k in range(_flat_size(w))]
        mat = _mat_from_scalars(slots, list(dtype_shape(w.dtype)), _flat_element_type(w))
        arms = re.findall(r"\| (\d+), (\d+) => s(\d+)", mat)
        placed = {(int(r), int(c)): int(k) for r, c, k in arms}
        if placed:  # a 1x1 wire has no `match`, just `fun _ _ =>`
            for k, (r, c) in enumerate(_flat_indices(w)):
                assert placed[(r, c)] == k, f"wire at offset {offset}"
        else:
            assert _flat_size(w) == 1


def test_scalar_bindings_hand_each_wire_its_own_slots(wires):
    """`_scalar_bindings_with_recon`'s `flat_slots` is the same tiling."""
    layout = flat_layout(wires)
    _lets, _bindings, flat_slots = _scalar_bindings_with_recon([("ctrl", wires)])
    for w in wires:
        offset, size = layout.span(w)
        positions = _flat_positions(flat_slots[w.id], "ctrl", layout.total)
        assert positions == list(range(offset, offset + size))


def test_na_state_variables_are_numbered_in_slot_order(wires):
    """The NA model's `var_k` and the flat tuple are the same numbering."""
    layout = flat_layout(wires)
    acc = _slot_accessors(wires)
    assert list(acc) == [f"s{i}" for i in range(len(wires))]
    seen = []
    for i, w in enumerate(wires):
        reads = acc[f"s{i}"]
        assert len(reads) == _flat_size(w)
        seen += [int(re.match(r"\(var_(\d+) state\)", r).group(1)) for r in reads]
    assert seen == list(range(layout.total))


def test_scalar_product_type_is_the_slot_types(wires):
    """The type of the flat tuple has one component per slot, in order."""
    layout = flat_layout(wires)
    assert _product_type_scalar(wires) == " × ".join(layout.element_types())
    assert len(layout.element_types()) == layout.total


# ── end to end, on a 2-D ctrl wire ──────────────────────────────────────────


def _matrix_module() -> Module:
    """`x : Int 2x3`, `x' = x + 1`, from a matrix whose entries are all different.

    The distinct entries are the point: an `init_k` list that is right up to a
    transposition still matches on a constant matrix of equal elements.
    """
    x = Var(Int([2, 3]))
    ones = Wire(Int([2, 3]))
    start = torch.tensor([[1, 2, 3], [4, 5, 6]])
    init = [Term(LIA.Int(start), [X(x)])]
    update = [
        Term(LIA.Int(torch.ones((2, 3), dtype=torch.int64)), [ones]),
        Term(LIA.Add(), [X(x)], [x, ones]),
    ]
    return Module.sequential([x], init, update)


def test_na_model_init_is_the_matrix_read_row_major():
    """`Definition.init_k` for a 2x3 ctrl wire, against the source tensor.

    The whole chain in one assertion: the Python tensor, `smt_encode`'s cvc5
    tuple, `_scalar_element`'s selection, and the slot numbering this module
    owns. Transpose any one of them and the constants come back permuted.
    """
    from zrth.lean.translate.fbk import atom_to_lean_na

    ctx = LeanContext(_matrix_module())
    src = atom_to_lean_na(ctx, "true", module_name="matrix")
    init = dict(re.findall(r"abbrev init_(\d+) : Int :=\n  \(?(-?\d+)", src))
    assert [init[str(k)] for k in range(6)] == ["1", "2", "3", "4", "5", "6"]


def test_scalar_unpack_of_a_2d_wire_reads_row_major():
    """`Scalar.unpack_ctrl` on the same wire, for the other half of the bridge."""
    ctx = LeanContext(_matrix_module())
    parts = _tuple_parts(_unpack_body("s", ctx.ctrl_next), 6)
    assert parts == [f"s {r} {c}" for r in range(2) for c in range(3)]


def test_bridge_reads_the_state_exactly_as_unpack_does():
    """`Bridge.toSlots` against `Scalar.unpack_ctrl`, arm by arm.

    These are the two halves of the FBK equivalence: `toSlots` says which
    model slot an element is, `unpack_ctrl` says which flat component it is.
    They used to be two functions written to match — the bridge's own
    docstring said "without either consulting the other" — and a
    disagreement makes `bridge_k` an obligation about the wrong element.
    """
    from zrth.lean.translate.fbk_bridge import atom_to_lean_fbk_bridge

    ctx = LeanContext(_matrix_module())
    src = atom_to_lean_fbk_bridge(ctx, na_module="NA")
    arms = re.findall(r"^  \| (?:_ \+ )?(\d+) => (.+)$", src, re.M)
    to_slots = [body for k, body in arms[:6]]
    assert to_slots == _tuple_parts(_unpack_body("s", ctx.ctrl_next), 6)
