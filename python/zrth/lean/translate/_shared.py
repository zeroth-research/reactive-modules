"""Shared helpers used by multiple translate sub-modules."""

from zrth.lean.ir.sorts import dtype_to_lean_type, flat_element_type, wire_shape
from zrth.lean.ir.layout import FlatLayout, product_accessor, mat_from_scalars, flat_layout
from zrth.lean.native import build_tuple, product_type_scalar


def has_block(ctx, block: str) -> bool:
    """Whether the module's atom has `block` ("init" or "update") terms. An
    encoding proves its theorems about a block only when the block has
    some, so whatever cites one of them asks this same question."""
    return bool(list(getattr(ctx.atom, block)))


def scalar_eq_name(block: str) -> str:
    """The theorem `scalar.py` proves for `block`: the functional block is
    `Scalar.pack` of `Scalar.<block>` of the unpacked inputs. It exists
    exactly when `has_block(ctx, block)`."""
    return f"{block}_scalar_eq"


def flat_slice(expr: str, layout: FlatLayout, wire) -> str:
    """The components of the flat tuple `expr` that `wire` owns.

    Takes the wire and the layout rather than an offset and a length: the
    only two numbers involved are a position in one index space and a length
    in another, and the defect `fe8600da` fixed was them mismatched at a
    call site.

    A wire covering the whole tuple slices to `expr` itself: projecting and
    re-tupling would be a no-op that repeats `expr` once per element, and
    `expr` is a whole `Scalar.update` call at the only call sites.
    """
    offset, size = layout.span(wire)
    if offset == 0 and size == layout.total:
        return expr
    return build_tuple(
        [f"{expr}{product_accessor(offset + k, layout.total)}" for k in range(size)]
    )


def effect_type(wire) -> str:
    """Codomain of a per-wire `effect_i`/`init_i`: its own flat element tuple.

    One component per element, so it can be compared to the matching slice of
    the scalar encoding's state tuple. Typing it as the wire's `Mat` instead
    put a `Mat Int 6 1` against an `Int x ... x Int`.
    """
    return product_type_scalar([wire])


def scalar_bindings_with_recon(
    params: "list[tuple[str, list]]",
) -> "tuple[list[str], dict[int, str], dict[int, list[str]]]":
    """Build wire bindings and flat scalar slot tracking for scalar encoding.

    For each wire:
    - Scalar (flat_size == 1): binding is the bare accessor (e.g. ``"ctrl.1"``).
    - Non-scalar (flat_size > 1): binding is a fresh let-var reconstructing the
      ``Mat T m n`` (for operations that need the full matrix); flat_slots also
      records the individual scalar accessor strings for argmax and similar.

    Returns ``(let_lines, bindings, flat_slots)`` where:
    - ``let_lines`` should be prepended to the function body (Mat reconstructions)
    - ``bindings`` maps wire_id → matrix var or scalar accessor
    - ``flat_slots`` maps wire_id → list of flat scalar accessor strings
    """
    let_lines: list[str] = []
    bindings: dict[int, str] = {}
    flat_slots: dict[int, list[str]] = {}
    let_counter = [0]

    for name, wires in params:
        layout = flat_layout(wires)
        flat_accrs = layout.flat_accessors(name)

        for w in wires:
            offset, size = layout.span(w)
            slots = flat_accrs[offset : offset + size]
            flat_slots[w.id] = slots
            if size == 1:
                bindings[w.id] = slots[0]
            else:
                mat_ty = dtype_to_lean_type(w)
                mat_expr = mat_from_scalars(slots, wire_shape(w), flat_element_type(w))
                var_name = f"_m{let_counter[0]}"
                let_counter[0] += 1
                let_lines.append(f"  let {var_name} : {mat_ty} := {mat_expr}")
                bindings[w.id] = var_name

    return let_lines, bindings, flat_slots


def _recon_var(let_line: str) -> str:
    """The variable a reconstruction let binds (`  let _m0 : ... := ...`)."""
    return let_line.split("let ", 1)[1].split(" ", 1)[0]


def _needed_recon(recon_lets: "list[str]", body: str) -> "list[str]":
    """The reconstruction lets `body` actually refers to, in order.

    An unused let still mentions its parameter group (`extl_l.1` and so on),
    and Lean's `variable` auto-binding would then add that parameter to the
    declaration -- while the call sites are built from what the body
    consumes, so they would under-apply it. Iterated to a fixed point in
    case one reconstruction ever refers to another.
    """
    kept: list[str] = []
    text = body
    changed = True
    while changed:
        changed = False
        for line in recon_lets:
            if line in kept:
                continue
            if _recon_var(line) in text:
                kept.append(line)
                text += "\n" + line
                changed = True
    return [line for line in recon_lets if line in kept]


def prepend_recon(recon_lets: "list[str]", body: str) -> str:
    """Prepend the Mat reconstruction let-lines this body needs."""
    needed = _needed_recon(recon_lets, body)
    if needed:
        return "\n".join(needed) + "\n" + body
    return body
