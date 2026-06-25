"""Shared helpers used by multiple translate sub-modules."""

from zrth.lean.common import (
    _accessor,
    _flat_size,
    _flat_element_type,
    dtype_to_lean_type,
    _mat_from_scalars,
)


def _scalar_bindings_with_recon(
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
        flat_sizes = [_flat_size(w) for w in wires]
        total = sum(flat_sizes)
        flat_accrs = [f"{name}{_accessor(k, total)}" for k in range(total)]

        offset = 0
        for w, fsize in zip(wires, flat_sizes):
            slots = flat_accrs[offset : offset + fsize]
            flat_slots[w.id] = slots
            if fsize == 1:
                bindings[w.id] = slots[0]
            else:
                mat_ty = dtype_to_lean_type(w)
                mat_expr = _mat_from_scalars(slots, list(w.dtype.shape), _flat_element_type(w))
                var_name = f"_m{let_counter[0]}"
                let_counter[0] += 1
                let_lines.append(f"  let {var_name} : {mat_ty} := {mat_expr}")
                bindings[w.id] = var_name
            offset += fsize

    return let_lines, bindings, flat_slots


def _prepend_recon(recon_lets: "list[str]", body: str) -> str:
    """Prepend Mat reconstruction let-lines to a function body."""
    if recon_lets:
        return "\n".join(recon_lets) + "\n" + body
    return body
