"""
Translate a Python Module (reactive module with init/update blocks)
into Lean4 code.

Python modules store computations as a flat list of Terms (SSA-style dataflow).
We translate each Term into a `let` binding in a Lean function, mapping each
IType operation to its Lean equivalent.
"""

from __future__ import annotations
from zrth.lean.common import (
    dtype_shape,
    ConstantRegistry,
    _constant_expr,
    itype_name,
    is_constant_name,
    _accessor,
    dtype_to_lean_type,
    tensor_to_mat_expr,
    _is_scalar_wire,
    _tensor_to_lean_scalar,
    _bind_wires_scalar,
    _tensor_to_lean_inline,
    _flat_element_type,
    _flat_size,
    _flat_indices,
)

from typing import Callable

from zrth import Wire, Sort


def _product_type(wires: list[Wire]) -> str:
    """Build a right-nested product type: [w1, w2] -> 'Bool × Bool', [w1] -> 'Bool'."""
    if not wires:
        return "Unit"
    if len(wires) == 1:
        return dtype_to_lean_type(wires[0])
    parts = [dtype_to_lean_type(w) for w in wires]
    return " × ".join(parts)


def _build_tuple(exprs: list[str]) -> str:
    """Build a tuple literal: [] -> '()', ['a'] -> 'a', ['a','b'] -> '(a, b)'."""
    if not exprs:
        return "()"
    if len(exprs) == 1:
        return exprs[0]
    return "(" + ", ".join(exprs) + ")"


def _append_expr(var1: str, count1: int, var2: str, count2: int) -> str:
    """Generate an expression that concatenates two tuples into one.

    E.g. _append_expr("x", 2, "e", 1) -> "(x.1, x.2, e)"
    """
    parts = []
    for i in range(count1):
        acc = _accessor(i, count1)
        parts.append(f"{var1}{acc}")
    for i in range(count2):
        acc = _accessor(i, count2)
        parts.append(f"{var2}{acc}")
    return _build_tuple(parts)


# Map operations to Lean expression builder (takes list of arg strings).
# Operands and results are all `Mat _ 1 1` (scalar)
# FIXME: element-wise ops extract position `0 0`, but
# it should be defined elem-wise on the whole matrix to match Rust
_LEAN_OP: dict[str, Callable] = {
    "Not": lambda a: f"(fun _ _ => !({a[0]} 0 0))",
    "And": lambda a: f"(fun _ _ => ({a[0]} 0 0 && {a[1]} 0 0))",
    "Or": lambda a: f"(fun _ _ => ({a[0]} 0 0 || {a[1]} 0 0))",
    "Ite": lambda a: f"(if {a[0]} 0 0 then {a[1]} else {a[2]})",
    "Add": lambda a: f"({a[0]} + {a[1]})",
    "Sub": lambda a: f"({a[0]} - {a[1]})",
    "Mul": lambda a: f"({a[0]} * {a[1]})",
    "Mod": lambda a: f"(fun _ _ => ({a[0]} 0 0 % {a[1]} 0 0))",
    "Neg": lambda a: f"(-{a[0]})",
    "Lt": lambda a: f"(fun _ _ => decide ({a[0]} 0 0 < {a[1]} 0 0))",
    "Le": lambda a: f"(fun _ _ => decide ({a[0]} 0 0 ≤ {a[1]} 0 0))",
    "Gt": lambda a: f"(fun _ _ => decide ({a[1]} 0 0 < {a[0]} 0 0))",
    "Ge": lambda a: f"(fun _ _ => decide ({a[1]} 0 0 ≤ {a[0]} 0 0))",
    "Eq": lambda a: f"(fun _ _ => decide ({a[0]} 0 0 = {a[1]} 0 0))",
    "Neq": lambda a: f"(fun _ _ => decide ({a[0]} 0 0 ≠ {a[1]} 0 0))",
    "Min": lambda a: f"(fun i j => Min.min ({a[0]} i j) ({a[1]} i j))",
    "Max": lambda a: f"(fun i j => Max.max ({a[0]} i j) ({a[1]} i j))",
    "MatMul": lambda a: f"MatMul {a[0]} {a[1]}",
    "Id": lambda a: a[0],
    # Linear is handled specially (its A, B are baked into the op, not read
    # wires) — see `_linear_expr`.
    "ReLU": lambda a: f"ReLu {a[0]}",
    "TensorGet": lambda a: f"({a[0]} 0 0)",
    "ToUnsigned": lambda a: f"(fun _ _ => Int.toNat ({a[0]} 0 0))",
}


_SCALAR_OP: dict[str, Callable] = {
    "Not": lambda a: f"(!{a[0]})",
    "And": lambda a: f"({a[0]} && {a[1]})",
    "Or": lambda a: f"({a[0]} || {a[1]})",
    "Ite": lambda a: f"(if {a[0]} then {a[1]} else {a[2]})",
    "Add": lambda a: f"({a[0]} + {a[1]})",
    "Sub": lambda a: f"({a[0]} - {a[1]})",
    "Mul": lambda a: f"({a[0]} * {a[1]})",
    "Mod": lambda a: f"({a[0]} % {a[1]})",
    "Neg": lambda a: f"(-{a[0]})",
    "Lt": lambda a: f"(decide ({a[0]} < {a[1]}))",
    "Le": lambda a: f"(decide ({a[0]} ≤ {a[1]}))",
    "Gt": lambda a: f"(decide ({a[1]} < {a[0]}))",
    "Ge": lambda a: f"(decide ({a[1]} ≤ {a[0]}))",
    "Eq": lambda a: f"(decide ({a[0]} = {a[1]}))",
    "Neq": lambda a: f"(decide ({a[0]} ≠ {a[1]}))",
    "Min": lambda a: f"(Min.min {a[0]} {a[1]})",
    "Max": lambda a: f"(Max.max {a[0]} {a[1]})",
    "MatMul": lambda a: f"({a[0]} * {a[1]})",
    "Id": lambda a: a[0],
    "TensorGet": lambda a: a[0],
    "ToUnsigned": lambda a: f"(Int.toNat {a[0]})",
}


def _argmax_expr(arg_expr: str, input_shape: list[int]) -> str:
    """Emit `argmax_1d` for 1-d input and `argmax` for generic 2-d input.
    Wrap output as `Mat Int 1 _` to match the Python IR's Int output type."""
    if len(input_shape) == 1 or (len(input_shape) == 2 and input_shape[0] == 1):
        return f"(fun i j => ((argmax_1d {arg_expr}) i j : Int))"
    if len(input_shape) == 2:
        return f"(fun i j => ((argmax {arg_expr}) i j : Int))"
    raise ValueError(
        f"argmax: unsupported input shape {input_shape}; expected 1-d or 2-d"
    )


def _linear_expr(term, wire_expr: dict[int, str]) -> str:
    """Emit `affineLinear A x b` for a baked-constant LIA/LRA `Linear` op.

    Convention `Y = A·X + B`: `A` (shape `[out, in]`) and `B` (`[out, batch]`,
    or empty for no bias) are baked into the op; the single read wire is `X`.
    """
    out_wire = term.write[0]
    x_expr = wire_expr[term.read[0].id]
    A = term.itype._0
    B = term.itype._1
    a_lit = tensor_to_mat_expr(A, out_wire.dtype, list(A.shape))
    if B.numel() == 0:
        b_lit = f"(MatZero : {dtype_to_lean_type(out_wire)})"
    else:
        b_lit = tensor_to_mat_expr(B, out_wire.dtype, list(B.shape))
    return f"(affineLinear {a_lit} {x_expr} {b_lit})"


def _reachable_terms(terms, output_wires: "list[Wire]") -> list:
    """Return the terms backward-reachable from output_wires, in original order.

    Traverses the term list in reverse: a term is kept if any wire it writes
    is needed; its read wires are then added to the needed set.
    """
    needed = {w.id for w in output_wires}
    selected = []
    for term in reversed(list(terms)):
        if {w.id for w in term.write} & needed:
            needed |= {w.id for w in term.read}
            selected.append(term)
    return list(reversed(selected))


def _translate_terms(
    terms,
    input_bindings: dict[int, str],
    block_outputs: list[Wire],
    constants: ConstantRegistry,
) -> str:
    """Compile a block of terms into a Lean function body with let-bindings.

    `input_bindings` maps block-input wire IDs to their Lean accessor expressions
    (e.g. ``{w.id: "ctrl.1"}``) — pre-built by the caller via `_bind_wires`.

    Returns the body string (let x0 := ...; ... ; (x1, x2)) or "sorry" if no terms.
    """
    term_list = list(terms)
    if not term_list:
        return "sorry /- no terms -/"

    wire_expr: dict[int, str] = dict(input_bindings)
    let_lines: list[str] = []

    for var_counter, term in enumerate(term_list):
        name = itype_name(term.itype)

        if is_constant_name(name):
            expr = _constant_expr(name, term, term.write[0], constants)
        elif name == "Argmax":
            arg_expr = wire_expr[term.read[0].id]
            expr = _argmax_expr(arg_expr, dtype_shape(term.read[0].dtype))
        elif name == "Linear":
            expr = _linear_expr(term, wire_expr)
        else:
            if name not in _LEAN_OP:
                raise ValueError(f"No Lean expression mapping for: {name}")
            input_exprs = [wire_expr[w.id] for w in term.read]
            expr = _LEAN_OP[name](input_exprs)

        # Each term writes exactly one wire
        write_wire = term.write[0]
        var = f"x{var_counter}"
        wire_expr[write_wire.id] = var
        ty = dtype_to_lean_type(write_wire)
        let_lines.append(f"  let {var} : {ty} := {expr}")

    # Build output tuple
    out_exprs = [wire_expr[w.id] for w in block_outputs]
    result_line = f"  {_build_tuple(out_exprs)}"

    return "\n".join(let_lines + [result_line])


def _product_type_scalar(wires: list[Wire]) -> str:
    """Build a flat scalar product type, expanding multi-element wires.

    Each wire contributes ``_flat_size(w)`` copies of its base type.
    E.g. [Int(1), Float(2)] → ``"Int × Real × Real"``.
    """
    if not wires:
        return "Unit"
    parts: list[str] = []
    for w in wires:
        ty = _flat_element_type(w)
        parts.extend([ty] * _flat_size(w))
    if len(parts) == 1:
        return parts[0]
    return " × ".join(parts)


def _constant_expr_scalar(
    const_name: str, term, w: "Wire", constants: ConstantRegistry
) -> str:
    """Like _constant_expr but returns bare scalar values (no Mat wrapper)."""
    if _is_scalar_wire(w):
        # Element type (Bool/Int/Real) is taken from the wire's dtype.
        return _tensor_to_lean_scalar(term.itype._0, w)
    # Non-scalar tensor: fall back to matrix constant from registry
    name = constants.lookup(w.id)
    if name is not None:
        return name
    return _tensor_to_lean_inline(term.itype._0, w)


def _argmax_scalar_name(n: int) -> str:
    """Name of the scalar axiom for 1-d argmax over n elements."""
    return f"argmax1d_scalar_{n}"


def _translate_terms_scalar(
    terms,
    input_bindings: dict[int, str],
    block_outputs: list[Wire],
    constants: ConstantRegistry,
    flat_slots: "dict[int, list[str]] | None" = None,
) -> str:
    """Compile terms into a Lean body using scalar types for 1×1 wires.

    ``flat_slots`` maps wire IDs to their flat scalar accessor strings.  When
    provided, ``Argmax`` on a wire whose slots are known emits a call to the
    scalar axiom ``argmax1d_scalar_n`` instead of reconstructing a matrix.

    Falls back to `_LEAN_OP` (matrix form) for non-scalar output wires or
    operations not in `_SCALAR_OP`.
    """
    term_list = list(terms)
    if not term_list:
        out_exprs = [input_bindings.get(w.id) for w in block_outputs]
        if all(e is not None for e in out_exprs):
            return f"  {_build_tuple(out_exprs)}"
        return "sorry /- no terms -/"

    wire_expr: dict[int, str] = dict(input_bindings)
    flat_slots = dict(flat_slots or {})
    let_lines: list[str] = []

    for var_counter, term in enumerate(term_list):
        name = itype_name(term.itype)
        write_wire = term.write[0]
        var = f"x{var_counter}"

        if is_constant_name(name):
            expr = _constant_expr_scalar(name, term, write_wire, constants)
        elif name == "Argmax":
            in_wire = term.read[0]
            slots = flat_slots.get(in_wire.id)
            if slots is not None:
                axiom_name = _argmax_scalar_name(len(slots))
                expr = f"({axiom_name} {' '.join(slots)})"
            else:
                mat_expr = _argmax_expr(wire_expr[in_wire.id], dtype_shape(in_wire.dtype))
                expr = f"({mat_expr} 0 0)" if _is_scalar_wire(write_wire) else mat_expr
        elif name == "Linear":
            expr = _linear_expr(term, wire_expr)
        elif _is_scalar_wire(write_wire) and name in _SCALAR_OP:
            input_exprs = [wire_expr[w.id] for w in term.read]
            expr = _SCALAR_OP[name](input_exprs)
        else:
            if name not in _LEAN_OP:
                raise ValueError(f"No Lean expression mapping for: {name}")
            input_exprs = [wire_expr[w.id] for w in term.read]
            expr = _LEAN_OP[name](input_exprs)

        wire_expr[write_wire.id] = var
        # Track flat scalar slots for the written wire so downstream Argmax
        # terms can reference individual elements without matrix reconstruction.
        if _is_scalar_wire(write_wire):
            flat_slots[write_wire.id] = [var]
        else:
            flat_slots[write_wire.id] = [
                f"{var} {r} {c}" for r, c in _flat_indices(write_wire)
            ]
        ty = dtype_to_lean_type(write_wire, simple_types=True)
        let_lines.append(f"  let {var} : {ty} := {expr}")

    out_exprs = [wire_expr[w.id] for w in block_outputs]
    result_line = f"  {_build_tuple(out_exprs)}"

    return "\n".join(let_lines + [result_line])
