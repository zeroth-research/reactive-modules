"""
Translate a Python Module (reactive module with init/update blocks)
into Lean4 code.

Python modules store computations as a flat list of Terms (SSA-style dataflow).
We translate each Term into a `let` binding in a Lean function, mapping each
IType operation to its Lean equivalent.
"""

from __future__ import annotations
from zrth.lean.common import (
    ConstantRegistry,
    _constant_expr,
    itype_name,
    _accessor,
    dtype_to_lean_type,
    _is_scalar_wire,
    _tensor_to_lean_scalar,
    _bind_wires_scalar,
    _tensor_to_lean_inline,
)

from typing import Callable

from zrth import Wire, DType


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
    "Linear": lambda a: f"affineLinear {a[0]} {a[1]} {a[2]}",
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

        if name in ("Tensor", "ConstBool", "ConstInt"):
            expr = _constant_expr(name, term, term.write[0], constants)
        elif name == "Argmax":
            arg_expr = wire_expr[term.read[0].id]
            expr = _argmax_expr(arg_expr, term.read[0].dtype.shape)
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
    """Build a product type using bare scalar types (Bool/Int/Real, no Mat)."""
    if not wires:
        return "Unit"
    if len(wires) == 1:
        return dtype_to_lean_type(wires[0], simple_types=True)
    parts = [dtype_to_lean_type(w, simple_types=True) for w in wires]
    return " × ".join(parts)


def _constant_expr_scalar(
    const_name: str, term, w: "Wire", constants: ConstantRegistry
) -> str:
    """Like _constant_expr but returns bare scalar values (no Mat wrapper)."""
    if const_name == "ConstBool":
        return "true" if bool(term.itype._0) else "false"
    if const_name == "ConstInt":
        return f"({int(term.itype._0)} : Int)"
    # Tensor
    if _is_scalar_wire(w):
        return _tensor_to_lean_scalar(term.itype._0, w)
    # Non-scalar tensor: fall back to matrix constant from registry
    name = constants.lookup(w.id)
    if name is not None:
        return name
    return _tensor_to_lean_inline(term.itype._0, w)


def _translate_terms_scalar(
    terms,
    input_bindings: dict[int, str],
    block_outputs: list[Wire],
    constants: ConstantRegistry,
) -> str:
    """Compile terms into a Lean body using scalar types for 1×1 wires.

    `input_bindings` maps wire IDs to Lean accessor expressions.  Pass
    bindings from ``_bind_wires`` — the Scalar.init/update signatures use
    bare scalar types (``Int``/``Bool``), so accessors like ``"ctrl.1"``
    already refer to bare scalars in that context.

    Falls back to `_LEAN_OP` (matrix form) for any non-scalar output wire or
    operations not in `_SCALAR_OP`.  Returns ``"sorry /- no terms -/"`` when
    the term list is empty.
    """
    term_list = list(terms)
    if not term_list:
        return "sorry /- no terms -/"

    wire_expr: dict[int, str] = dict(input_bindings)
    let_lines: list[str] = []

    for var_counter, term in enumerate(term_list):
        name = itype_name(term.itype)
        write_wire = term.write[0]
        var = f"x{var_counter}"

        if name in ("Tensor", "ConstBool", "ConstInt"):
            expr = _constant_expr_scalar(name, term, write_wire, constants)
        elif name == "Argmax":
            expr = _argmax_expr(wire_expr[term.read[0].id], term.read[0].dtype.shape)
        elif _is_scalar_wire(write_wire) and name in _SCALAR_OP:
            input_exprs = [wire_expr[w.id] for w in term.read]
            expr = _SCALAR_OP[name](input_exprs)
        else:
            if name not in _LEAN_OP:
                raise ValueError(f"No Lean expression mapping for: {name}")
            input_exprs = [wire_expr[w.id] for w in term.read]
            expr = _LEAN_OP[name](input_exprs)

        wire_expr[write_wire.id] = var
        ty = dtype_to_lean_type(write_wire, simple_types=True)
        let_lines.append(f"  let {var} : {ty} := {expr}")

    out_exprs = [wire_expr[w.id] for w in block_outputs]
    result_line = f"  {_build_tuple(out_exprs)}"

    return "\n".join(let_lines + [result_line])
