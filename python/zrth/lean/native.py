"""
Translate a Python Module (reactive module with init/update blocks)
into Lean4 code.

Python modules store computations as a flat list of Terms (SSA-style dataflow).
We translate each Term into a `let` binding in a Lean function, mapping each
IType operation to its Lean equivalent.
"""

from __future__ import annotations
from zrth.lean.common import _constant_expr, itype_name, _accessor, dtype_to_lean_type

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


# Map operations to Lean expression builder (takes list of arg strings)
_LEAN_OP: dict[str, Callable] = {
    "Not": lambda a: f"!{a[0]}",
    "And": lambda a: f"({a[0]} && {a[1]})",
    "Or": lambda a: f"({a[0]} || {a[1]})",
    "Ite": lambda a: f"if {a[0]} then {a[1]} else {a[2]}",
    "Add": lambda a: f"({a[0]} + {a[1]})",
    "Sub": lambda a: f"({a[0]} - {a[1]})",
    "Mul": lambda a: f"({a[0]} * {a[1]})",
    "Mod": lambda a: f"({a[0]} % {a[1]})",
    "Neg": lambda a: f"(-{a[0]})",
    "Lt": lambda a: f"decide ({a[0]} 0 0 < {a[1]} 0 0)",
    "Le": lambda a: f"decide ({a[0]} 0 0 ≤ {a[1]} 0 0)",
    "Gt": lambda a: f"decide ({a[1]} 0 0 < {a[0]} 0 0)",
    "Ge": lambda a: f"decide ({a[1]} 0 0 ≤ {a[0]} 0 0)",
    "Eq": lambda a: f"({a[0]} == {a[1]})",
    "Neq": lambda a: f"({a[0]} != {a[1]})",
    "Min": lambda a: f"Min.min {a[0]} {a[1]}",
    "Max": lambda a: f"Max.max {a[0]} {a[1]}",
    "MatMul": lambda a: f"MatMul {a[0]} {a[1]}",
    "Id": lambda a: a[0],
    "Linear": lambda a: f"affineLinear {a[0]} {a[1]} {a[2]}",
    "ReLU": lambda a: f"Max.max 0 {a[0]}",
    "TensorGet": lambda a: f"({a[0]} 0 0)",
    "ToUnsigned": lambda a: f"({a[0]}).toNat",
    "Argmax": lambda a: f"argmax({a[0]})",
}


def _translate_terms(
    terms,
    block_inputs: tuple[list[Wire], ...],
    block_outputs: list[Wire],
    constants: dict[int, str],
    param_names: list[str],
) -> str:
    """Compile a block of terms into a Lean function body with let-bindings.

    Returns the body string (let x0 := ...; ... ; (x1, x2)) or "sorry" if no terms.
    """
    term_list = [terms[i] for i in range(len(terms))]
    if not term_list:
        return "sorry /- no terms -/"

    # Map wire_id -> Lean expression (variable name or input accessor)
    wire_expr: dict[int, str] = {}
    all_inputs = [w for wires in block_inputs for w in wires]
    for name, wires in zip(param_names, block_inputs):
        n_inputs = len(wires)
        for i, w in enumerate(wires):
            acc = _accessor(i, n_inputs)
            wire_expr[w.id] = f"{name}{acc}"
            print(f"{w.id} => {wire_expr[w.id]}")

    var_counter = 0
    let_lines: list[str] = []

    for term in term_list:
        read_wires = [term.read[i] for i in range(len(term.read))]
        write_wires = [term.write[i] for i in range(len(term.write))]
        name = itype_name(term.itype)

        if name in ("Tensor", "ConstBool", "ConstInt"):
            expr = _constant_expr(name, term, write_wires[0], constants)
        else:
            input_exprs = [wire_expr[w.id] for w in read_wires]
            it_name = itype_name(term.itype)
            if it_name not in _LEAN_OP:
                raise ValueError(f"No Lean expression mapping for: {it_name}")
            expr = _LEAN_OP[it_name](input_exprs)

        # Each term writes exactly one wire
        var = f"x{var_counter}"
        var_counter += 1
        wire_expr[write_wires[0].id] = var
        let_lines.append(f"  let {var} := {expr}")

    # Build output tuple
    out_exprs = [wire_expr[w.id] for w in block_outputs]
    result_line = f"  {_build_tuple(out_exprs)}"

    return "\n".join(let_lines + [result_line])
