"""
Translate a Python Module (reactive module with init/update blocks)
into Lean4 code.

Python modules store computations as a flat list of Terms (SSA-style dataflow).
We translate each Term into a `let` binding in a Lean function, mapping each
IType operation to its Lean equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

from zrth import Module, Wire, DType, IType, Term


def dtype_to_lean_native(wire: Wire) -> str:
    """Map a Wire's DType to a native Lean type (Bool, Int, Fin m → Fin n → Int)."""
    dt = wire.dtype
    shape = dt.shape

    if isinstance(dt, DType.Bool):
        if shape == [1] or shape == []:
            return "Bool"
        raise ValueError(f"Unsupported Bool shape: {shape}")

    if isinstance(dt, DType.Int):
        if shape == [1] or shape == []:
            return "Int"
        if len(shape) == 1:
            return f"(Fin {shape[0]} → Fin 1 → Int)"
        if len(shape) == 2:
            return f"(Fin {shape[0]} → Fin {shape[1]} → Int)"
        raise ValueError(f"Unsupported Int shape: {shape}")

    raise ValueError(f"Unsupported DType for Lean conversion: {dt}")


def _product_type(wires: list[Wire]) -> str:
    """Build a right-nested product type: [w1, w2] -> 'Bool × Bool', [w1] -> 'Bool'."""
    if not wires:
        return "Unit"
    if len(wires) == 1:
        return dtype_to_lean_native(wires[0])
    parts = [dtype_to_lean_native(w) for w in wires]
    return " × ".join(parts)


def _tensor_to_lean_def(name: str, tensor, wire: Wire) -> str:
    """
    Generate a top-level Lean definition for a constant tensor.

    E.g.:
        @[simp] def A : Fin 3 → Fin 2 → Int := fun i j =>
          match i, j with
          | 0, 0 => 0 | 0, 1 => 1
          ...
    """
    shape = wire.dtype.shape

    # Scalar bool constant
    if isinstance(wire.dtype, DType.Bool):
        val = bool(tensor.item())
        lean_val = "true" if val else "false"
        return f"@[simp] def {name} : Bool := {lean_val}\n"

    # Scalar int constant
    if isinstance(wire.dtype, DType.Int) and (shape == [1] or shape == []):
        val = int(tensor.item())
        return f"@[simp] def {name} : Int := {val}\n"

    # Matrix constant
    if isinstance(wire.dtype, DType.Int) and len(shape) >= 1:
        if len(shape) == 1:
            m, n = shape[0], 1
        else:
            m, n = shape[0], shape[1]

        lines = [f"@[simp] def {name} : Fin {m} → Fin {n} → Int := fun i j =>"]
        lines.append("  match i, j with")

        data = tensor.reshape(m, n)
        for i in range(m):
            row_entries = []
            for j in range(n):
                val = int(data[i, j].item())
                row_entries.append(f"| {i}, {j} => {val}")
            lines.append("  " + " ".join(row_entries))

        return "\n".join(lines) + "\n"

    raise ValueError(
        f"Cannot generate Lean constant for dtype={wire.dtype}, shape={shape}"
    )


# ══════════════════════════════════════════════════════════════════════════
# Part E: Block Compiler (core algorithm)
# ══════════════════════════════════════════════════════════════════════════


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
    "Neg": lambda a: f"(-{a[0]})",
    "Lt": lambda a: f"decide ({a[0]} < {a[1]})",
    "Le": lambda a: f"decide ({a[0]} ≤ {a[1]})",
    "Gt": lambda a: f"decide ({a[1]} < {a[0]})",
    "Ge": lambda a: f"decide ({a[1]} ≤ {a[0]})",
    "Eq": lambda a: f"({a[0]} == {a[1]})",
    "Neq": lambda a: f"({a[0]} != {a[1]})",
    "Min": lambda a: f"Min.min {a[0]} {a[1]}",
    "Max": lambda a: f"Max.max {a[0]} {a[1]}",
    "MatMul": lambda a: f"MatMul {a[0]} {a[1]}",
    "Id": lambda a: a[0],
}


def itype_name(itype) -> str:
    """Get the variant name of an IType, e.g. IType.Add() -> 'Add'."""
    name = type(itype).__name__
    # PyO3 exports variants as IType_Add, IType_Tensor, etc.
    if name.startswith("IType_"):
        return name[6:]
    return name


def _compile_block_functional(
    terms,
    block_inputs: list[Wire],
    block_outputs: list[Wire],
    constants: dict[int, str],
    param_name: str = "s",
) -> Optional[str]:
    """Compile a block into a Lean function body with let-bindings.

    Returns the body string (let x0 := ...; ... ; (x1, x2)) or None if no terms.
    """
    term_list = [terms[i] for i in range(len(terms))]
    if not term_list:
        return None

    # Map wire_id -> Lean expression (variable name or input accessor)
    n_inputs = len(block_inputs)
    wire_expr: dict[int, str] = {}
    for i, w in enumerate(block_inputs):
        acc = _accessor(i, n_inputs)
        wire_expr[w.id] = f"{param_name}{acc}"

    var_counter = 0
    let_lines: list[str] = []

    for term in term_list:
        read_wires = [term.read[i] for i in range(len(term.read))]
        write_wires = [term.write[i] for i in range(len(term.write))]
        name = itype_name(term.itype)

        if name in ("Tensor", "ConstBool", "ConstInt"):
            if name == "Tensor":
                expr = constants[write_wires[0].id]
            elif name == "ConstBool":
                val = bool(term.itype._0)
                expr = "true" if val else "false"
            else:
                expr = str(int(term.itype._0))
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


# ══════════════════════════════════════════════════════════════════════════
# ModuleToLean4 Class
# ══════════════════════════════════════════════════════════════════════════


class ModuleToLean4:
    """Convert a Python Module into Lean4 Box wiring diagram code."""

    def __init__(self, module: Module):
        self.module = module
        self._const_counter = 0
        self._const_defs: list[str] = []
        self._constants: dict[int, str] = {}  # wire_id -> const name
        self.const_names: list[str] = []  # populated after to_lean()

    def _next_const_name(self) -> str:
        """Generate sequential constant names: c0, c1, c2, ..."""
        name = f"c{self._const_counter}"
        self._const_counter += 1
        return name

    def _extract_constants(self, terms) -> None:
        """Pre-scan terms for Tensor constants and generate top-level definitions."""
        for term in terms:
            name_str = itype_name(term.itype)
            if name_str == "Tensor":
                out_wire = term.write[0]
                const_name = self._next_const_name()
                self._constants[out_wire.id] = const_name
                tensor = term.itype._0
                lean_def = _tensor_to_lean_def(const_name, tensor, out_wire)
                self._const_defs.append(lean_def)

    def to_lean(self) -> str:
        """Generate the full Lean4 source for this module."""
        m = self.module

        # Extract single atom (assume single atom for now)
        atoms = list(m.atoms)
        if len(atoms) != 1:
            raise ValueError(
                f"ModuleToLean4 currently supports single-atom modules, got {len(atoms)}"
            )
        atom = atoms[0]

        extl_next = [pair[1] for pair in m.extl]
        ctrl_latched = [pair[0] for pair in m.ctrl]
        ctrl_next = [pair[1] for pair in m.ctrl]

        # Extract constants from both blocks
        init_terms = atom.init
        update_terms = atom.update
        self._extract_constants(init_terms)
        self._extract_constants(update_terms)

        # Compile both blocks
        init_inputs = list(extl_next)
        init_outputs = list(ctrl_next)
        init_body = _compile_block_functional(
            init_terms, init_inputs, init_outputs, self._constants
        )

        update_inputs = ctrl_latched + extl_next
        update_outputs = list(ctrl_next)
        update_body = _compile_block_functional(
            update_terms, update_inputs, update_outputs, self._constants
        )

        # Store metadata for certificate generation
        self.const_names = list(self._constants.values())

        # Render output
        lines = []

        # Constants
        if self._const_defs:
            lines.append("/- Concrete constants -/")
            lines.append("")
            for cdef in self._const_defs:
                lines.append(cdef)

        # Init definition
        if init_body:
            init_dom = _product_type(init_inputs)
            init_cod = _product_type(init_outputs)
            lines.append(f"@[simp] def init (s : {init_dom}) : {init_cod} :=")
            lines.append(init_body)
            lines.append("")

        # Update definition
        if update_body:
            upd_dom = _product_type(update_inputs)
            upd_cod = _product_type(update_outputs)
            lines.append(f"@[simp] def update (s : {upd_dom}) : {upd_cod} :=")
            lines.append(update_body)
            lines.append("")

        return "\n".join(lines)
