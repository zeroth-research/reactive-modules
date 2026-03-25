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

from zrth import Module, Wire, DType, Term


def dtype_to_lean_ty(wire: Wire) -> str:
    """Map a Wire's DType to a Lean Ty expression (.bool, .int, .mat m n)."""
    dt = wire.dtype
    shape = dt.shape

    if isinstance(dt, DType.Bool):
        if shape == [1] or shape == []:
            return ".bool"
        raise ValueError(f"Unsupported Bool shape: {shape}")

    if isinstance(dt, DType.Int):
        if shape == [1] or shape == []:
            return ".int"
        if len(shape) == 1:
            return f".mat {shape[0]} 1"
        if len(shape) == 2:
            return f".mat {shape[0]} {shape[1]}"
        raise ValueError(f"Unsupported Int shape: {shape}")

    raise ValueError(f"Unsupported DType for Lean conversion: {dt}")


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


def _ty_list(wires: list[Wire]) -> str:
    """Build a right-nested product type: [w1, w2] -> 'Bool × Bool', [w1] -> 'Bool'."""
    if not wires:
        return "[]"
    parts = [dtype_to_lean_ty(w) for w in wires]
    return "[{}]".format(" , ".join(parts))


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


def _is_scalar_tensor(wire: Wire) -> bool:
    """True if the wire carries a scalar Bool or Int (not a matrix)."""
    dt = wire.dtype
    if isinstance(dt, DType.Bool):
        return True
    if isinstance(dt, DType.Int):
        shape = dt.shape
        return shape == [] or shape == [1]
    return False


def _tensor_to_lean_inline(tensor, wire: Wire) -> str:
    """Return an inline Lean literal for a scalar tensor."""
    if isinstance(wire.dtype, DType.Bool):
        return "true" if bool(tensor.item()) else "false"
    if isinstance(wire.dtype, DType.Int):
        return str(int(tensor.item()))
    raise ValueError(f"Cannot inline tensor with dtype={wire.dtype}")


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

# Map operations to Lean expression builder (takes list of arg strings)
_LEAN_OP_BOX: dict[str, str] = {
    "Not": "Box.not",
    "And": "Box.and",
    "Or": "Box.or",
    "Ite": "Box.ite",
    "Add": "Box.add",
    "Sub": "Box.sub",
    "Mul": "Box.mul",
    "Neg": "Box.neg",
    "Lt": "Box.lt",
    "Le": "Box.le",
    "Gt": "Box.gt",
    "Ge": "Box.ge",
    "Eq": "Box.eq",
    "Neq": "Box.neq",
    "Min": "Box.min",
    "Max": "Box.max",
    "MatMul": "Box.matmul",
    "Id": "Box.id",
}


def itype_name(itype) -> str:
    """Get the variant name of an IType, e.g. IType.Add() -> 'Add'."""
    name = type(itype).__name__
    # PyO3 exports variants as IType_Add, IType_Tensor, etc.
    if name.startswith("IType_"):
        return name[6:]
    return name


def _translate_terms(
    terms,
    block_inputs: list[Wire],
    block_outputs: list[Wire],
    constants: dict[int, str],
    param_name: str = "s",
) -> str:
    """Compile a block of terms into a Lean function body with let-bindings.

    Returns the body string (let x0 := ...; ... ; (x1, x2)) or "sorry" if no terms.
    """
    term_list = [terms[i] for i in range(len(terms))]
    if not term_list:
        return "sorry /- no terms -/"

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
                wire_id = write_wires[0].id
                if wire_id in constants:
                    expr = constants[wire_id]
                else:
                    expr = _tensor_to_lean_inline(term.itype._0, write_wires[0])
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


def _translate_terms_circ(
    terms,
    block_inputs: list[Wire],
    block_outputs: list[Wire],
    constants: dict[int, str],
    param_name: str = "s",
) -> str:
    """Compile a block of terms into a Lean Boxes"""
    term_list = [terms[i] for i in range(len(terms))]
    if not term_list:
        return "sorry /- no terms -/"

    # Map wire_id -> Lean expression (variable name or input accessor)
    # TODO: do this once (we use it also in `_translate_terms`)
    n_inputs = len(block_inputs)
    wire_expr: dict[int, str] = {}
    for i, w in enumerate(block_inputs):
        acc = _accessor(i, n_inputs)
        wire_expr[w.id] = f"{param_name}{acc}"

    # output wires to terms
    wire_to_term: dict[Wire, Term] = {t.write[0]: t for t in term_list}

    # output layer are just the terms writing to output wires. However,
    # there are no terms e.g., for inputs (unless we want to create new
    # terms). So we use `(Wire, Term | None)` where if term is None we still
    # have the information about the element in the layer in the Wire
    output_layer: list[Wire] = block_outputs
    layers: list[list[Wire]] = [output_layer]

    # repeatedly take the last layer and cerate new layer
    # representing the inputs to the last layer.
    have_new = True
    while have_new:
        have_new = False

        layer: list[Wire] = layers[-1]
        new_layer: list[Wire] = []

        for w in layer:
            # if this is the input wire, just copy it
            if w in block_inputs:
                new_layer.append(w)
                continue

            term = wire_to_term[w]
            name = itype_name(term.itype)
            # constant terms have no inputs, the do not propagate into the upper layers
            if name in ("Tensor", "ConstBool", "ConstInt"):
                continue

            new_layer.extend(term.read)
            have_new = bool(new_layer)

        if new_layer != layer:
            layers.append(new_layer)

        if not have_new:
            break

    # compute swapping wires
    swapping = []
    wires_ord = {w: n for n, w in enumerate(block_inputs)}
    layer = layers[-1]
    while True:
        new_layer: list[Wire] = []
        boxes: list[str] = []
        N = len(layer)
        changed = False
        in_ty, out_ty = [], []
        i = 0
        while i < N:
            if i >= N - 1:
                # emit ID, this is the last wire or this and next wire are correctly ordered
                boxes.append(f"@Box.id {dtype_to_lean_ty(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_ty(layer[i]))
                out_ty.append(dtype_to_lean_ty(layer[i]))
                i += 1
            elif wires_ord[layer[i]] <= wires_ord[layer[i + 1]]:
                boxes.append(f"@Box.id {dtype_to_lean_ty(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_ty(layer[i]))
                out_ty.append(dtype_to_lean_ty(layer[i]))
                i += 1
            else:
                changed = True
                boxes.append(
                    f"@Box.swap {dtype_to_lean_ty(layer[i])} {dtype_to_lean_ty(layer[i + 1])}"
                )
                new_layer.append(layer[i + 1])
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_ty(layer[i]))
                in_ty.append(dtype_to_lean_ty(layer[i + 1]))
                out_ty.append(dtype_to_lean_ty(layer[i + 1]))
                out_ty.append(dtype_to_lean_ty(layer[i]))
                # `swap` consumes two wires
                i += 2

        if not changed:
            break
        assert boxes
        swapping.append(
            "  (({}): Box [{}] [{}])".format(
                " ⊗ ".join(boxes), ", ".join(in_ty), ", ".join(out_ty)
            )
        )
        assert len(new_layer) == len(layer)
        layer = new_layer

    # duplication of wires
    while True:
        new_layer: list[Wire] = []
        boxes: list[str] = []
        N = len(layer)
        changed = False
        in_ty, out_ty = [], []
        i = 0
        while i < N:
            if i >= N - 1:
                # emit ID, this is the last wire or this and next wire are correctly ordered
                boxes.append(f"@Box.id {dtype_to_lean_ty(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_ty(layer[i]))
                out_ty.append(dtype_to_lean_ty(layer[i]))
                i += 1
            elif wires_ord[layer[i]] != wires_ord[layer[i + 1]]:
                boxes.append(f"@Box.id {dtype_to_lean_ty(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_ty(layer[i]))
                out_ty.append(dtype_to_lean_ty(layer[i]))
                i += 1
            else:
                changed = True
                boxes.append(f"@Box.dup {dtype_to_lean_ty(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_ty(layer[i]))
                out_ty.append(dtype_to_lean_ty(layer[i]))
                out_ty.append(dtype_to_lean_ty(layer[i]))
                # `dup` consumes two wires
                i += 2

        if not changed:
            break
        assert boxes

        swapping.append(
            "  (({}): Box [{}] [{}])".format(
                " ⊗ ".join(boxes), ", ".join(in_ty), ", ".join(out_ty)
            )
        )
        assert len(new_layer) < len(layer)
        layer = new_layer

    # deletion of unused wires
    if len(block_inputs) != len(layer):
        boxes = []
        for w in block_inputs:
            if w in layer:
                boxes.append(f"@Box.id {dtype_to_lean_ty(layer[i])}")
            else:
                boxes.append("Box.destr")
        assert boxes
        swapping.append("  ({})".format(" ⊗ ".join(boxes)))

    lines: list[str] = list(reversed(swapping))

    for n, layer in enumerate(reversed(layers)):
        if not layer:
            # if there are unused variables, the first layer can be empty
            assert n == 0
            continue
        # lines.append("/- layer -/")
        boxes: list[str] = []
        in_ty, out_ty = [], []
        for w in layer:
            if w in block_inputs:
                boxes.append(f"@Box.id {dtype_to_lean_ty(w)}")
                in_ty.append(dtype_to_lean_ty(w))
                out_ty.append(dtype_to_lean_ty(w))
            else:
                term = wire_to_term[w]
                name = itype_name(term.itype)

                if name in ("Tensor", "ConstBool", "ConstInt"):
                    if name == "Tensor":
                        wire_id = w.id
                        if wire_id in constants:
                            expr = constants[wire_id]
                        else:
                            expr = _tensor_to_lean_inline(term.itype._0, w)
                    elif name == "ConstBool":
                        val = bool(term.itype._0)
                        expr = "true" if val else "false"
                    else:
                        expr = str(int(term.itype._0))

                    boxes.append(f"Box.const {dtype_to_lean_ty(w)} {expr}")
                    out_ty.append(dtype_to_lean_ty(w))
                else:
                    boxes.append(_LEAN_OP_BOX[name])
                    in_ty.extend([dtype_to_lean_ty(u) for u in term.read])
                    out_ty.extend([dtype_to_lean_ty(u) for u in term.write])
        assert boxes
        lines.append(
            "  (({}: Box [{}] [{}]))".format(
                " ⊗ ".join(boxes), ", ".join(in_ty), ", ".join(out_ty)
            )
        )

    return " ≫ \n".join(lines)
    ## return "\n".join(_lines + [result_line])


# ====================================================================
# ModuleToLean4 Class
#
# TODO: change to `GenerateLean4Cert` and cover also the certificate
# ====================================================================
class ModuleToLean4:
    """Convert a Python Module into Lean4 wiring diagram code."""

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
        """Pre-scan terms for matrix Tensor constants and generate top-level definitions.

        Scalar Bool/Int tensors are inlined directly in the function body.
        Only matrix tensors (shape with dim >= 2 or vector with size > 1) get
        top-level named definitions.
        """
        for term in terms:
            name_str = itype_name(term.itype)
            if name_str == "Tensor":
                out_wire = term.write[0]
                if _is_scalar_tensor(out_wire):
                    continue
                const_name = self._next_const_name()
                self._constants[out_wire.id] = const_name
                tensor = term.itype._0
                lean_def = _tensor_to_lean_def(const_name, tensor, out_wire)
                self._const_defs.append(lean_def)

    def to_lean_circuit(self, atom) -> str:
        """Generate the full Lean4 source for this module as a combinational circuit"""
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

        # Compile both blocks
        init_inputs = list(extl_next)
        init_outputs = list(ctrl_next)
        init_body = _translate_terms_circ(
            init_terms, init_inputs, init_outputs, self._constants
        )

        update_inputs = ctrl_latched + extl_next
        update_outputs = list(ctrl_next)
        update_body = _translate_terms_circ(
            update_terms, update_inputs, update_outputs, self._constants
        )

        # Render output
        lines = []

        lines.append("namespace Circ")

        # Init definition
        if init_body:
            init_dom = _ty_list(init_inputs)
            init_cod = _ty_list(init_outputs)
            lines.append(f"@[simp] def init : Box {init_dom} {init_cod} :=")
            lines.append(init_body)
            lines.append("")

        # Update definition
        if update_body:
            upd_dom = _ty_list(update_inputs)
            upd_cod = _ty_list(update_outputs)
            lines.append(f"@[simp] def update : Box {upd_dom} {upd_cod} :=")
            lines.append(update_body)
            lines.append("")

        lines.append("end Circ")

        return "\n".join(lines)

    def translate_constants(self, atom) -> str:
        # Extract constants from both blocks
        init_terms = atom.init
        update_terms = atom.update
        self._extract_constants(init_terms)
        self._extract_constants(update_terms)

        # Store metadata for certificate generation
        self.const_names = list(self._constants.values())

        lines: list[str] = []

        # Constants
        if self._const_defs:
            lines.append("/- Concrete constants -/")
            lines.append("")
            for cdef in self._const_defs:
                lines.append(cdef)

        return "\n".join(lines)

    def to_lean(self, circuit: bool = False) -> str:
        """Generate the full Lean4 source for this module."""
        m = self.module

        # Extract single atom (assume single atom for now)
        atoms = list(m.atoms)
        if len(atoms) != 1:
            raise ValueError(
                f"ModuleToLean4 currently supports single-atom modules, got {len(atoms)}"
            )

        atom = atoms[0]

        return "{}\n\n{}\n\n{}".format(
            self.translate_constants(atom),
            self.to_lean_circuit(atom),
            self.to_lean_functional(atom),
        )

    def to_lean_functional(self, atom) -> str:
        m = self.module

        extl_next = [pair[1] for pair in m.extl]
        ctrl_latched = [pair[0] for pair in m.ctrl]
        ctrl_next = [pair[1] for pair in m.ctrl]

        init_terms = atom.init
        update_terms = atom.update

        # Compile both blocks
        init_inputs = list(extl_next)
        init_outputs = list(ctrl_next)
        init_body = _translate_terms(
            init_terms, init_inputs, init_outputs, self._constants
        )

        update_inputs = ctrl_latched + extl_next
        update_outputs = list(ctrl_next)
        update_body = _translate_terms(
            update_terms, update_inputs, update_outputs, self._constants
        )

        # Render output
        lines = []

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
