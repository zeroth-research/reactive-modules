"""
Translate a Python Module (reactive module with init/update blocks)
into Lean4 code using the Box wiring-diagram type system.

The core challenge: Python modules store computations as a flat list of Terms
(SSA-style dataflow). We convert this into point-free categorical composition
using ≫ (sequential) and ⊗ (parallel) Box operators, inserting routing
primitives (dup, destr, id, swap) as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from zrth import Module, Wire, DType, IType, Term


# ══════════════════════════════════════════════════════════════════════════
# Part A: Type Mapping
# ══════════════════════════════════════════════════════════════════════════


def dtype_to_lean_ty(wire: Wire) -> str:
    """Map a Wire's DType to a Lean Ty expression."""
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


# ══════════════════════════════════════════════════════════════════════════
# Part B: Box Expression AST
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class BoxExpr:
    """Base class for Box expression AST nodes."""

    dom: list[str]  # list of Lean Ty expressions
    cod: list[str]

    def render(self) -> str:
        raise NotImplementedError


@dataclass
class Prim(BoxExpr):
    """A primitive box: add, mul, lt, dup, id, swap, destr, etc."""

    name: str
    type_params: str = ""  # e.g. for typed id: "@id (.int)"

    def render(self) -> str:
        if self.type_params:
            return f"{self.name} {self.type_params}"
        # Prefix with @ for primitives that need explicit type params
        if self.name == "swap":
            if len(self.dom) == 2:
                return f"@{self.name} ({self.dom[0]}) ({self.dom[1]})"
            return self.name
        if self.name == "ite":
            # ite : Box [.bool, t, t] [t] — provide the value type
            if len(self.dom) >= 2:
                return f"@{self.name} ({self.dom[1]})"
            return self.name
        if self.name in ("id", "dup", "destr", "eq", "neq"):
            if len(self.dom) > 0:
                ty = self.dom[0]
                return f"@{self.name} ({ty})"
            return self.name
        return self.name


@dataclass
class Const(BoxExpr):
    """A constant box: const (.mat 3 2) A"""

    ty: str
    const_name: str

    def render(self) -> str:
        return f"const ({self.ty}) {self.const_name}"


@dataclass
class Seq(BoxExpr):
    """Sequential composition: left ≫ right"""

    left: BoxExpr
    right: BoxExpr

    def render(self) -> str:
        return f"{self.left.render()} ≫\n  {self.right.render()}"


@dataclass
class Par(BoxExpr):
    """Parallel composition: left ⊗ right"""

    parts: list[BoxExpr]

    def render(self) -> str:
        if len(self.parts) == 1:
            return self.parts[0].render()
        return " ⊗ ".join(p.render() for p in self.parts)


@dataclass
class Annotated(BoxExpr):
    """Type-annotated expression: (expr : Box [dom...] [cod...])"""

    expr: BoxExpr

    def render(self) -> str:
        dom_str = ", ".join(self.dom)
        cod_str = ", ".join(self.cod)
        inner = self.expr.render()
        return f"(({inner}): Box [{dom_str}] [{cod_str}])"


# ══════════════════════════════════════════════════════════════════════════
# Part C: IType-to-Box Mapping
# ══════════════════════════════════════════════════════════════════════════

# Map IType variant name -> (lean primitive name, is_unary)
_ITYPE_TO_BOX: dict[str, tuple[str, bool]] = {
    "Add": ("add", False),
    "Sub": ("sub", False),
    "Mul": ("mul", False),
    "Neg": ("neg", True),
    "Lt": ("lt", False),
    "Le": ("le", False),
    "Gt": ("gt", False),
    "Ge": ("ge", False),
    "Eq": ("eq", False),
    "Neq": ("neq", False),
    "Or": ("or", False),
    "And": ("and", False),
    "Not": ("not", True),
    "Ite": ("ite", False),  # special: 3 inputs
    "MatMul": ("matMul", False),
    "MatAdd": ("matAdd", False),
    "Id": ("id", True),
    "Min": ("min", False),
    "Max": ("max", False),
}


def itype_name(itype) -> str:
    """Get the variant name of an IType, e.g. IType.Add() -> 'Add'."""
    name = type(itype).__name__
    # PyO3 exports variants as IType_Add, IType_Tensor, etc.
    if name.startswith("IType_"):
        return name[6:]
    return name


def itype_to_box_name(itype, input_wires: list[Wire] = None) -> str:
    """Map an IType to its Lean Box primitive name.

    For Add/Mul on matrix types, automatically maps to matAdd/matMul.
    """
    name = itype_name(itype)

    # Check if inputs are matrix types — if so, use mat-prefixed ops
    if input_wires and name in ("Add", "Mul"):
        first = input_wires[0]
        if isinstance(first.dtype, DType.Int) and len(first.dtype.shape) >= 2:
            return "matAdd" if name == "Add" else "matMul"
        if isinstance(first.dtype, DType.Int) and len(first.dtype.shape) == 1 and first.dtype.shape[0] > 1:
            return "matAdd" if name == "Add" else "matMul"

    if name in _ITYPE_TO_BOX:
        return _ITYPE_TO_BOX[name][0]
    if name in ("Tensor", "ConstBool", "ConstInt"):
        return "const"
    raise ValueError(f"Unsupported IType for Lean conversion: {itype} ({name})")


# ══════════════════════════════════════════════════════════════════════════
# Part D: Constant Extraction
# ══════════════════════════════════════════════════════════════════════════


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

    raise ValueError(f"Cannot generate Lean constant for dtype={wire.dtype}, shape={shape}")


# ══════════════════════════════════════════════════════════════════════════
# Part E: Block Compiler (core algorithm)
# ══════════════════════════════════════════════════════════════════════════


def _vt_accessor(pos: int) -> str:
    """ValTuple accessor for position pos: 0->'.1', 1->'.2.1', 2->'.2.2.1'"""
    return ".2" * pos + ".1"


# Map lean primitive name -> (n_inputs, n_outputs, output_expr_builder)
# Each builder takes a list of input expressions and returns a list of output expressions
_PRIM_EXPR_MAP: dict[str, tuple[int, int]] = {
    "id": (1, 1),
    "dup": (1, 2),
    "swap": (2, 2),
    "destr": (1, 0),
    "not": (1, 1),
    "and": (2, 1),
    "or": (2, 1),
    "ite": (3, 1),
    "add": (2, 1),
    "sub": (2, 1),
    "mul": (2, 1),
    "neg": (1, 1),
    "lt": (2, 1),
    "le": (2, 1),
    "gt": (2, 1),
    "ge": (2, 1),
    "eq": (2, 1),
    "neq": (2, 1),
    "min": (2, 1),
    "max": (2, 1),
    "matMul": (2, 1),
    "matAdd": (2, 1),
}


def _prim_output_exprs(name: str, inputs: list[str]) -> list[str]:
    """Given a primitive name and its input expressions, return output expressions."""
    if name == "id":
        return [inputs[0]]
    if name == "dup":
        return [inputs[0], inputs[0]]
    if name == "swap":
        return [inputs[1], inputs[0]]
    if name == "destr":
        return []
    if name == "not":
        return [f"¬{inputs[0]}"]
    if name == "and":
        return [f"({inputs[0]} ∧ {inputs[1]})"]
    if name == "or":
        return [f"({inputs[0]} ∨ {inputs[1]})"]
    if name == "ite":
        return [f"(if {inputs[0]} then {inputs[1]} else {inputs[2]})"]
    if name == "add":
        return [f"({inputs[0]} + {inputs[1]})"]
    if name == "sub":
        return [f"({inputs[0]} - {inputs[1]})"]
    if name == "mul":
        return [f"({inputs[0]} * {inputs[1]})"]
    if name == "neg":
        return [f"(-{inputs[0]})"]
    if name == "lt":
        return [f"({inputs[0]} < {inputs[1]})"]
    if name == "le":
        return [f"({inputs[0]} ≤ {inputs[1]})"]
    if name == "gt":
        return [f"({inputs[1]} < {inputs[0]})"]
    if name == "ge":
        return [f"({inputs[1]} ≤ {inputs[0]})"]
    if name == "eq":
        return [f"({inputs[0]} == {inputs[1]})"]
    if name == "neq":
        return [f"({inputs[0]} != {inputs[1]})"]
    if name == "min":
        return [f"(Min.min {inputs[0]} {inputs[1]})"]
    if name == "max":
        return [f"(Max.max {inputs[0]} {inputs[1]})"]
    if name == "matMul":
        return [f"(MatMul {inputs[0]} {inputs[1]})"]
    if name == "matAdd":
        return [f"(MatAdd {inputs[0]} {inputs[1]})"]
    return None


def _layer_output_exprs(layer: BoxExpr) -> "list[str] | None":
    """Compute output expressions for a layer in terms of input 's'.

    Returns a list of strings, one per output position, or None if unrecognized.
    """
    # Unwrap Annotated
    expr = layer
    if isinstance(expr, Annotated):
        expr = expr.expr

    # Handle single Prim
    if isinstance(expr, Prim):
        parts = [expr]
    elif isinstance(expr, Par):
        parts = expr.parts
    elif isinstance(expr, Const):
        parts = [expr]
    else:
        return None

    input_offset = 0
    outputs = []
    for part in parts:
        if isinstance(part, Const):
            outputs.append(part.const_name)
            # Consts consume 0 inputs
        elif isinstance(part, Prim):
            # Strip @ prefix and type params to get base name
            base_name = part.name.lstrip("@").split()[0]
            n_in = len(part.dom)
            inp_exprs = [f"s{_vt_accessor(input_offset + i)}" for i in range(n_in)]
            out = _prim_output_exprs(base_name, inp_exprs)
            if out is None:
                return None
            outputs.extend(out)
            input_offset += n_in
        else:
            return None

    return outputs


@dataclass
class _WireInfo:
    """Track a wire's position and usage in the compilation."""

    wire: Wire
    ty: str  # Lean type string
    remaining_reads: int  # how many more times this wire will be read


class BlockCompiler:
    """
    Compile a list of Terms (one block: init or update) into a Box expression.

    The algorithm maintains a "stack" (ordered list) of live wires.
    For each term, it generates a layer that:
    1. Dups wires that are read but still needed later
    2. Routes read-wires to the operation, passes others through with id
    3. Appends output wires to the stack
    """

    def __init__(
        self,
        terms: list[Term],
        block_inputs: list[Wire],
        block_outputs: list[Wire],
        constants: dict[int, str],  # wire_id -> constant name
    ):
        self.terms = terms
        self.block_inputs = block_inputs
        self.block_outputs = block_outputs
        self.constants = constants  # wire_id -> Lean constant name
        self.layers: list[BoxExpr] = []

    def compile(self) -> Optional[BoxExpr]:
        if not self.terms:
            return None

        # Build wire-write map and compute fan-out
        wire_write_map: dict[int, Term] = {}  # wire_id -> term that writes it
        wire_future_reads: dict[int, int] = {}  # wire_id -> total read count

        for term in self.terms:
            for i in range(len(term.write)):
                w = term.write[i]
                wire_write_map[w.id] = term

        # Count reads across all terms
        for term in self.terms:
            for i in range(len(term.read)):
                w = term.read[i]
                wire_future_reads[w.id] = wire_future_reads.get(w.id, 0) + 1

        # Count reads from block outputs
        for w in self.block_outputs:
            wire_future_reads[w.id] = wire_future_reads.get(w.id, 0) + 1

        # Initialize wire stack with block inputs
        stack: list[int] = [w.id for w in self.block_inputs]  # wire ids
        wire_types: dict[int, str] = {}
        wire_remaining: dict[int, int] = {}

        for w in self.block_inputs:
            wire_types[w.id] = dtype_to_lean_ty(w)
            wire_remaining[w.id] = wire_future_reads.get(w.id, 0)

        layers: list[BoxExpr] = []

        for term in self.terms:
            read_ids = [term.read[i].id for i in range(len(term.read))]
            read_wires = [term.read[i] for i in range(len(term.read))]
            write_wires = [term.write[i] for i in range(len(term.write))]

            # Register output wire types
            for w in write_wires:
                wire_types[w.id] = dtype_to_lean_ty(w)
                wire_remaining[w.id] = wire_future_reads.get(w.id, 0)

            name = itype_name(term.itype)

            # Handle constants (no reads, just produce a wire)
            if name == "Tensor" or name == "ConstBool" or name == "ConstInt":
                out_wire = write_wires[0]
                out_ty = wire_types[out_wire.id]

                if out_wire.id in self.constants:
                    const_name = self.constants[out_wire.id]
                else:
                    # Inline scalar constants
                    if name == "ConstBool":
                        val = bool(term.itype._0)
                        const_name = "true" if val else "false"
                    elif name == "ConstInt":
                        const_name = str(int(term.itype._0))
                    else:
                        raise ValueError(
                            f"Tensor constant without pre-extracted name: wire {out_wire.id}"
                        )

                const_box = Const(
                    dom=[],
                    cod=[out_ty],
                    ty=out_ty,
                    const_name=const_name,
                )

                # Build layer: all existing stack wires pass through with id, const appended
                if stack:
                    stack_dom = [wire_types[wid] for wid in stack]
                    stack_cod = [wire_types[wid] for wid in stack] + [out_ty]

                    id_parts = [
                        Prim(dom=[wire_types[wid]], cod=[wire_types[wid]], name="id")
                        for wid in stack
                    ]
                    id_parts.append(const_box)
                    layer_expr = Par(dom=stack_dom, cod=stack_cod, parts=id_parts)
                    layer = Annotated(dom=stack_dom, cod=stack_cod, expr=layer_expr)
                else:
                    layer = const_box

                layers.append(layer)
                stack.append(out_wire.id)
                continue

            # For non-constant terms: route inputs, apply op, pass-through others

            # Step 1: Figure out which stack positions are consumed by this term
            # and which need duplication
            consumed_positions: list[int] = []  # stack indices consumed by this term
            for rid in read_ids:
                # Find this wire in the stack
                for idx, sid in enumerate(stack):
                    if sid == rid and idx not in consumed_positions:
                        consumed_positions.append(idx)
                        break

            # Step 2: Determine which consumed wires need duplication
            # (still have future reads after this term)
            dup_layers: list[BoxExpr] = []
            for pos in consumed_positions:
                wid = stack[pos]
                wire_remaining[wid] -= 1
                if wire_remaining[wid] > 0:
                    # Need to dup this wire
                    ty = wire_types[wid]
                    # Build a layer that dups wire at position `pos`
                    new_stack = list(stack)
                    parts = []
                    new_dom = [wire_types[sid] for sid in stack]
                    new_cod = []
                    for i, sid in enumerate(stack):
                        sty = wire_types[sid]
                        if i == pos:
                            parts.append(Prim(dom=[sty], cod=[sty, sty], name="dup"))
                            new_cod.append(sty)
                            new_cod.append(sty)
                        else:
                            parts.append(Prim(dom=[sty], cod=[sty], name="id"))
                            new_cod.append(sty)

                    dup_expr = Par(dom=new_dom, cod=new_cod, parts=parts)
                    dup_layer = Annotated(dom=new_dom, cod=new_cod, expr=dup_expr)
                    dup_layers.append(dup_layer)

                    # Update stack: insert duplicate after position
                    # The duplicate occupies a new slot; shift consumed_positions
                    stack.insert(pos + 1, wid)
                    # Shift consumed positions that come after
                    for j in range(len(consumed_positions)):
                        if consumed_positions[j] > pos:
                            consumed_positions[j] += 1

            layers.extend(dup_layers)

            # Step 3: We now need to rearrange the stack so that the consumed
            # wires are contiguous and in the right order for the operation.
            # Strategy: use swaps to bring consumed wires together.
            # For simplicity, we'll permute consumed wires to be at the front.

            # Build the desired order: consumed wires first (in read order), then rest
            read_positions = []
            used_positions = set()
            for rid in read_ids:
                for idx, sid in enumerate(stack):
                    if sid == rid and idx not in used_positions:
                        read_positions.append(idx)
                        used_positions.add(idx)
                        break

            pass_positions = [i for i in range(len(stack)) if i not in used_positions]
            desired_order = read_positions + pass_positions

            # Generate swap layers to achieve the permutation
            # We use bubble sort on the stack to bring wires into desired order
            current_perm = list(desired_order)
            # We need to find swaps that transform identity perm into desired_order
            # Actually we need inverse: from current stack to desired arrangement
            # Let's just do adjacent swaps
            temp_stack = list(stack)
            swap_layers = self._generate_permutation_layers(
                temp_stack, read_positions, pass_positions, wire_types
            )
            layers.extend(swap_layers)

            # Update stack to reflect the permutation
            new_stack = [stack[i] for i in desired_order]
            stack = new_stack

            # Step 4: Build the operation layer
            # Consumed wires (first N) go to the op, rest pass through with id
            n_consumed = len(read_ids)
            op_name = itype_to_box_name(term.itype, read_wires)

            op_dom = [wire_types[stack[i]] for i in range(n_consumed)]
            op_cod = [wire_types[w.id] for w in write_wires]

            if name == "Ite":
                # ite needs explicit type param
                t = wire_types[write_wires[0].id]
                op_box = Prim(dom=op_dom, cod=op_cod, name="ite")
            elif name in ("Eq", "Neq"):
                op_box = Prim(dom=op_dom, cod=op_cod, name=op_name)
            else:
                op_box = Prim(dom=op_dom, cod=op_cod, name=op_name)

            pass_dom = [wire_types[stack[i]] for i in range(n_consumed, len(stack))]
            pass_cod = pass_dom[:]

            parts = [op_box]
            for i in range(n_consumed, len(stack)):
                ty = wire_types[stack[i]]
                parts.append(Prim(dom=[ty], cod=[ty], name="id"))

            full_dom = [wire_types[sid] for sid in stack]
            full_cod = op_cod + pass_cod

            if len(parts) == 1:
                layer_expr = parts[0]
            else:
                layer_expr = Par(dom=full_dom, cod=full_cod, parts=parts)

            layer = Annotated(dom=full_dom, cod=full_cod, expr=layer_expr)
            layers.append(layer)

            # Update stack: remove consumed, add outputs
            stack = [w.id for w in write_wires] + stack[n_consumed:]

        # Step 5: Final routing - reorder stack to match block_outputs,
        # destroy any wires not in block_outputs
        output_ids = [w.id for w in self.block_outputs]

        # Remove wires not needed in output
        wires_to_destroy = [wid for wid in stack if wid not in output_ids]
        for wid in wires_to_destroy:
            idx = stack.index(wid)
            ty = wire_types[wid]
            dom = [wire_types[sid] for sid in stack]
            cod = [wire_types[sid] for sid in stack if sid != wid]

            parts = []
            for i, sid in enumerate(stack):
                sty = wire_types[sid]
                if i == idx:
                    parts.append(Prim(dom=[sty], cod=[], name="destr"))
                else:
                    parts.append(Prim(dom=[sty], cod=[sty], name="id"))

            destr_expr = Par(dom=dom, cod=cod, parts=parts)
            destr_layer = Annotated(dom=dom, cod=cod, expr=destr_expr)
            layers.append(destr_layer)
            stack.remove(wid)

        # Reorder to match output order (generate swaps if needed)
        if stack != output_ids:
            swap_layers = self._reorder_stack_to_output(
                stack, output_ids, wire_types
            )
            layers.extend(swap_layers)

        # Store layers for per-layer definitions
        self.layers = list(layers)

        # Combine all layers with sequential composition
        if not layers:
            return None

        result = layers[0]
        for layer in layers[1:]:
            result_cod = layer.dom if hasattr(layer, "dom") else result.cod
            result = Seq(
                dom=result.dom,
                cod=layer.cod,
                left=result,
                right=layer,
            )

        return result

    def _generate_permutation_layers(
        self,
        stack: list[int],
        read_positions: list[int],
        pass_positions: list[int],
        wire_types: dict[int, str],
    ) -> list[BoxExpr]:
        """Generate swap layers to move read wires to the front of the stack."""
        layers = []
        current = list(range(len(stack)))
        target = read_positions + pass_positions

        # Use selection-sort style: for each target position, find the wire
        # and bubble it into place using adjacent swaps
        working_stack = list(stack)
        for target_pos in range(len(target)):
            # Find where target[target_pos] currently is
            wanted_wire = stack[target[target_pos]]
            current_pos = working_stack.index(wanted_wire)

            # Swap it into position with adjacent swaps
            while current_pos > target_pos:
                # Swap current_pos-1 and current_pos
                ty1 = wire_types[working_stack[current_pos - 1]]
                ty2 = wire_types[working_stack[current_pos]]

                dom = [wire_types[wid] for wid in working_stack]
                cod = list(dom)
                cod[current_pos - 1], cod[current_pos] = (
                    cod[current_pos],
                    cod[current_pos - 1],
                )

                parts = []
                i = 0
                while i < len(working_stack):
                    if i == current_pos - 1:
                        parts.append(
                            Prim(dom=[ty1, ty2], cod=[ty2, ty1], name="swap")
                        )
                        i += 2
                    else:
                        sty = wire_types[working_stack[i]]
                        parts.append(Prim(dom=[sty], cod=[sty], name="id"))
                        i += 1

                swap_par = Par(dom=dom, cod=cod, parts=parts)
                swap_layer = Annotated(dom=dom, cod=cod, expr=swap_par)
                layers.append(swap_layer)

                working_stack[current_pos - 1], working_stack[current_pos] = (
                    working_stack[current_pos],
                    working_stack[current_pos - 1],
                )
                current_pos -= 1

        return layers

    def _reorder_stack_to_output(
        self,
        stack: list[int],
        output_ids: list[int],
        wire_types: dict[int, str],
    ) -> list[BoxExpr]:
        """Generate swap layers to reorder stack to match output order."""
        layers = []
        working = list(stack)

        for target_pos, wanted_id in enumerate(output_ids):
            current_pos = working.index(wanted_id)
            while current_pos > target_pos:
                ty1 = wire_types[working[current_pos - 1]]
                ty2 = wire_types[working[current_pos]]

                dom = [wire_types[wid] for wid in working]
                cod = list(dom)
                cod[current_pos - 1], cod[current_pos] = (
                    cod[current_pos],
                    cod[current_pos - 1],
                )

                parts = []
                i = 0
                while i < len(working):
                    if i == current_pos - 1:
                        parts.append(
                            Prim(dom=[ty1, ty2], cod=[ty2, ty1], name="swap")
                        )
                        i += 2
                    else:
                        sty = wire_types[working[i]]
                        parts.append(Prim(dom=[sty], cod=[sty], name="id"))
                        i += 1

                swap_par = Par(dom=dom, cod=cod, parts=parts)
                swap_layer = Annotated(dom=dom, cod=cod, expr=swap_par)
                layers.append(swap_layer)

                working[current_pos - 1], working[current_pos] = (
                    working[current_pos],
                    working[current_pos - 1],
                )
                current_pos -= 1

        return layers


# ══════════════════════════════════════════════════════════════════════════
# Part F: ModuleToLean4 Class
# ══════════════════════════════════════════════════════════════════════════


class ModuleToLean4:
    """Convert a Python Module into Lean4 Box wiring diagram code."""

    def __init__(self, module: Module):
        self.module = module
        self._const_counter = 0
        self._const_defs: list[str] = []
        self._constants: dict[int, str] = {}  # wire_id -> const name
        self.const_names: list[str] = []  # populated after to_lean()
        self.update_layer_count: int = 0  # populated after to_lean()

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

    def _compile_block(
        self,
        terms,
        block_inputs: list[Wire],
        block_outputs: list[Wire],
    ) -> Optional[str]:
        """Compile a block (init or update) into a Lean Box expression string."""
        term_list = [terms[i] for i in range(len(terms))]
        compiler = BlockCompiler(term_list, block_inputs, block_outputs, self._constants)
        expr = compiler.compile()
        if expr is None:
            return None
        return expr.render()

    def _compile_block_with_layers(
        self,
        terms,
        block_inputs: list[Wire],
        block_outputs: list[Wire],
    ) -> tuple[Optional[str], list[BoxExpr]]:
        """Compile a block and also return individual layers."""
        term_list = [terms[i] for i in range(len(terms))]
        compiler = BlockCompiler(term_list, block_inputs, block_outputs, self._constants)
        expr = compiler.compile()
        if expr is None:
            return None, []
        return expr.render(), compiler.layers

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

        # Determine block inputs/outputs
        # Init block: inputs are extl next-wires, outputs are ctrl next-wires
        # Update block: inputs are ctrl latched-wires + extl next-wires,
        #               outputs are ctrl next-wires (+ obs next-wires if different)

        extl_next = [pair[1] for pair in m.extl]
        ctrl_latched = [pair[0] for pair in m.ctrl]
        ctrl_next = [pair[1] for pair in m.ctrl]

        # Extract constants from both blocks
        init_terms = atom.init
        update_terms = atom.update
        self._extract_constants(init_terms)
        self._extract_constants(update_terms)

        # Compile init block
        # Init reads from extl (external inputs) and writes ctrl next-wires
        init_inputs = list(extl_next)
        init_outputs = list(ctrl_next)
        init_expr = self._compile_block(init_terms, init_inputs, init_outputs)

        # Compile update block (with layers for per-layer definitions)
        update_inputs = ctrl_latched + extl_next
        update_outputs = list(ctrl_next)
        update_expr, update_layers = self._compile_block_with_layers(
            update_terms, update_inputs, update_outputs
        )

        # Store metadata for certificate generation
        self.const_names = list(self._constants.values())
        self.update_layer_count = len(update_layers)

        # Render output
        lines = []
        lines.append("open Box")
        lines.append("")

        # Constants
        if self._const_defs:
            lines.append("/- Concrete constants -/")
            lines.append("")
            for cdef in self._const_defs:
                lines.append(cdef)

        # Init definition
        if init_expr:
            lines.append("/-- Wiring diagram encoding init of module -/")
            init_dom = ", ".join(dtype_to_lean_ty(w) for w in init_inputs)
            init_cod = ", ".join(dtype_to_lean_ty(w) for w in init_outputs)
            if init_inputs:
                lines.append(
                    f"def init : Box [{init_dom}] [{init_cod}] :="
                )
            else:
                lines.append(f"def init : Box [] [{init_cod}] :=")
            lines.append(f"  {init_expr}")
            lines.append("")

        # Update definition — per-layer definitions with accessor lemmas
        if update_expr and update_layers:
            upd_dom = ", ".join(
                dtype_to_lean_ty(w) for w in update_inputs
            )
            upd_cod = ", ".join(
                dtype_to_lean_ty(w) for w in update_outputs
            )

            lines.append("/-- Wiring diagram encoding update of module -/")
            lines.append("")

            # Emit per-layer definitions and accessor lemmas
            for i, layer in enumerate(update_layers):
                layer_dom = ", ".join(layer.dom)
                layer_cod = ", ".join(layer.cod)

                # Unwrap Annotated to avoid redundant type annotation in body
                inner = layer
                if isinstance(inner, Annotated):
                    inner = inner.expr

                lines.append(
                    f"def L{i+1} : Box [{layer_dom}] [{layer_cod}] :="
                )
                lines.append(f"  {inner.render()}")
                lines.append("")

                # Generate accessor lemmas
                out_exprs = _layer_output_exprs(layer)
                if out_exprs is not None:
                    for k, expr in enumerate(out_exprs):
                        accessor = _vt_accessor(k)
                        lines.append(
                            f"theorem L{i+1}_{k+1} : (L{i+1}.fn s){accessor} = {expr} := by rfl"
                        )
                    lines.append("")

            # Emit composed update definition
            layer_names = " ≫ ".join(f"L{i+1}" for i in range(len(update_layers)))
            lines.append(
                f"def update : Box [{upd_dom}] [{upd_cod}] :="
            )
            lines.append(f"  {layer_names}")
            lines.append("")
        elif update_expr:
            # Fallback: monolithic update (no layers available)
            lines.append("/-- Wiring diagram encoding update of module -/")
            upd_dom = ", ".join(
                dtype_to_lean_ty(w) for w in update_inputs
            )
            upd_cod = ", ".join(
                dtype_to_lean_ty(w) for w in update_outputs
            )
            lines.append(
                f"def update : Box [{upd_dom}] [{upd_cod}] :="
            )
            lines.append(f"  {update_expr}")
            lines.append("")

        return "\n".join(lines)
