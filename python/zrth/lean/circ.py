"""
Translate a Python Module (reactive module with init/update blocks)
into Lean4 code.

Python modules store computations as a flat list of Terms (SSA-style dataflow).
We translate each Term into a `let` binding in a Lean function, mapping each
IType operation to its Lean equivalent.
"""

from __future__ import annotations
from zrth.lean.common import _accessor, dtype_to_lean_type, _tensor_to_lean_inline

from dataclasses import dataclass

from zrth import Wire, Term


def _ty_list(wires: list[Wire]) -> str:
    """Build a right-nested product type: [w1, w2] -> 'Bool × Bool', [w1] -> 'Bool'."""
    if not wires:
        return "[]"
    parts = [dtype_to_lean_type(w) for w in wires]
    return "[{}]".format(" , ".join(parts))


@dataclass
class CircLayer:
    """A single layer in a circuit composition."""

    in_tys: list[str]
    out_tys: list[str]
    body: str


def _native_to_vt(param: str, n_wires: int) -> str:
    """Generate native-to-ValTuple conversion expression.

    n=0: "()"
    n=1: "(s, ())"
    n=2: "(s.1, (s.2, ()))"
    n=3: "(s.1, (s.2.1, (s.2.2, ())))"
    """
    if n_wires == 0:
        return "()"
    parts = [f"{param}{_accessor(i, n_wires)}" for i in range(n_wires)]
    result = "()"
    for p in reversed(parts):
        result = f"({p}, {result})"
    return result


def _vt_to_native(param: str, n_wires: int) -> str:
    """Generate ValTuple-to-native conversion expression.

    n=0: "()"
    n=1: "v.1"
    n=2: "(v.1, v.2.1)"
    n=3: "(v.1, v.2.1, v.2.2.1)"
    """
    if n_wires == 0:
        return "()"
    if n_wires == 1:
        return f"{param}.1"
    parts = [f"{param}{'.2' * i}.1" for i in range(n_wires)]
    return "(" + ", ".join(parts) + ")"


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
    "MatMul": "Box.mul",
    "MatAdd": "Box.add",
    "Id": "Box.id",
    "Linear": "Box.nnLinear",
    "ReLU": "Box.relu",
}


def itype_name(itype) -> str:
    """Get the variant name of an IType, e.g. IType.Add() -> 'Add'."""
    name = type(itype).__name__
    # PyO3 exports variants as IType_Add, IType_Tensor, etc.
    if name.startswith("IType_"):
        return name[6:]
    return name


def _constant_expr(
    const_name: str, term: Term, w: Wire, constants: dict[int, str]
) -> str:
    if const_name == "Tensor":
        wire_id = w.id
        if wire_id in constants:
            return constants[wire_id]
        else:
            return _tensor_to_lean_inline(term.itype._0, w)
    elif const_name == "ConstBool":
        val = bool(term.itype._0)
        return "true" if val else "false"
    else:
        return str(int(term.itype._0))

    raise NotImplementedError


def _circ_translate_body(
    block_inputs: list[Wire], block_outputs: list[Wire], wire_to_term: dict[Wire, Term]
) -> list[list[Wire]]:

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
            return layers

    raise RuntimeError("Unreachable")


def _cicr_compute_swapping(
    block_inputs: list[Wire],
    block_outputs: list[Wire],
    wires_ord: dict[Wire, int],
    layer: list[Wire],
) -> tuple[list[CircLayer], list[Wire]]:
    swapping: list[CircLayer] = []
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
                boxes.append(f"@Box.id {dtype_to_lean_type(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                i += 1
            elif wires_ord[layer[i]] <= wires_ord[layer[i + 1]]:
                boxes.append(f"@Box.id {dtype_to_lean_type(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                i += 1
            else:
                changed = True
                boxes.append(
                    f"@Box.swap {dtype_to_lean_type(layer[i])} {dtype_to_lean_type(layer[i + 1])}"
                )
                new_layer.append(layer[i + 1])
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_type(layer[i]))
                in_ty.append(dtype_to_lean_type(layer[i + 1]))
                out_ty.append(dtype_to_lean_type(layer[i + 1]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                # `swap` consumes two wires
                i += 2

        if not changed:
            return swapping, layer
        assert boxes
        swapping.append(
            CircLayer(
                in_tys=list(in_ty),
                out_tys=list(out_ty),
                body=" ⊗ ".join(boxes),
            )
        )
        assert len(new_layer) == len(layer)
        layer = new_layer

    raise RuntimeError("Unreachable")


def _circ_compute_dups(
    block_inputs: list[Wire], wires_ord: dict[Wire, int], layer: list[Wire]
) -> tuple[list[CircLayer], list[Wire]]:
    layers = []
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
                boxes.append(f"@Box.id {dtype_to_lean_type(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                i += 1
            elif wires_ord[layer[i]] != wires_ord[layer[i + 1]]:
                boxes.append(f"@Box.id {dtype_to_lean_type(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                i += 1
            else:
                changed = True
                boxes.append(f"@Box.dup {dtype_to_lean_type(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                # `dup` consumes two wires
                i += 2

        if not changed:
            return layers, layer

        assert boxes

        layers.append(
            CircLayer(
                in_tys=list(in_ty),
                out_tys=list(out_ty),
                body=" ⊗ ".join(boxes),
            )
        )
        assert len(new_layer) < len(layer)
        layer = new_layer

    raise RuntimeError("Unreachable")


def _circ_comput_dels(block_inputs: list[Wire], last_layer: list[Wire]):
    dels = []
    if len(block_inputs) != len(last_layer):
        boxes = []
        in_ty, out_ty = [], []
        for w in block_inputs:
            if w in last_layer:
                boxes.append(f"@Box.id {dtype_to_lean_type(w)}")
                in_ty.append(dtype_to_lean_type(w))
                out_ty.append(dtype_to_lean_type(w))
            else:
                boxes.append(f"@Box.destr {dtype_to_lean_type(w)}")
                in_ty.append(dtype_to_lean_type(w))
        assert boxes
        dels.append(
            CircLayer(
                in_tys=list(in_ty),
                out_tys=list(out_ty),
                body=" ⊗ ".join(boxes),
            )
        )
    return dels


def _translate_terms_circ(
    terms,
    block_inputs: list[Wire],
    block_outputs: list[Wire],
    constants: dict[int, str],
    param_name: str = "s",
) -> list[CircLayer]:
    """Compile a block of terms into a list of CircLayer objects.

    Each CircLayer represents one layer of the circuit composition.
    Returns empty list if there are no terms.
    """
    term_list = [terms[i] for i in range(len(terms))]
    if not term_list:
        return []

    # Map wire_id -> Lean expression (variable name or input accessor)
    # TODO: do this once (we use it also in `_translate_terms`)
    n_inputs = len(block_inputs)
    wire_expr: dict[int, str] = {}
    for i, w in enumerate(block_inputs):
        acc = _accessor(i, n_inputs)
        wire_expr[w.id] = f"{param_name}{acc}"

    # output wires to terms
    wire_to_term: dict[Wire, Term] = {t.write[0]: t for t in term_list}
    # order of the input wires
    wires_ord: dict[Wire, int] = {w: n for n, w in enumerate(block_inputs)}

    layers = _circ_translate_body(block_inputs, block_outputs, wire_to_term)

    # compute swapping wires
    swapping, last_layer = _cicr_compute_swapping(
        block_inputs, block_outputs, wires_ord, layers[-1]
    )

    # duplication of wires
    dups, last_layer = _circ_compute_dups(block_inputs, wires_ord, last_layer)

    # deletion of unused wires
    dels = _circ_comput_dels(block_inputs, last_layer)

    result: list[CircLayer] = list(reversed(swapping + dups + dels))

    for n, layer in enumerate(reversed(layers)):
        if not layer:
            # if there are unused variables, the first layer can be empty
            assert n == 0
            continue
        boxes: list[str] = []
        in_ty, out_ty = [], []
        for w in layer:
            if w in block_inputs:
                boxes.append(f"@Box.id {dtype_to_lean_type(w)}")
                in_ty.append(dtype_to_lean_type(w))
                out_ty.append(dtype_to_lean_type(w))
            else:
                term = wire_to_term[w]
                name = itype_name(term.itype)

                if name in ("Tensor", "ConstBool", "ConstInt"):
                    expr = _constant_expr(name, term, w, constants)

                    boxes.append(f"@Box.const {dtype_to_lean_type(w)} {expr}")
                    out_ty.append(dtype_to_lean_type(w))
                else:
                    boxes.append(_LEAN_OP_BOX[name])
                    in_ty.extend([dtype_to_lean_type(u) for u in term.read])
                    out_ty.extend([dtype_to_lean_type(u) for u in term.write])
        assert boxes
        result.append(
            CircLayer(
                in_tys=list(in_ty),
                out_tys=list(out_ty),
                body=" ⊗ ".join(boxes),
            )
        )

    return result
