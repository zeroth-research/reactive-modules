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
    _is_scalar_shape,
    ConstantRegistry,
    _constant_expr,
    itype_name,
    is_constant_name,
    dtype_to_lean_type,
    linear_list_literals,
    _is_scalar_wire,
    _tensor_to_lean_scalar,
    _bind_wires_scalar,
    _tensor_to_lean_inline,
    _flat_element_type,
    _flat_indices,
    flat_layout,
)
from zrth.lean.ops import lean_gap, mat_emitter, scalar_emitter

from zrth import Wire


def lean_gaps(ctx) -> list[str]:
    """Every operator in this module that no Lean column can write.

    One sentence per operator, in the order the module first reaches for it:
    which block, which wire, and the reason the op table records.

    Asked before anything is generated, which is the whole point. The same
    module met during codegen raises about a missing table cell, half way
    through a file, for a route that may never have wanted the functional
    encoding -- and that is what the limit matrix's only `GEN-FAIL` was. A
    module that generates has nothing to report here, because an operator
    with no matrix form has no emitter in the other two Lean columns either
    (:func:`ops.lean_gap`), so this cannot refuse anything that works.
    """
    out, seen = [], set()
    for block in ("init", "update"):
        # Only what the next state is built from, which is what the encoders
        # walk: a term nothing reads is pruned before it is ever emitted, and
        # refusing a module over one would refuse a module that generates.
        terms = _reachable_terms(getattr(ctx.atom, block), ctx.ctrl_next)
        for term in terms:
            name = itype_name(term.itype)
            why = None if name in seen else lean_gap(term.itype)
            if why is None:
                continue
            seen.add(name)
            out.append(
                f"the `{block}` block writes wire #{term.write[0].id} with "
                f"`{name}`, and {why}"
            )
    return out


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


def _argmax_expr(
    arg_expr: str, input_shape: list[int], output_shape: "list[int] | None" = None
) -> str:
    """Emit `argmax_1d` for 1-d input and `argmax` for generic 2-d input.

    Both yield a single index in a `Mat _ 1 1` -- the column index for one
    row, the row-major flat index otherwise -- so the two agree wherever
    they overlap. Wrap output as `Mat Int 1 _` to match the Python IR's Int
    output type."""
    _check_argmax_output(output_shape)
    if len(input_shape) == 1 or (len(input_shape) == 2 and input_shape[0] == 1):
        return f"(fun i j => ((argmax_1d {arg_expr}) i j : Int))"
    if len(input_shape) == 2:
        return f"(fun i j => ((argmax {arg_expr}) i j : Int))"
    raise ValueError(
        f"argmax: unsupported input shape {input_shape}; expected 1-d or 2-d"
    )


def _check_argmax_output(output_shape: "list[int] | None") -> None:
    """Reject an Argmax output wider than the single index we can emit.

    Argmax follows `torch.argmax`, which returns one row-major flat index,
    so both Lean variants produce a `Mat _ 1 1`. The theory is looser -- it
    accepts any vector output (see the FIXME on `LIA::Argmax` in
    theory/src/lia.rs) -- and a wider wire would previously get that 1x1
    body ascribed to it, which does not elaborate. Fail here with the reason
    instead of emitting Lean that cannot compile.
    """
    if output_shape is None:
        return
    if not _is_scalar_shape(output_shape):
        raise ValueError(
            f"argmax: output shape {output_shape} holds more than one element, "
            "but argmax yields a single flat index (torch.argmax semantics); "
            "declare a [1, 1] output wire"
        )


def _lift_scalar_reads(term, wire_expr: dict[int, str]) -> dict[int, str]:
    """`wire_expr` with this term's scalar-bound reads lifted to 1x1 matrices.

    In the scalar encoding a 1x1 wire is bound to a bare `Bool`/`Int`, but
    the matrix-form emitters apply their operands to `0 0`. Wrapping as
    `fun _ _ => x` makes that reduce back to `x` and typecheck.
    """
    lifted = dict(wire_expr)
    for w in term.read:
        if _is_scalar_wire(w):
            # The ascription is load-bearing: `matVecAffine`'s batch
            # dimension is not determined by `fun _ _ => x`, so an
            # un-ascribed lift leaves `n` unsolved ("don't know how to
            # synthesize implicit argument `n`").
            ty = f"Mat {_flat_element_type(w)} 1 1"
            lifted[w.id] = f"((fun _ _ => {wire_expr[w.id]} : {ty}))"
    return lifted


def _linear_expr(term, wire_expr: dict[int, str]) -> str:
    """Emit a baked-constant LIA/LRA `Linear` op, `Y = A·X + B`, in the reflected
    form `matVecAffine m A b X`.

    `A` (shape `[out, in]`) and `B` (`[out, 1]`, or empty) are constants baked
    into the op; the read wire supplies `X` (`[in, batch]`). They are emitted as
    plain `List (List t)` / `List t` literals: cheap to elaborate (no dense
    `match`), and `matVecAffine` reduces to a linear expression under `simp`
    (no symbolic `∑`). `Core.Mat.matVecAffine_eq` proves this equals
    `affineLinear (matrixOf A) X (colOf b)` for any batch width, so the
    contraction is machine-checked rather than trusted.
    """
    x_expr = wire_expr[term.read[0].id]
    out_m, a_lit, b_lit, _ = linear_list_literals(term)
    return f"(matVecAffine {out_m} {a_lit} {b_lit} {x_expr})"


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
            expr = _constant_expr(term, term.write[0], constants)
        elif name == "Argmax":
            arg_expr = wire_expr[term.read[0].id]
            expr = _argmax_expr(
                arg_expr,
                dtype_shape(term.read[0].dtype),
                dtype_shape(term.write[0].dtype),
            )
        elif name == "Linear":
            expr = _linear_expr(term, wire_expr)
        else:
            input_exprs = [wire_expr[w.id] for w in term.read]
            expr = mat_emitter(term.itype)(input_exprs)

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

    One component per slot, in slot order: the type view of the same
    flattening `flat_layout` owns, so the tuple's shape and the slot
    numbering cannot disagree.
    E.g. [Int(1), Float(2)] → ``"Int × Real × Real"``.
    """
    if not wires:
        return "Unit"
    parts = flat_layout(wires).element_types()
    if len(parts) == 1:
        return parts[0]
    return " × ".join(parts)


def _constant_expr_scalar(term, w: "Wire", constants: ConstantRegistry) -> str:
    """Like _constant_expr but returns bare scalar values (no Mat wrapper)."""
    if _is_scalar_wire(w):
        # Element type (Bool/Int/Real) is taken from the wire's dtype.
        return _tensor_to_lean_scalar(term.itype._0, w)
    # Non-scalar tensor: fall back to matrix constant from registry
    name = constants.lookup(w.id)
    if name is not None:
        return name
    return _tensor_to_lean_inline(term.itype._0, w)


def _argmax_scalar_name(elem_ty: str, n: int) -> str:
    """Name of the scalar axiom for 1-d argmax over n elements of `elem_ty`.

    The element type belongs in the name: variants are collected per
    `(elem_ty, n)`, so naming them by `n` alone made an Argmax over
    `Mat Int 1 4` and one over `Mat Real 1 4` emit two definitions and two
    `_eq` theorems under one name.
    """
    return f"argmax1d_scalar_{_elem_ty_slug(elem_ty)}_{n}"


def _elem_ty_slug(elem_ty: str) -> str:
    """Lean-identifier-safe form of an element type, e.g. `(BitVec 8)` -> `bv8`."""
    if elem_ty.startswith("(BitVec "):
        return f"bv{elem_ty[len('(BitVec '):-1].strip()}"
    return elem_ty.lower()


def _translate_terms_scalar(
    terms,
    input_bindings: dict[int, str],
    block_outputs: list[Wire],
    constants: ConstantRegistry,
    flat_slots: "dict[int, list[str]] | None" = None,
    flatten_outputs: bool = False,
) -> str:
    """Compile terms into a Lean body using scalar types for 1×1 wires.

    ``flat_slots`` maps wire IDs to their flat scalar accessor strings.  When
    provided, ``Argmax`` on a wire whose slots are known emits a call to the
    scalar axiom ``argmax1d_scalar_n`` instead of reconstructing a matrix.

    Falls back to the op table's matrix column for non-scalar output wires
    and for the ops whose scalar cell is `ViaMat`.
    """
    term_list = list(terms)
    if not term_list:
        slots = dict(flat_slots or {})
        passthrough: list[str] = []
        for w in block_outputs:
            if flatten_outputs and w.id in slots:
                passthrough.extend(slots[w.id])
            elif w.id in input_bindings:
                passthrough.append(input_bindings[w.id])
            else:
                return "sorry /- no terms -/"
        return f"  {_build_tuple(passthrough)}"

    wire_expr: dict[int, str] = dict(input_bindings)
    flat_slots = dict(flat_slots or {})
    let_lines: list[str] = []

    for var_counter, term in enumerate(term_list):
        name = itype_name(term.itype)
        write_wire = term.write[0]
        var = f"x{var_counter}"

        if is_constant_name(name):
            expr = _constant_expr_scalar(term, write_wire, constants)
        elif name == "Argmax":
            in_wire = term.read[0]
            _check_argmax_output(dtype_shape(write_wire.dtype))
            slots = flat_slots.get(in_wire.id)
            if slots is not None:
                axiom_name = _argmax_scalar_name(
                    _flat_element_type(in_wire), len(slots)
                )
                expr = f"({axiom_name} {' '.join(slots)})"
            else:
                mat_expr = _argmax_expr(
                    wire_expr[in_wire.id],
                    dtype_shape(in_wire.dtype),
                    dtype_shape(write_wire.dtype),
                )
                expr = f"({mat_expr} 0 0)" if _is_scalar_wire(write_wire) else mat_expr
        elif name == "Linear":
            # `matVecAffine` consumes a `Mat t n batch` and yields a
            # `Mat t out batch`, but a 1x1 wire is bound to a bare scalar
            # here and ascribed one. Lift the operand and take `0 0` off
            # the result, as the Argmax branch above does.
            expr = _linear_expr(term, _lift_scalar_reads(term, wire_expr))
            if _is_scalar_wire(write_wire):
                expr = f"({expr} 0 0)"
        elif _is_scalar_wire(write_wire) and scalar_emitter(term.itype) is not None:
            input_exprs = [wire_expr[w.id] for w in term.read]
            expr = scalar_emitter(term.itype)(input_exprs)
        else:
            # The `mat` column holds the matrix forms, which apply operands
            # to `0 0`. Scalar-bound reads must be lifted first: a matrix
            # `Ite` with a scalar Bool condition emitted
            # `if <Bool> 0 0 then ...`, which does not elaborate.
            lifted = _lift_scalar_reads(term, wire_expr)
            input_exprs = [lifted[w.id] for w in term.read]
            expr = mat_emitter(term.itype)(input_exprs)
            # A matrix form yields a `Mat`; a scalar write wire is ascribed
            # the bare element type, so project it — the same step the
            # `Argmax` and `Linear` branches above already take. Without it
            # a 1x1 `Min`/`Max` emitted `let x : Int := matMax …`.
            if _is_scalar_wire(write_wire):
                expr = f"({expr} 0 0)"

        wire_expr[write_wire.id] = var
        # Track flat scalar slots for the written wire so downstream Argmax
        # terms can reference individual elements without matrix reconstruction.
        if _is_scalar_wire(write_wire):
            flat_slots[write_wire.id] = [var]
        else:
            # Parenthesised: these are spliced into an application
            # (`argmax1d_scalar_int_3 <slots>`), where a bare `x0 0 0`
            # would be read as three separate arguments.
            flat_slots[write_wire.id] = [
                f"({var} {r} {c})" for r, c in _flat_indices(write_wire)
            ]
        ty = dtype_to_lean_type(write_wire, simple_types=True)
        let_lines.append(f"  let {var} : {ty} := {expr}")

    # `flatten_outputs` says which codomain the caller declared. The scalar
    # encoding uses `_product_type_scalar` -- one component per *element* --
    # so it wants the flat slots; `rel.py` and `fbk.py` type their `effect_i`
    # as the wire's own `Mat`, so they want the matrix the terms built.
    out_exprs: list[str] = []
    for w in block_outputs:
        if flatten_outputs:
            out_exprs.extend(flat_slots.get(w.id, [wire_expr[w.id]]))
        else:
            out_exprs.append(wire_expr[w.id])
    result_line = f"  {_build_tuple(out_exprs)}"

    return "\n".join(let_lines + [result_line])
