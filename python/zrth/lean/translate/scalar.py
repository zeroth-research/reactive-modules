"""Scalar encoding translation."""

from zrth.lean.native import (
    _product_type,
    _product_type_scalar,
    _translate_terms_scalar,
    _build_tuple,
    _argmax_scalar_name,
)
from zrth.lean.common import (
    dtype_shape,
    LeanContext,
    _accessor,
    _flat_size,
    _flat_indices,
    _flat_element_type,
    _mat_from_scalars,
    itype_name,
)
from zrth.lean.translate._shared import _scalar_bindings_with_recon, _prepend_recon
from zrth import Wire


def _scalar_dom(param: str, wires: "list[Wire]") -> str:
    ty = _product_type_scalar(wires) if wires else "Unit"
    return f"({param}: {ty})"


def _unpack_body(param: str, wires: "list[Wire]") -> str:
    """Body of ``unpack_<param>``: extract all flat elements from the wire."""
    n = len(wires)
    parts = []
    for i, w in enumerate(wires):
        base = f"{param}{_accessor(i, n)}"
        for row, col in _flat_indices(w):
            parts.append(f"{base} {row} {col}")
    return _build_tuple(parts)


def _pack_body(var: str, wires: "list[Wire]") -> str:
    """Body of ``pack``: reconstruct each wire's Mat from flat scalar slots."""
    total = sum(_flat_size(w) for w in wires)
    flat_accrs = [f"{var}{_accessor(k, total)}" for k in range(total)]
    parts = []
    offset = 0
    for w in wires:
        n = _flat_size(w)
        slots = flat_accrs[offset : offset + n]
        parts.append(_mat_from_scalars(slots, list(dtype_shape(w.dtype)), _flat_element_type(w)))
        offset += n
    return _build_tuple(parts)


def _collect_argmax_variants(terms) -> "list[tuple[str, int]]":
    """Return (elem_ty, n) pairs for each distinct Argmax input shape in terms."""
    seen: set[tuple[str, int]] = set()
    result: list[tuple[str, int]] = []
    for term in terms:
        if itype_name(term.itype) != "Argmax":
            continue
        in_wire = term.read[0]
        n = _flat_size(in_wire)
        ety = _flat_element_type(in_wire)
        key = (ety, n)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _argmax_scalar_def_lines(elem_ty: str, n: int) -> "list[str]":
    """Emit a scalar def and simp theorem for argmax over n elements."""
    name = _argmax_scalar_name(n)
    noncomp = "noncomputable " if elem_ty == "Real" else ""
    params = " ".join(f"(s{i} : {elem_ty})" for i in range(n))
    default_map = {"Real": "(0 : Real)", "Int": "(0 : Int)", "Bool": "false"}
    default = default_map.get(elem_ty, f"(Inhabited.default : {elem_ty})")
    body = [f"  let b0 : Nat × {elem_ty} := (0, {default})"]
    for i in range(n):
        body.append(
            f"  let b{i+1} : Nat × {elem_ty} := if b{i}.2 ≤ s{i} then ({i}, s{i}) else b{i}"
        )
    body.append(f"  (b{n}.1 : Int)")
    eq_params = " ".join(f"(v 0 {i})" for i in range(n))
    return [
        f"{noncomp}def {name} {params} : Int :=",
        *body,
        "",
        f"@[simp] theorem {name}_eq (v : Mat {elem_ty} 1 {n}) :",
        f"    {name} {eq_params} = (↑(argmax_1d v 0 0) : Int) := by",
        f"  simp only [{name}, argmax_1d, List.finRange, List.foldl]",
        f"  rfl",
        "",
    ]


def atom_to_lean_scalar(ctx: LeanContext) -> "tuple[str, list[tuple[str, int]]]":
    """Generate the scalar encoding inside ``namespace Scalar``.

    Returns (source_string, argmax_variants).
    """
    atom = ctx.atom
    noncomp = "noncomputable " if ctx.uses_real else ""
    cod = _product_type_scalar(ctx.ctrl_next)

    all_terms = list(atom.init) + list(atom.update)
    argmax_variants = _collect_argmax_variants(all_terms)

    input_groups = [
        ("ctrl", ctx.ctrl_latched),
        ("extl_l", ctx.extl_latched),
        ("extl_n", ctx.extl_next),
    ]

    lines: list[str] = []
    for ety, n in argmax_variants:
        lines.extend(_argmax_scalar_def_lines(ety, n))

    lines += ["namespace Scalar", ""]

    # 1. Unpack functions
    for param, wires in input_groups:
        if not wires:
            continue
        mat_ty = _product_type(wires)
        scalar_ty = _product_type_scalar(wires)
        body = _unpack_body(param, wires)
        lines.append(
            f"@[simp] {noncomp}def unpack_{param} ({param} : {mat_ty}) : {scalar_ty} :="
        )
        lines.append(f"  {body}")
        lines.append("")

    # 2. Pack function
    pack_body = _pack_body("r", ctx.ctrl_next)
    out_mat_ty = _product_type(ctx.ctrl_next)
    lines.append(f"@[simp] {noncomp}def pack (r : {cod}) : {out_mat_ty} :=")
    lines.append(f"  {pack_body}")
    lines.append("")

    # 3. Scalar init / update
    init_recon, init_bindings, init_flat = _scalar_bindings_with_recon(
        [("extl_n", ctx.extl_next)]
    )
    init_body = _translate_terms_scalar(
        atom.init,
        init_bindings,
        ctx.ctrl_next,
        ctx.constants,
        flat_slots=init_flat,
    )
    if init_body:
        lines.append(
            f"@[simp] {noncomp}def init {_scalar_dom('extl_n', ctx.extl_next)} : {cod} :="
        )
        lines.append(_prepend_recon(init_recon, init_body))
        lines.append("")

    update_recon, update_bindings_scalar, update_flat = _scalar_bindings_with_recon(
        [
            ("ctrl", ctx.ctrl_latched),
            ("extl_l", ctx.extl_latched),
            ("extl_n", ctx.extl_next),
        ]
    )
    update_body = _translate_terms_scalar(
        atom.update,
        update_bindings_scalar,
        ctx.ctrl_next,
        ctx.constants,
        flat_slots=update_flat,
    )
    if update_body:
        upd_dom = " ".join(_scalar_dom(p, w) for p, w in input_groups)
        lines.append(f"@[simp] {noncomp}def update {upd_dom} : {cod} :=")
        lines.append(_prepend_recon(update_recon, update_body))
        lines.append("")

    lines.append("end Scalar")
    return "\n".join(lines), argmax_variants


def to_lean_scalar_equiv(
    ctx: LeanContext,
    argmax_variants: "list[tuple[str, int]]",
) -> str:
    """Generate theorems proving init/update = pack ∘ Scalar.init/update ∘ unpack."""
    lines: list[str] = []

    def _var(binder: str) -> str:
        return binder.split(" : ")[0].strip("( )")

    def _call_arg(param: str, wires) -> str:
        return f"(Scalar.unpack_{param} {param})" if wires else param

    argmax_eq_lemmas = [
        f"{_argmax_scalar_name(n)}_eq"
        for _, n in argmax_variants
    ]

    def _proof(
        binders: "list[str]",
        func_name: str,
        scalar_args: "list[str]",
        simp_extras: "list[str]",
    ) -> "list[str]":
        vars_ = [_var(b) for b in binders]
        scalar_call = f"Scalar.{func_name} {' '.join(scalar_args)}"
        rhs = f"Scalar.pack ({scalar_call})"
        simp_names = (
            ["Scalar.pack", f"Scalar.{func_name}", func_name]
            + simp_extras
            + argmax_eq_lemmas
            + ["Fin.cons_zero", "Fin.cons_succ"]
        )
        return [
            f"theorem {func_name}_scalar_eq : ∀ {' '.join(binders)},",
            f"    {func_name} {' '.join(vars_)} = {rhs} := by",
            f"  intro {' '.join(vars_)}",
            f"  simp only [{', '.join(simp_names)}]",
            f"  try rfl",
            f"  try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])",
            f"  try (funext i j; simp [Fin.fin_one_eq_zero])",
            f"  try simp [Fin.fin_one_eq_zero]",
        ]

    input_groups = [
        ("ctrl", ctx.ctrl_latched),
        ("extl_l", ctx.extl_latched),
        ("extl_n", ctx.extl_next),
    ]

    if list(ctx.atom.init):
        init_binder = f"(extl_n : {_product_type(ctx.extl_next)})"
        extl_n_arg = _call_arg("extl_n", ctx.extl_next)
        extras = [f"Scalar.unpack_extl_n"] if ctx.extl_next else []
        lines.extend(_proof([init_binder], "init", [extl_n_arg], extras))
        lines.append("")

    if list(ctx.atom.update):
        binders = [
            f"({p} : {_product_type(w) if w else 'Unit'})" for p, w in input_groups
        ]
        scalar_args = [_call_arg(p, w) for p, w in input_groups]
        extras = [f"Scalar.unpack_{p}" for p, w in input_groups if w]
        lines.extend(_proof(binders, "update", scalar_args, extras))
        lines.append("")

    return "\n".join(lines)
