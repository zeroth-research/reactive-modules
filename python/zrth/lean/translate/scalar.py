"""Scalar encoding translation."""

from zrth.lean.native import (
    elem_ty_slug,
    product_type,
    product_type_scalar,
    translate_terms_scalar,
    build_tuple,
    argmax_scalar_name,
)
from zrth.lean.sorts import (
    dtype_to_lean_type,
    flat_element_type,
    itype_name,
    wire_shape,
)
from zrth.lean.layout import flat_size, mat_from_scalars, flat_layout
from zrth.lean.context import LeanContext
from zrth.lean.translate._shared import (
    prepend_recon,
    scalar_bindings_with_recon,
    has_block,
    scalar_eq_name,
)
from zrth import Wire


def _scalar_dom(param: str, wires: "list[Wire]") -> str:
    ty = product_type_scalar(wires) if wires else "Unit"
    return f"({param}: {ty})"


def _unpack_body(param: str, wires: "list[Wire]") -> str:
    """Body of ``unpack_<param>``: extract all flat elements from the wire."""
    layout = flat_layout(wires)
    return build_tuple([layout.wire_accessor(param, s) for s in layout.slots])


def _pack_body(var: str, wires: "list[Wire]") -> str:
    """Body of ``pack``: reconstruct each wire's Mat from flat scalar slots.

    The exact inverse of `_unpack_body`, and it has to be: `update_scalar_eq`
    and the `ScalarRel` bridges all rest on `pack (unpack x) = x`. Both walk
    the one layout, so the round trip is by construction rather than by two
    functions agreeing.
    """
    layout = flat_layout(wires)
    flat_accrs = layout.flat_accessors(var)
    parts = []
    for w in wires:
        offset, size = layout.span(w)
        parts.append(
            mat_from_scalars(
                flat_accrs[offset : offset + size],
                wire_shape(w),
                flat_element_type(w),
            )
        )
    return build_tuple(parts)


def _collect_argmax_variants(terms) -> "list[tuple[str, int]]":
    """Return (elem_ty, n) pairs for each distinct Argmax input shape in terms."""
    seen: set[tuple[str, int]] = set()
    result: list[tuple[str, int]] = []
    for term in terms:
        if itype_name(term.itype) != "Argmax":
            continue
        in_wire = term.read[0]
        n = flat_size(in_wire)
        ety = flat_element_type(in_wire)
        key = (ety, n)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _recon_lemma_lines(w) -> "list[str]":
    """Lemma: rebuilding a wire from its own elements gives the wire back.

    `Scalar.update` reconstructs a `Mat` from the unpacked scalars whenever a
    term needs the whole matrix (`matVecAffine`, say). In `update_scalar_eq`
    that reconstruction sits inside the argument, so the two sides differ by
    exactly `pack (unpack ctrl)` vs `ctrl` and `simp` cannot close the gap
    without this. Built by the same `mat_from_scalars` as the reconstruction
    itself, so the shapes match syntactically.
    """
    shape = wire_shape(w)
    elem = flat_element_type(w)
    m, n = shape.m, shape.n
    slots = [f"c {i} {j}" for i in range(m) for j in range(n)]
    lhs = mat_from_scalars(slots, shape, elem)
    name = f"recon_{elem_ty_slug(elem)}_{m}_{n}"
    return [
        "set_option maxHeartbeats 2000000 in",
        "set_option maxRecDepth 100000 in",
        f"@[simp] theorem {name} (c : {dtype_to_lean_type(w)}) :",
        f"    {lhs} = c := by",
        "  funext i j",
        "  fin_cases i <;> fin_cases j <;> simp",
        "",
    ]


def _argmax_scalar_def_lines(elem_ty: str, n: int) -> "list[str]":
    """Emit a scalar def and simp theorem for argmax over n elements.

    Mirrors ``Core.Mat.argmax_1d`` step for step -- seeded with element 0 and
    updating on a strict ``<`` so the first maximal index wins -- because
    ``{name}_eq`` below proves the two equal by unfolding both. A neutral
    seed or a non-strict comparison here would either break that proof or,
    worse, keep it provable against an equally wrong matrix definition.
    """
    name = argmax_scalar_name(elem_ty, n)
    noncomp = "noncomputable " if elem_ty == "Real" else ""
    params = " ".join(f"(s{i} : {elem_ty})" for i in range(n))
    body = [f"  let b0 : Nat × {elem_ty} := (0, s0)"]
    for i in range(1, n):
        body.append(
            f"  let b{i} : Nat × {elem_ty} := if b{i-1}.2 < s{i} then ({i}, s{i}) else b{i-1}"
        )
    body.append(f"  (b{n-1}.1 : Int)")
    eq_params = " ".join(f"(v 0 {i})" for i in range(n))
    return [
        f"{noncomp}def {name} {params} : Int :=",
        *body,
        "",
        f"@[simp] theorem {name}_eq (v : Mat {elem_ty} 1 {n}) :",
        f"    {name} {eq_params} = (↑(argmax_1d v 0 0) : Int) := by",
        f"  simp only [{name}, argmax_1d, List.ofFn_succ, List.ofFn_zero]",
        f"  rfl",
        "",
    ]


def atom_to_lean_scalar(ctx: LeanContext) -> "tuple[str, list[tuple[str, int]]]":
    """Generate the scalar encoding inside ``namespace Scalar``.

    Returns (source_string, argmax_variants).
    """
    atom = ctx.atom
    noncomp = "noncomputable " if ctx.uses_real else ""
    cod = product_type_scalar(ctx.ctrl_next)

    all_terms = list(atom.init) + list(atom.update)
    argmax_variants = _collect_argmax_variants(all_terms)

    input_groups = [
        ("ctrl", ctx.ctrl_latched),
        ("extl_l", ctx.extl_latched),
        ("extl_n", ctx.extl_next),
    ]

    lines: list[str] = []
    seen_recon: set[tuple] = set()
    for _, wires in input_groups:
        for w in wires:
            if flat_size(w) == 1:
                continue
            key = (wire_shape(w), flat_element_type(w))
            if key in seen_recon:
                continue
            seen_recon.add(key)
            lines.extend(_recon_lemma_lines(w))
    for ety, n in argmax_variants:
        lines.extend(_argmax_scalar_def_lines(ety, n))

    lines += ["namespace Scalar", ""]

    # 1. Unpack functions
    for param, wires in input_groups:
        if not wires:
            continue
        mat_ty = product_type(wires)
        scalar_ty = product_type_scalar(wires)
        body = _unpack_body(param, wires)
        lines.append(
            f"@[simp] {noncomp}def unpack_{param} ({param} : {mat_ty}) : {scalar_ty} :="
        )
        lines.append(f"  {body}")
        lines.append("")

    # 2. Pack function
    pack_body = _pack_body("r", ctx.ctrl_next)
    out_mat_ty = product_type(ctx.ctrl_next)
    lines.append(f"@[simp] {noncomp}def pack (r : {cod}) : {out_mat_ty} :=")
    lines.append(f"  {pack_body}")
    lines.append("")

    # 3. Scalar init / update
    init_recon, init_bindings, init_flat = scalar_bindings_with_recon(
        [("extl_n", ctx.extl_next)]
    )
    init_body = translate_terms_scalar(
        atom.init,
        init_bindings,
        ctx.ctrl_next,
        ctx.constants,
        flat_slots=init_flat,
    )
    lines.append(
        f"@[simp] {noncomp}def init {_scalar_dom('extl_n', ctx.extl_next)} : {cod} :="
    )
    lines.append(prepend_recon(init_recon, init_body))
    lines.append("")

    update_recon, update_bindings_scalar, update_flat = scalar_bindings_with_recon(
        [
            ("ctrl", ctx.ctrl_latched),
            ("extl_l", ctx.extl_latched),
            ("extl_n", ctx.extl_next),
        ]
    )
    update_body = translate_terms_scalar(
        atom.update,
        update_bindings_scalar,
        ctx.ctrl_next,
        ctx.constants,
        flat_slots=update_flat,
    )
    upd_dom = " ".join(_scalar_dom(p, w) for p, w in input_groups)
    lines.append(f"@[simp] {noncomp}def update {upd_dom} : {cod} :=")
    lines.append(prepend_recon(update_recon, update_body))
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
        f"{argmax_scalar_name(ety, n)}_eq"
        for ety, n in argmax_variants
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
            # `fin_cases` enumerates every index of every wire, so the cost
            # grows with the state width: a 32-wide module exceeds the default
            # budget. Raised per theorem rather than file-wide.
            f"set_option maxHeartbeats 2000000 in",
            f"set_option maxRecDepth 100000 in",
            f"theorem {scalar_eq_name(func_name)} : ∀ {' '.join(binders)},",
            f"    {func_name} {' '.join(vars_)} = {rhs} := by",
            f"  intro {' '.join(vars_)}",
            f"  simp only [{', '.join(simp_names)}]",
            f"  try rfl",
            # Enumerate the matrix indices first. `Fin.fin_one_eq_zero` alone
            # only collapses a `Fin 1`, so a wire with several elements is
            # left with a `match i, j with ...` on a symbolic `i`; and once an
            # earlier step has introduced `i`/`j`, a later `funext` can no
            # longer fire, so the general form has to come before the narrow
            # ones rather than after.
            f"  try (apply Prod.ext <;> funext i j <;> fin_cases i <;> fin_cases j <;> simp)",
            f"  try (funext i j <;> fin_cases i <;> fin_cases j <;> simp)",
            f"  try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])",
            f"  try (funext i j; simp [Fin.fin_one_eq_zero])",
            f"  try simp [Fin.fin_one_eq_zero]",
            # The scalar side repacks a 1×1 wire as `fun _ _ => w 0 0`, and
            # the matrix side hands `w` itself to an operator that consumes
            # the whole matrix (`matVecAffine ... w`). Neither `fin_cases`
            # nor `Fin.fin_one_eq_zero` touches the argument, so what is left
            # is `op w 0 0 = op (fun _ _ => w 0 0) 0 0` -- true, and closed
            # only by rewriting the repacked wire back into the wire. Over
            # every goal: with two or more wires `Prod.ext` leaves one per
            # component, each with its own copy of the residue.
            f"  all_goals try simp [← Mat_1_1_eq]",
        ]

    input_groups = [
        ("ctrl", ctx.ctrl_latched),
        ("extl_l", ctx.extl_latched),
        ("extl_n", ctx.extl_next),
    ]

    if has_block(ctx, "init"):
        init_binder = f"(extl_n : {product_type(ctx.extl_next)})"
        extl_n_arg = _call_arg("extl_n", ctx.extl_next)
        extras = [f"Scalar.unpack_extl_n"] if ctx.extl_next else []
        lines.extend(_proof([init_binder], "init", [extl_n_arg], extras))
        lines.append("")

    if has_block(ctx, "update"):
        binders = [
            f"({p} : {product_type(w) if w else 'Unit'})" for p, w in input_groups
        ]
        scalar_args = [_call_arg(p, w) for p, w in input_groups]
        extras = [f"Scalar.unpack_{p}" for p, w in input_groups if w]
        lines.extend(_proof(binders, "update", scalar_args, extras))
        lines.append("")

    return "\n".join(lines)
