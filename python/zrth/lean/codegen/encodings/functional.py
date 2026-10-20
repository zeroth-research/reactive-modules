"""Functional (init/update def) translation."""

from zrth.lean.codegen.terms import translate_terms_lean, product_type
from zrth.lean.codegen.context import LeanContext


def constants_block(ctx: LeanContext) -> str:
    """Render the top-level ``@[simp] def c0 : ...`` constant definitions."""
    defs = ctx.constants.defs()
    if not defs:
        return ""
    lines = ["/- Concrete constants -/", ""]
    lines.extend(defs)
    return "\n".join(lines)


def atom_to_lean_functional(ctx: LeanContext) -> str:
    """Generate the functional init/update definitions."""
    atom = ctx.atom

    init_body = translate_terms_lean(
        atom.init,
        ctx.init_wire_names,
        ctx.ctrl_next,
        ctx.constants,
    )
    update_body = translate_terms_lean(
        atom.update,
        ctx.update_wire_names,
        ctx.ctrl_next,
        ctx.constants,
    )

    lines = []
    cod = product_type(ctx.ctrl_next)
    noncomp = "noncomputable " if ctx.uses_real else ""

    init_dom = f"(extl_n: {product_type(ctx.extl_next)})"
    lines.append(f"@[simp] {noncomp}def init {init_dom} : {cod} :=")
    lines.append(init_body)
    lines.append("")

    upd_dom = (
        f"(ctrl: {product_type(ctx.ctrl_latched)}) "
        f"(extl_l: {product_type(ctx.extl_latched)}) "
        f"(extl_n: {product_type(ctx.extl_next)})"
    )
    lines.append(f"@[simp] {noncomp}def update {upd_dom} : {cod} :=")
    lines.append(update_body)
    lines.append("")

    return "\n".join(lines)
