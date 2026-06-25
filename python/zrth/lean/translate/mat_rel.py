"""Matrix-domain Rel translation."""

from zrth.lean.native import (
    _product_type,
    _translate_terms,
    _reachable_terms,
)
from zrth.lean.common import (
    LeanContext,
    _accessor,
    _bind_wires,
    dtype_to_lean_type,
)


def atom_to_lean_mat_rel(ctx: LeanContext) -> str:
    """Generate the matrix-domain relational encoding inside ``namespace Rel``."""
    noncomp = "noncomputable " if ctx.uses_real else ""

    n_ctrl = len(ctx.ctrl_next)
    state_ty = _product_type(ctx.ctrl_next)

    def _mat_ty(wires):
        return _product_type(wires) if wires else "Unit"

    update_groups = [
        ("ctrl", ctx.ctrl_latched),
        ("extl_l", ctx.extl_latched),
        ("extl_n", ctx.extl_next),
    ]
    update_binders = " ".join(f"({p} : {_mat_ty(w)})" for p, w in update_groups)
    update_args = " ".join(p for p, _ in update_groups)
    update_intro = " ".join(p for p, _ in update_groups)

    extl_groups = [("extl_l", ctx.extl_latched), ("extl_n", ctx.extl_next)]
    extl_binders = " ".join(f"({p} : {_mat_ty(w)})" for p, w in extl_groups)
    extl_args = " ".join(p for p, _ in extl_groups)

    extl_n_binder = f"(extl_n : {_mat_ty(ctx.extl_next)})"

    update_bindings = _bind_wires(update_groups)
    init_bindings = _bind_wires([("extl_n", ctx.extl_next)])

    def _proj_theorem(
        func_name: str,
        ref_func: str,
        all_binders: str,
        all_args: str,
        intro_vars: str,
        acc: str,
    ) -> "list[str]":
        ref_proj = f"({ref_func} {all_args}){acc}"
        return [
            f"theorem {func_name}_eq : ∀ {all_binders},",
            f"    {func_name} {all_args} = {ref_proj} := by",
            f"  intro {intro_vars}",
            f"  simp only [{ref_func}, {func_name}]",
            f"  try rfl",
            f"  try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])",
            f"  try simp [Fin.fin_one_eq_zero]",
            f"  try omega",
            "",
        ]

    lines = ["namespace Rel", ""]

    has_update = bool(list(ctx.atom.update))
    has_init = bool(list(ctx.atom.init))

    if has_update:
        for i, w in enumerate(ctx.ctrl_next):
            ty = dtype_to_lean_type(w)
            body = _translate_terms(
                _reachable_terms(ctx.atom.update, [w]),
                update_bindings,
                [w],
                ctx.constants,
            )
            lines.append(f"@[simp] {noncomp}def effect_{i} {update_binders} : {ty} :=")
            lines.append(body)
            lines.append("")

        for i in range(n_ctrl):
            acc = _accessor(i, n_ctrl)
            lines.extend(
                _proj_theorem(
                    f"effect_{i}",
                    "update",
                    update_binders,
                    update_args,
                    update_intro,
                    acc,
                )
            )

        for i in range(n_ctrl):
            acc = _accessor(i, n_ctrl)
            lines.append(f"def R_{i} (old new : {state_ty}) {extl_binders} : Prop :=")
            lines.append(f"  new{acc} = effect_{i} old {extl_args}")
            lines.append("")

        r_calls = [f"R_{i} old new {extl_args}" for i in range(n_ctrl)]
        lines.append(f"def TransRel (old new : {state_ty}) {extl_binders} : Prop :=")
        lines.append(f"  " + " ∧\n  ".join(r_calls))
        lines.append("")

        trans_simp = (
            ["TransRel"]
            + [f"R_{i}" for i in range(n_ctrl)]
            + [f"effect_{i}_eq" for i in range(n_ctrl)]
            + ["Prod.ext_iff"]
        )
        trans_intro = f"old new {extl_args}"
        lines.append(
            f"theorem TransRel_func_eq : ∀ (old new : {state_ty}) {extl_binders},"
        )
        lines.append(
            f"    TransRel old new {extl_args} ↔ new = update old {extl_args} := by"
        )
        lines.append(f"  intro {trans_intro}")
        lines.append(f"  simp only [{', '.join(trans_simp)}]")
        lines.append("  try tauto")
        lines.append("")

    if has_init:
        for i, w in enumerate(ctx.ctrl_next):
            ty = dtype_to_lean_type(w)
            body = _translate_terms(
                _reachable_terms(ctx.atom.init, [w]),
                init_bindings,
                [w],
                ctx.constants,
            )
            lines.append(f"@[simp] {noncomp}def init_{i} {extl_n_binder} : {ty} :=")
            lines.append(body)
            lines.append("")

        for i in range(n_ctrl):
            acc = _accessor(i, n_ctrl)
            lines.extend(
                _proj_theorem(
                    f"init_{i}",
                    "init",
                    extl_n_binder,
                    "extl_n",
                    "extl_n",
                    acc,
                )
            )

        for i in range(n_ctrl):
            acc = _accessor(i, n_ctrl)
            lines.append(f"def Init_{i} (s : {state_ty}) {extl_n_binder} : Prop :=")
            lines.append(f"  s{acc} = init_{i} extl_n")
            lines.append("")

        init_calls = [f"Init_{i} s extl_n" for i in range(n_ctrl)]
        lines.append(f"def InitCond (s : {state_ty}) {extl_n_binder} : Prop :=")
        lines.append(f"  " + " ∧\n  ".join(init_calls))
        lines.append("")

        init_simp = (
            ["InitCond"]
            + [f"Init_{i}" for i in range(n_ctrl)]
            + [f"init_{i}_eq" for i in range(n_ctrl)]
            + ["Prod.ext_iff"]
        )
        lines.append(
            f"theorem InitCond_func_eq : ∀ (s : {state_ty}) {extl_n_binder},"
        )
        lines.append("    InitCond s extl_n ↔ s = init extl_n := by")
        lines.append("  intro s extl_n")
        lines.append(f"  simp only [{', '.join(init_simp)}]")
        lines.append("  try tauto")
        lines.append("")

    lines.append("end Rel")
    return "\n".join(lines)
