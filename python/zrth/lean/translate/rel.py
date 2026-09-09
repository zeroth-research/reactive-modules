"""ScalarRel (relational, scalar domain) translation."""

from zrth.lean.native import (
    _product_type,
    _product_type_scalar,
    _translate_terms_scalar,
    _reachable_terms,
)
from zrth.lean.common import LeanContext
from zrth.lean.translate._shared import (
    _scalar_bindings_with_recon,
    _prepend_recon,
    _flat_layout,
    _flat_slice,
    _effect_type,
)


def atom_to_lean_rel(ctx: LeanContext) -> str:
    """Generate the relational encoding inside ``namespace ScalarRel``."""
    noncomp = "noncomputable " if ctx.uses_real else ""

    # One `effect_i`/`R_i` per ctrl *wire*, but one state component per
    # *element*: `Scalar.update`'s codomain is the flattened tuple, so wire
    # `i`'s data sits at `spans[i]` in it, not at position `i`.
    spans, n_slots = _flat_layout(ctx.ctrl_next)
    n_ctrl = len(spans)
    state_ty = _product_type_scalar(ctx.ctrl_next)

    def _ty(wires):
        return _product_type_scalar(wires) if wires else "Unit"

    update_groups = [
        ("ctrl", ctx.ctrl_latched),
        ("extl_l", ctx.extl_latched),
        ("extl_n", ctx.extl_next),
    ]
    update_binders = " ".join(f"({p} : {_ty(w)})" for p, w in update_groups)
    update_args = " ".join(p for p, _ in update_groups)
    update_intro = " ".join(p for p, _ in update_groups)

    extl_groups = [("extl_l", ctx.extl_latched), ("extl_n", ctx.extl_next)]
    extl_binders = " ".join(f"({p} : {_ty(w)})" for p, w in extl_groups)
    extl_args = " ".join(p for p, _ in extl_groups)

    extl_n_binder = f"(extl_n : {_ty(ctx.extl_next)})"

    update_recon, update_bindings, update_flat = _scalar_bindings_with_recon(
        [
            ("ctrl", ctx.ctrl_latched),
            ("extl_l", ctx.extl_latched),
            ("extl_n", ctx.extl_next),
        ]
    )
    init_recon, init_bindings, init_flat = _scalar_bindings_with_recon(
        [("extl_n", ctx.extl_next)]
    )

    def _proj_theorem(
        func_name: str,
        scalar_func: str,
        all_binders: str,
        all_args: str,
        intro_vars: str,
        span: "tuple[int, int]",
    ) -> "list[str]":
        offset, size = span
        scalar_proj = _flat_slice(
            f"({scalar_func} {all_args})", offset, size, n_slots
        )
        return [
            f"theorem {func_name}_eq : ∀ {all_binders},",
            f"    {func_name} {all_args} = {scalar_proj} := by",
            f"  intro {intro_vars}",
            f"  simp only [{scalar_func}, {func_name}, Fin.cons_zero, Fin.cons_succ]",
            f"  try rfl",
            f"  try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])",
            f"  try simp [Fin.fin_one_eq_zero]",
            f"  try omega",
            "",
        ]

    def _pack_roundtrip(var: str) -> "list[str]":
        """Tactics for ``var = Scalar.pack (Scalar.unpack_ctrl var)``.

        The functional and scalar states differ by exactly this round trip.
        Widest tactic first, as in `scalar.py`: a multi-element wire leaves a
        `match i, j with` on a symbolic `i` that only `fin_cases` clears, and
        once `i`/`j` are introduced a later `funext` can no longer fire.
        """
        return [
            f"    have hpack : {var} = Scalar.pack (Scalar.unpack_ctrl {var}) := by",
            "      simp only [Scalar.pack, Scalar.unpack_ctrl, ← Mat_1_1_eq, Prod.eta]",
            "      try rfl",
            "      try (apply Prod.ext <;> funext i j <;> fin_cases i <;> fin_cases j <;> simp)",
            "      try (funext i j <;> fin_cases i <;> fin_cases j <;> simp)",
            "      try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])",
            "      try (funext i j; simp [Fin.fin_one_eq_zero])",
        ]

    lines = ["namespace ScalarRel", ""]

    has_update = bool(list(ctx.atom.update))
    has_init = bool(list(ctx.atom.init))

    if has_update:
        for i, w in enumerate(ctx.ctrl_next):
            ty = _effect_type(w)
            body = _translate_terms_scalar(
                _reachable_terms(ctx.atom.update, [w]),
                update_bindings,
                [w],
                ctx.constants,
                flat_slots=update_flat,
                flatten_outputs=True,
            )
            lines.append(
                f"@[simp] {noncomp}def effect_{i} {update_binders} : {ty} :="
            )
            lines.append(_prepend_recon(update_recon, body))
            lines.append("")

        for i, span in enumerate(spans):
            lines.extend(
                _proj_theorem(
                    f"effect_{i}",
                    "Scalar.update",
                    update_binders,
                    update_args,
                    update_intro,
                    span,
                )
            )

        for i, (offset, size) in enumerate(spans):
            lines.append(
                f"def R_{i} (old new : {state_ty}) {extl_binders} : Prop :="
            )
            new_slice = _flat_slice("new", offset, size, n_slots)
            lines.append(f"  {new_slice} = effect_{i} old {extl_args}")
            lines.append("")

        r_calls = [f"R_{i} old new {extl_args}" for i in range(n_ctrl)]
        lines.append(
            f"def TransRel (old new : {state_ty}) {extl_binders} : Prop :="
        )
        lines.append(f"  " + " ∧\n  ".join(r_calls))
        lines.append("")

    if has_init:
        for i, w in enumerate(ctx.ctrl_next):
            ty = _effect_type(w)
            body = _translate_terms_scalar(
                _reachable_terms(ctx.atom.init, [w]),
                init_bindings,
                [w],
                ctx.constants,
                flat_slots=init_flat,
                flatten_outputs=True,
            )
            lines.append(f"@[simp] {noncomp}def init_{i} {extl_n_binder} : {ty} :=")
            lines.append(_prepend_recon(init_recon, body))
            lines.append("")

        for i, span in enumerate(spans):
            lines.extend(
                _proj_theorem(
                    f"init_{i}",
                    "Scalar.init",
                    extl_n_binder,
                    "extl_n",
                    "extl_n",
                    span,
                )
            )

        for i, (offset, size) in enumerate(spans):
            lines.append(f"def Init_{i} (s : {state_ty}) {extl_n_binder} : Prop :=")
            s_slice = _flat_slice("s", offset, size, n_slots)
            lines.append(f"  {s_slice} = init_{i} extl_n")
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
            f"theorem InitCond_scalar_eq : ∀ (s : {state_ty}) {extl_n_binder},"
        )
        lines.append("    InitCond s extl_n ↔ s = Scalar.init extl_n := by")
        lines.append("  intro s extl_n")
        lines.append(f"  simp only [{', '.join(init_simp)}]")
        lines.append("  try tauto")
        lines.append("")

        mat_ctrl_ty = _product_type(ctx.ctrl_next)

        def _unpack(param, wires):
            return f"(Scalar.unpack_{param} {param})" if wires else param

        new_unpack    = f"(Scalar.unpack_ctrl ctrl')" if ctx.ctrl_next else "ctrl'"
        extl_n_unpack = _unpack("extl_n", ctx.extl_next)
        init_cond_call = f"InitCond {new_unpack} {extl_n_unpack}"
        mat_extl_n_ty = _product_type(ctx.extl_next) if ctx.extl_next else "Unit"
        init_mat_binders = f"(ctrl' : {mat_ctrl_ty}) (extl_n : {mat_extl_n_ty})"

        lines.append(
            f"theorem InitCond_func_eq : ∀ {init_mat_binders},"
        )
        lines.append(f"    {init_cond_call} ↔ ctrl' = init extl_n := by")
        lines.append("  intro ctrl' extl_n")
        lines.append("  rw [InitCond_scalar_eq, init_scalar_eq]")
        lines.append("  constructor")
        lines.append("  · intro h")
        lines.extend(_pack_roundtrip("ctrl'"))
        lines.append("    rw [h] at hpack; exact hpack")
        lines.append("  · intro h")
        lines.append(f"    simp [Scalar.pack, Scalar.unpack_ctrl, h]")
        lines.append("")

    if has_update:
        scalar_update_args = " ".join(
            "old" if p == "ctrl" else p for p, _ in update_groups
        )
        trans_simp = (
            ["TransRel"]
            + [f"R_{i}" for i in range(n_ctrl)]
            + [f"effect_{i}_eq" for i in range(n_ctrl)]
            + ["Prod.ext_iff"]
        )
        trans_intro = f"old new {extl_args}"
        lines.append(
            f"theorem TransRel_scalar_eq : ∀ (old new : {state_ty}) {extl_binders},"
        )
        lines.append(
            f"    TransRel old new {extl_args} ↔ new = Scalar.update {scalar_update_args} := by"
        )
        lines.append(f"  intro {trans_intro}")
        lines.append(f"  simp only [{', '.join(trans_simp)}]")
        lines.append("  try tauto")
        lines.append("")

        mat_ctrl_ty = _product_type(ctx.ctrl_latched)

        def _mat_extl_ty(wires):
            return _product_type(wires) if wires else "Unit"

        def _unpack(param, wires):
            return f"(Scalar.unpack_{param} {param})" if wires else param

        func_extl_binders = " ".join(
            f"({p} : {_mat_extl_ty(w)})" for p, w in extl_groups
        )
        func_extl_args = " ".join(p for p, _ in extl_groups)
        func_extl_intro = " ".join(p for p, _ in extl_groups)

        old_unpack = f"(Scalar.unpack_ctrl ctrl)" if ctx.ctrl_latched else "ctrl"
        new_unpack = f"(Scalar.unpack_ctrl ctrl')" if ctx.ctrl_next else "ctrl'"
        extl_unpacks = " ".join(_unpack(p, w) for p, w in extl_groups)
        trans_func_call = f"TransRel {old_unpack} {new_unpack} {extl_unpacks}"

        lines.append(
            f"theorem TransRel_func_eq : ∀ (ctrl ctrl' : {mat_ctrl_ty}) {func_extl_binders},"
        )
        lines.append(
            f"    {trans_func_call} ↔ ctrl' = update ctrl {func_extl_args} := by"
        )
        lines.append(f"  intro ctrl ctrl' {func_extl_intro}")
        lines.append("  rw [TransRel_scalar_eq, update_scalar_eq]")
        lines.append("  constructor")
        lines.append("  · intro h")
        lines.extend(_pack_roundtrip("ctrl'"))
        lines.append("    rw [h] at hpack; exact hpack")
        lines.append("  · intro h")
        lines.append(f"    simp [Scalar.pack, Scalar.unpack_ctrl, h]")
        lines.append("")

    lines.append("end ScalarRel")
    return "\n".join(lines)
