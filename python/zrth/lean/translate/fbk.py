"""FBK (Bool-valued relational) translation."""

from zrth.lean.native import (
    _product_type_scalar,
    _translate_terms_scalar,
    _reachable_terms,
)
from zrth.lean.common import (
    LeanContext,
    _accessor,
    dtype_to_lean_type,
)
from zrth.lean.translate._shared import _scalar_bindings_with_recon, _prepend_recon


def atom_to_lean_bool_rel(ctx: LeanContext) -> str:
    """Generate a Bool-valued relational encoding inside ``namespace FBK``."""
    noncomp = "noncomputable " if ctx.uses_real else ""

    n_ctrl = len(ctx.ctrl_next)
    state_ty = _product_type_scalar(ctx.ctrl_next)

    def _ty(wires):
        return _product_type_scalar(wires) if wires else "Unit"

    extl_groups = [("extl_l", ctx.extl_latched), ("extl_n", ctx.extl_next)]
    extl_args = " ".join(p for p, _ in extl_groups)

    update_recon, update_bindings, update_flat = _scalar_bindings_with_recon(
        [
            ("state", ctx.ctrl_latched),
            ("extl_l", ctx.extl_latched),
            ("extl_n", ctx.extl_next),
        ]
    )
    init_recon, init_bindings, init_flat = _scalar_bindings_with_recon(
        [("extl_n", ctx.extl_next)]
    )

    lines = ["namespace FBK", ""]

    has_update = bool(list(ctx.atom.update))
    has_init = bool(list(ctx.atom.init))

    def _consumed(atom_terms, target_wire):
        reach = _reachable_terms(atom_terms, [target_wire])
        return {r.id for t in reach for r in t.read}

    def _effect_args(consumed_ids):
        args = []
        if any(w.id in consumed_ids for w in ctx.ctrl_latched):
            args.append("state")
        if any(w.id in consumed_ids for w in ctx.extl_latched):
            args.append("extl_l")
        if any(w.id in consumed_ids for w in ctx.extl_next):
            args.append("extl_n")
        return args

    def _init_args(consumed_ids):
        return ["extl_n"] if any(w.id in consumed_ids for w in ctx.extl_next) else []

    var_parts = []
    if has_update or has_init:
        var_parts.append(f"(state newstate s : {state_ty})")
    var_parts.append(f"(extl_l : {_ty(ctx.extl_latched)})")
    var_parts.append(f"(extl_n : {_ty(ctx.extl_next)})")
    lines.append(f"variable {' '.join(var_parts)}")
    lines.append("")

    thm_lines: list[str] = []

    if has_update:
        update_data: list[tuple[int, str, str, list[str]]] = []
        for i, w in enumerate(ctx.ctrl_next):
            ty = dtype_to_lean_type(w, simple_types=True)
            body = _translate_terms_scalar(
                _reachable_terms(ctx.atom.update, [w]),
                update_bindings,
                [w],
                ctx.constants,
                flat_slots=update_flat,
            )
            eargs = _effect_args(_consumed(ctx.atom.update, w))
            update_data.append((i, ty, _prepend_recon(update_recon, body), eargs))

        effect_arg_lists = [d[3] for d in update_data]
        r_arg_lists = [
            [v for v in ["state", "newstate", "extl_l", "extl_n"]
             if v == "newstate" or v in effect_arg_lists[i]]
            for i in range(n_ctrl)
        ]
        all_trans_vars = {v for ra in r_arg_lists for v in ra}
        trans_lhs_args = [v for v in ["state", "newstate", "extl_l", "extl_n"] if v in all_trans_vars]

        for i, ty, full_body, eargs in update_data:
            lines.append(f"abbrev {noncomp}effect_{i} : {ty} :=")
            lines.append(full_body)
            lines.append("")

        for i in range(n_ctrl):
            acc = _accessor(i, n_ctrl)
            eargs = effect_arg_lists[i]
            eargs_str = (" " + " ".join(eargs)) if eargs else ""
            lines.append(f"abbrev R_{i} : Bool :=")
            lines.append(f"  newstate{acc} == effect_{i}{eargs_str}")
            lines.append("")

        r_calls = [
            f"R_{i}" + ((" " + " ".join(r_arg_lists[i])) if r_arg_lists[i] else "")
            for i in range(n_ctrl)
        ]
        lines.append(f"abbrev TransRel : Bool :=")
        lines.append("  " + " &&\n  ".join(r_calls))
        lines.append("")

        for i, _ty, _body, eargs in update_data:
            lhs = f"effect_{i}" + ((" " + " ".join(eargs)) if eargs else "")
            thm_lines.append(f"theorem effect_{i}_eq : {lhs} = ScalarRel.effect_{i} state extl_l extl_n := by")
            thm_lines.append(f"  simp [effect_{i}, ScalarRel.effect_{i}]")
            thm_lines.append("")

        for i in range(n_ctrl):
            r_str = " ".join(r_arg_lists[i])
            thm_lines.append(f"theorem R_{i}_iff : (R_{i} {r_str} = true) ↔ ScalarRel.R_{i} state newstate extl_l extl_n := by")
            thm_lines.append(f"  simp only [R_{i}, ScalarRel.R_{i}, ← effect_{i}_eq, beq_iff_eq]")
            thm_lines.append("")

        trans_lhs = "TransRel" + ((" " + " ".join(trans_lhs_args)) if trans_lhs_args else "")
        bool_and = ", Bool.and_eq_true" if n_ctrl > 1 else ""
        r_expand = ", ".join(f"R_{i}, ScalarRel.R_{i}" for i in range(n_ctrl))
        eff_back = ", ".join(f"← effect_{i}_eq" for i in range(n_ctrl))
        thm_lines.append(f"theorem TransRel_iff : ({trans_lhs} = true) ↔ ScalarRel.TransRel state newstate extl_l extl_n := by")
        thm_lines.append(f"  simp only [TransRel, ScalarRel.TransRel{bool_and}, {r_expand}, {eff_back}, beq_iff_eq]")
        thm_lines.append("")

    if has_init:
        init_data: list[tuple[int, str, str, list[str]]] = []
        for i, w in enumerate(ctx.ctrl_next):
            ty = dtype_to_lean_type(w, simple_types=True)
            body = _translate_terms_scalar(
                _reachable_terms(ctx.atom.init, [w]),
                init_bindings,
                [w],
                ctx.constants,
                flat_slots=init_flat,
            )
            iargs = _init_args(_consumed(ctx.atom.init, w))
            init_data.append((i, ty, _prepend_recon(init_recon, body), iargs))

        init_arg_lists = [d[3] for d in init_data]
        init_cond_arg_lists = [
            [v for v in ["s", "extl_n"] if v == "s" or v in init_arg_lists[i]]
            for i in range(n_ctrl)
        ]
        all_initcond_vars = {v for ia in init_cond_arg_lists for v in ia}
        initcond_lhs_args = [v for v in ["s", "extl_n"] if v in all_initcond_vars]

        for i, ty, full_body, iargs in init_data:
            lines.append(f"abbrev {noncomp}init_{i} : {ty} :=")
            lines.append(full_body)
            lines.append("")

        for i in range(n_ctrl):
            acc = _accessor(i, n_ctrl)
            iargs = init_arg_lists[i]
            iargs_str = (" " + " ".join(iargs)) if iargs else ""
            lines.append(f"abbrev Init_{i} : Bool :=")
            lines.append(f"  s{acc} == init_{i}{iargs_str}")
            lines.append("")

        init_calls = [
            f"Init_{i}" + ((" " + " ".join(init_cond_arg_lists[i])) if init_cond_arg_lists[i] else "")
            for i in range(n_ctrl)
        ]
        lines.append(f"abbrev InitCond : Bool :=")
        lines.append("  " + " &&\n  ".join(init_calls))
        lines.append("")

        for i, _ty, _body, iargs in init_data:
            lhs = f"init_{i}" + ((" " + " ".join(iargs)) if iargs else "")
            thm_lines.append(f"theorem init_{i}_eq : {lhs} = ScalarRel.init_{i} extl_n := by")
            thm_lines.append(f"  simp [init_{i}, ScalarRel.init_{i}]")
            thm_lines.append("")

        for i in range(n_ctrl):
            ic_str = " ".join(init_cond_arg_lists[i])
            thm_lines.append(f"theorem Init_{i}_iff : (Init_{i} {ic_str} = true) ↔ ScalarRel.Init_{i} s extl_n := by")
            thm_lines.append(f"  simp only [Init_{i}, ScalarRel.Init_{i}, ← init_{i}_eq, beq_iff_eq]")
            thm_lines.append("")

        initcond_lhs = "InitCond" + ((" " + " ".join(initcond_lhs_args)) if initcond_lhs_args else "")
        bool_and_init = ", Bool.and_eq_true" if n_ctrl > 1 else ""
        ini_expand = ", ".join(f"Init_{i}, ScalarRel.Init_{i}" for i in range(n_ctrl))
        init_back = ", ".join(f"← init_{i}_eq" for i in range(n_ctrl))
        thm_lines.append(f"theorem InitCond_iff : ({initcond_lhs} = true) ↔ ScalarRel.InitCond s extl_n := by")
        thm_lines.append(f"  simp only [InitCond, ScalarRel.InitCond{bool_and_init}, {ini_expand}, {init_back}, beq_iff_eq]")
        thm_lines.append("")

    lines.extend(thm_lines)

    lines.append("end FBK")
    return "\n".join(lines)
