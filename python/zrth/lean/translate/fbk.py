"""FBK (Bool-valued relational) translation."""

from zrth.lean.native import (
    _product_type_scalar,
    _translate_terms_scalar,
    _reachable_terms,
)
from zrth.lean.common import (
    LeanContext,
    dtype_to_lean_type,
    _flat_size,
    _flat_indices,
)
from zrth.lean.translate._shared import _scalar_bindings_with_recon, _prepend_recon


def _build_state_bindings(
    ctrl_wires: list,
) -> "tuple[dict[int, str], dict[int, list[str]]]":
    """Build wire bindings and flat_slots for a StateType state variable.

    Each ctrl wire at position i maps to ``state i`` (function application).
    For scalar wires (flat_size == 1) the binding IS ``state i``.
    For matrix wires the binding is also ``state i`` (the full matrix), and
    flat_slots carries individual element accessors ``state i r c``.
    """
    bindings: dict[int, str] = {}
    flat_slots: dict[int, list[str]] = {}
    for i, w in enumerate(ctrl_wires):
        expr = f"(state {i})"
        bindings[w.id] = expr
        if _flat_size(w) == 1:
            flat_slots[w.id] = [expr]
        else:
            flat_slots[w.id] = [f"(state {i}) {r} {c}" for r, c in _flat_indices(w)]
    return bindings, flat_slots


def _state_tuple(varname: str, n: int) -> str:
    """Build the ScalarRel-compatible tuple from a StateType variable.

    n=1 → ``varname 0``
    n=2 → ``(varname 0, varname 1)``
    n=3 → ``(varname 0, (varname 1, varname 2))``  (right-nested)
    """
    items = [f"({varname} {i})" for i in range(n)]
    if len(items) == 1:
        return items[0]
    result = items[-1]
    for item in reversed(items[:-1]):
        result = f"({item}, {result})"
    return result


def atom_to_lean_bool_rel(ctx: LeanContext) -> str:
    """Generate a Bool-valued relational encoding inside ``namespace FBK``."""
    noncomp = "noncomputable " if ctx.uses_real else ""

    n_ctrl = len(ctx.ctrl_next)

    def _ty(wires):
        return _product_type_scalar(wires) if wires else "Unit"

    # Build state bindings for update (ctrl_latched wires accessed as state i).
    state_bindings, state_flat = _build_state_bindings(ctx.ctrl_latched)

    # Extl bindings via the shared helper (tuple accessors for extl parameters).
    extl_recon, extl_bindings, extl_flat = _scalar_bindings_with_recon(
        [("extl_l", ctx.extl_latched), ("extl_n", ctx.extl_next)]
    )

    update_bindings = {**state_bindings, **extl_bindings}
    update_flat = {**state_flat, **extl_flat}
    update_recon = extl_recon  # state wires need no Mat reconstruction

    init_recon, init_bindings, init_flat = _scalar_bindings_with_recon(
        [("extl_n", ctx.extl_next)]
    )

    lines = ["namespace FBK", ""]

    has_update = bool(list(ctx.atom.update))
    has_init = bool(list(ctx.atom.init))

    # --- TypeMap / StateType ---
    ctrl_types = [dtype_to_lean_type(w, simple_types=True) for w in ctx.ctrl_next]
    all_same = len(set(ctrl_types)) == 1
    lines.append("abbrev TypeMap : Nat → Type")
    if all_same:
        lines.append(f"  | _ => {ctrl_types[0]}")
    else:
        for i, ty in enumerate(ctrl_types):
            lines.append(f"  | {i} => {ty}")
        lines.append(f"  | _ => {ctrl_types[-1]}")
    lines.append("")
    lines.append("abbrev StateType := (n : Nat) → TypeMap n")
    lines.append("")

    # --- variable declaration ---
    var_parts = []
    if has_update or has_init:
        var_parts.append("(state newstate s : StateType)")
    var_parts.append(f"(extl_l : {_ty(ctx.extl_latched)})")
    var_parts.append(f"(extl_n : {_ty(ctx.extl_next)})")
    lines.append(f"variable {' '.join(var_parts)}")
    lines.append("")

    # --- named variable abbrevs ---
    for i in range(n_ctrl):
        lines.append(f"abbrev var_{i} := state {i}")
    lines.append("")

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

        # Emit effect abbrevs.
        for i, ty, full_body, eargs in update_data:
            lines.append(f"abbrev {noncomp}effect_{i} : {ty} :=")
            lines.append(full_body)
            lines.append("")

        # Emit R_i abbrevs — access newstate via function application.
        for i in range(n_ctrl):
            eargs = effect_arg_lists[i]
            eargs_str = (" " + " ".join(eargs)) if eargs else ""
            lines.append(f"abbrev R_{i} : Bool :=")
            lines.append(f"  (newstate {i}) == effect_{i}{eargs_str}")
            lines.append("")

        # Emit TransRel abbrev.
        r_calls = [
            f"R_{i}" + ((" " + " ".join(r_arg_lists[i])) if r_arg_lists[i] else "")
            for i in range(n_ctrl)
        ]
        lines.append("abbrev TransRel : Bool :=")
        lines.append("  " + " &&\n  ".join(r_calls))
        lines.append("")

        # Collect effect_i_eq theorems.
        # ScalarRel.effect_i takes a tuple state; build it from state i.
        st = _state_tuple("state", n_ctrl)
        for i, _ty, _body, eargs in update_data:
            lhs = f"effect_{i}" + ((" " + " ".join(eargs)) if eargs else "")
            thm_lines.append(f"theorem effect_{i}_eq : {lhs} = ScalarRel.effect_{i} {st} extl_l extl_n := by")
            thm_lines.append(f"  simp [effect_{i}, ScalarRel.effect_{i}]")
            thm_lines.append("")

        # Collect R_i_iff theorems.
        st = _state_tuple("state", n_ctrl)
        nst = _state_tuple("newstate", n_ctrl)
        for i in range(n_ctrl):
            r_str = " ".join(r_arg_lists[i])
            thm_lines.append(f"theorem R_{i}_iff : (R_{i} {r_str} = true) ↔ ScalarRel.R_{i} {st} {nst} extl_l extl_n := by")
            thm_lines.append(f"  simp only [R_{i}, ScalarRel.R_{i}, ← effect_{i}_eq, beq_iff_eq]")
            thm_lines.append("")

        # Collect TransRel_iff theorem.
        trans_lhs = "TransRel" + ((" " + " ".join(trans_lhs_args)) if trans_lhs_args else "")
        bool_and = ", Bool.and_eq_true" if n_ctrl > 1 else ""
        r_expand = ", ".join(f"R_{i}, ScalarRel.R_{i}" for i in range(n_ctrl))
        eff_back = ", ".join(f"← effect_{i}_eq" for i in range(n_ctrl))
        thm_lines.append(f"theorem TransRel_iff : ({trans_lhs} = true) ↔ ScalarRel.TransRel {st} {nst} extl_l extl_n := by")
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

        # Emit init abbrevs.
        for i, ty, full_body, iargs in init_data:
            lines.append(f"abbrev {noncomp}init_{i} : {ty} :=")
            lines.append(full_body)
            lines.append("")

        # Emit Init_i abbrevs — access s via function application.
        for i in range(n_ctrl):
            iargs = init_arg_lists[i]
            iargs_str = (" " + " ".join(iargs)) if iargs else ""
            lines.append(f"abbrev Init_{i} : Bool :=")
            lines.append(f"  (s {i}) == init_{i}{iargs_str}")
            lines.append("")

        # Emit InitCond abbrev.
        init_calls = [
            f"Init_{i}" + ((" " + " ".join(init_cond_arg_lists[i])) if init_cond_arg_lists[i] else "")
            for i in range(n_ctrl)
        ]
        lines.append("abbrev InitCond : Bool :=")
        lines.append("  " + " &&\n  ".join(init_calls))
        lines.append("")

        # Collect init_i_eq theorems (no state involved).
        for i, _ty, _body, iargs in init_data:
            lhs = f"init_{i}" + ((" " + " ".join(iargs)) if iargs else "")
            thm_lines.append(f"theorem init_{i}_eq : {lhs} = ScalarRel.init_{i} extl_n := by")
            thm_lines.append(f"  simp [init_{i}, ScalarRel.init_{i}]")
            thm_lines.append("")

        # Collect Init_i_iff theorems.
        ss = _state_tuple("s", n_ctrl)
        for i in range(n_ctrl):
            ic_str = " ".join(init_cond_arg_lists[i])
            thm_lines.append(f"theorem Init_{i}_iff : (Init_{i} {ic_str} = true) ↔ ScalarRel.Init_{i} {ss} extl_n := by")
            thm_lines.append(f"  simp only [Init_{i}, ScalarRel.Init_{i}, ← init_{i}_eq, beq_iff_eq]")
            thm_lines.append("")

        # Collect InitCond_iff theorem.
        initcond_lhs = "InitCond" + ((" " + " ".join(initcond_lhs_args)) if initcond_lhs_args else "")
        bool_and_init = ", Bool.and_eq_true" if n_ctrl > 1 else ""
        ini_expand = ", ".join(f"Init_{i}, ScalarRel.Init_{i}" for i in range(n_ctrl))
        init_back = ", ".join(f"← init_{i}_eq" for i in range(n_ctrl))
        thm_lines.append(f"theorem InitCond_iff : ({initcond_lhs} = true) ↔ ScalarRel.InitCond {ss} extl_n := by")
        thm_lines.append(f"  simp only [InitCond, ScalarRel.InitCond{bool_and_init}, {ini_expand}, {init_back}, beq_iff_eq]")
        thm_lines.append("")

    lines.extend(thm_lines)

    lines.append("end FBK")
    return "\n".join(lines)
