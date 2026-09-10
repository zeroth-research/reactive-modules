"""FBK (Bool-valued relational) translation."""

from zrth.lean.native import (
    _product_type_scalar,
    _translate_terms_scalar,
    _reachable_terms,
)
from zrth.lean.common import (
    LeanContext,
    dtype_shape,
    dtype_to_lean_type,
    _flat_element_type,
    _flat_size,
    _mat_from_scalars,
)
from zrth.lean.translate._shared import (
    _scalar_bindings_with_recon,
    _prepend_recon,
    _flat_layout,
    _effect_type,
)


def _build_state_bindings(
    ctrl_wires: list,
) -> "tuple[list[str], dict[int, str], dict[int, list[str]]]":
    """Build recon lets, wire bindings and flat_slots for a StateType variable.

    `StateType` carries one slot per *element*, matching the flattened state
    `ScalarRel` uses -- `ScalarRel.effect_i` takes an `Int × ... × Int`, so a
    slot per wire holding a whole `Mat` could not be passed to it. A wire of
    several elements therefore has no single slot and is rebuilt from its
    own, the same way `_scalar_bindings_with_recon` does for the parameter
    groups.
    """
    recon: list[str] = []
    bindings: dict[int, str] = {}
    flat_slots: dict[int, list[str]] = {}
    slot = 0
    for w in ctrl_wires:
        size = _flat_size(w)
        slots = [f"(state {slot + k})" for k in range(size)]
        flat_slots[w.id] = slots
        if size == 1:
            bindings[w.id] = slots[0]
        else:
            var = f"_s{len(recon)}"
            mat = _mat_from_scalars(
                slots, list(dtype_shape(w.dtype)), _flat_element_type(w)
            )
            recon.append(f"  let {var} : {dtype_to_lean_type(w)} := {mat}")
            bindings[w.id] = var
        slot += size
    return recon, bindings, flat_slots


def _state_slot_types(wires: list) -> "list[str]":
    """One `TypeMap` entry per element of each wire."""
    return [_flat_element_type(w) for w in wires for _ in range(_flat_size(w))]


def _state_tuple(varname: str, n: int, offset: int = 0) -> str:
    """Build the ScalarRel-compatible tuple from a StateType variable.

    n=1 → ``varname 0``
    n=2 → ``(varname 0, varname 1)``
    n=3 → ``(varname 0, (varname 1, varname 2))``  (right-nested)

    ``offset`` starts the run later, which is what one wire's slice of the
    state needs: `ScalarRel` groups the state per element, so wire `i`'s
    components begin after every element before it.
    """
    items = [f"({varname} {offset + i})" for i in range(n)]
    if len(items) == 1:
        return items[0]
    result = items[-1]
    for item in reversed(items[:-1]):
        result = f"({item}, {result})"
    return result


def atom_to_lean_bool_rel(ctx: LeanContext) -> str:
    """Generate a Bool-valued relational encoding inside ``namespace FBK``."""
    noncomp = "noncomputable " if ctx.uses_real else ""

    # One `effect_i`/`R_i` per ctrl *wire*, but one state slot per *element*:
    # `ScalarRel` draws the same distinction and the two counts differ for a
    # multi-element wire. `spans[i]` is where wire `i`'s elements sit in the
    # state, so `R_i` compares that run of slots -- not slot `i` -- against
    # `effect_i`.
    n_ctrl = len(ctx.ctrl_next)
    spans, n_slots = _flat_layout(ctx.ctrl_next)

    def _ty(wires):
        return _product_type_scalar(wires) if wires else "Unit"

    # Build state bindings for update (ctrl_latched elements as state k).
    state_recon, state_bindings, state_flat = _build_state_bindings(ctx.ctrl_latched)

    # Extl bindings via the shared helper (tuple accessors for extl parameters).
    extl_recon, extl_bindings, extl_flat = _scalar_bindings_with_recon(
        [("extl_l", ctx.extl_latched), ("extl_n", ctx.extl_next)]
    )

    update_bindings = {**state_bindings, **extl_bindings}
    update_flat = {**state_flat, **extl_flat}
    # A multi-element state wire is rebuilt from its slots, like the extl ones.
    update_recon = state_recon + extl_recon

    init_recon, init_bindings, init_flat = _scalar_bindings_with_recon(
        [("extl_n", ctx.extl_next)]
    )

    lines = ["namespace FBK", ""]

    has_update = bool(list(ctx.atom.update))
    has_init = bool(list(ctx.atom.init))

    # --- TypeMap / StateType ---
    ctrl_types = _state_slot_types(ctx.ctrl_next)
    if not ctrl_types:
        # `TypeMap` is a total function `Nat -> Type`, so it needs at least
        # one case to fall back on. A module with no controlled state has no
        # state relation to encode; say so rather than indexing an empty list.
        return "-- FBK encoding not available: module has no ctrl wires"
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
    for i in range(n_slots):
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
            ty = _effect_type(w)
            body = _translate_terms_scalar(
                _reachable_terms(ctx.atom.update, [w]),
                update_bindings,
                [w],
                ctx.constants,
                flat_slots=update_flat,
                flatten_outputs=True,
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
            lines.append(f"{noncomp}abbrev effect_{i} : {ty} :=")
            lines.append(full_body)
            lines.append("")

        # Emit R_i abbrevs — access newstate via function application.
        for i, (offset, size) in enumerate(spans):
            eargs = effect_arg_lists[i]
            eargs_str = (" " + " ".join(eargs)) if eargs else ""
            lines.append(f"{noncomp}abbrev R_{i} : Bool :=")
            new_slice = _state_tuple("newstate", size, offset)
            lines.append(f"  {new_slice} == effect_{i}{eargs_str}")
            lines.append("")

        # Emit TransRel abbrev.
        r_calls = [
            f"R_{i}" + ((" " + " ".join(r_arg_lists[i])) if r_arg_lists[i] else "")
            for i in range(n_ctrl)
        ]
        lines.append(f"{noncomp}abbrev TransRel : Bool :=")
        lines.append("  " + " &&\n  ".join(r_calls))
        lines.append("")

        # Collect effect_i_eq theorems.
        # ScalarRel.effect_i takes a tuple state; build it from state i.
        #
        # `rfl` leads, and `simp` is only the fallback. The two bodies are the
        # same terms over different bindings -- `state k` here, `ctrl.2.1`
        # there -- so applying `ScalarRel.effect_i` to the literal state tuple
        # makes them definitionally equal. But a multi-element wire is rebuilt
        # with `fun i j => match i, j with ...`, and every `match` in a
        # definition elaborates to an auxiliary matcher named after its
        # enclosing declaration: `FBK.effect_i.match_1` on one side,
        # `ScalarRel.effect_i.match_1` on the other (likewise `_proof_1` for
        # the `Fin` literal bounds). simp closes a goal only up to *syntactic*
        # equality after rewriting, so it left `X = X` unsolved -- the two
        # matchers print identically and are defeq, but are not the same
        # constant. `rfl` checks defeq at default transparency and unfolds
        # them. Modules whose ctrl wires are all 1x1 emit no `match` at all,
        # which is why this only ever bit the multi-element ones.
        st = _state_tuple("state", n_slots)
        for i, _ty, _body, eargs in update_data:
            lhs = f"effect_{i}" + ((" " + " ".join(eargs)) if eargs else "")
            thm_lines.append(f"theorem effect_{i}_eq : {lhs} = ScalarRel.effect_{i} {st} extl_l extl_n := by")
            thm_lines.append(
                f"  first | rfl | simp [effect_{i}, ScalarRel.effect_{i}]"
            )
            thm_lines.append("")

        # Collect R_i_iff theorems.
        st = _state_tuple("state", n_slots)
        nst = _state_tuple("newstate", n_slots)
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
            ty = _effect_type(w)
            body = _translate_terms_scalar(
                _reachable_terms(ctx.atom.init, [w]),
                init_bindings,
                [w],
                ctx.constants,
                flat_slots=init_flat,
                flatten_outputs=True,
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
            lines.append(f"{noncomp}abbrev init_{i} : {ty} :=")
            lines.append(full_body)
            lines.append("")

        # Emit Init_i abbrevs — access s via function application.
        for i, (offset, size) in enumerate(spans):
            iargs = init_arg_lists[i]
            iargs_str = (" " + " ".join(iargs)) if iargs else ""
            lines.append(f"{noncomp}abbrev Init_{i} : Bool :=")
            s_slice = _state_tuple("s", size, offset)
            lines.append(f"  {s_slice} == init_{i}{iargs_str}")
            lines.append("")

        # Emit InitCond abbrev.
        init_calls = [
            f"Init_{i}" + ((" " + " ".join(init_cond_arg_lists[i])) if init_cond_arg_lists[i] else "")
            for i in range(n_ctrl)
        ]
        lines.append(f"{noncomp}abbrev InitCond : Bool :=")
        lines.append("  " + " &&\n  ".join(init_calls))
        lines.append("")

        # Collect init_i_eq theorems (no state involved).
        for i, _ty, _body, iargs in init_data:
            lhs = f"init_{i}" + ((" " + " ".join(iargs)) if iargs else "")
            thm_lines.append(f"theorem init_{i}_eq : {lhs} = ScalarRel.init_{i} extl_n := by")
            thm_lines.append(f"  first | rfl | simp [init_{i}, ScalarRel.init_{i}]")
            thm_lines.append("")

        # Collect Init_i_iff theorems.
        ss = _state_tuple("s", n_slots)
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
