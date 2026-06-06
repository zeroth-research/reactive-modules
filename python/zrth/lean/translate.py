from zrth.lean.native import (
    _product_type,
    _product_type_scalar,
    _translate_terms,
    _translate_terms_scalar,
    _build_tuple,
    _reachable_terms,
    _argmax_scalar_name,
)
from zrth.lean.circ import (
    CircLayer,
    _translate_terms_circ,
    _ty_list,
    _native_to_vt,
    _natives_to_vt,
)
from zrth.lean.common import (
    LeanContext,
    _accessor,
    _is_scalar_wire,
    _bind_wires,
    dtype_to_lean_type,
    _flat_size,
    _flat_indices,
    _flat_element_type,
    _mat_from_scalars,
    itype_name,
)
from zrth import Module, Wire


def _scalar_bindings_with_recon(
    params: "list[tuple[str, list]]",
) -> "tuple[list[str], dict[int, str], dict[int, list[str]]]":
    """Build wire bindings and flat scalar slot tracking for scalar encoding.

    For each wire:
    - Scalar (flat_size == 1): binding is the bare accessor (e.g. ``"ctrl.1"``).
    - Non-scalar (flat_size > 1): binding is a fresh let-var reconstructing the
      ``Mat T m n`` (for operations that need the full matrix); flat_slots also
      records the individual scalar accessor strings for argmax and similar.

    Returns ``(let_lines, bindings, flat_slots)`` where:
    - ``let_lines`` should be prepended to the function body (Mat reconstructions)
    - ``bindings`` maps wire_id → matrix var or scalar accessor
    - ``flat_slots`` maps wire_id → list of flat scalar accessor strings
    """
    let_lines: list[str] = []
    bindings: dict[int, str] = {}
    flat_slots: dict[int, list[str]] = {}
    let_counter = [0]

    for name, wires in params:
        flat_sizes = [_flat_size(w) for w in wires]
        total = sum(flat_sizes)
        flat_accrs = [f"{name}{_accessor(k, total)}" for k in range(total)]

        offset = 0
        for w, fsize in zip(wires, flat_sizes):
            slots = flat_accrs[offset : offset + fsize]
            flat_slots[w.id] = slots
            if fsize == 1:
                bindings[w.id] = slots[0]
            else:
                mat_ty = dtype_to_lean_type(w)
                mat_expr = _mat_from_scalars(slots, list(w.dtype.shape), _flat_element_type(w))
                var_name = f"_m{let_counter[0]}"
                let_counter[0] += 1
                let_lines.append(f"  let {var_name} : {mat_ty} := {mat_expr}")
                bindings[w.id] = var_name
            offset += fsize

    return let_lines, bindings, flat_slots


class ModuleToLean4:
    """Convert a Python Module into Lean4 wiring diagram code.

    Read-only consumer of a pre-populated LeanContext — does not mutate it.
    Accepts a Module for convenience; in that case builds a fresh context.
    """

    def __init__(self, src: "LeanContext | Module"):
        self.ctx = src if isinstance(src, LeanContext) else LeanContext(src)
        # Layer names become available after atom_to_lean_circuit runs; the
        # equivalence-theorem emitter reads them back.
        self._init_layer_names: list[str] = []
        self._update_layer_names: list[str] = []

    @property
    def module(self):
        return self.ctx.module

    def _emit_named_layers(
        self,
        block_name: str,
        circ_layers: list[CircLayer],
        dom: str,
        cod: str,
    ) -> tuple[list[str], list[str]]:
        """Emit named @[simp] definitions for each layer and a composed definition."""
        lines: list[str] = []
        layer_names: list[str] = []
        # `Box.eq` / `Box.neq` on Real-valued wires drag in `Real.decidableEq`,
        # which is classical. Mark all layers (and the composed def) as
        # `noncomputable` when any Real wires are present so Lean's IR checker
        # doesn't try to emit executable code.
        noncomp = "noncomputable " if self.ctx.uses_real else ""

        for i, layer in enumerate(circ_layers):
            name = f"{block_name}_l{i}"
            in_tys = ", ".join(layer.in_tys)
            out_tys = ", ".join(layer.out_tys)
            lines.append(f"@[simp] {noncomp}def {name} : Box [{in_tys}] [{out_tys}] :=")
            lines.append(f"  {layer.body}")
            lines.append("")
            layer_names.append(name)

        lines.append(f"@[simp] {noncomp}def {block_name} : Box {dom} {cod} :=")
        lines.append(f"  {' ≫ '.join(layer_names)}")
        lines.append("")

        return lines, layer_names

    def atom_to_lean_circuit(self) -> str:
        """Generate the full Lean4 source for this module as a combinational circuit."""
        ctx = self.ctx
        atom = ctx.atom

        init_inputs = ctx.extl_next
        init_outputs = ctx.ctrl_next
        init_layers = _translate_terms_circ(
            atom.init,
            (init_inputs,),
            init_outputs,
            ctx.constants,
        )

        update_inputs = (ctx.ctrl_latched, ctx.extl_latched, ctx.extl_next)
        update_outputs = ctx.ctrl_next
        update_layers = _translate_terms_circ(
            atom.update,
            update_inputs,
            update_outputs,
            ctx.constants,
        )

        lines = ["namespace Circ"]

        if init_layers:
            init_dom = _ty_list(init_inputs)
            init_cod = _ty_list(init_outputs)
            layer_lines, self._init_layer_names = self._emit_named_layers(
                "init",
                init_layers,
                init_dom,
                init_cod,
            )
            lines.extend(layer_lines)

        if update_layers:
            upd_dom = _ty_list(ctx.ctrl_latched + ctx.extl_latched + ctx.extl_next)
            upd_cod = _ty_list(update_outputs)
            layer_lines, self._update_layer_names = self._emit_named_layers(
                "update",
                update_layers,
                upd_dom,
                upd_cod,
            )
            lines.extend(layer_lines)

        lines.append("end Circ")
        return "\n".join(lines)

    def _constants_block(self) -> str:
        """Render the top-level `@[simp] def c0 : ...` constant definitions."""
        defs = self.ctx.constants.defs()
        if not defs:
            return ""
        lines = ["/- Concrete constants -/", ""]
        lines.extend(defs)
        return "\n".join(lines)

    # Simp lemmas for reducing a single circuit layer
    _LAYER_SIMP = [
        "Box.par",
        "ValTuple.split",
        "ValTuple.append",
        "ValTuple.append_split",
        "ValTuple.append_ite",
        "ValTuple.split_singleton_fst",
        "ValTuple.split_singleton_snd",
        "ValTuple.split_cons_fst_fst",
        "ValTuple.split_cons_fst_snd",
        "ValTuple.split_2_fst",
        "ValTuple.split_2_snd",
        "ValTuple.split_3_fst",
        "ValTuple.split_3_snd",
        "ValTuple.split_nil",
        "ValTuple.split_nil_snd",
        "Box.id",
        "Box.dup",
        "Box.swap",
        "Box.destr",
        "Box.const",
        "Box.not",
        "Box.and",
        "Box.or",
        "Box.ite",
        "Box.add",
        "Box.sub",
        "Box.mul",
        "Box.neg",
        "Box.lt",
        "Box.le",
        "Box.gt",
        "Box.ge",
        "Box.eq",
        "Box.neq",
        "Box.min",
        "Box.max",
        "Box.nnLinear",
        "Box.relu",
        "Box.argmax_1d",
        "Box.argmax",
        "ite_pair",
        "Mat_1_1_lt_iff",
        "Mat_1_1_le_iff",
    ]

    def _simp_circ_macro(self) -> str:
        """Generate the simp_circ helper tactic macro."""
        layer_simp = ",\n    ".join(self._LAYER_SIMP)
        lines = []
        lines.append("/-- Reduce one circuit layer: unfolds the given lemma,")
        lines.append("    then simplifies all Box/ValTuple plumbing. -/")
        lines.append(
            'macro "simp_circ" "[" ls:Lean.Parser.Tactic.simpLemma,* "]" : tactic =>'
        )
        lines.append("  `(tactic| simp only [$ls,*,")
        lines.append(f"    {layer_simp}])")
        return "\n".join(lines)

    def _equiv_proof_tactic(
        self,
        intro_vars: list[str],
        layer_names: list[str],
        block_name: str,
    ) -> str:
        """Generate layer-by-layer proof tactic for equivalence theorem."""
        const_names = ", ".join(self.ctx.constants.names())

        proof = []
        proof.append(f"  intro {' '.join(intro_vars)}")
        proof.append(f"  simp_circ [Circ.{block_name}, Box.seq]")
        for name in layer_names:
            proof.append(f"  simp_circ [Circ.{name}]")
        final_simp = [block_name, "ite_pair"]
        if const_names:
            final_simp.append(const_names)
        proof.append(f"  simp [{', '.join(final_simp)}]")
        proof.append("  try exact List.ofFn_inj.mp rfl")
        proof.append("  try grind")
        proof.append("  try omega")
        proof.append("  try simp")
        proof.append("  try grind")
        proof.append(f"  try simp [{', '.join(final_simp)}]")
        proof.append("  try exact List.ofFn_inj.mp rfl")
        proof.append("  try omega")
        return "\n".join(proof)

    def to_lean_equiv_theorems(self) -> str:
        """Generate theorems proving circuit ≡ functional."""
        ctx = self.ctx
        lines: list[str] = []

        has_theorems = bool(self._init_layer_names or self._update_layer_names)
        if has_theorems:
            lines.append(self._simp_circ_macro())
            lines.append("")

        n_ctrl = len(ctx.ctrl_next)

        if self._init_layer_names:
            n_extl_n = len(ctx.extl_next)
            init_binder = f"(extl_n : {_product_type(ctx.extl_next)})"
            lhs_input = _natives_to_vt([("extl_n", n_extl_n)])
            rhs_output = _native_to_vt("r", n_ctrl)

            lines.append(f"theorem init_circ_eq : ∀ {init_binder},")
            lines.append(f"    Circ.init.fn {lhs_input} =")
            lines.append("    let r := init extl_n")
            lines.append(f"    {rhs_output} := by")
            lines.append(
                self._equiv_proof_tactic(
                    ["extl_n"],
                    self._init_layer_names,
                    "init",
                )
            )
            lines.append("")

        if self._update_layer_names:
            n_ctrl_l = len(ctx.ctrl_latched)
            n_extl_l = len(ctx.extl_latched)
            n_extl_n = len(ctx.extl_next)
            update_binders = (
                f"(ctrl : {_product_type(ctx.ctrl_latched)}) "
                f"(extl_l : {_product_type(ctx.extl_latched)}) "
                f"(extl_n : {_product_type(ctx.extl_next)})"
            )
            lhs_input = _natives_to_vt(
                [("ctrl", n_ctrl_l), ("extl_l", n_extl_l), ("extl_n", n_extl_n)]
            )
            rhs_output = _native_to_vt("r", n_ctrl)

            lines.append(f"theorem update_circ_eq : ∀ {update_binders},")
            lines.append(f"    Circ.update.fn {lhs_input} =")
            lines.append("    let r := update ctrl extl_l extl_n")
            lines.append(f"    {rhs_output} := by")
            lines.append(
                self._equiv_proof_tactic(
                    ["ctrl", "extl_l", "extl_n"],
                    self._update_layer_names,
                    "update",
                )
            )
            lines.append("")

        return "\n".join(lines)

    def to_lean_functional(self) -> str:
        """Generate the functional init/update definitions (plus constants)."""
        return "{}\n\n{}".format(
            self._constants_block(), self.atom_to_lean_functional()
        )

    def to_lean_circ(self) -> str:
        return "{}\n\n{}".format(
            self.atom_to_lean_circuit(),
            self.to_lean_equiv_theorems(),
        )

    # ------------------------------------------------------------------
    # Scalar encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_body(param: str, wires: "list[Wire]") -> str:
        """Body of ``unpack_<param>``: extract all flat elements from the wire."""
        n = len(wires)
        parts = []
        for i, w in enumerate(wires):
            base = f"{param}{_accessor(i, n)}"
            for row, col in _flat_indices(w):
                parts.append(f"{base} {row} {col}")
        return _build_tuple(parts)

    @staticmethod
    def _pack_body(var: str, wires: "list[Wire]") -> str:
        """Body of ``pack``: reconstruct each wire's Mat from flat scalar slots."""
        total = sum(_flat_size(w) for w in wires)
        flat_accrs = [f"{var}{_accessor(k, total)}" for k in range(total)]
        parts = []
        offset = 0
        for w in wires:
            n = _flat_size(w)
            slots = flat_accrs[offset : offset + n]
            parts.append(_mat_from_scalars(slots, list(w.dtype.shape), _flat_element_type(w)))
            offset += n
        return _build_tuple(parts)

    @staticmethod
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

    @staticmethod
    def _argmax_scalar_def_lines(elem_ty: str, n: int) -> list[str]:
        """Emit a scalar def and simp theorem for argmax over n elements.

        The def mirrors the ``foldl`` in ``argmax_1d`` step by step so the
        equality theorem closes by ``simp`` + ``rfl`` (definitional equality
        after unfolding ``List.finRange`` and ``List.foldl``).
        """
        name = _argmax_scalar_name(n)
        noncomp = "noncomputable " if elem_ty == "Real" else ""
        params = " ".join(f"(s{i} : {elem_ty})" for i in range(n))
        # Default value mirrors Inhabited.default for each type.
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

    def atom_to_lean_scalar(self) -> str:
        """Generate the scalar encoding inside ``namespace Scalar``.

        Emits, in order:
        0. Scalar argmax axioms (one per distinct Argmax input shape used).
        1. ``unpack_<param>`` for each non-Unit input group (mat → scalars).
        2. ``pack`` that wraps scalar outputs back into matrices.
        3. ``init`` / ``update`` with fully scalar signatures.
        """
        ctx = self.ctx
        atom = ctx.atom
        noncomp = "noncomputable " if ctx.uses_real else ""
        cod = _product_type_scalar(ctx.ctrl_next)

        all_terms = list(atom.init) + list(atom.update)
        argmax_variants = self._collect_argmax_variants(all_terms)
        # Store for use in the equivalence proof simp set.
        self._argmax_scalar_variants: list[tuple[str, int]] = argmax_variants

        # Groups that have actual wires (non-Unit parameters).
        input_groups = [
            ("ctrl", ctx.ctrl_latched),
            ("extl_l", ctx.extl_latched),
            ("extl_n", ctx.extl_next),
        ]

        lines: list[str] = []
        for ety, n in argmax_variants:
            lines.extend(self._argmax_scalar_def_lines(ety, n))

        lines += ["namespace Scalar", ""]

        # 1. Unpack functions
        for param, wires in input_groups:
            if not wires:
                continue
            mat_ty = _product_type(wires)
            scalar_ty = _product_type_scalar(wires)
            body = self._unpack_body(param, wires)
            lines.append(
                f"@[simp] {noncomp}def unpack_{param} ({param} : {mat_ty}) : {scalar_ty} :="
            )
            lines.append(f"  {body}")
            lines.append("")

        # 2. Pack function
        pack_body = self._pack_body("r", ctx.ctrl_next)
        out_mat_ty = _product_type(ctx.ctrl_next)
        lines.append(f"@[simp] {noncomp}def pack (r : {cod}) : {out_mat_ty} :=")
        lines.append(f"  {pack_body}")
        lines.append("")

        # 3. Scalar init / update — flat scalar inputs, Mat reconstruction via let-lines
        def _scalar_dom(param: str, wires: "list[Wire]") -> str:
            ty = _product_type_scalar(wires) if wires else "Unit"
            return f"({param}: {ty})"

        def _prepend_recon(recon_lets: "list[str]", body: str) -> str:
            if recon_lets:
                return "\n".join(recon_lets) + "\n" + body
            return body

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
        return "\n".join(lines)

    def to_lean_scalar_equiv(self) -> str:
        """Generate theorems proving init/update = pack ∘ Scalar.init/update ∘ unpack."""
        ctx = self.ctx
        lines: list[str] = []

        def _var(binder: str) -> str:
            return binder.split(" : ")[0].strip("( )")

        def _call_arg(param: str, wires: "list[Wire]") -> str:
            return f"(Scalar.unpack_{param} {param})" if wires else param

        argmax_eq_lemmas = [
            f"{_argmax_scalar_name(n)}_eq"
            for _, n in getattr(self, "_argmax_scalar_variants", [])
        ]

        def _proof(
            binders: list[str],
            func_name: str,
            scalar_args: list[str],
            simp_extras: list[str],
        ) -> list[str]:
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

    def to_lean_scalar(self) -> str:
        if not self._can_scalarize():
            return (
                "-- scalar encoding not available: module has non-scalar (matrix) wires"
            )
        return "{}\n\n{}".format(
            self.atom_to_lean_scalar(),
            self.to_lean_scalar_equiv(),
        )

    # ------------------------------------------------------------------
    # Relational encoding helpers
    # ------------------------------------------------------------------

    def atom_to_lean_rel(self) -> str:
        """Generate the relational encoding inside ``namespace ScalarRel``.

        Emits, in order:
        1. ``effect_i`` — one function per ctrl_next wire, independently compiled
           from the update term list targeting only that wire.
        2. ``effect_i_eq`` — theorem: ``effect_i ctrl extl_l extl_n =
           (Scalar.update ctrl extl_l extl_n).i``.
        3. ``R_i`` — per-variable transition relation
           (``new.i = effect_i old ...``).
        4. ``TransRel`` — conjunction of all ``R_i``.
        5. ``init_i`` / ``init_i_eq`` / ``Init_i`` / ``InitCond`` — same
           structure for the init block (only when init terms exist).
        """
        ctx = self.ctx
        noncomp = "noncomputable " if ctx.uses_real else ""

        n_ctrl = len(ctx.ctrl_next)
        state_ty = _product_type_scalar(ctx.ctrl_next)

        def _ty(wires):
            return _product_type_scalar(wires) if wires else "Unit"

        # All update input groups (matches Scalar.update signature)
        update_groups = [
            ("ctrl", ctx.ctrl_latched),
            ("extl_l", ctx.extl_latched),
            ("extl_n", ctx.extl_next),
        ]
        update_binders = " ".join(f"({p} : {_ty(w)})" for p, w in update_groups)
        update_args = " ".join(p for p, _ in update_groups)
        update_intro = " ".join(p for p, _ in update_groups)

        # Extl-only groups for R_i / TransRel (old/new replace ctrl)
        extl_groups = [("extl_l", ctx.extl_latched), ("extl_n", ctx.extl_next)]
        extl_binders = " ".join(f"({p} : {_ty(w)})" for p, w in extl_groups)
        extl_args = " ".join(p for p, _ in extl_groups)

        # Init input (just extl_n, matches Scalar.init signature)
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
            acc: str,
        ) -> list[str]:
            scalar_proj = f"({scalar_func} {all_args}){acc}"
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

        def _prepend_recon(recon_lets: "list[str]", body: str) -> str:
            if recon_lets:
                return "\n".join(recon_lets) + "\n" + body
            return body

        lines = ["namespace ScalarRel", ""]

        has_update = bool(list(ctx.atom.update))
        has_init = bool(list(ctx.atom.init))

        if has_update:
            # Per-variable effect functions (independently compiled, dead-code pruned)
            for i, w in enumerate(ctx.ctrl_next):
                ty = dtype_to_lean_type(w, simple_types=True)
                body = _translate_terms_scalar(
                    _reachable_terms(ctx.atom.update, [w]),
                    update_bindings,
                    [w],
                    ctx.constants,
                    flat_slots=update_flat,
                )
                lines.append(
                    f"@[simp] {noncomp}def effect_{i} {update_binders} : {ty} :="
                )
                lines.append(_prepend_recon(update_recon, body))
                lines.append("")

            # Equality theorems vs Scalar.update
            for i in range(n_ctrl):
                acc = _accessor(i, n_ctrl)
                lines.extend(
                    _proj_theorem(
                        f"effect_{i}",
                        "Scalar.update",
                        update_binders,
                        update_args,
                        update_intro,
                        acc,
                    )
                )

            # Per-variable transition relations
            for i in range(n_ctrl):
                acc = _accessor(i, n_ctrl)
                lines.append(
                    f"def R_{i} (old new : {state_ty}) {extl_binders} : Prop :="
                )
                lines.append(f"  new{acc} = effect_{i} old {extl_args}")
                lines.append("")

            # Full transition relation
            r_calls = [f"R_{i} old new {extl_args}" for i in range(n_ctrl)]
            lines.append(
                f"def TransRel (old new : {state_ty}) {extl_binders} : Prop :="
            )
            lines.append(f"  " + " ∧\n  ".join(r_calls))
            lines.append("")

        if has_init:
            # Per-variable init value functions (independently compiled, dead-code pruned)
            for i, w in enumerate(ctx.ctrl_next):
                ty = dtype_to_lean_type(w, simple_types=True)
                body = _translate_terms_scalar(
                    _reachable_terms(ctx.atom.init, [w]),
                    init_bindings,
                    [w],
                    ctx.constants,
                    flat_slots=init_flat,
                )
                lines.append(f"@[simp] {noncomp}def init_{i} {extl_n_binder} : {ty} :=")
                lines.append(_prepend_recon(init_recon, body))
                lines.append("")

            # Equality theorems vs Scalar.init
            for i in range(n_ctrl):
                acc = _accessor(i, n_ctrl)
                lines.extend(
                    _proj_theorem(
                        f"init_{i}",
                        "Scalar.init",
                        extl_n_binder,
                        "extl_n",
                        "extl_n",
                        acc,
                    )
                )

            # Per-variable init conditions
            for i in range(n_ctrl):
                acc = _accessor(i, n_ctrl)
                lines.append(f"def Init_{i} (s : {state_ty}) {extl_n_binder} : Prop :=")
                lines.append(f"  s{acc} = init_{i} extl_n")
                lines.append("")

            # Full init condition
            init_calls = [f"Init_{i} s extl_n" for i in range(n_ctrl)]
            lines.append(f"def InitCond (s : {state_ty}) {extl_n_binder} : Prop :=")
            lines.append(f"  " + " ∧\n  ".join(init_calls))
            lines.append("")

            # InitCond ↔ s = Scalar.init extl_n
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

            # InitCond ↔ ctrl' = init extl_n  (functional / Mat domain)
            mat_ctrl_ty = _product_type(ctx.ctrl_next)

            def _unpack(param, wires):
                return f"(Scalar.unpack_{param} {param})" if wires else param

            unpack_pack_simp = (
                ["Scalar.pack", "Scalar.unpack_ctrl"]
                + (["Scalar.unpack_extl_n"] if ctx.extl_next else [])
            )
            unpack_pack_simp_str = ", ".join(unpack_pack_simp)

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
            lines.append(f"    have hpack : ctrl' = Scalar.pack (Scalar.unpack_ctrl ctrl') := by")
            lines.append(f"      simp only [Scalar.pack, Scalar.unpack_ctrl, ← Mat_1_1_eq, Prod.eta]")
            lines.append(f"      try (funext i j; simp [Fin.fin_one_eq_zero])")
            lines.append("    rw [h] at hpack; exact hpack")
            lines.append("  · intro h")
            lines.append(f"    simp [Scalar.pack, Scalar.unpack_ctrl, h]")
            lines.append("")

        if has_update:
            # TransRel ↔ new = Scalar.update old extl_l extl_n
            # Use `old` in place of `ctrl` when calling Scalar.update.
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

            # TransRel ↔ ctrl' = update ctrl extl_l extl_n  (functional / Mat domain)
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

            unpack_pack_simp = (
                ["Scalar.pack", "Scalar.unpack_ctrl"]
                + (["Scalar.unpack_extl_l"] if ctx.extl_latched else [])
                + (["Scalar.unpack_extl_n"] if ctx.extl_next else [])
            )
            unpack_pack_simp_str = ", ".join(unpack_pack_simp)

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
            lines.append(f"    have hpack : ctrl' = Scalar.pack (Scalar.unpack_ctrl ctrl') := by")
            lines.append(f"      simp only [Scalar.pack, Scalar.unpack_ctrl, ← Mat_1_1_eq, Prod.eta]")
            lines.append(f"      try (funext i j; simp [Fin.fin_one_eq_zero])")
            lines.append("    rw [h] at hpack; exact hpack")
            lines.append("  · intro h")
            lines.append(f"    simp [Scalar.pack, Scalar.unpack_ctrl, h]")
            lines.append("")

        lines.append("end ScalarRel")
        return "\n".join(lines)

    def atom_to_lean_mat_rel(self) -> str:
        """Generate the matrix-domain relational encoding inside ``namespace Rel``.

        Same structure as ``atom_to_lean_rel`` (ScalarRel) but operates on the
        original matrix types — no unpacking/scalarization.  Uses
        ``_translate_terms`` (the functional translator) and ``_bind_wires`` to
        keep everything in the ``Mat T m n`` domain.

        Emits, in order:
        1. ``effect_i`` — one function per ctrl_next wire, translated directly
           from the update term list in matrix domain.
        2. ``effect_i_eq`` — theorem: ``effect_i ctrl extl_l extl_n =
           (update ctrl extl_l extl_n).i``.
        3. ``R_i`` — per-variable transition relation (``new.i = effect_i old ...``).
        4. ``TransRel`` — conjunction of all ``R_i``.
        5. ``TransRel_func_eq`` — ``TransRel old new ... ↔ new = update old ...``.
        6. ``init_i`` / ``init_i_eq`` / ``Init_i`` / ``InitCond`` / ``InitCond_func_eq``
           — same structure for the init block (only when init terms exist).
        """
        ctx = self.ctx
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
        ) -> list[str]:
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

    def to_lean_mat_rel(self) -> str:
        return self.atom_to_lean_mat_rel()

    def to_lean_rel(self) -> str:
        if not self._can_scalarize():
            return "-- relational encoding not available: module has non-scalar (matrix) wires"
        return self.atom_to_lean_rel()

    def _can_scalarize(self) -> bool:
        """True when every wire has a shape that can be flattened to scalars (≤ 2-D)."""
        ctx = self.ctx
        all_wires = ctx.ctrl_latched + ctx.ctrl_next + ctx.extl_latched + ctx.extl_next
        return all(len(w.dtype.shape) <= 2 for w in all_wires)

    def to_lean(
        self, circuit: bool = True, scalar: bool = True, rel: bool = True
    ) -> str:
        """Return all enabled encodings joined by blank lines.

        ``circuit=True`` appends the circuit (Box algebra) encoding and its
        equivalence theorems.  ``scalar=True`` appends the scalar namespace
        (``Scalar.init``/``update``, ``unpack_*``, ``pack``) and the
        composition theorems.  ``rel=True`` appends the relational namespace
        (``ScalarRel.effect_i``, ``ScalarRel.R_i``, ``ScalarRel.TransRel``, ``ScalarRel.InitCond``).
        All default to True; pass ``False`` to omit.  The scalar and
        relational encodings are silently skipped when the module has
        non-scalar (matrix) wires.
        """
        parts = [self.to_lean_functional()]
        if circuit:
            parts.append(self.to_lean_circ())
        if scalar:
            parts.append(self.to_lean_scalar())
        if rel:
            parts.append(self.to_lean_rel())
        return "\n\n".join(parts)

    def atom_to_lean_functional(self) -> str:
        ctx = self.ctx
        atom = ctx.atom

        init_body = _translate_terms(
            atom.init,
            ctx.init_wire_names,
            ctx.ctrl_next,
            ctx.constants,
        )
        update_body = _translate_terms(
            atom.update,
            ctx.update_wire_names,
            ctx.ctrl_next,
            ctx.constants,
        )

        lines = []
        cod = _product_type(ctx.ctrl_next)
        noncomp = "noncomputable " if ctx.uses_real else ""

        if init_body:
            init_dom = f"(extl_n: {_product_type(ctx.extl_next)})"
            lines.append(f"@[simp] {noncomp}def init {init_dom} : {cod} :=")
            lines.append(init_body)
            lines.append("")

        if update_body:
            upd_dom = (
                f"(ctrl: {_product_type(ctx.ctrl_latched)}) "
                f"(extl_l: {_product_type(ctx.extl_latched)}) "
                f"(extl_n: {_product_type(ctx.extl_next)})"
            )
            lines.append(f"@[simp] {noncomp}def update {upd_dom} : {cod} :=")
            lines.append(update_body)
            lines.append("")

        return "\n".join(lines)
