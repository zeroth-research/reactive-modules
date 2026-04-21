from zrth.lean.native import _product_type, _translate_terms
from zrth.lean.circ import (
    CircLayer,
    _translate_terms_circ,
    _ty_list,
    _native_to_vt,
    _natives_to_vt,
)
from zrth.lean.common import LeanContext
from zrth import Module, Wire


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
            upd_dom = _ty_list(
                ctx.ctrl_latched + ctx.extl_latched + ctx.extl_next
            )
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
        proof.append("  try simp")
        proof.append("  try grind")
        proof.append(f"  try simp [{', '.join(final_simp)}]")
        proof.append("  try exact List.ofFn_inj.mp rfl")
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
        return "{}\n\n{}".format(self._constants_block(), self.atom_to_lean_functional())

    def to_lean_circ(self) -> str:
        return "{}\n\n{}".format(
            self.atom_to_lean_circuit(),
            self.to_lean_equiv_theorems(),
        )

    def to_lean(self, circuit: bool = True) -> str:
        if circuit:
            return f"{self.to_lean_functional()}\n\n{self.to_lean_circ()}"
        return self.to_lean_functional()

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
