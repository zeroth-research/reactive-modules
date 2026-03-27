from zrth.lean.native import _product_type, _translate_terms
from zrth.lean.circ import CircLayer, _translate_terms_circ, _ty_list, _native_to_vt
from zrth.lean.common import itype_name, _is_scalar_tensor, _tensor_to_lean_def
from zrth import Module, Wire


# ====================================================================
# ModuleToLean4 Class
#
# TODO: change to `GenerateLean4Cert` and cover also the certificate
# ====================================================================
class ModuleToLean4:
    """Convert a Python Module into Lean4 wiring diagram code."""

    def __init__(self, module: Module):
        self.module = module
        self._const_counter = 0
        self._const_defs: list[str] = []
        self._constants: dict[int, str] = {}  # wire_id -> const name
        self.const_names: list[str] = []  # populated after to_lean()

    def _next_const_name(self) -> str:
        """Generate sequential constant names: c0, c1, c2, ..."""
        name = f"c{self._const_counter}"
        self._const_counter += 1
        return name

    def _extract_constants(self, terms) -> None:
        """Pre-scan terms for matrix Tensor constants and generate top-level definitions.

        Scalar Bool/Int tensors are inlined directly in the function body.
        Only matrix tensors (shape with dim >= 2 or vector with size > 1) get
        top-level named definitions.
        """
        for term in terms:
            name_str = itype_name(term.itype)
            if name_str == "Tensor":
                out_wire = term.write[0]
                if _is_scalar_tensor(out_wire):
                    continue
                const_name = self._next_const_name()
                self._constants[out_wire.id] = const_name
                tensor = term.itype._0
                lean_def = _tensor_to_lean_def(const_name, tensor, out_wire)
                self._const_defs.append(lean_def)

    def _emit_named_layers(
        self,
        block_name: str,
        circ_layers: list[CircLayer],
        dom: str,
        cod: str,
    ) -> tuple[list[str], list[str]]:
        """Emit named @[simp] definitions for each layer and a composed definition.

        Returns (lines, layer_names) where layer_names includes both
        individual layer names and the composed definition name.
        """
        lines: list[str] = []
        layer_names: list[str] = []

        for i, layer in enumerate(circ_layers):
            name = f"{block_name}_l{i}"
            in_tys = ", ".join(layer.in_tys)
            out_tys = ", ".join(layer.out_tys)
            lines.append(f"@[simp] def {name} : Box [{in_tys}] [{out_tys}] :=")
            lines.append(f"  {layer.body}")
            lines.append("")
            layer_names.append(name)

        # Composed definition
        if len(layer_names) == 1:
            lines.append(f"@[simp] def {block_name} : Box {dom} {cod} :=")
            lines.append(f"  {layer_names[0]}")
        else:
            lines.append(f"@[simp] def {block_name} : Box {dom} {cod} :=")
            lines.append(f"  {' ≫ '.join(layer_names)}")
        lines.append("")

        return lines, layer_names

    def to_lean_circuit(self, atom) -> str:
        """Generate the full Lean4 source for this module as a combinational circuit"""
        m = self.module

        # Extract single atom (assume single atom for now)
        atoms = list(m.atoms)
        if len(atoms) != 1:
            raise ValueError(
                f"ModuleToLean4 currently supports single-atom modules, got {len(atoms)}"
            )
        atom = atoms[0]

        extl_next: list[Wire] = [pair[1] for pair in m.extl]
        ctrl_latched: list[Wire] = [pair[0] for pair in m.ctrl]
        ctrl_next: list[Wire] = [pair[1] for pair in m.ctrl]
        params: list[Wire] = []  # [x for x in m.param]

        # Extract constants from both blocks
        init_terms = atom.init
        update_terms = atom.update

        # Compile both blocks
        init_inputs = extl_next + params
        init_outputs = ctrl_next
        init_layers = _translate_terms_circ(
            init_terms, init_inputs, init_outputs, self._constants
        )

        update_inputs = ctrl_latched + extl_next + params
        update_outputs = ctrl_next
        update_layers = _translate_terms_circ(
            update_terms, update_inputs, update_outputs, self._constants
        )

        # Render output
        lines = []
        lines.append("namespace Circ")

        # Init definition with named layers
        self._init_layer_names: list[str] = []
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

        # Update definition with named layers
        self._update_layer_names: list[str] = []
        if update_layers:
            upd_dom = _ty_list(update_inputs)
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

    def translate_constants(self, atom) -> str:
        # Extract constants from both blocks
        init_terms = atom.init
        update_terms = atom.update
        self._extract_constants(init_terms)
        self._extract_constants(update_terms)

        # Store metadata for certificate generation
        self.const_names = list(self._constants.values())

        lines: list[str] = []

        # Constants
        if self._const_defs:
            lines.append("/- Concrete constants -/")
            lines.append("")
            for cdef in self._const_defs:
                lines.append(cdef)

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
        input_wires: list[Wire],
        layer_names: list[str],
        block_name: str,
    ) -> str:
        """Generate layer-by-layer proof tactic for equivalence theorem.

        Unfolds one circuit layer at a time to keep intermediate terms small.
        """
        const_names = ", ".join(self.const_names) if self.const_names else ""

        proof = []
        proof.append("  intro s")
        # Step 1: unfold the composed circuit and sequential composition
        proof.append(f"  simp_circ [Circ.{block_name}, Box.seq]")
        # Step 2: reduce each layer from innermost to outermost
        for name in layer_names:
            proof.append(f"  simp_circ [Circ.{name}]")
        # Step 3: match with functional definition
        final_simp = [block_name, "ite_pair"]
        if const_names:
            final_simp.append(const_names)
        proof.append(f"  simp [{', '.join(final_simp)}]")
        proof.append("  try simp only [Mat_1_1_eq]; simp")
        proof.append("  try grind")
        return "\n".join(proof)

    def to_lean_equiv_theorems(self, atom) -> str:
        """Generate theorems proving circuit ≡ functional."""
        m = self.module

        extl_next: list[Wire] = [pair[1] for pair in m.extl]
        ctrl_latched: list[Wire] = [pair[0] for pair in m.ctrl]
        ctrl_next: list[Wire] = [pair[1] for pair in m.ctrl]
        params: list[Wire] = []  # [x for x in m.param]

        lines: list[str] = []

        # Emit simp_circ helper tactic
        has_theorems = bool(self._init_layer_names or self._update_layer_names)
        if has_theorems:
            lines.append(self._simp_circ_macro())
            lines.append("")

        # Init equivalence theorem
        init_inputs = extl_next + params
        n_inputs = len(init_inputs)
        n_ctrl = len(ctrl_next)

        if self._init_layer_names:
            init_native = _product_type(init_inputs)
            lhs_input = _native_to_vt("s", n_inputs)
            rhs_output = _native_to_vt("r", n_ctrl)

            lines.append(f"theorem init_circ_eq : ∀ (s : {init_native}),")
            lines.append(f"    Circ.init.fn {lhs_input} =")
            lines.append("    let r := init s")
            lines.append(f"    {rhs_output} := by")
            lines.append(
                self._equiv_proof_tactic(
                    init_inputs,
                    self._init_layer_names,
                    "init",
                )
            )
            lines.append("")

        # Update equivalence theorem
        update_inputs = ctrl_latched + extl_next + params
        n_update = len(update_inputs)

        if self._update_layer_names:
            update_native = _product_type(update_inputs)
            lhs_input = _native_to_vt("s", n_update)
            rhs_output = _native_to_vt("r", n_ctrl)

            lines.append(f"theorem update_circ_eq : ∀ (s : {update_native}),")
            lines.append(f"    Circ.update.fn {lhs_input} =")
            lines.append("    let r := update s")
            lines.append(f"    {rhs_output} := by")
            lines.append(
                self._equiv_proof_tactic(
                    update_inputs,
                    self._update_layer_names,
                    "update",
                )
            )
            lines.append("")

        return "\n".join(lines)

    def to_lean(self, circuit: bool = False) -> str:
        """Generate the full Lean4 source for this module."""
        m = self.module

        # Extract single atom (assume single atom for now)
        atoms = list(m.atoms)
        if len(atoms) != 1:
            raise ValueError(
                f"ModuleToLean4 currently supports single-atom modules, got {len(atoms)}"
            )

        atom = atoms[0]

        return "{}\n\n{}\n\n{}\n\n{}".format(
            self.translate_constants(atom),
            self.to_lean_circuit(atom),
            self.to_lean_functional(atom),
            self.to_lean_equiv_theorems(atom),
        )

    def to_lean_functional(self, atom) -> str:
        m = self.module

        # TODO: we should handle also `extl_latched`
        extl_next: list[Wire] = [pair[1] for pair in m.extl]
        ctrl_latched: list[Wire] = [pair[0] for pair in m.ctrl]
        ctrl_next: list[Wire] = [pair[1] for pair in m.ctrl]
        params: list[Wire] = []  # [x for x in m.param]

        init_terms = atom.init
        update_terms = atom.update

        # Compile both blocks
        init_inputs = extl_next + params
        init_outputs = ctrl_next
        init_body = _translate_terms(
            init_terms, init_inputs, init_outputs, self._constants
        )

        update_inputs = ctrl_latched + extl_next + params
        update_outputs = ctrl_next
        update_body = _translate_terms(
            update_terms, update_inputs, update_outputs, self._constants
        )

        # Render output
        lines = []

        # Init definition
        if init_body:
            init_dom = _product_type(init_inputs)
            init_cod = _product_type(init_outputs)
            lines.append(f"@[simp] def init (s : {init_dom}) : {init_cod} :=")
            lines.append(init_body)
            lines.append("")

        # Update definition
        if update_body:
            upd_dom = _product_type(update_inputs)
            upd_cod = _product_type(update_outputs)
            lines.append(f"@[simp] def update (s : {upd_dom}) : {upd_cod} :=")
            lines.append(update_body)
            lines.append("")

        return "\n".join(lines)
