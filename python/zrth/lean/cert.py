from zrth.lean.native import _product_type, _append_expr, _translate_terms
from zrth import Module
from .translate import ModuleToLean4


def generate_certificate_lean(
    project_name: str,
    module: Module,
    module_name: str,
    m2l: ModuleToLean4,
    inv_terms: list | None = None,
    init_pre_terms: list | None = None,
    update_pre_terms: list | None = None,
    ranking_terms: list | None = None,
    p_terms: list | None = None,
) -> str:
    """Generate Certificate.lean with compiled or placeholder definitions."""
    extl_next = [pair[1] for pair in module.extl]
    ctrl_next = [pair[1] for pair in module.ctrl]

    ctrl_latched = [p[0] for p in module.ctrl]
    extl_native = _product_type(extl_next)
    ctrl_native = _product_type(ctrl_next)
    append = _append_expr("x", len(ctrl_latched), "e", len(extl_next))

    # Extract constants from certificate term lists
    # TODO: remove the dependency on ModuleToLean4
    # (make generating the certificate part of that class)
    existing_const_count = len(m2l._const_defs)
    for terms in [inv_terms, init_pre_terms, update_pre_terms, ranking_terms, p_terms]:
        if terms is not None:
            m2l._extract_constants(terms)
    cert_const_defs = m2l._const_defs[existing_const_count:]
    const_names = list(m2l._constants.values())

    def _cert_body(terms, block_inputs, param_name):
        """Compile a certificate term list into a Lean function body."""
        output = [terms[-1].write[0]]
        return _translate_terms(terms, block_inputs, output, m2l._constants, param_name)

    lines: list[str] = []

    # Imports
    lines.append("import Mathlib.Algebra.BigOperators.Fin")
    lines.append("import Core.Basic")
    lines.append(f"import {project_name}.{module_name}")
    lines.append("")

    # Certificate-specific constants (if any)
    if cert_const_defs:
        for cdef in cert_const_defs:
            lines.append(cdef)
        lines.append("")

    # init_pre
    if init_pre_terms is not None:
        body = _cert_body(init_pre_terms, extl_next, "e")
        lines.append(f"def init_pre (e : {extl_native}) : Prop :=")
        lines.append(body)
    else:
        lines.append(f"def init_pre (e : {extl_native}) : Prop := True")
    lines.append("")

    # update_pre
    if update_pre_terms is not None:
        body = _cert_body(update_pre_terms, extl_next, "e")
        lines.append(f"def update_pre (e : {extl_native}) : Prop :=")
        lines.append(body)
    else:
        lines.append(f"def update_pre (e : {extl_native}) : Prop := True")
    lines.append("")

    # inv
    if inv_terms is not None:
        body = _cert_body(inv_terms, ctrl_next, "s")
        lines.append(f"def inv (s : {ctrl_native}) : Prop :=")
        lines.append(body)
    else:
        lines.append(f"def inv (s : {ctrl_native}) : Prop := True")
    lines.append("")

    # P
    if p_terms is not None:
        body = _cert_body(p_terms, ctrl_next, "s")
        lines.append(f"def P (s : {ctrl_native}) : Prop :=")
        lines.append(body)
    else:
        lines.append(f"def P (s : {ctrl_native}) : Prop := sorry")
    lines.append("")

    # DecidablePred P — must come after P and before ranking
    lines.append("instance : DecidablePred P := inferInstance")
    lines.append("")

    # ranking
    if ranking_terms is not None:
        body = _cert_body(ranking_terms, ctrl_next, "s")
        lines.append(f"def ranking (s : {ctrl_native}) : Nat :=")
        lines.append(body)
    else:
        lines.append(f"def ranking (s : {ctrl_native}) : Nat := sorry")
    lines.append("")
    lines.append("")

    # ReactiveModule definition
    lines.append("def RM : ReactiveModule")
    lines.append(f"          ({extl_native}) ({ctrl_native})")
    lines.append(":= {")
    lines.append("    init := init")
    lines.append(f"    update := fun x e => update {append}")
    lines.append("    init_pre := init_pre")
    lines.append("    update_pre := update_pre")
    lines.append("}")
    lines.append("")

    # simp_mod tactic — unfolds module definitions for proof automation
    const_list = ", ".join(const_names) if const_names else ""

    lines.append("-- tactic that unfolds module definitions and simplifies")
    lines.append('macro "simp_mod" : tactic =>')
    lines.append("  `(tactic| (")
    simp_lemmas = "init, update, inv"
    if const_list:
        simp_lemmas += f",\n               {const_list}"
    simp_lemmas += (
        ",\n               MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply"
    )
    simp_lemmas += ",\n               Bool.or_eq_true, decide_eq_true_eq"
    simp_lemmas += ",\n               Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue"
    lines.append(f"    simp only [{simp_lemmas}]")
    lines.append("    <;> (try omega)")
    lines.append("    <;> (try (split <;> simp_all <;> omega))")
    lines.append("    <;> (try (split <;> split <;> simp_all <;> omega))))")
    lines.append("")

    # simp_inv tactic — reduces module defs then solves CNF goals
    lines.append(
        "-- tactic that reduces module definitions and solves CNF invariant goals"
    )
    lines.append('macro "simp_inv" : tactic =>')
    lines.append("  `(tactic| (")
    inv_lemmas = "RM, ReactiveModule.init, ReactiveModule.update,\n"
    inv_lemmas += "               ReactiveModule.init_pre, ReactiveModule.update_pre,\n"
    inv_lemmas += "               init, update, inv, init_pre, update_pre"
    if const_list:
        inv_lemmas += f",\n               {const_list}"
    inv_lemmas += (
        ",\n               MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply"
    )
    inv_lemmas += ",\n               Bool.or_eq_true, decide_eq_true_eq"
    inv_lemmas += ",\n               Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue"
    lines.append(f"    simp only [{inv_lemmas}] at *")
    lines.append("    <;> first")
    lines.append("      | trivial")
    lines.append("      | omega")
    lines.append("      | (simp_all; omega)")
    lines.append("      | (repeat' constructor)")
    lines.append("        <;> first")
    lines.append("          | trivial | omega")
    lines.append("          | (simp_all; omega)")
    lines.append("          | (left; omega) | (right; omega)")
    lines.append("          | (left; simp_all; omega) | (right; simp_all; omega)")
    lines.append("          | (right; right; omega) | (right; right; simp_all; omega)")
    lines.append("      | (split <;> simp_all <;> omega)")
    lines.append("      | (split <;> split <;> simp_all <;> omega)))")
    lines.append("")

    # init_inv theorem
    lines.append("theorem init_inv :")
    lines.append("  ∀ s, RM.init_pre s → inv (RM.init s) := by")
    lines.append("   intro s hpre")
    lines.append("   simp_inv")
    lines.append("")

    # step_inv theorem
    lines.append("theorem step_inv :")
    lines.append("  ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e) := by")
    lines.append("   intro s e ⟨hpre, hinv⟩")
    lines.append("   simp_inv")
    lines.append("")
    lines.append("")

    # LTS section
    lines.append("section LTS")
    lines.append("")
    lines.append("def lts := RM.toLTS'")
    lines.append("")

    # hinv' theorem
    lines.append("theorem hinv' : lts.StateSet_isInductiveInitial inv := by")
    lines.append("  unfold LTS'.StateSet_isInductiveInitial")
    lines.append("  unfold LTS'.StateSet_isInductive")
    lines.append("  constructor")
    lines.append("  · intro s hs")
    lines.append("    unfold lts at hs")
    lines.append(
        "    simp [ReactiveModule.toLTS', ReactiveModule.LTS_init, RM, init_pre] at hs"
    )
    lines.append("    unfold inv")
    lines.append("    simp [Membership.mem]")
    lines.append("  · intro s s' ⟨hs, l, hstep⟩")
    lines.append("    unfold lts at hstep")
    lines.append(
        "    simp only [ReactiveModule.toLTS', ReactiveModule.LTS_update] at hstep"
    )
    lines.append("    rw [← hstep.2]")
    lines.append("    exact step_inv s l ⟨hstep.1, hs⟩")
    lines.append("")

    # hinv theorem
    lines.append("theorem hinv : lts.StateSet_isInvariant inv := by")
    lines.append("  apply LTS'.StateSet_ind_init_is_inv lts")
    lines.append("  exact hinv'")
    lines.append("")
    lines.append("")

    # hrank theorem
    lines.append("theorem hrank : ∀ s s', inv s → ¬(P s) → (∃ l, lts.Tr s l s') →")
    lines.append("    ranking s' < ranking s := by")
    lines.append("    intro s s' hi hP htr")
    lines.append("    unfold lts at htr")
    lines.append(
        "    simp only [ReactiveModule.toLTS', ReactiveModule.LTS_update] at htr"
    )
    lines.append("    obtain ⟨l, hpre, heq⟩ := htr")
    lines.append("    rw [← heq]")
    lines.append("    unfold ranking P at *")
    lines.append("    unfold inv at *")
    lines.append("    simp only [RM, ReactiveModule.update]")
    lines.append("    unfold update")
    lines.append("    simp_mod")
    lines.append("")
    lines.append("def buchi := rule_buchi")
    lines.append("  lts")
    lines.append("  P")
    lines.append("  inv")
    lines.append("  hinv")
    lines.append("  ranking")
    lines.append("  hrank")
    lines.append("")
    lines.append("end LTS")
    lines.append("")

    return "\n".join(lines)
