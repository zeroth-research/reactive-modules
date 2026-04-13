from dataclasses import dataclass

from zrth.lean.native import _product_type, _append_expr, _translate_terms
from zrth import Module, Wire
from .translate import ModuleToLean4
from ..expr import Expr


@dataclass
class CertificateData:
    """Data needed to generate a Lean certificate."""

    prp: str | Expr | None = None
    inv: Expr | str | None = None
    init_pre: Expr | str | None = None
    update_pre: Expr | str | None = None
    ranking: Expr | str | None = None


# TODO: move the parts that does not change into .lean file.
# Or use templating engine, e.g., `Jinja2`.
#
def generate_certificate_lean(
    project_name: str,
    module: Module,
    module_name: str,
    m2l: ModuleToLean4,
    cert_data: CertificateData | None = None,
) -> str:
    """Generate Certificate.lean with compiled or placeholder definitions."""
    if cert_data is None:
        cert_data = CertificateData()

    def _as_terms(v):
        """Return v if it is a list of Terms, else None (strings are not yet compiled)."""
        return v if isinstance(v, list) else None

    inv_terms = _as_terms(cert_data.inv)
    init_pre_terms = _as_terms(cert_data.init_pre)
    update_pre_terms = _as_terms(cert_data.update_pre)
    ranking_terms = _as_terms(cert_data.ranking)
    p_terms = _as_terms(cert_data.prp)

    extl_next: list[Wire] = [pair[1] for pair in module.extl]
    ctrl_next: list[Wire] = [pair[1] for pair in module.ctrl]
    ctrl_latched: list[Wire] = [p[0] for p in module.ctrl]
    params: list[Wire] = []  # [x for x in module.param]

    extl_native = _product_type(extl_next + params)
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
    if p_terms is not None:
        lines.append(
            "instance : DecidablePred P := fun s => by unfold P; dsimp; infer_instance"
        )
    else:
        lines.append("instance : DecidablePred P := sorry")
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
    lines.append(f"""\
def RM : ReactiveModule ({extl_native}) ({ctrl_native}) := {{
    init := init
    update := fun x e => update {append}
    init_pre := init_pre
    update_pre := update_pre
}}
""")

    # simp_mod tactic — unfolds module definitions for proof automation
    const_list = ", ".join(const_names) if const_names else ""

    simp_lemmas = "init, update, inv"
    if const_list:
        simp_lemmas += f",\n               {const_list}"
    simp_lemmas += (
        ",\n               MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply"
        ",\n               mul_Mat_apply, add_Mat_apply"
    )
    simp_lemmas += ",\n               Bool.or_eq_true, decide_eq_true_eq"
    simp_lemmas += (
        ",\n               Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue"
        ",\n               Fin.sum_univ_one, Fin.sum_univ_two, Fin.sum_univ_three"
    )

    lines.append(f"""\
-- tactic that unfolds module definitions and simplifies
macro "simp_mod" : tactic =>
  `(tactic| (
    simp only [{simp_lemmas}]
    <;> (try omega)
    <;> (try (split <;> simp_all <;> omega))
    <;> (try (split <;> split <;> simp_all <;> omega))))
""")

    inv_lemmas = "RM, ReactiveModule.init, ReactiveModule.update,\n"
    inv_lemmas += "               ReactiveModule.init_pre, ReactiveModule.update_pre,\n"
    inv_lemmas += "               init, update, inv, init_pre, update_pre"
    if const_list:
        inv_lemmas += f",\n               {const_list}"
    inv_lemmas += (
        ",\n               MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply"
        ",\n               mul_Mat_apply, add_Mat_apply"
    )
    inv_lemmas += ",\n               Bool.or_eq_true, decide_eq_true_eq"
    inv_lemmas += (
        ",\n               Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue"
        ",\n               Fin.sum_univ_one, Fin.sum_univ_two, Fin.sum_univ_three"
    )

    # simp_inv tactic — reduces module defs then solves CNF goals
    # Split into two phases: (1) unfold module wrappers, (2) simplify arithmetic
    arith_lemmas = "init, update, inv, init_pre, update_pre, P, ranking"
    if const_list:
        arith_lemmas += f",\n               {const_list}"
    arith_lemmas += (
        ",\n               MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply"
        ",\n               mul_Mat_apply, add_Mat_apply"
    )
    arith_lemmas += ",\n               Bool.or_eq_true, decide_eq_true_eq"
    arith_lemmas += (
        ",\n               Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue"
        ",\n               Fin.sum_univ_one, Fin.sum_univ_two, Fin.sum_univ_three"
    )

    lines.append(f"""\
-- tactic that reduces module definitions and solves CNF invariant goals
macro "simp_inv" : tactic =>
  `(tactic| (
    -- Phase 1: unfold module wrappers
    try simp only [RM, ReactiveModule.init, ReactiveModule.update,
               ReactiveModule.init_pre, ReactiveModule.update_pre] at *
    -- Phase 2: unfold and simplify
    try dsimp only [inv, init, update, init_pre, update_pre] at *
    simp [{arith_lemmas}] at *
    <;> first
      | trivial
      | omega
      | (simp_all; omega)
      | (repeat' constructor)
        <;> first
          | trivial | omega
          | (simp_all; omega)
          | (left; omega) | (right; omega)
          | (left; simp_all; omega) | (right; simp_all; omega)
          | (right; right; omega) | (right; right; simp_all; omega)
      | (split <;> simp_all <;> omega)
      | (split <;> split <;> simp_all <;> omega)
      | (split <;> split <;> split <;> simp_all <;> omega)))
""")

    # init_inv theorem
    lines.append("""\
theorem init_inv : ∀ s, RM.init_pre s → inv (RM.init s) := by
   intro s hpre
   simp_inv

theorem step_inv : ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e) := by
   intro s e ⟨hpre, hinv⟩
   simp_inv
""")

    # LTS section
    lines.append("section LTS\n")
    lines.append("def lts := RM.toLTS'\n")

    # hinv' theorem
    lines.append("""\
theorem hinv' : lts.StateSet_isInductiveInitial inv := by
  unfold LTS'.StateSet_isInductiveInitial
  unfold LTS'.StateSet_isInductive
  constructor
  · intro s hs
    unfold lts at hs
    simp [ReactiveModule.toLTS', ReactiveModule.LTS_init, RM] at hs
    obtain ⟨l, hpre, hl⟩ := hs
    rw [← hl]
    exact init_inv l hpre
  · intro s s' ⟨hs, l, hstep⟩
    unfold lts at hstep
    simp only [ReactiveModule.toLTS', ReactiveModule.LTS_update] at hstep
    rw [← hstep.2]
    exact step_inv s l ⟨hstep.1, hs⟩

theorem hinv : lts.StateSet_isInvariant inv := by
  apply LTS'.StateSet_ind_init_is_inv lts
  exact hinv'

theorem hrank : ∀ s s', inv s → ¬(P s) → (∃ l, lts.Tr s l s') →
    ranking s' < ranking s := by
    intro s s' hi hP htr
    unfold lts at htr
    simp only [ReactiveModule.toLTS', ReactiveModule.LTS_update] at htr
    obtain ⟨l, hpre, heq⟩ := htr
    rw [← heq]
    try simp_inv
    all_goals sorry

def buchi := rule_buchi
  lts
  P
  inv
  hinv
  ranking
  hrank

""")

    lines.append("end LTS\n")

    return "\n".join(lines)
