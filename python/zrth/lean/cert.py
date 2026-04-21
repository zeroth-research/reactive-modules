from dataclasses import dataclass

from zrth.lean.native import _product_type, _translate_terms
from zrth.lean.common import LeanContext, _bind_wires
from ..expr import Expr


@dataclass
class CertificateData:
    """Data needed to generate a Lean certificate."""

    prp: str | Expr | None = None
    inv: Expr | str | None = None
    init_pre: Expr | str | None = None
    update_pre: Expr | str | None = None
    ranking: Expr | str | None = None


# TODO: move the parts that do not change into a .lean file.
# Or use a templating engine, e.g. `Jinja2`.
def generate_certificate_lean(
    project_name: str,
    module_name: str,
    ctx: LeanContext,
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

    extl_latched = ctx.extl_latched
    extl_next = ctx.extl_next
    ctrl_next = ctx.ctrl_next

    # `ReactiveModule.Extl` is a single label per step. The generated
    # `update` takes three curried args — `ctrl`, `extl_l`, `extl_n` — so
    # the RM's `Extl` is a pair `(extl_latched_tuple, extl_next_tuple)`:
    # `e.1` feeds `extl_l`, `e.2` feeds `extl_n`.
    extl_latched_native = _product_type(extl_latched)
    extl_next_native = _product_type(extl_next)
    extl_native = f"({extl_latched_native}) × ({extl_next_native})"
    ctrl_native = _product_type(ctrl_next)

    # Precomputed wire bindings for cert-local param names ("e" and "s").
    # Certificate predicates receive `e : Extl` (the pair above), so route
    # latched wires through `e.1.*` and next wires through `e.2.*`.
    e_bindings = {
        **_bind_wires([("e.1", extl_latched)]),
        **_bind_wires([("e.2", extl_next)]),
    }
    s_bindings = _bind_wires([("s", ctrl_next)])

    def _cert_body(terms, bindings):
        """Compile a certificate term list into a Lean function body."""
        output = [terms[-1].write[0]]
        return _translate_terms(terms, bindings, output, ctx.constants)

    const_names = ctx.constants.names()

    lines: list[str] = []

    lines.append("import Smt")
    lines.append("import Mathlib.Algebra.BigOperators.Fin")
    lines.append("import Core.Basic")
    lines.append(f"import {project_name}.{module_name}")
    lines.append("")
    lines.append("open Lean Elab Tactic Smt")
    lines.append("")

    # init_pre
    if init_pre_terms is not None:
        body = _cert_body(init_pre_terms, e_bindings)
        lines.append(f"def init_pre (e : {extl_native}) : Prop :=")
        lines.append(body)
    elif isinstance(cert_data.init_pre, str):
        lines.append(f"def init_pre : {extl_native} → Prop := {cert_data.init_pre}")
    else:
        lines.append(f"def init_pre (e : {extl_native}) : Prop := True")
    lines.append("")

    # update_pre
    if update_pre_terms is not None:
        body = _cert_body(update_pre_terms, e_bindings)
        lines.append(f"def update_pre (e : {extl_native}) : Prop :=")
        lines.append(body)
    elif isinstance(cert_data.update_pre, str):
        lines.append(
            f"def update_pre : {extl_native} → Prop := {cert_data.update_pre}"
        )
    else:
        lines.append(f"def update_pre (e : {extl_native}) : Prop := True")
    lines.append("")

    # inv
    if inv_terms is not None:
        body = _cert_body(inv_terms, s_bindings)
        lines.append(f"def inv (s : {ctrl_native}) : Prop :=")
        lines.append(body)
    elif isinstance(cert_data.inv, str):
        lines.append(f"def inv : {ctrl_native} → Prop := {cert_data.inv}")
    else:
        lines.append(f"def inv (s : {ctrl_native}) : Prop := True")
    lines.append("")

    # P
    if p_terms is not None:
        body = _cert_body(p_terms, s_bindings)
        lines.append(f"def P (s : {ctrl_native}) : Prop :=")
        lines.append(body)
    elif isinstance(cert_data.prp, str):
        lines.append(f"def P : {ctrl_native} → Prop := {cert_data.prp}")
    else:
        lines.append(f"def P (s : {ctrl_native}) : Prop := sorry")
    lines.append("")

    # DecidablePred P — must come after P and before ranking
    if p_terms is not None or isinstance(cert_data.prp, str):
        lines.append(
            "instance : DecidablePred P := fun s => by unfold P; dsimp; infer_instance"
        )
    else:
        lines.append("instance : DecidablePred P := sorry")
    lines.append("")

    # ranking
    if ranking_terms is not None:
        body = _cert_body(ranking_terms, s_bindings)
        lines.append(f"def ranking (s : {ctrl_native}) : Nat :=")
        lines.append(body)
    elif isinstance(cert_data.ranking, str):
        lines.append(f"def ranking : {ctrl_native} → Nat := {cert_data.ranking}")
    else:
        lines.append(f"def ranking (s : {ctrl_native}) : Nat := sorry")
    lines.append("")
    lines.append("")

    # ReactiveModule definition.
    # `init` takes only `extl_n`; unpack `e.2` for it. `update` takes
    # `ctrl extl_l extl_n` — feed `x`, `e.1`, `e.2` respectively.
    rm_noncomp = "noncomputable " if ctx.uses_real else ""
    lines.append(f"""\
{rm_noncomp}def RM : ReactiveModule ({extl_native}) ({ctrl_native}) := {{
    init := fun e => init e.2
    update := fun x e => update x e.1 e.2
    init_pre := init_pre
    update_pre := update_pre
}}
""")

    const_list = ", ".join(const_names) if const_names else ""

    # All definitions that need unfolding for proof automation
    # Note: ReactiveModule fields are handled by dsimp/simp, not unfold (universe polymorphic)
    unfold_defs = [
        "RM", "init", "update", "inv", "init_pre", "update_pre", "P", "ranking",
    ]
    unfold_defs.extend(const_names)

    unfold_list = ", ".join(f"``{d}" for d in unfold_defs)

    simp_mat = "MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply"
    simp_mat += ", mul_Mat_apply, add_Mat_apply"
    simp_mat += ", Bool.or_eq_true, decide_eq_true_eq"
    simp_mat += ", Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue"
    simp_mat += ", Fin.sum_univ_one, Fin.sum_univ_two, Fin.sum_univ_three"

    all_defs = "init, update, inv, init_pre, update_pre, P, ranking"
    if const_list:
        all_defs += f", {const_list}"

    lines.append(f"""\
-- Unfold all module and certificate definitions
elab "unfold_all" : tactic => do
  for f in [{unfold_list}] do
    try
      evalTactic (← `(tactic| unfold $(mkIdent f)))
    catch _ =>
      continue

-- Simplify matrix expressions to bare Int arithmetic
-- Phase 1: unfold RM projections and all definitions (no matrix reduction)
macro "unfold_mod" : tactic =>
  `(tactic| (
    first | (unfold RM at *; dsimp at *) | skip
    unfold_all; unfold_all; unfold_all))

-- Phase 2: reduce matrix arithmetic on the goal only (avoids exponential blowup on hypotheses)
macro "simp_mat" : tactic =>
  `(tactic| simp [{all_defs}, {simp_mat}])
""")

    lines.append("""\
syntax "zeroth_hammer" : tactic

elab_rules : tactic
  | `(tactic| zeroth_hammer) => do
      -- Pre-step: clear ReactiveModule wrappers if present
      try evalTactic (← `(tactic| unfold RM at *)) catch _ => pure ()
      -- 1. simp with all defs + matrix lemmas on goal, then omega
      try
        evalTactic (← `(tactic| simp_mat; omega))
        return
      catch _ => pure ()
      -- 2. case-split cascade — handles branching on ite
      try
        evalTactic (← `(tactic|
          simp_mat
          <;> first
            | omega
            | (simp_all; omega)
            | (split <;> simp_all <;> omega)
            | (split <;> split <;> simp_all <;> omega)
            | (split <;> split <;> split <;> simp_all <;> omega)))
        return
      catch _ => pure ()
      -- 3. smt fallback (cvc5) after full reduction
      try
        evalTactic (← `(tactic| simp_mat; smt))
        return
      catch _ => pure ()
      -- 4. bare smt
      try
        evalTactic (← `(tactic| smt))
        return
      catch _ => pure ()
      -- 5. last-resort fallback
      evalTactic (← `(tactic| sorry))
""")

    lines.append("""\

theorem init_inv : ∀ s, RM.init_pre s → inv (RM.init s) := by
   intro s hpre
   zeroth_hammer

theorem step_inv : ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e) := by
   intro s e ⟨hpre, hinv⟩
   zeroth_hammer
""")

    lines.append("section LTS\n")
    lines.append(f"{rm_noncomp}def lts := RM.toLTS'\n")

    lines.append("""\
theorem hinv' : lts.StateSet_isInductiveInitial inv := by
  unfold LTS'.StateSet_isInductiveInitial
  unfold LTS'.StateSet_isInductive
  constructor
  · intro s hs
    unfold lts at hs
    simp only [ReactiveModule.toLTS', ReactiveModule.LTS_init] at hs
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

theorem hrank : ∀ s s', (inv s ∧ ¬(P s) ∧ (∃ l, lts.Tr s l s')) →
    ranking s' < ranking s := by
    intro s s' ⟨hi, hP, htr⟩
    unfold lts at htr
    simp only [ReactiveModule.toLTS', ReactiveModule.LTS_update] at htr
    obtain ⟨l, hpre, heq⟩ := htr
    rw [← heq]
    zeroth_hammer

""")

    lines.append(f"""\
{rm_noncomp}def buchi := rule_buchi
  lts
  P
  inv
  hinv
  ranking
  hrank

""")

    lines.append("end LTS\n")

    return "\n".join(lines)
