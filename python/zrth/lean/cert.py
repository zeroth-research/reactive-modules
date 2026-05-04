from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zrth.lean.native import _product_type, _translate_terms
from zrth.lean.common import LeanContext, _bind_wires
from ..expr import Expr

if TYPE_CHECKING:
    from zrth import Module

_ZEROTH_HAMMER = """\
syntax "zeroth_hammer" : tactic

/-- Zeroth hammer: cascading automated prover for reactive module goals.
    Phase 0: simp + omega (fast, closes init_inv/step_inv)
    Phase 1: simp + case-split + omega (branching goals)
    Phase 2: full reduction + scalar collapse + case-split + omega (hrank)
    Phase 3: smt fallback (cvc5)
    Phase 4: sorry (explicit give-up) -/
elab_rules : tactic
  | `(tactic| zeroth_hammer) => do
      -- Phase 0: simp_mat + omega (closes init_inv, step_inv)
      try
        evalTactic (← `(tactic| simp_mat; omega))
        return
      catch _ => pure ()
      -- Phase 1: simp_mat + case-split cascade
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
      -- Phase 2: full pipeline for ranking proofs
      -- Unfold defs in hypotheses → reduce matrices in goal →
      -- collapse Mat 1 1 to scalar → case-split all ite → omega
      try evalTactic (← `(tactic| simp_defs)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      -- Check for contradictory hypotheses (e.g., ¬True from vacuous hrank)
      try evalTactic (← `(tactic| contradiction)); return catch _ => pure ()
      -- Try decide/native_decide after full reduction (works for finite Bool state)
      try evalTactic (← `(tactic| decide)); return catch _ => pure ()
      try evalTactic (← `(tactic| native_decide)); return catch _ => pure ()
      -- Reduce matrices and case-split
      try evalTactic (← `(tactic| simp_mat)) catch _ => pure ()
      try evalTactic (← `(tactic| mat_collapse)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      try
        evalTactic (← `(tactic|
          split_ifs at *
          <;> first | omega | simp_all | (simp_all; omega)))
        return
      catch _ => pure ()
      -- Phase 3: smt after full reduction
      try
        evalTactic (← `(tactic| smt))
        return
      catch _ => pure ()
      -- Phase 4: sorry (explicit give-up)
      evalTactic (← `(tactic| sorry))
"""

_LTS_THEOREMS = """\
theorem hinv' : lts.StateSet_isInductiveInitial inv := by
  constructor
  · intro s hs
    unfold lts at hs; simp only [RM, ReactiveModule.toLTS', ReactiveModule.LTS_init] at hs
    obtain ⟨l, hpre, hl⟩ := hs
    rw [← hl]
    exact init_inv l hpre
  · intro s s' ⟨hs, l, hstep⟩
    unfold lts at hstep; simp only [RM, ReactiveModule.toLTS', ReactiveModule.LTS_update] at hstep
    rw [← hstep.2]
    exact step_inv s l ⟨hstep.1, hs⟩

theorem hinv : lts.StateSet_isInvariant inv := by
  apply LTS'.StateSet_ind_init_is_inv lts
  exact hinv'

theorem hrank : ∀ s s', (inv s ∧ ¬(P s) ∧ (∃ l, lts.Tr s l s')) →
    ranking s' < ranking s := by
    intro s s' ⟨hi, hP, htr⟩
    unfold lts at htr; simp only [RM, ReactiveModule.toLTS', ReactiveModule.LTS_update] at htr
    obtain ⟨l, hpre, heq⟩ := htr
    rw [← heq]
    zeroth_hammer

"""


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
        lines.append(f"def update_pre : {extl_native} → Prop := {cert_data.update_pre}")
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
            "instance : DecidablePred P := fun s => by unfold P; first | infer_instance | dsimp; infer_instance"
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
        "RM",
        "init",
        "update",
        "inv",
        "init_pre",
        "update_pre",
        "P",
        "ranking",
    ]
    unfold_defs.extend(const_names)

    unfold_list = ", ".join(f"``{d}" for d in unfold_defs)

    simp_mat = "MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply"
    simp_mat += ", mul_Mat_apply, add_Mat_apply"
    simp_mat += ", Bool.or_eq_true, decide_eq_true_eq"
    simp_mat += ", Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue"
    simp_mat += ", Fin.sum_univ_one, Fin.sum_univ_two, Fin.sum_univ_three"

    all_defs = "RM, init, update, inv, init_pre, update_pre, P, ranking"
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

-- Reduce matrix arithmetic on the goal only (avoids exponential blowup on hypotheses)
macro "simp_mat" : tactic =>
  `(tactic| simp [{all_defs}, {simp_mat}])

-- Unfold definitions everywhere (cheap: no matrix arithmetic reduction)
-- Gives omega/decide access to invariant and property conditions in hypotheses
macro "simp_defs" : tactic =>
  `(tactic| (simp only [{all_defs}] at *; try dsimp at *))

-- Collapse Mat 1 1 types to scalars everywhere (cheap: no Fin.sum_univ)
-- Rewrites Mat 1 1 comparisons, ite through functions, Bool/decide normalization
macro "mat_collapse" : tactic =>
  `(tactic| simp only [Mat_1_1_lt_iff, Mat_1_1_le_iff, Mat_1_1_eq_iff, Mat_1_1_ne_iff,
                        ite_fun_apply,
                        decide_eq_true_eq, Bool.or_eq_true, Bool.and_eq_true,
                        Bool.not_eq_true'] at *)
""")

    lines.append(_ZEROTH_HAMMER)

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

    lines.append(_LTS_THEOREMS)

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


def smt_predicates_to_lean(cert_data: CertificateData, module: "Module") -> CertificateData:
    """Translate SMT-LIB string fields in *cert_data* to Lean expression strings.

    None and compiled term-list fields pass through unchanged.
    """
    fields = (cert_data.prp, cert_data.inv, cert_data.init_pre, cert_data.update_pre, cert_data.ranking)
    if not any(isinstance(f, str) for f in fields):
        return cert_data

    # Lazy imports keep cvc5 off the critical path when this function is unused.
    import cvc5

    from .smt_module import ModuleSMT
    from .smt_prompt import CegarPromptEnv, parse_predicate
    from .smt_to_lean import smt_to_lean, smt_to_lean_nat

    tm = cvc5.TermManager()
    msmt = ModuleSMT(tm=tm, module=module)
    env = CegarPromptEnv(msmt)

    def translate(smt_src: str | None, mode: str) -> str | None:
        if not isinstance(smt_src, str):
            return smt_src
        term = parse_predicate(env, smt_src)
        if mode in ("property", "invariant"):
            return smt_to_lean(term, msmt.ctrl_next, param_name="s")
        if mode == "ranking":
            return smt_to_lean_nat(term, msmt.ctrl_next, param_name="s")
        # preconditions
        return smt_to_lean(
            term,
            state_wires=[],
            param_name="e",
            extra=[
                ("e", "e.2", msmt.extl_next),
                ("el", "e.1", msmt.extl_latched),
            ],
        )

    return CertificateData(
        prp=translate(cert_data.prp, "property"),
        init_pre=translate(cert_data.init_pre, "pre"),
        update_pre=translate(cert_data.update_pre, "pre"),
        inv=translate(cert_data.inv, "invariant"),
        ranking=translate(cert_data.ranking, "ranking"),
    )
