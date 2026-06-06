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
    Phase 0: simp alone (closes trivially-True invariants)
    Phase 1: fast arithmetic — omega, norm_cast+omega, simp+omega, simp+linarith
    Phase 2: push_neg + simp + omega (negated arithmetic)
    Phase 3: simp + deep case-split + omega/linarith/norm_cast (branching goals)
    Phase 4: simp_defs + case-split (unfold defs before splitting)
    Phase 5: full reduction + mat_collapse + split_ifs + omega/linarith (hrank)
    Phase 6: aesop (general-purpose proof search)
    Phase 7: smt fallback (cvc5)
    Phase 8: sorry (explicit give-up) -/
elab_rules : tactic
  | `(tactic| zeroth_hammer) => do
      -- Phase 0: simp_mat alone (closes trivial True goals without needing omega)
      -- Note: simp never throws when it makes partial progress, so we must
      -- check goals explicitly rather than relying on try/return/catch.
      try evalTactic (← `(tactic| simp_mat)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      -- Phase 1: fast arithmetic passes
      -- 1a: omega alone (goal already in linear arithmetic fragment after intros)
      try evalTactic (← `(tactic| omega)); return catch _ => pure ()
      -- 1b: norm_cast + omega (normalises Nat/Int coercions, e.g. Int.toNat in rankings)
      try evalTactic (← `(tactic| norm_cast <;> omega)); return catch _ => pure ()
      -- 1c: simp_mat + omega (main fast path)
      try
        evalTactic (← `(tactic| simp_mat <;> omega))
        return
      catch _ => pure ()
      -- 1d: simp_mat + linarith (ordered-ring arithmetic; fallback when omega is too weak)
      try
        evalTactic (← `(tactic| simp_mat <;> linarith))
        return
      catch _ => pure ()
      -- Phase 2: push_neg normalises negated arithmetic before omega
      -- (e.g. ¬(x > 0) → x ≤ 0; useful when inv or P contains negated comparisons)
      try
        evalTactic (← `(tactic| push_neg; simp_mat <;> omega))
        return
      catch _ => pure ()
      -- Phase 3: simp_mat + deep case-split cascade (branching state machines)
      -- Four levels of if-branch splitting; tries omega/linarith/norm_cast at each leaf
      try
        evalTactic (← `(tactic|
          simp_mat
          <;> first
            | omega
            | linarith
            | (norm_cast; omega)
            | (push_neg; omega)
            | (simp_all; omega)
            | (split <;> simp_all <;> omega)
            | (split <;> split <;> simp_all <;> omega)
            | (split <;> split <;> split <;> simp_all <;> omega)
            | (split <;> split <;> split <;> split <;> simp_all <;> omega)))
        return
      catch _ => pure ()
      -- Phase 4: unfold definitions everywhere first, then case-split
      -- Useful when inv/P/ranking are not yet visible to the split heuristic
      try
        evalTactic (← `(tactic|
          simp_defs
          <;> first
            | omega
            | linarith
            | (norm_cast; omega)
            | (simp_all; omega)
            | (split <;> simp_all <;> omega)
            | (split <;> split <;> simp_all <;> omega)
            | (split <;> split <;> split <;> simp_all <;> omega)))
        return
      catch _ => pure ()
      -- Phase 5: full pipeline for ranking proofs
      -- Unfold defs in hypotheses → reduce matrices → collapse Mat 1 1 to scalar
      -- → split all ite → omega / linarith / norm_cast
      try evalTactic (← `(tactic| simp_defs)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      -- Check for contradictory hypotheses (e.g. ¬True from vacuous hrank)
      try evalTactic (← `(tactic| contradiction)); return catch _ => pure ()
      -- Try decide/native_decide after full reduction (works for finite Bool state)
      try evalTactic (← `(tactic| decide)); return catch _ => pure ()
      try evalTactic (← `(tactic| native_decide)); return catch _ => pure ()
      -- Reduce matrices and collapse Mat 1 1 to bare scalar arithmetic
      try evalTactic (← `(tactic| simp_mat)) catch _ => pure ()
      try evalTactic (← `(tactic| mat_collapse)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      try
        evalTactic (← `(tactic|
          split_ifs at *
          <;> first
            | omega
            | linarith
            | (norm_cast; omega)
            | (norm_cast; linarith)
            | simp_all
            | (simp_all; omega)
            | (simp_all; linarith)
            | positivity))
        return
      catch _ => pure ()
      -- Phase 6: aesop (general-purpose proof search before SMT)
      try
        evalTactic (← `(tactic| aesop))
        return
      catch _ => pure ()
      -- Phase 7: smt after full reduction
      try
        evalTactic (← `(tactic| smt))
        return
      catch _ => pure ()
      -- Phase 8: sorry (explicit give-up)
      evalTactic (← `(tactic| sorry))
"""


def generate_zeroth_hammer_lean() -> str:
    """Generate a standalone ZerothHammer.lean with only the zeroth_hammer tactic.

    The tactic references ``simp_mat``, ``simp_defs``, and ``mat_collapse`` by
    name.  Importers must define those macros for their specific module before
    invoking ``zeroth_hammer``.
    """
    return "\n".join(
        [
            "import Lean",
            "import Mathlib.Tactic",
            "import Smt",
            "",
            "open Lean Elab Tactic",
            "",
            "-- Default stub macros; certificate files redefine these for their specific module.",
            'macro "simp_mat"     : tactic => `(tactic| simp)',
            'macro "simp_defs"    : tactic => `(tactic| simp only [])',
            'macro "mat_collapse" : tactic => `(tactic| simp only [])',
            "",
            _ZEROTH_HAMMER,
        ]
    )


_TS_THEOREMS = """\
theorem hinv' : lts.StateSet_isInductiveInitial inv := by
  constructor
  · intro s hs
    unfold lts at hs; simp only [RM, ReactiveModule.toTS, ReactiveModule.TS_init] at hs
    obtain ⟨l, hpre, hl⟩ := hs
    rw [← hl]
    exact init_inv l hpre
  · intro s s' ⟨hs, l, hstep⟩
    unfold lts at hstep; simp only [RM, ReactiveModule.toTS, ReactiveModule.TS_update] at hstep
    rw [← hstep.2]
    exact step_inv s l ⟨hstep.1, hs⟩

theorem hinv : lts.StateSet_isInvariant inv := by
  apply TS.StateSet_ind_init_is_inv lts
  exact hinv'

theorem hrank : ∀ s s', (inv s ∧ ¬(P s) ∧ (∃ l, lts.Tr s l s')) →
    ranking s' < ranking s := by
    intro s s' ⟨hi, hP, htr⟩
    unfold lts at htr; simp only [RM, ReactiveModule.toTS, ReactiveModule.TS_update] at htr
    obtain ⟨l, hpre, heq⟩ := htr
    rw [← heq]
    simp_defs
    simp_mat
    split_ifs <;> first | omega | (norm_cast; omega)

"""


@dataclass
class CertificateData:
    """Data needed to generate a Lean certificate."""

    prp: str | Expr | None = None
    inv: Expr | str | None = None

    init_pre: Expr | str | None = None
    update_pre: Expr | str | None = None
    ranking: Expr | str | None = None


def _cert_def_lines(
    ctx: LeanContext,
    cert_data: CertificateData,
) -> list[str]:
    """Emit the five certificate definition lines (init_pre, update_pre, inv, P, ranking).

    Shared by ``generate_data_lean`` and the inline path of ``generate_certificate_lean``.
    """
    def _as_terms(v):
        return v if isinstance(v, list) else None

    inv_terms = _as_terms(cert_data.inv)
    init_pre_terms = _as_terms(cert_data.init_pre)
    update_pre_terms = _as_terms(cert_data.update_pre)
    ranking_terms = _as_terms(cert_data.ranking)
    p_terms = _as_terms(cert_data.prp)

    extl_latched = ctx.extl_latched
    extl_next = ctx.extl_next
    ctrl_next = ctx.ctrl_next

    extl_latched_native = _product_type(extl_latched)
    extl_next_native = _product_type(extl_next)
    extl_native = f"({extl_latched_native}) × ({extl_next_native})"
    ctrl_native = _product_type(ctrl_next)

    e_bindings = {
        **_bind_wires([("e.1", extl_latched)]),
        **_bind_wires([("e.2", extl_next)]),
    }
    s_bindings = _bind_wires([("s", ctrl_next)])

    def _cert_body(terms, bindings):
        output = [terms[-1].write[0]]
        return _translate_terms(terms, bindings, output, ctx.constants)

    lines: list[str] = []

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

    # DecidablePred P
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

    return lines


def generate_data_lean(
    project_name: str,
    module_name: str,
    ctx: LeanContext,
    cert_data: CertificateData | None = None,
) -> str:
    """Generate XXXData.lean content: init_pre, update_pre, inv, P, ranking.

    Imports only ``Core.Basic`` — no dependency on the module's init/update.
    ``Certificate.lean`` imports this file so that only the data file needs to
    be regenerated when the certificate data changes (e.g. after --infer).
    """
    if cert_data is None:
        cert_data = CertificateData()
    lines: list[str] = [
        "import Core.Basic",
        "",
        *_cert_def_lines(ctx, cert_data),
    ]
    return "\n".join(lines)


def generate_certificate_lean(
    project_name: str,
    module_name: str,
    ctx: LeanContext,
    cert_data: CertificateData | None = None,
    *,
    hammer_import: str | None = None,
    module_inline: str | None = None,
    data_import: str | None = None,
) -> str:
    """Generate Certificate.lean.

    Args:
        hammer_import: When set (e.g. ``"ZerothHammer"``), add
            ``import <hammer_import>`` and omit the inlined ``zeroth_hammer``
            definition.
        module_inline: When set, inline this Lean source (the ``init`` /
            ``update`` functions) instead of emitting
            ``import {project_name}.{module_name}``.
        data_import: When set (e.g. ``"Rea.ReaData"``), import that module
            for ``init_pre``, ``inv``, ``P``, ``ranking`` instead of emitting
            them inline.  Use this for project mode so only ``XXXData.lean``
            needs regeneration when cert_data changes.
    """
    if cert_data is None:
        cert_data = CertificateData()

    extl_latched = ctx.extl_latched
    extl_next = ctx.extl_next
    ctrl_next = ctx.ctrl_next

    extl_latched_native = _product_type(extl_latched)
    extl_next_native = _product_type(extl_next)
    extl_native = f"({extl_latched_native}) × ({extl_next_native})"
    ctrl_native = _product_type(ctrl_next)

    const_names = ctx.constants.names()

    lines: list[str] = []

    lines.append("import Mathlib.Algebra.BigOperators.Fin")
    lines.append("import Core.Basic")
    if module_inline is not None:
        lines.append("import Core.Box")
    elif data_import is not None:
        lines.append(f"import {project_name}.{module_name}")
        lines.append(f"import {data_import}")
    else:
        lines.append(f"import {project_name}.{module_name}")
    if hammer_import is not None:
        lines.append(f"import {hammer_import}")
    else:
        lines.append("import Smt")
    lines.append("")
    lines.append("")
    if module_inline is not None:
        lines.append(module_inline)
        lines.append("")

    if data_import is None:
        # Inline definitions (standalone or legacy path)
        lines.extend(_cert_def_lines(ctx, cert_data))
        lines.append("")
        has_ranking = cert_data.ranking is not None
    else:
        # Definitions live in data_import; always include ranking in simp set
        has_ranking = True

    # ReactiveModule definition.
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

    simp_mat = "MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply"
    simp_mat += ", mul_Mat_apply, add_Mat_apply"
    simp_mat += ", Bool.or_eq_true, decide_eq_true_eq"
    simp_mat += ", Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue"
    simp_mat += ", Fin.sum_univ_one, Fin.sum_univ_two, Fin.sum_univ_three"

    all_defs = "RM, init, update, inv, init_pre, update_pre, P"
    if has_ranking:
        all_defs += ", ranking"
    if const_list:
        all_defs += f", {const_list}"

    lines.append(f"""\
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

    if hammer_import is None:
        lines.append(_ZEROTH_HAMMER)

    lines.append("""\

theorem init_inv : ∀ s, RM.init_pre s → inv (RM.init s) := by
   intro s hpre
   try simp_mat
   try simp_defs
   try split_ifs <;> omega

theorem step_inv : ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e) := by
   intro s e ⟨hpre, hinv⟩
   try simp_defs
   try simp_mat
   try split_ifs <;> omega
""")

    lines.append("section TS\n")
    lines.append(f"{rm_noncomp}def lts := RM.toTS\n")

    lines.append(_TS_THEOREMS)

    lines.append(f"""\
{rm_noncomp}def buchi := rule_buchi
  lts
  P
  inv
  hinv
  ranking
  hrank

""")

    lines.append("end TS\n")

    return "\n".join(lines)


def smt_predicates_to_lean(
    cert_data: CertificateData, module: "Module"
) -> CertificateData:
    """Translate SMT-LIB string fields in *cert_data* to Lean expression strings.

    None and compiled term-list fields pass through unchanged.
    """
    fields = (
        cert_data.prp,
        cert_data.inv,
        cert_data.init_pre,
        cert_data.update_pre,
        cert_data.ranking,
    )
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
