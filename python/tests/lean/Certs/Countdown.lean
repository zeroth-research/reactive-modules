import Mathlib.Algebra.BigOperators.Fin
import Core.Basic
import Core.Box
import ZerothHammer



@[simp] def init (extl_n: Unit) : (Mat Int 1 1) :=
  let x0 : (Mat Int 1 1) := (fun _ _ => (100 : Int))
  let x1 : (Mat Int 1 1) := x0
  x1

@[simp] def update (ctrl: (Mat Int 1 1)) (extl_l: Unit) (extl_n: Unit) : (Mat Int 1 1) :=
  let x0 : (Mat Int 1 1) := (fun _ _ => (0 : Int))
  let x1 : (Mat Bool 1 1) := (fun _ _ => decide (ctrl 0 0 = x0 0 0))
  let x2 : (Mat Int 1 1) := (fun _ _ => (100 : Int))
  let x3 : (Mat Int 1 1) := (fun _ _ => (1 : Int))
  let x4 : (Mat Int 1 1) := (ctrl - x3)
  let x5 : (Mat Int 1 1) := (if x1 0 0 then x2 else x4)
  let x6 : (Mat Int 1 1) := x5
  x6


def init_pre (e : (Unit) × (Unit)) : Prop := True

def update_pre (e : (Unit) × (Unit)) : Prop := True

def inv : (Mat Int 1 1) → Prop := fun s => (((s 0 0) ≥ 0) ∧ ((s 0 0) ≤ 100))

def P : (Mat Int 1 1) → Prop := fun s => ((s 0 0) = 0)

instance : DecidablePred P := fun s => by unfold P; first | infer_instance | dsimp; infer_instance

def ranking : (Mat Int 1 1) → Nat := fun s => (((if ((s 0 0) = 0) then 0 else (s 0 0)) : Int)).toNat


def RM : ReactiveModule ((Unit) × (Unit)) ((Mat Int 1 1)) := {
    init := fun e => init e.2
    update := fun x e => update x e.1 e.2
    init_pre := init_pre
    update_pre := update_pre
}

-- Reduce matrix arithmetic on the goal only (avoids exponential blowup on hypotheses)
macro "simp_mat" : tactic =>
  `(tactic| simp [RM, init, update, inv, init_pre, update_pre, P, ranking, MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply, mul_Mat_apply, add_Mat_apply, Bool.or_eq_true, decide_eq_true_eq, Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue, Fin.sum_univ_one, Fin.sum_univ_two, Fin.sum_univ_three])

-- Unfold definitions everywhere (cheap: no matrix arithmetic reduction)
-- Gives omega/decide access to invariant and property conditions in hypotheses
macro "simp_defs" : tactic =>
  `(tactic| (simp only [RM, init, update, inv, init_pre, update_pre, P, ranking] at *; try dsimp at *))

-- Collapse Mat 1 1 types to scalars everywhere (cheap: no Fin.sum_univ)
-- Rewrites Mat 1 1 comparisons, ite through functions, Bool/decide normalization
macro "mat_collapse" : tactic =>
  `(tactic| simp only [Mat_1_1_lt_iff, Mat_1_1_le_iff, Mat_1_1_eq_iff, Mat_1_1_ne_iff,
                        ite_fun_apply,
                        decide_eq_true_eq, Bool.or_eq_true, Bool.and_eq_true,
                        Bool.not_eq_true'] at *)


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

section TS

def lts := RM.toTS

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

def buchi := rule_buchi
  lts
  P
  inv
  hinv
  ranking
  hrank


end TS
