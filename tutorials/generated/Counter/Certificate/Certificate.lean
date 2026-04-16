import Smt
import Mathlib.Algebra.BigOperators.Fin
import Core.Basic
import Counter.Counter

open Lean Elab Tactic Smt

@[simp] def c6 : (Mat Int 1 3) := fun i j =>
  match i, j with
  | 0, 0 => 1 | 0, 1 => 0 | 0, 2 => 0

@[simp] def c7 : (Mat Int 1 3) := fun i j =>
  match i, j with
  | 0, 0 => 0 | 0, 1 => 1 | 0, 2 => 0

@[simp] def c8 : (Mat Int 1 3) := fun i j =>
  match i, j with
  | 0, 0 => 0 | 0, 1 => 0 | 0, 2 => 1

@[simp] def c9 : (Mat Int 1 1) := fun i j =>
  match i, j with
  | 0, 0 => 0

@[simp] def c10 : (Mat Int 1 2) := fun i j =>
  match i, j with
  | 0, 0 => 1 | 0, 1 => 0

@[simp] def c11 : (Mat Int 1 2) := fun i j =>
  match i, j with
  | 0, 0 => 0 | 0, 1 => 1

@[simp] def c12 : (Mat Int 1 1) := fun i j =>
  match i, j with
  | 0, 0 => 0

@[simp] def c13 : (Mat Int 1 3) := fun i j =>
  match i, j with
  | 0, 0 => 1 | 0, 1 => 0 | 0, 2 => 0

@[simp] def c14 : (Mat Int 1 3) := fun i j =>
  match i, j with
  | 0, 0 => 0 | 0, 1 => 1 | 0, 2 => 0

@[simp] def c15 : (Mat Int 1 3) := fun i j =>
  match i, j with
  | 0, 0 => 0 | 0, 1 => 0 | 0, 2 => 1

@[simp] def c16 : (Mat Int 1 1) := fun i j =>
  match i, j with
  | 0, 0 => 0

@[simp] def c17 : (Mat Int 1 1) := fun i j =>
  match i, j with
  | 0, 0 => 1

@[simp] def c18 : (Mat Int 1 3) := fun i j =>
  match i, j with
  | 0, 0 => 1 | 0, 1 => 0 | 0, 2 => 0

@[simp] def c19 : (Mat Int 1 1) := fun i j =>
  match i, j with
  | 0, 0 => 0


def init_pre (e : (Mat Int 2 1)) : Prop :=
  let x0 := c10
  let x1 := MatMul x0 e
  let x2 := c11
  let x3 := MatMul x2 e
  let x4 := c12
  let x5 := decide (x4 0 0 ≤ x1 0 0)
  let x6 := decide (x4 0 0 ≤ x3 0 0)
  let x7 := (x5 && x6)
  x7

def update_pre (e : (Mat Int 2 1)) : Prop := True

def inv (s : (Mat Int 3 1)) : Prop :=
  let x0 := c6
  let x1 := MatMul x0 s
  let x2 := c7
  let x3 := MatMul x2 s
  let x4 := c8
  let x5 := MatMul x4 s
  let x6 := c9
  let x7 := decide (x6 0 0 ≤ x1 0 0)
  let x8 := decide (x1 0 0 ≤ x3 0 0)
  let x9 := decide (x1 0 0 ≤ x5 0 0)
  let x10 := (x8 || x9)
  let x11 := (x7 && x10)
  x11

def P (s : (Mat Int 3 1)) : Prop :=
  let x0 := c18
  let x1 := MatMul x0 s
  let x2 := c19
  let x3 := (x1 == x2)
  x3

instance : DecidablePred P := fun s => by unfold P; dsimp; infer_instance

def ranking (s : (Mat Int 3 1)) : Nat :=
  let x0 := c13
  let x1 := MatMul x0 s
  let x2 := c14
  let x3 := MatMul x2 s
  let x4 := c15
  let x5 := MatMul x4 s
  let x6 := c16
  let x7 := (x1 == x6)
  let x8 := decide (x5 0 0 ≤ x3 0 0)
  let x9 := if x8 then x3 else x5
  let x10 := (x9 - x1)
  let x11 := c17
  let x12 := (x10 + x11)
  let x13 := if x7 then x6 else x12
  let x14 := (x13 0 0)
  let x15 := (x14).toNat
  x15


def RM : ReactiveModule ((Mat Int 2 1)) ((Mat Int 3 1)) := {
    init := init
    update := fun x e => update (x, e)
    init_pre := init_pre
    update_pre := update_pre
}

-- Unfold all module and certificate definitions
elab "unfold_all" : tactic => do
  for f in [``RM, ``init, ``update, ``inv, ``init_pre, ``update_pre, ``P, ``ranking, ``c0, ``c1, ``c2, ``c3, ``c4, ``c5, ``c6, ``c7, ``c8, ``c9, ``c10, ``c11, ``c12, ``c13, ``c14, ``c15, ``c16, ``c17, ``c18, ``c19] do
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
  `(tactic| simp [init, update, inv, init_pre, update_pre, P, ranking, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply, mul_Mat_apply, add_Mat_apply, Bool.or_eq_true, decide_eq_true_eq, Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue, Fin.sum_univ_one, Fin.sum_univ_two, Fin.sum_univ_three])

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
      evalTactic (← `(tactic| smt))


theorem init_inv : ∀ s, RM.init_pre s → inv (RM.init s) := by
   intro s hpre
   zeroth_hammer

theorem step_inv : ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e) := by
   intro s e ⟨hpre, hinv⟩
   zeroth_hammer

section LTS

def lts := RM.toLTS'

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

theorem hrank : ∀ s s', (inv s ∧ ¬(P s) ∧ (∃ l, lts.Tr s l s')) →
    ranking s' < ranking s := by
    intro s s' ⟨hi, hP, htr⟩
    unfold lts at htr
    simp only [ReactiveModule.toLTS', ReactiveModule.LTS_update] at htr
    obtain ⟨l, hpre, heq⟩ := htr
    rw [← heq]
    zeroth_hammer

def buchi := rule_buchi
  lts
  P
  inv
  hinv
  ranking
  hrank


end LTS
