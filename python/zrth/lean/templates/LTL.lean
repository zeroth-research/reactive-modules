import Cslib.Foundations.Semantics.LTS.Basic

/-  ----------------------------------------------------------------
 LTL formulas
 ----------------------------------------------------------------- -/

inductive LTLFormula {State: Type u} where
  | AP: (State → Prop) → LTLFormula -- Atomic proposition
  | Next: LTLFormula → LTLFormula
  | Until: LTLFormula → LTLFormula → LTLFormula
  | Or: LTLFormula → LTLFormula → LTLFormula
  -- we do not need both And and Or, but it is more comfortable to have them both
  | And: LTLFormula → LTLFormula → LTLFormula
  | Not: LTLFormula → LTLFormula

namespace LTLFormula


instance : Coe (State → Prop) (@LTLFormula State) := ⟨AP⟩

/- Shallow embedding of LTL -/
namespace Shallow

def AP (prop: State → Prop) (trace: Cslib.ωSequence State): Prop
  := prop (trace.get 0)

def Next (F: Cslib.ωSequence State → Prop) (trace: Cslib.ωSequence State): Prop
  := F (trace.tail)

def Until (A B: Cslib.ωSequence State → Prop) (trace: Cslib.ωSequence State): Prop
  := ∃ n: Nat, B (trace.drop n) ∧ (∀ j < n, A (trace.drop j))

def Or (A B: Cslib.ωSequence State → Prop) (trace: Cslib.ωSequence State): Prop
  := (A trace) ∨ (B trace)

def And (A B: Cslib.ωSequence State → Prop) (trace: Cslib.ωSequence State): Prop
  := (A trace) ∧ (B trace)

def Not (F: Cslib.ωSequence State → Prop) (trace: Cslib.ωSequence State): Prop
  := ¬(F trace)

end Shallow

/-- Translate LTL formula into its shallow embedding -/
def toShallow {State: Type u} (F: @LTLFormula State):
  Cslib.ωSequence State → Prop :=
 match F with
 | AP prop => Shallow.AP prop
 | Next F' => Shallow.Next (F'.toShallow)
 | Until A B => Shallow.Until (A.toShallow) (B.toShallow)
 | Or A B => Shallow.Or (A.toShallow) (B.toShallow)
 | And A B => Shallow.And (A.toShallow) (B.toShallow)
 | Not A => Shallow.Not (A.toShallow)


/-- Semantics of LTL formulas wrt. arbitrary point in the trace -/
def semₙ {State: Type u} (F: @LTLFormula State) (trace: Cslib.ωSequence State) (i: Nat) : Prop :=
  match F with
  | AP prop => prop (trace.get i)
  | Next F' => F'.semₙ trace (i + 1)
  | Until A B => ∃ n: Nat, i ≤ n ∧ (B.semₙ trace n) ∧
                    (∀ j, i ≤ j ∧ j < n → A.semₙ trace j)
  | Or A B => (A.semₙ trace i) ∨ (B.semₙ trace i)
  | And A B => (A.semₙ trace i) ∧ (B.semₙ trace i)
  | Not F' => ¬ (F'.semₙ trace i)

/-- Semantics of LTL formulas wrt. initial point in the trace -/
def sem {State: Type u} (F: @LTLFormula State) (trace: Cslib.ωSequence State) : Prop :=
  match F with
  | AP prop => prop (trace.get 0)
  | Next F' => F'.sem trace.tail
  | Until A B => ∃ n: Nat, (B.sem (trace.drop n)) ∧ (∀ j < n, A.sem (trace.drop j))
  | Or A B => (A.sem trace) ∨ (B.sem trace)
  | And A B => (A.sem trace) ∧ (B.sem trace)
  | Not F' => ¬ (F'.sem trace)

/-- shallow embedding and the initial semantics of LTLFormulas are the same -/
theorem shallow_sem: ∀ F: (@LTLFormula State), F.toShallow = F.sem := by
  intro F
  induction F
  case AP => rfl
  case Next =>
    simp [toShallow, sem]
    unfold Shallow.Next
    simp_all
  case Until =>
    simp [toShallow, sem]
    unfold Shallow.Until
    simp_all
  case Or =>
    simp [toShallow, sem]
    unfold Shallow.Or
    simp_all
  case And =>
    simp [toShallow, sem]
    unfold Shallow.And
    simp_all
  case Not =>
    simp [toShallow, sem]
    unfold Shallow.Not
    simp_all

lemma drop_get_zero (ss: Cslib.ωSequence α)
  : ∀ n: Nat, ss.get n = (ss.drop n).get 0 := by
    intro n
    simp [Cslib.ωSequence.drop]
    rfl

lemma drop_get_n (ss: Cslib.ωSequence α)
  : ∀ n x: Nat, ss.get (n + x) = (ss.drop n).get x := by
    intro n x
    simp_all [Cslib.ωSequence.drop]
    rw [Nat.add_comm]
    rfl

-- lemma semₙ_tail: ∀ F: (@LTLFormula State), F.semₙ tr.tail n ↔ F.semₙ tr (n + 1) := by
--   intro F
--   induction F generalizing n with
--   | AP a =>
--     simp [semₙ]
--     simp [Cslib.ωSequence.tail]
--     rfl
--   | Next a ih =>
--     simp [semₙ]
--     exact ih (n := n + 1)
--   | Or a b iha ihb =>
--     simp [semₙ]
--     exact ⟨fun h => h.elim (Or.inl ∘ iha.mp) (Or.inr ∘ ihb.mp),
--            fun h => h.elim (Or.inl ∘ iha.mpr) (Or.inr ∘ ihb.mpr)⟩
--   | And a b iha ihb =>
--     simp [semₙ]
--     exact ⟨fun ⟨ha, hb⟩ => ⟨iha.mp ha, ihb.mp hb⟩,
--            fun ⟨ha, hb⟩ => ⟨iha.mpr ha, ihb.mpr hb⟩⟩
--   | Not a ih => simp [semₙ]; exact ⟨fun h c => h (ih.mpr c), fun h c => h (ih.mp c)⟩
--   | Until a b iha ihb =>
--     simp [semₙ]
--     constructor
--     · rintro ⟨m, hm, hb, ha⟩
--       refine ⟨m + 1, by omega, ihb.mp hb, ?_⟩
--       refine fun j a a_1 => ?_
--       have ha_j1 := ha (j - 1) (by omega) (by omega)
--       have iha_j1 := @iha (j - 1)
--       rw [iha_j1] at ha_j1
--       have : j - 1 + 1 = j := by omega
--       rw [this] at ha_j1
--       exact ha_j1
--     · rintro ⟨m, hm, hb, ha⟩
--       refine ⟨m - 1, by omega, ?_, fun j hj1 hj2 => ?_⟩
--       · rw [ihb]
--         have : m - 1 + 1 = m := by omega
--         rwa [this]
--       · rw [iha]
--         have : j + 1 < m := by omega
--         have : n < j + 1 := by omega
--         exact ha (j + 1) this (by omega)

theorem sem_semₙ_suffixes:
∀ F: (@LTLFormula State), F.sem (tr.drop n) ↔ F.semₙ tr n := by
  intro F
  induction F generalizing tr n
  case AP a =>
    simp_all [sem, semₙ]
    simp [drop_get_zero]
  case Or => simp_all [sem, semₙ]
  case And => simp_all [sem, semₙ]
  case Not => simp_all [sem, semₙ]
  case Next a ih =>
    simp_all [sem, semₙ]
  case Until a ah ah2 =>
    simp_all [sem, semₙ]
    constructor
    . rintro ⟨m, hb, ha⟩
      refine ⟨n + m, by omega, ?_, fun j hj1 hj2 => ?_⟩
      · exact hb
      · have := ha (j - n) (by omega)
        rw [show n + (j - n) = j from by omega] at this
        exact this
    · rintro ⟨m, hm, hb, ha⟩
      refine ⟨m - n, ?_, fun j hj => ?_⟩
      ·
        rw [show n + (m - n) = m from by omega]
        exact hb
      ·
        rw [show n + j = n + j from rfl]
        exact ha (n + j) (by omega) (by omega)


/-- The relation of initial and arbitrary-point semantics of LTL -/
theorem sem_semₙ:
∀ F: (@LTLFormula State), F.sem tr ↔ F.semₙ tr 0 := by
  intro F
  have := @sem_semₙ_suffixes State tr 0 F
  rw [Cslib.ωSequence.drop_zero] at this
  exact this

/- classical notation and abbreviations -/
section Notation

scoped notation:50 φ " U " ψ => Until φ ψ
scoped prefix:75 "X" => Next

-- Standard logic notation
scoped infixl:65 " ∧' " => And
scoped infixl:60 " ∨' " => Or
scoped prefix:80 "¬' " => Not

-- Derived operators
abbrev Finally (φ : @LTLFormula State) : @LTLFormula State
  := (AP fun _ => True) U φ
abbrev Globally (φ : @LTLFormula State) : @LTLFormula State
  := Not (Finally (Not φ))

scoped prefix:75 "F" => Finally
scoped prefix:75 "G" => Globally

-- Semantic notation: trace, i ⊨ φ
scoped notation:30 tr " ⊧ " φ => LTLFormula.sem φ tr
end Notation

theorem Globally.explicit (ψ: @LTLFormula State) (tr: Cslib.ωSequence State):
  ((Globally ψ).sem tr) ↔ (∀ i, ψ.sem (tr.drop i)) := by
  constructor
  case mp =>
    unfold Globally
    rw [sem]
    rw [Finally]
    rw [sem]
    simp_all
    intro h x
    have hh := h x
    rw [sem] at hh
    simp [sem] at hh
    exact hh
  case mpr =>
    unfold Globally
    rw [sem]
    rw [Finally]
    rw [sem]
    simp_all
    intro h x hs
    simp [sem]
    rw [sem] at hs
    have A := h x
    contradiction

theorem Finally.explicit (ψ: @LTLFormula State) (tr: Cslib.ωSequence State):
  ((Finally ψ).sem tr) ↔ (∃ i, ψ.sem (tr.drop i)) := by
  simp [sem]

end LTLFormula

/-  ----------------------------------------------------------------
 Examples
 ----------------------------------------------------------------- -/
section Examples

open LTLFormula

def T: Cslib.ωSequence Nat := ⟨ fun (i: Nat) => if i % 2 = 0 then 0 else 1 ⟩
def Ax := fun x => x > 0
def Ay := fun x => x > 1

example : T ⊧ ¬'Ax := by
  unfold T
  simp_all [LTLFormula.sem]
  simp [Ax]

example : T ⊧ X Ax := by
  unfold T
  simp_all [Cslib.ωSequence.tail, LTLFormula.sem]
  simp [Ax]

def Az := fun x => x ≤ 1

example : T ⊧ G Az := by
  unfold T
  simp_all [LTLFormula.sem]
  simp [Az]
  intro x
  by_cases x % 2 = 0
  . simp_all [Cslib.ωSequence.drop]
  . simp_all [Cslib.ωSequence.drop]

end Examples
