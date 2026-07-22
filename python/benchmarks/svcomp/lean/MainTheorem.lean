import Init.Data.Fin
import Init.Classical

/-!
The abstract coverage meta-theorem, parameterised over a program's state type
and its pieces — invariant `Φ`, transition relation `Trans`, activation cell
family `C'` (over the *pair* `(s, s')`, since the rank is read at both ends of a
step), decrease margin `δ`, ranking function `V`.

Two filters, matching the emitted artifacts:
  `Φ s`        the inductive invariant (external)   — emitted `invariants`
  `Trans s s'` the loop-path relation               — emitted `guard`

`Trans` is the *whole* path relation (the composed sequence of guard-assumes,
assignments, and havocs) — one object, exactly as the CFA composes it. The loop
guard is its domain; we do not split it out. `Φ` is the only thing kept separate,
because it is genuinely external to the path (a Houdini-inferred fact).

`mainCoverageTheorem` reduces the decrease obligation to (1) coverage and (2) a
per-cell sufficient condition — exactly what our emitted per-path proofs produce
(`<path>_covered` ↦ coverage, `cellK_decrease` ↦ the sufficient condition).
-/

section
variable {State : Type}
variable {k : Nat}
variable (Φ : State → Prop)
variable (Trans : State → State → Prop)
variable (C' : Fin k → State → State → Prop)
variable (δ : Int)
variable (V : State → Int)

/-- ## Preliminaries -/

theorem deMorgan₁ {s s' : State} : ¬ (∃ i : Fin k , C' i s s') → ∀ i, ¬ C' i s s' := by
  intro ϑ i h
  exact ϑ ⟨i, h⟩

theorem deMorgan₂ {s s' : State} : (∀ i, ¬ C' i s s') → ¬ (∃ i : Fin k , C' i s s') := by
  intro ϑ ⟨i, h⟩
  exact ϑ i h

/- ## The coverage condition

Every guarded reachable step `(s, s')` lands in some certified cell. -/

/-- The main form of the coverage condition. -/
def Coverage₁ : Prop :=
  ∀ s s' : State, (Φ s ∧ Trans s s') → ∃ i : Fin k , C' i s s'

/-- By applying de Morgan duality, we obtain `Coverage₂`. -/
def Coverage₂ : Prop :=
  ∀ s s' : State, (Φ s ∧ Trans s s') → ¬ (∀ i : Fin k , ¬ C' i s s')

theorem coverage₁_iff_coverage₂ : Coverage₁ Φ Trans C' ↔ Coverage₂ Φ Trans C' := by
  apply Iff.intro
  . intro cov s s' h
    exact Classical.not_forall_not.mpr (cov s s' h)
  . intro cov s s' h
    exact Classical.not_forall_not.mp (cov s s' h)

/-- By the law of contraposition, a third equivalent form. -/
def Coverage₃ : Prop :=
  ∀ s s' : State, ¬ ¬ (∀ i : Fin k , ¬ C' i s s') → ¬ (Φ s ∧ Trans s s')

theorem coverage₃_iff_coverage₂ : Coverage₃ Φ Trans C' ↔ Coverage₂ Φ Trans C' := by
  apply Iff.intro
  . intro cov₃ s s' h
    apply Classical.byContradiction
    intro ϑ
    exact cov₃ s s' ϑ h
  . intro cov₂ s s'
    exact mt (cov₂ s s')

/-- Double-negation elimination gives `Coverage₄`. -/
def Coverage₄ : Prop :=
  ∀ s s' : State, (∀ i : Fin k , ¬ C' i s s') → ¬ (Φ s ∧ Trans s s')

theorem coverage₄_iff_coverage₃ : Coverage₄ Φ Trans C' ↔ Coverage₃ Φ Trans C' := by
  apply Iff.intro
  . intro cov₄ s s' h
    exact cov₄ s s' (Classical.not_not.mp h)
  . intro cov₃ s s' h
    exact cov₃ s s' (Classical.not_not.mpr h)

def Coverage₅ : Prop :=
  ∀ s s' : State, (∀ i : Fin k , ¬ C' i s s') → (¬ Φ s) ∨ (¬ Trans s s')

/- ## The verification condition -/

/-- The decrease obligation we want: every guarded step drops the rank. -/
def ProofObligation₁ : Prop :=
  ∀ s s' : State, (Φ s ∧ Trans s s') → V s' - V s ≤ -δ

/--
Assuming coverage, the condition below suffices to conclude `ProofObligation₁`.
-/
def SufficientCondition₁ : Prop :=
  ∀ s s' : State, (Φ s ∧ Trans s s') → ¬ (∀ i : Fin k , ¬ C' i s s') → V s' - V s ≤ -δ

theorem sufficientCondition₁_implies_obligation₁_assuming_coverage :
    Coverage₁ Φ Trans C' → SufficientCondition₁ Φ Trans C' δ V →
    ProofObligation₁ Φ Trans δ V := by
  intro cov p s s' h
  apply p s s' h
  exact (coverage₁_iff_coverage₂ Φ Trans C').mp cov s s' h

/- ## Sufficient condition assuming coverage -/

/-- By de Morgan duality, an equivalent form of the sufficient condition. -/
def SufficientCondition₂ : Prop :=
  ∀ s s' : State, (Φ s ∧ Trans s s') → (∃ i : Fin k , C' i s s') → V s' - V s ≤ -δ

theorem sufficientCondition₁_iff_sufficientCondition₂ :
    SufficientCondition₂ Φ Trans C' δ V ↔ SufficientCondition₁ Φ Trans C' δ V := by
  apply Iff.intro
  . intro sc₂ s s' h hne
    exact sc₂ s s' h (Classical.not_forall_not.mp hne)
  . intro sc₁ s s' h hex
    exact sc₁ s s' h (Classical.not_forall_not.mpr hex)

/-- By contraposition, a third equivalent form. -/
def SufficientCondition₃ : Prop :=
  ∀ s s' : State, (Φ s ∧ Trans s s') → ¬ (V s' - V s ≤ -δ) → ¬ (∃ i : Fin k , C' i s s')

theorem sufficientCondition₂_iff_sufficientCondition₃ :
    SufficientCondition₂ Φ Trans C' δ V ↔ SufficientCondition₃ Φ Trans C' δ V := by
  apply Iff.intro
  . intro sc₂ s s' h
    exact mt (sc₂ s s' h)
  . intro sc₃ s s' h hex
    apply Classical.byContradiction
    intro hnd
    exact sc₃ s s' h hnd hex

/-- Pushing the negation inside the existential gives `SufficientCondition₄`. -/
def SufficientCondition₄ : Prop :=
  ∀ s s' : State, (Φ s ∧ Trans s s') → ¬ (V s' - V s ≤ -δ) → (∀ i : Fin k , ¬ C' i s s')

theorem sufficientCondition₃_iff_sufficientCondition₄ :
    SufficientCondition₃ Φ Trans C' δ V ↔ SufficientCondition₄ Φ Trans C' δ V := by
  apply Iff.intro
  . intro sc₃ s s' h hnd
    exact deMorgan₁ C' (sc₃ s s' h hnd)
  . intro sc₄ s s' h hnd
    exact deMorgan₂ C' (sc₄ s s' h hnd)

/--
The logical underpinning of the decision procedure: the decrease obligation
reduces to

  1. the coverage condition, and
  2. the sufficient condition (checked by the CEGAR loop).
-/
theorem mainCoverageTheorem :
    Coverage₁ Φ Trans C' → SufficientCondition₄ Φ Trans C' δ V →
    ProofObligation₁ Φ Trans δ V := by
  intro cov₁ sc₄ s s' h
  have sc₃ := (sufficientCondition₃_iff_sufficientCondition₄ Φ Trans C' δ V).mpr sc₄
  have sc₂ := (sufficientCondition₂_iff_sufficientCondition₃ Φ Trans C' δ V).mpr sc₃
  have sc₁ := (sufficientCondition₁_iff_sufficientCondition₂ Φ Trans C' δ V).mp sc₂
  exact sufficientCondition₁_implies_obligation₁_assuming_coverage Φ Trans C' δ V cov₁ sc₁ s s' h

end
