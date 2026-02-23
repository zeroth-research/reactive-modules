structure State where

structure ExternalParams where
  x0 : Int
  x5 : Int

def init (params : ExternalParams) : State :=
  {  }

noncomputable def update (s : State) : State :=
  { /* identity */ }

def buchi_condition : State → Prop :=
  fun ⟨x0⟩ ↦
    x0 = x0 ∨ x0 = x0

def invariant : State → Prop :=
  fun ⟨x0⟩ ↦
    x0 ≤ x0 ∨ x0 ≤ x0

def relu : Int → Int := fun x ↦ max x 0

def variant : State → Int :=
  fun ⟨x0, x1⟩ ↦
    relu (x1 - x0) + relu (x1 - x0)