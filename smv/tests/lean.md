structure State where
  x0 : Int
  x1 : Int
  x2 : Int

structure ExternalParams where
  x3 : Int
  x4 : Int

def init (params : ExternalParams) : State :=
  { x0 := 0, x1 := Int.natAbs params.x3, x2 := Int.natAbs params.x4 }

noncomputable def update (s : State) : State :=
  let x1 := s.x1
  let x2 := s.x2
  if s.x0 < s.x1 ∨ s.x0 < s.x2 then
    { s with x0 := s.x0 + 1 }
  else
    { s with x0 := 0 }

def buchi_condition : State → Prop :=
  fun ⟨x0, x1, x2⟩ ↦
    x0 = x1 ∨ x0 = x2

def invariant : State → Prop :=
  fun ⟨x0, x1, x2⟩ ↦
    x0 ≤ x1 ∨ x0 ≤ x2

def relu : Int → Int := fun x ↦ max x 0

def variant : State → Int :=
  fun ⟨x0, x1, x2⟩ ↦
    relu (x1 - x0) + relu (x2 - x0)