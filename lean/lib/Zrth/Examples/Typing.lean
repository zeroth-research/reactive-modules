import Zrth.Theory.LIA
import Zrth.Theory.LRA
import Zrth.DataFlow.Diagram

/-!
# Typing

The doc-tests of `theory::lia` and `theory::lra`, checked by `decide`: the
generators are polymorphic, the wires pick the instance.
-/

namespace Zrth.Examples.Typing

open Elab (WellTyped)

/-! ## LIA -/

-- pointwise less-than on scalars: `Int(1,1), Int(1,1) -> Bool(1,1)`
example : WellTyped LIA.Gen.lt ⟨[.int 1 1, .int 1 1], [.bool 1 1]⟩ := by decide
-- ReLU preserves shape and stays in the integer fragment
example : WellTyped LIA.Gen.relu ⟨[.int 3 4], [.int 3 4]⟩ := by decide
example : ¬ WellTyped LIA.Gen.relu ⟨[.bool 1 1], [.bool 1 1]⟩ := by decide
-- `A · X + B` with `A : 2 × 3`, `X : 3 × 5`
example : WellTyped (LIA.Gen.linear ⟨2, 3, fun _ _ => 1⟩ (some ⟨2, 1, fun _ _ => 0⟩))
    ⟨[.int 3 5], [.int 2 5]⟩ := by decide
example : ¬ WellTyped (LIA.Gen.linear ⟨2, 3, fun _ _ => 1⟩ (some ⟨3, 1, fun _ _ => 0⟩))
    ⟨[.int 3 5], [.int 2 5]⟩ := by decide

/-! ## LRA -/

-- pointwise less-than on scalars: `Real(1,1), Real(1,1) -> Bool(1,1)`
example : WellTyped LRA.Gen.lt ⟨[.real 1 1, .real 1 1], [.bool 1 1]⟩ := by decide
example : WellTyped LRA.Gen.relu ⟨[.real 3 4], [.real 3 4]⟩ := by decide
example : ¬ WellTyped LRA.Gen.relu ⟨[.bool 1 1], [.bool 1 1]⟩ := by decide
-- linear operations keep the grade: `Y' = A · X'`
example : WellTyped (LRA.Gen.linear ⟨2, 3, fun _ _ => 1⟩ none)
    ⟨[Tangent.T (.real 3 1)], [Tangent.T (.real 2 1)]⟩ := by decide
example : ¬ WellTyped (LRA.Gen.linear ⟨2, 3, fun _ _ => 1⟩ none)
    ⟨[Tangent.T (.real 3 1)], [.real 2 1]⟩ := by decide
-- a literal is a value, never a derivative: use `realZerograd`
example : ¬ WellTyped (LRA.Gen.real ⟨1, 1, fun _ _ => 0⟩) ⟨[], [Tangent.T (.real 1 1)]⟩ := by
  decide
example : WellTyped (LRA.Gen.realZerograd 1 1) ⟨[], [Tangent.T (.real 1 1)]⟩ := by decide
example : ¬ WellTyped (LRA.Gen.realZerograd 1 1) ⟨[], [.real 1 1]⟩ := by decide
-- the derivative of a boolean is silenced by `zero`
example : WellTyped LRA.Gen.zero ⟨[], [Tangent.T (.bool 2 2)]⟩ := by decide

/-! ## Elaboration -/

/-- `add` used at `2 × 2` derivatives. -/
def box : Box thrLRA :=
  { gen := .add, read := [⟨0, .real 2 2 1⟩, ⟨1, .real 2 2 1⟩], write := [⟨2, .real 2 2 1⟩] }

-- the box elaborates to the instance of `add` at the sorts of its wires
example : box.inst = LRA.Inst.add 2 2 1 := rfl

end Zrth.Examples.Typing
