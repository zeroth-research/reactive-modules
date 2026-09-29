import Zrth.Theory.LIA
import Zrth.Theory.LRA
import Zrth.Theory.BV
import Zrth.Theory.Any
import Zrth.DataFlow.Diagram

/-!
# Typing

The doc-tests of `theory::lia`, `theory::lra` and `theory::bv`, checked by
`decide`: the generators are polymorphic, the wires pick the instance.
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

/-! ## BV -/

-- matrix multiply: `(2 × 3) · (3 × 4) → (2 × 4)`
example : WellTyped BV.Gen.matmul ⟨[.bv 8 2 3, .bv 8 3 4], [.bv 8 2 4]⟩ := by decide
-- element-wise `add` requires matching shapes
example : WellTyped BV.Gen.add ⟨[.bv 8 2 3, .bv 8 2 3], [.bv 8 2 3]⟩ := by decide
example : ¬ WellTyped BV.Gen.add ⟨[.bv 8 2 3, .bv 8 3 4], [.bv 8 2 4]⟩ := by decide
-- literals must fit the width
example : WellTyped (BV.Gen.const ⟨1, 2, fun _ j => j.val⟩) ⟨[], [.bv 1 1 2]⟩ := by decide
example : ¬ WellTyped (BV.Gen.const ⟨1, 2, fun _ _ => 2⟩) ⟨[], [.bv 1 1 2]⟩ := by decide
-- the bits `[3..=0]` of an 8-bit bit-vector
example : WellTyped (BV.Gen.bitSelect 3 0) ⟨[.bv 8 1 1], [.bv 4 1 1]⟩ := by decide
example : ¬ WellTyped (BV.Gen.bitSelect 8 0) ⟨[.bv 8 1 1], [.bv 9 1 1]⟩ := by decide

/-! ## Any -/

-- a generator of a base theory, on sorts of that theory
example : WellTyped (Any.Gen.lia .lt) ⟨[.int 1 1, .int 1 1], [.bool 1 1]⟩ := by decide
example : ¬ WellTyped (Any.Gen.lia .lt) ⟨[.real 1 1, .real 1 1], [.bool 1 1]⟩ := by decide
example : WellTyped (Any.Gen.lra .lt) ⟨[.real 1 1, .real 1 1], [.bool 1 1]⟩ := by decide
-- the sub-signatures: `skip` is sequential but not combinatorial
example : (Any.Gen.skip (.int 1 1)).isSequential := rfl
example : ¬ (Any.Gen.skip (.int 1 1)).isCombinatorial := by decide

/-! ## Structural generators -/

-- `skip` fits every sort
example : WellTyped (Sequential.skip (G := LRA.Gen) (.real 2 2 1)) ⟨[.real 2 2 1], [.real 2 2 1]⟩ := by
  decide
-- `zero` writes the zero of a tangent sort
example : WellTyped (Differential.zero (G := LRA.Gen) (.real 2 2 1)) ⟨[], [.real 2 2 1]⟩ := by decide
example : WellTyped (Differential.zero (G := LRA.Gen) .zero) ⟨[], [.zero]⟩ := by decide
-- As in the Rust crate, `havoc` of a derivative and `zero` of a value do not
-- type-check: `AnyReal` writes values, `RealZerograd` writes derivatives.
example : ¬ WellTyped (Combinatorial.havoc (G := LRA.Gen) (.real 2 2 1)) ⟨[], [.real 2 2 1]⟩ := by
  decide
example : ¬ WellTyped (Differential.zero (G := LRA.Gen) (.real 2 2)) ⟨[], [.real 2 2]⟩ := by decide

/-! ## Elaboration -/

/-- `add` used at `2 × 2` derivatives. -/
def box : Box thrLRA :=
  { gen := .add, read := [⟨0, .real 2 2 1⟩, ⟨1, .real 2 2 1⟩], write := [⟨2, .real 2 2 1⟩] }

-- the box elaborates to the instance of `add` at the sorts of its wires
example : box.inst = LRA.Inst.add 2 2 1 := rfl

/-- `skip` in the next of an atom. -/
def skipBox : Box thrSequential :=
  { gen := Sequential.skip (.bool 1 1), read := [⟨0, .bool 1 1⟩], write := [⟨1, .bool 1 1⟩] }

example : skipBox.inst.1 = Any.Inst.skip (.bool 1 1) := rfl

end Zrth.Examples.Typing
