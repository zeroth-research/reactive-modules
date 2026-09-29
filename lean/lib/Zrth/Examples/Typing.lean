import Zrth.Theory.LIA
import Zrth.Theory.LRA
import Zrth.Theory.BV
import Zrth.Theory.Any
import Zrth.DataFlow.Diagram

/-!
# Typing

The doc-tests of `theory::lia`, `theory::lra` and `theory::bv`, as boxes: a
box is accepted iff its wires fit the signature of its generator.
-/

namespace Zrth.Examples.Typing

open HasSignature (signature)

/-! ## LIA -/

-- pointwise less-than on scalars: `Int(1,1), Int(1,1) -> Bool(1,1)`
example : Box thrLIA := { gen := .lt 1 1, read := [⟨0, .int 1 1⟩, ⟨1, .int 1 1⟩], write := [⟨2, .bool 1 1⟩] }
-- ReLU preserves shape and stays in the integer fragment
example : Box thrLIA := { gen := .relu 3 4, read := [⟨0, .int 3 4⟩], write := [⟨1, .int 3 4⟩] }
example : ∀ m n, (signature (LIA.Gen.relu m n)).dom ≠ [.bool 1 1] := by
  intro m n; simp [signature]
-- `A · X + B` with `A : 2 × 3`, `X : 3 × 5`
example : Box thrLIA :=
  { gen := .linear ⟨2, 3, fun _ _ => 1⟩ (some ⟨2, 1, fun _ _ => 0⟩) 5 (by decide)
    read := [⟨0, .int 3 5⟩], write := [⟨1, .int 2 5⟩] }
example : ¬ LIA.linearFits ⟨2, 3, fun _ _ => 1⟩ (some ⟨3, 1, fun _ _ => 0⟩) := by decide

/-! ## LRA -/

-- pointwise less-than on scalars: `Real(1,1), Real(1,1) -> Bool(1,1)`
example : Box thrLRA :=
  { gen := .lt 1 1, read := [⟨0, .real 1 1⟩, ⟨1, .real 1 1⟩], write := [⟨2, .bool 1 1⟩] }
-- linear operations keep the order: `Y' = A · X'`
example : Box thrLRA :=
  { gen := .linear ⟨2, 3, fun _ _ => 1⟩ none 1 1 (by decide)
    read := [⟨0, Tangent.T (.real 3 1)⟩], write := [⟨1, Tangent.T (.real 2 1)⟩] }
-- a literal is a value, never a derivative: use `realZerograd`
example : ∀ c, (signature (LRA.Gen.real c)).cod ≠ [Tangent.T (.real c.rows c.cols)] := by
  intro c; simp [signature, Tangent.T]
-- comparisons and the other non-linear operations take values only
example : ∀ m n g, g ∈ [LRA.Gen.lt m n, .relu m n, .min m n] →
    .δreal m n 0 ∉ (signature g).dom := by
  intro m n g hg; simp at hg; rcases hg with rfl | rfl | rfl <;> simp [signature]
example : Box thrLRA := { gen := .realZerograd 1 1, read := [], write := [⟨0, Tangent.T (.real 1 1)⟩] }
-- the derivative of a boolean is silenced by `zero`
example : Box thrLRA := { gen := .zero, read := [], write := [⟨0, Tangent.T (.bool 2 2)⟩] }

/-! ## BV -/

-- matrix multiply: `(2 × 3) · (3 × 4) → (2 × 4)`
example : Box thrBV :=
  { gen := .matmul 8 2 3 4, read := [⟨0, .bv 8 2 3⟩, ⟨1, .bv 8 3 4⟩], write := [⟨2, .bv 8 2 4⟩] }
-- literals must fit the width
example : Box thrBV := { gen := .const 1 ⟨1, 2, fun _ j => j.val⟩ rfl, read := [], write := [⟨0, .bv 1 1 2⟩] }
example : BV.constFits 1 ⟨1, 2, fun _ _ => 2⟩ = false := by decide
-- the bits `[3..=0]` of an 8-bit bit-vector
example : Box thrBV :=
  { gen := .bitSelect 3 0 8 1 1 (by decide), read := [⟨0, .bv 8 1 1⟩], write := [⟨1, .bv 4 1 1⟩] }

/-! ## Any -/

-- a generator of a base theory, on the embedded sorts
example : Box thrAny :=
  { gen := .lia (.lt 1 1), read := [⟨0, .int 1 1⟩, ⟨1, .int 1 1⟩], write := [⟨2, .bool 1 1⟩] }
-- the sub-signatures: `skip` is sequential but not combinatorial
example : Box thrSequential :=
  { gen := Sequential.skip (.bool 1 1), read := [⟨0, .bool 1 1⟩], write := [⟨1, .bool 1 1⟩] }
example : ¬ (Any.Gen.skip (.int 1 1)).isCombinatorial := by decide

/-! ## Structural generators -/

-- `havoc` writes its range, derivatives included
example : Box thrLRA :=
  { gen := Combinatorial.havoc (.δreal 2 2 0), read := [], write := [⟨0, .δreal 2 2 0⟩] }

end Zrth.Examples.Typing
