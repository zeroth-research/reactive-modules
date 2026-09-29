/-!
# Sorts

Collections of sorts; the semantics of a sort is given by `MultiSort.toType`.
-/

namespace Zrth

universe u v

/-- A collection of sorts, each denoting a Lean type. -/
class MultiSort (α : Type u) where
  [decEq : DecidableEq α]
  toType : α → Type u

instance [MultiSort α] : DecidableEq α := MultiSort.decEq

/-- Sorts with tangent sorts: `T s` is the sort of the derivatives of `s`. -/
class Differentiable (α : Type u) extends MultiSort α where
  T : α → α
  zero: α

end Zrth
