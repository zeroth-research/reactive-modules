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

/-- Sorts closed under the tangent former: `T s` is the sort of the rates of
change of values of sort `s` (mirrors `theory::Tangent`).

Only the action of `T` on sorts lives here. The zero section is a generator
of the signature, not a sort. Discrete sorts have a trivial tangent (an
inhabited singleton sort), not a missing one. -/
class Tangent (α : Type u) extends MultiSort α where
  T : α → α

end Zrth
