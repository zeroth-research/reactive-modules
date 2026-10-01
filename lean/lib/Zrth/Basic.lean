/-!
# Basic definitions

Definitions shared by the rest of the library.
-/

namespace Zrth

universe u

/-- An `m × n` matrix with entries in `α`. -/
def Mat (α : Type u) (m n : Nat) := Fin m → Fin n → α

/-- A matrix together with its dimensions: the counterpart of a 2-D `PyTensor`
in the `theory` crate, whose shape is only known at runtime. -/
structure Tensor (α : Type u) where
  rows : Nat
  cols : Nat
  data : Mat α rows cols

/-- The entries of a tensor, row by row. -/
def Tensor.entries (t : Tensor α) : List α :=
  (List.finRange t.rows).flatMap fun i => (List.finRange t.cols).map fun j => t.data i j

end Zrth
