/-!
# Basic definitions

Definitions shared by the rest of the library.
-/

namespace Zrth

universe u

/-- An `m × n` matrix with entries in `α`. -/
def Mat (α : Type u) (m n : Nat) := Fin m → Fin n → α

end Zrth
