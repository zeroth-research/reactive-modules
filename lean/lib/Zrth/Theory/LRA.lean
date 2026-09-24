import Zrth.Theory

/-!
# Linear real arithmetic

Sorts and (a fragment of) the signature of linear real arithmetic over matrices.
-/

namespace Zrth


universe u

/-- An `m × n` matrix with entries in `α`. -/
def Mat (α : Type u) (m n : Nat) := Fin m → Fin n → α


/-- Sorts of LRA: `m × n` matrices of reals or booleans, and the tangent sorts
(`tan N m n` is the `N`-th tangent; `tan 0` is the zero tangent space). -/
inductive SortsLRA where | real (m n : Nat) | bool (m n : Nat) | tan (N m n : Nat) deriving DecidableEq, Repr

instance: MultiSort SortsLRA where
  toType := fun (s: SortsLRA) =>
    match s with
    | .real m n => Mat Float m n
    | .bool m n => Mat Bool m n
    | .tan 0 m n => {x : Mat Float m n // ∀ m' n', x m' n' = 0}
    | .tan _ m n => Mat Float m n

instance: Differentiable SortsLRA where
  T := fun (s: SortsLRA) =>
    match s with
    | .bool m n => SortsLRA.tan 0 m n
    | .tan 0 m n => SortsLRA.tan 0 m n
    | .tan N m n => SortsLRA.tan (N + 1) m n
    | .real m n => SortsLRA.tan 1 m n

  zero := SortsLRA.tan 0 1 1

namespace LRA

/-- Generators of LRA, indexed by the dimensions of the matrices they work on. -/
inductive Gen where
  | add (m n: Nat)
  | mul (m n m': Nat)
  | id (m n: Nat)

/-- The signatures of the LRA generators. -/
instance : HasSignature SortsLRA Gen where
  signature := fun g => match g with
    | .add m n => { dom := [SortsLRA.real m n, SortsLRA.real m n], cod := [SortsLRA.real m n]}
    | .mul m n m' => { dom := [SortsLRA.real m n, SortsLRA.real n m'], cod := [SortsLRA.real m m']}
    | .id m n  => { dom := [SortsLRA.real m n], cod := [SortsLRA.real m n]}

end LRA

/-- The theory of LRA. -/
def thrLRA : Theory SortsLRA := { gen := LRA.Gen }

end Zrth
