import Zrth.Theory

/-!
# Linear real arithmetic

Sorts and (a fragment of) the signature of linear real arithmetic over matrices
(mirrors `theory::lra` in the `theory` crate).

The differential grade is part of the sort: `real m n r` is the `r`-th
derivative of an `m × n` real matrix, and `T` raises `r`. Booleans are constant
sorts, so their tangent is the trivial sort `zero`, whose only writer is the
`zero` generator.
-/

namespace Zrth

/-- Sorts of LRA: `m × n` real matrices of differential grade `rank`
(0 = value, 1 = first derivative, ...), `m × n` boolean matrices, and the
trivial tangent `zero` (a singleton: terminal, not empty). -/
inductive SortsLRA where
  | real (m n : Nat) (rank : Nat := 0)
  | bool (m n : Nat)
  | zero
deriving DecidableEq, Repr

instance: Tangent SortsLRA where
  toType := fun (s: SortsLRA) =>
    match s with
    | .real m n _ => Mat Float m n
    | .bool m n => Mat Bool m n
    | .zero => Unit
  T := fun (s: SortsLRA) =>
    match s with
    -- the shape is unchanged, the grade goes up
    | .real m n r => .real m n (r + 1)
    -- constant sorts have the trivial tangent, which is a fixed point
    | .bool _ _ | .zero => .zero

namespace LRA

/-- Generators of LRA, indexed by the dimensions (and grades) of the matrices
they work on. -/
inductive Gen where
  | add (m n: Nat) (rank : Nat := 0)
  /-- `X × y` for a constant `X`; linear, so `y` may be a derivative -/
  | mul (m n m': Nat) (rank : Nat := 0)
  | id (s : SortsLRA)
  /-- the unique inhabitant of `zero` -/
  | zero
  /-- the zero derivative of an `m × n` real matrix of grade `rank` -/
  | realZerograd (m n : Nat) (rank : Nat := 0)

/-- The signatures of the LRA generators. -/
instance : HasSignature SortsLRA Gen where
  signature := fun g => match g with
    | .add m n r => { dom := [.real m n r, .real m n r], cod := [.real m n r] }
    | .mul m n m' r => { dom := [.real m n, .real n m' r], cod := [.real m m' r] }
    | .id s => { dom := [s], cod := [s] }
    | .zero => { dom := [], cod := [.zero] }
    | .realZerograd m n r => { dom := [], cod := [.real m n (r + 1)] }

end LRA

/-- The theory of LRA. -/
def thrLRA : Theory SortsLRA := { gen := LRA.Gen }

end Zrth
