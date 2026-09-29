import Zrth.Theory

/-!
# Linear real arithmetic

Sorts and signature of linear real arithmetic over matrices, mixing real and
boolean matrices (mirrors `theory::lra` in the `theory` crate).

The differential grade is part of the sort: `real m n r` is the `r`-th
derivative of an `m × n` real matrix, and `T` raises `r`. Booleans are constant
sorts, so their tangent is the trivial sort `zero`, whose only writer is the
`zero` generator.

The generators are those of `theory::lra::LRA` at concrete sorts, so each has
a fixed signature.
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

/-- `A` is not empty, and `B` is empty or a column with as many rows as `A`. -/
def linearFits (A : Tensor Float) (B : Option (Tensor Float)) : Prop :=
  A.rows ≠ 0 ∧ ∀ b ∈ B, b.rows = A.rows ∧ b.cols = 1

instance {A : Tensor Float} : ∀ {B}, Decidable (linearFits A B)
  | none => decidable_of_iff (A.rows ≠ 0) (by simp [linearFits])
  | some b => decidable_of_iff (A.rows ≠ 0 ∧ b.rows = A.rows ∧ b.cols = 1) (by simp [linearFits])

/-- The generators of LRA (mirrors `theory::lra::LRA`), at concrete sorts.
Operations on reals take the grade `r` of their operands. -/
inductive Gen where
  | real (c : Tensor Float)
  | bool (c : Tensor Bool)
  | and (m n : Nat)
  | or (m n : Nat)
  | xor (m n : Nat)
  | not (m n : Nat)
  | le (m n : Nat) (r : Nat := 0)
  | lt (m n : Nat) (r : Nat := 0)
  | ge (m n : Nat) (r : Nat := 0)
  | gt (m n : Nat) (r : Nat := 0)
  | eq (m n : Nat) (r : Nat := 0)
  | ne (m n : Nat) (r : Nat := 0)
  /-- `A · X + B` for `X : real A.cols b r`; linear, so it keeps the grade -/
  | linear (A : Tensor Float) (B : Option (Tensor Float)) (b r : Nat) (h : linearFits A B)
  | add (m n : Nat) (r : Nat := 0)
  | sub (m n : Nat) (r : Nat := 0)
  | relu (m n : Nat) (r : Nat := 0)
  /-- A reduction of an `i × j` matrix of grade `ri` to an `m × n` vector of
  grade `r`. The shapes and grades are not related, as in the Rust
  `check_mat_ops` (FIXME there). -/
  | argmax (i j ri m n r : Nat) (h : m = 1 ∨ n = 1)
  | min (i j ri m n r : Nat) (h : m = 1 ∨ n = 1)
  | max (i j ri m n r : Nat) (h : m = 1 ∨ n = 1)
  | transpose (m n : Nat) (r : Nat := 0)
  | ite (s : SortsLRA)
  | id (s : SortsLRA)
  /-- a sink (`read = true`) or a source (`read = false`) of sort `s` -/
  | uninterpreted (name : String) (read : Bool) (s : SortsLRA)
  | zero
  /-- the zero derivative of grade `r + 1` -/
  | realZerograd (m n : Nat) (r : Nat := 0)
  | anyBool (m n : Nat)
  /-- an arbitrary value or derivative -/
  | anyReal (m n : Nat) (r : Nat := 0)

/-- The signatures of the LRA generators. -/
instance : HasSignature SortsLRA Gen where
  signature := fun g => match g with
    | .real c => { dom := [], cod := [.real c.rows c.cols] }
    | .bool c => { dom := [], cod := [.bool c.rows c.cols] }
    | .and m n | .or m n | .xor m n =>
        { dom := [.bool m n, .bool m n], cod := [.bool m n] }
    | .not m n => { dom := [.bool m n], cod := [.bool m n] }
    | .le m n r | .lt m n r | .ge m n r | .gt m n r | .eq m n r | .ne m n r =>
        { dom := [.real m n r, .real m n r], cod := [.bool m n] }
    | .linear A _ b r _ => { dom := [.real A.cols b r], cod := [.real A.rows b r] }
    | .add m n r | .sub m n r =>
        { dom := [.real m n r, .real m n r], cod := [.real m n r] }
    | .relu m n r => { dom := [.real m n r], cod := [.real m n r] }
    | .argmax i j ri m n r _ | .min i j ri m n r _ | .max i j ri m n r _ =>
        { dom := [.real i j ri], cod := [.real m n r] }
    | .transpose m n r => { dom := [.real m n r], cod := [.real n m r] }
    | .ite s => { dom := [.bool 1 1, s, s], cod := [s] }
    | .id s => { dom := [s], cod := [s] }
    | .uninterpreted _ true s => { dom := [s], cod := [] }
    | .uninterpreted _ false s => { dom := [], cod := [s] }
    | .zero => { dom := [], cod := [.zero] }
    | .realZerograd m n r => { dom := [], cod := [.real m n (r + 1)] }
    | .anyBool m n => { dom := [], cod := [.bool m n] }
    | .anyReal m n r => { dom := [], cod := [.real m n r] }

instance : Sequential SortsLRA Gen where
  skip s := .id s
  skip_sig _ := rfl

instance : Combinatorial SortsLRA Gen where
  havoc
    | .bool m n => .anyBool m n
    | .real m n r => .anyReal m n r
    -- havoc over a singleton is the singleton
    | .zero => .zero
  havoc_sig s := by cases s <;> rfl

instance : Differential SortsLRA Gen where
  zero
    | .real m n r => .realZerograd m n (r - 1)
    | .bool .. | .zero => .zero
  zero_sig s := by cases s <;> rfl

end LRA

/-- The theory of LRA. -/
abbrev thrLRA : Theory SortsLRA := { gen := LRA.Gen }

end Zrth
