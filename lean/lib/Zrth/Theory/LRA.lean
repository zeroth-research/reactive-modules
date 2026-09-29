import Zrth.Theory

/-!
# Linear real arithmetic

Sorts and signature of linear real arithmetic over matrices, mixing real and
boolean matrices (mirrors `theory::lra` in the `theory` crate).

Values and their derivatives are different sorts: `real m n` is an `m × n` real
matrix, and `δreal m n k` its `(k+1)`-th derivative. `T` sends a value to its
first derivative and raises the order of a derivative. Booleans are constant
sorts, so their tangent is the trivial sort `zero`, whose only writer is the
`zero` generator.

The generators are those of `theory::lra::LRA` at concrete sorts, so each has
a fixed signature. The linear ones (`add`, `sub`, `linear`, `transpose`, `ite`,
`id`) work on values and derivatives alike; the others on values only.
-/

namespace Zrth

/-- Sorts of LRA: `m × n` real matrices, their `(k+1)`-th derivatives, `m × n`
boolean matrices, and the trivial tangent `zero` (a singleton: terminal, not
empty). -/
inductive SortsLRA where
  | real (m n : Nat)
  | δreal (m n k : Nat)
  | bool (m n : Nat)
  | zero
deriving DecidableEq, Repr

/-- The `r`-th derivative of an `m × n` real matrix (the matrix itself for
`r = 0`): the sorts the linear generators work on. -/
def SortsLRA.graded (m n : Nat) : Nat → SortsLRA
  | 0 => .real m n
  | k + 1 => .δreal m n k

instance: Tangent SortsLRA where
  toType := fun (s: SortsLRA) =>
    match s with
    | .real m n | .δreal m n _ => Mat Float m n
    | .bool m n => Mat Bool m n
    | .zero => Unit
  T := fun (s: SortsLRA) =>
    match s with
    -- the shape is unchanged, the order goes up
    | .real m n => .δreal m n 0
    | .δreal m n k => .δreal m n (k + 1)
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
The linear ones take the order `r` of their operands (`0` for values). -/
inductive Gen where
  | real (c : Tensor Float)
  | bool (c : Tensor Bool)
  | and (m n : Nat)
  | or (m n : Nat)
  | xor (m n : Nat)
  | not (m n : Nat)
  -- pointwise comparisons of values
  | le (m n : Nat)
  | lt (m n : Nat)
  | ge (m n : Nat)
  | gt (m n : Nat)
  | eq (m n : Nat)
  | ne (m n : Nat)
  /-- `A · X + B` for `X : graded A.cols b r`; linear, so it keeps the order -/
  | linear (A : Tensor Float) (B : Option (Tensor Float)) (b r : Nat) (h : linearFits A B)
  | add (m n : Nat) (r : Nat := 0)
  | sub (m n : Nat) (r : Nat := 0)
  | relu (m n : Nat)
  /-- the index of the maximum of an `i × j` matrix, flattened -/
  | argmax (i j : Nat)
  -- element-wise minimum and maximum
  | min (m n : Nat)
  | max (m n : Nat)
  | transpose (m n : Nat) (r : Nat := 0)
  | ite (s : SortsLRA)
  | id (s : SortsLRA)
  /-- a sink (`read = true`) or a source (`read = false`) of sort `s` -/
  | uninterpreted (name : String) (read : Bool) (s : SortsLRA)
  | zero
  /-- the zero `(k+1)`-th derivative -/
  | realZerograd (m n : Nat) (k : Nat := 0)
  | anyBool (m n : Nat)
  /-- an arbitrary value (`r = 0`) or `r`-th derivative -/
  | anyReal (m n : Nat) (r : Nat := 0)

open SortsLRA (graded) in
/-- The signatures of the LRA generators. -/
instance : HasSignature SortsLRA Gen where
  signature := fun g => match g with
    | .real c => { dom := [], cod := [.real c.rows c.cols] }
    | .bool c => { dom := [], cod := [.bool c.rows c.cols] }
    | .and m n | .or m n | .xor m n =>
        { dom := [.bool m n, .bool m n], cod := [.bool m n] }
    | .not m n => { dom := [.bool m n], cod := [.bool m n] }
    | .le m n | .lt m n | .ge m n | .gt m n | .eq m n | .ne m n =>
        { dom := [.real m n, .real m n], cod := [.bool m n] }
    | .linear A _ b r _ => { dom := [graded A.cols b r], cod := [graded A.rows b r] }
    | .add m n r | .sub m n r => { dom := [graded m n r, graded m n r], cod := [graded m n r] }
    | .min m n | .max m n => { dom := [.real m n, .real m n], cod := [.real m n] }
    | .relu m n => { dom := [.real m n], cod := [.real m n] }
    | .argmax i j => { dom := [.real i j], cod := [.real 1 1] }
    | .transpose m n r => { dom := [graded m n r], cod := [graded n m r] }
    | .ite s => { dom := [.bool 1 1, s, s], cod := [s] }
    | .id s => { dom := [s], cod := [s] }
    | .uninterpreted _ true s => { dom := [s], cod := [] }
    | .uninterpreted _ false s => { dom := [], cod := [s] }
    | .zero => { dom := [], cod := [.zero] }
    | .realZerograd m n k => { dom := [], cod := [.δreal m n k] }
    | .anyBool m n => { dom := [], cod := [.bool m n] }
    | .anyReal m n r => { dom := [], cod := [graded m n r] }

instance : Sequential SortsLRA Gen where
  skip s := .id s
  skip_sig _ := rfl

instance : Combinatorial SortsLRA Gen where
  havoc
    | .bool m n => .anyBool m n
    | .real m n => .anyReal m n
    | .δreal m n k => .anyReal m n (k + 1)
    -- havoc over a singleton is the singleton
    | .zero => .zero
  havoc_sig s := by cases s <;> rfl

instance : Differential SortsLRA Gen where
  zero
    | .δreal m n k => .realZerograd m n k
    -- not a tangent sort: `zero` is only asked for derivative wires
    | .real m n => .realZerograd m n
    | .bool .. | .zero => .zero
  zero_sig s := by cases s <;> rfl

end LRA

/-- The theory of LRA. -/
abbrev thrLRA : Theory SortsLRA := { gen := LRA.Gen }

end Zrth
