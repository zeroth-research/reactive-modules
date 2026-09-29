import Zrth.Theory

/-!
# Linear integer arithmetic

Sorts and signature of linear integer arithmetic over matrices, mixing integer
and boolean matrices (mirrors `theory::lia` in the `theory` crate).

The generators are those of `theory::lia::LIA` at concrete shapes, so each has
a fixed signature. All sorts are constant: their tangent is the trivial sort
`zero`, whose only writer is the `zero` generator.
-/

namespace Zrth

/-- Sorts of LIA: `m × n` matrices of integers or booleans, and the trivial
tangent `zero`. -/
inductive SortsLIA where | int (m n : Nat) | bool (m n : Nat) | zero deriving DecidableEq, Repr

instance: Tangent SortsLIA where
  toType := fun (s: SortsLIA) =>
    match s with
    | .int m n => Mat Int m n
    | .bool m n => Mat Bool m n
    | .zero => Unit
  -- values cannot move during delay
  T := fun _ => .zero

namespace LIA

/-- `A` is not empty, and `B` is empty or a column with as many rows as `A`. -/
def linearFits (A : Tensor Int) (B : Option (Tensor Int)) : Prop :=
  A.rows ≠ 0 ∧ ∀ b ∈ B, b.rows = A.rows ∧ b.cols = 1

instance {A : Tensor Int} : ∀ {B}, Decidable (linearFits A B)
  | none => decidable_of_iff (A.rows ≠ 0) (by simp [linearFits])
  | some b => decidable_of_iff (A.rows ≠ 0 ∧ b.rows = A.rows ∧ b.cols = 1) (by simp [linearFits])

/-- The generators of LIA (mirrors `theory::lia::LIA`), at concrete shapes. -/
inductive Gen where
  | int (c : Tensor Int)
  | bool (c : Tensor Bool)
  | and (m n : Nat)
  | or (m n : Nat)
  | xor (m n : Nat)
  | not (m n : Nat)
  | le (m n : Nat)
  | lt (m n : Nat)
  | ge (m n : Nat)
  | gt (m n : Nat)
  | eq (m n : Nat)
  | ne (m n : Nat)
  /-- `A · X + B` for `X : int A.cols b` -/
  | linear (A : Tensor Int) (B : Option (Tensor Int)) (b : Nat) (h : linearFits A B)
  | add (m n : Nat)
  | sub (m n : Nat)
  | relu (m n : Nat)
  /-- A reduction of `s` to an `m × n` vector. The input is unconstrained, as
  in the Rust `check_mat_ops` (FIXME there). -/
  | argmax (s : SortsLIA) (m n : Nat) (h : m = 1 ∨ n = 1)
  | min (s : SortsLIA) (m n : Nat) (h : m = 1 ∨ n = 1)
  | max (s : SortsLIA) (m n : Nat) (h : m = 1 ∨ n = 1)
  | transpose (m n : Nat)
  | ite (s : SortsLIA)
  | id (s : SortsLIA)
  /-- a sink (`read = true`) or a source (`read = false`) of sort `s` -/
  | uninterpreted (name : String) (read : Bool) (s : SortsLIA)
  | anyInt (m n : Nat)
  | anyBool (m n : Nat)
  | zero

/-- The signatures of the LIA generators. -/
instance : HasSignature SortsLIA Gen where
  signature := fun g => match g with
    | .int c => { dom := [], cod := [.int c.rows c.cols] }
    | .bool c => { dom := [], cod := [.bool c.rows c.cols] }
    | .and m n | .or m n | .xor m n =>
        { dom := [.bool m n, .bool m n], cod := [.bool m n] }
    | .not m n => { dom := [.bool m n], cod := [.bool m n] }
    | .le m n | .lt m n | .ge m n | .gt m n | .eq m n | .ne m n =>
        { dom := [.int m n, .int m n], cod := [.bool m n] }
    | .linear A _ b _ => { dom := [.int A.cols b], cod := [.int A.rows b] }
    | .add m n | .sub m n => { dom := [.int m n, .int m n], cod := [.int m n] }
    | .relu m n => { dom := [.int m n], cod := [.int m n] }
    | .argmax s m n _ | .min s m n _ | .max s m n _ => { dom := [s], cod := [.int m n] }
    | .transpose m n => { dom := [.int m n], cod := [.int n m] }
    | .ite s => { dom := [.bool 1 1, s, s], cod := [s] }
    | .id s => { dom := [s], cod := [s] }
    | .uninterpreted _ true s => { dom := [s], cod := [] }
    | .uninterpreted _ false s => { dom := [], cod := [s] }
    | .anyInt m n => { dom := [], cod := [.int m n] }
    | .anyBool m n => { dom := [], cod := [.bool m n] }
    | .zero => { dom := [], cod := [.zero] }

instance : Sequential SortsLIA Gen where
  skip s := .id s
  skip_sig _ := rfl

instance : Combinatorial SortsLIA Gen where
  havoc
    | .int m n => .anyInt m n
    | .bool m n => .anyBool m n
    -- havoc over a singleton is the singleton
    | .zero => .zero

end LIA

/-- The theory of LIA. -/
abbrev thrLIA : Theory SortsLIA := { gen := LIA.Gen }

end Zrth
