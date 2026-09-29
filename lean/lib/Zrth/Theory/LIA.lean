import Zrth.Theory

/-!
# Linear integer arithmetic

Sorts and signature of linear integer arithmetic over matrices, mixing integer
and boolean matrices (mirrors `theory::lia` in the `theory` crate).

Unlike the Rust `check`, the shapes are indices of the generators, so each
generator has a fixed signature. Wire degrees are not modelled: all
operands are of degree 0.
-/

namespace Zrth

/-- Sorts of LIA: `m × n` matrices of integers or booleans. -/
inductive SortsLIA where | int (m n : Nat) | bool (m n : Nat) deriving DecidableEq, Repr

instance: MultiSort SortsLIA where
  toType := fun (s: SortsLIA) =>
    match s with
    | .int m n => Mat Int m n
    | .bool m n => Mat Bool m n

namespace LIA

/-- Generators of LIA, indexed by the dimensions of the matrices they work on. -/
inductive Gen where
  -- constant matrix literals
  | int (m n : Nat) (c : Mat Int m n)
  | bool (m n : Nat) (c : Mat Bool m n)
  -- boolean operations
  | and (m n : Nat)
  | or (m n : Nat)
  | xor (m n : Nat)
  | not (m n : Nat)
  -- pointwise integer comparisons
  | le (m n : Nat)
  | lt (m n : Nat)
  | ge (m n : Nat)
  | gt (m n : Nat)
  | eq (m n : Nat)
  | ne (m n : Nat)
  /-- `X ↦ A·X + B` for `X : int i b`; `B` is a column added to every column of `A·X`. -/
  | linear (o i b : Nat) (A : Mat Int o i) (B : Mat Int o 1)
  | add (m n : Nat)
  | sub (m n : Nat)
  | relu (m n : Nat)
  /-- Column-wise reductions of an `m × n` matrix to a `1 × n` row.
  (The Rust check accepts either orientation of the output vector.) -/
  | argmax (m n : Nat)
  | min (m n : Nat)
  | max (m n : Nat)
  | transpose (m n : Nat)
  -- control flow
  | ite (s : SortsLIA)
  | id (s : SortsLIA)
  /-- An uninterpreted source (`read = false`, writes one value) or sink
  (`read = true`, reads one value). -/
  | uninterpreted (name : String) (read : Bool) (s : SortsLIA)
  -- havoc: an arbitrary value
  | anyInt (m n : Nat)
  | anyBool (m n : Nat)

/-- The signatures of the LIA generators. -/
instance : HasSignature SortsLIA Gen where
  signature := fun g => match g with
    | .int m n _ => { dom := [], cod := [.int m n] }
    | .bool m n _ => { dom := [], cod := [.bool m n] }
    | .and m n | .or m n | .xor m n =>
        { dom := [.bool m n, .bool m n], cod := [.bool m n] }
    | .not m n => { dom := [.bool m n], cod := [.bool m n] }
    | .le m n | .lt m n | .ge m n | .gt m n | .eq m n | .ne m n =>
        { dom := [.int m n, .int m n], cod := [.bool m n] }
    | .linear o i b _ _ => { dom := [.int i b], cod := [.int o b] }
    | .add m n | .sub m n => { dom := [.int m n, .int m n], cod := [.int m n] }
    | .relu m n => { dom := [.int m n], cod := [.int m n] }
    | .argmax m n | .min m n | .max m n => { dom := [.int m n], cod := [.int 1 n] }
    | .transpose m n => { dom := [.int m n], cod := [.int n m] }
    | .ite s => { dom := [.bool 1 1, s, s], cod := [s] }
    | .id s => { dom := [s], cod := [s] }
    | .uninterpreted _ true s => { dom := [s], cod := [] }
    | .uninterpreted _ false s => { dom := [], cod := [s] }
    | .anyInt m n => { dom := [], cod := [.int m n] }
    | .anyBool m n => { dom := [], cod := [.bool m n] }

end LIA

/-- The theory of LIA. -/
def thrLIA : Theory SortsLIA := { gen := LIA.Gen }

end Zrth
