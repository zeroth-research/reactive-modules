import Zrth.Theory

/-!
# Linear real arithmetic

Sorts and signature of linear real arithmetic over matrices, mixing real and
boolean matrices (mirrors `theory::lra` in the `theory` crate).

The differential grade is part of the sort: `real m n r` is the `r`-th
derivative of an `m × n` real matrix, and `T` raises `r`. Booleans are constant
sorts, so their tangent is the trivial sort `zero`, whose only writer is the
`zero` generator.

`LRA.Gen` are the generators of `theory::lra::LRA`, polymorphic in the shapes
and grades; `LRA.Inst` are their instances at concrete sorts, with signatures.

Where the typing differs from `LRA::check`:
- comparisons must read exactly two values and write one (`check_cmp` ignores
  surplus wires);
- a literal writes the sort of its tensor (`Real(t)` with a boolean `t` writes
  a `Bool` wire in Rust).
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

/-- The generators of LRA (mirrors `theory::lra::LRA`). -/
inductive Gen where
  -- constant matrix literals
  | real (c : Tensor Float)
  | bool (c : Tensor Bool)
  -- boolean operations
  | and | or | xor | not
  -- pointwise real comparisons
  | le | lt | ge | gt | eq | ne
  /-- `X ↦ A·X + B`; `B` is a column added to every column of `A·X`, and
  `none` stands for the empty tensor. -/
  | linear (A : Tensor Float) (B : Option (Tensor Float))
  | add | sub | relu
  /-- reductions of a matrix to a vector -/
  | argmax | min | max
  | transpose
  -- control flow
  | ite | id
  /-- an uninterpreted source (writes one value) or sink (reads one value) -/
  | uninterpreted (name : String)
  /-- the unique inhabitant of `zero` -/
  | zero
  /-- the zero derivative of an `m × n` real matrix -/
  | realZerograd (m n : Nat)
  -- havoc: an arbitrary value
  | anyBool (m n : Nat)
  | anyReal (m n : Nat)

/-- `A` is not empty, and `B` is empty or a column with as many rows as `A`. -/
def linearFits (A : Tensor Float) (B : Option (Tensor Float)) : Prop :=
  A.rows ≠ 0 ∧ ∀ b ∈ B, b.rows = A.rows ∧ b.cols = 1

instance {A : Tensor Float} : ∀ {B}, Decidable (linearFits A B)
  | none => decidable_of_iff (A.rows ≠ 0) (by simp [linearFits])
  | some b => decidable_of_iff (A.rows ≠ 0 ∧ b.rows = A.rows ∧ b.cols = 1) (by simp [linearFits])

/-- The generators of LRA at concrete sorts. Operations on reals take the
grade `r` of their operands. -/
inductive Inst where
  | real (c : Tensor Float)
  | bool (c : Tensor Bool)
  | and (m n : Nat)
  | or (m n : Nat)
  | xor (m n : Nat)
  | not (m n : Nat)
  | le (m n r : Nat)
  | lt (m n r : Nat)
  | ge (m n r : Nat)
  | gt (m n r : Nat)
  | eq (m n r : Nat)
  | ne (m n r : Nat)
  /-- `A · X + B` for `X : real A.cols b r`; linear, so it keeps the grade -/
  | linear (A : Tensor Float) (B : Option (Tensor Float)) (b r : Nat) (h : linearFits A B)
  | add (m n r : Nat)
  | sub (m n r : Nat)
  | relu (m n r : Nat)
  /-- A reduction of `s` to an `m × n` vector. `check` leaves the input
  unconstrained (FIXME in the Rust `check_mat_ops`). -/
  | argmax (s : SortsLRA) (m n r : Nat) (h : m = 1 ∨ n = 1)
  | min (s : SortsLRA) (m n r : Nat) (h : m = 1 ∨ n = 1)
  | max (s : SortsLRA) (m n r : Nat) (h : m = 1 ∨ n = 1)
  | transpose (m n r : Nat)
  | ite (s : SortsLRA)
  | id (s : SortsLRA)
  /-- a sink (`read = true`) or a source (`read = false`) of sort `s` -/
  | uninterpreted (name : String) (read : Bool) (s : SortsLRA)
  | zero
  /-- the zero derivative of grade `r + 1` -/
  | realZerograd (m n r : Nat)
  | anyBool (m n : Nat)
  | anyReal (m n : Nat)

/-- The signatures of the LRA instances. -/
instance : HasSignature SortsLRA Inst where
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
    | .argmax s m n r _ | .min s m n r _ | .max s m n r _ =>
        { dom := [s], cod := [.real m n r] }
    | .transpose m n r => { dom := [.real m n r], cod := [.real n m r] }
    | .ite s => { dom := [.bool 1 1, s, s], cod := [s] }
    | .id s => { dom := [s], cod := [s] }
    | .uninterpreted _ true s => { dom := [s], cod := [] }
    | .uninterpreted _ false s => { dom := [], cod := [s] }
    | .zero => { dom := [], cod := [.zero] }
    | .realZerograd m n r => { dom := [], cod := [.real m n (r + 1)] }
    | .anyBool m n => { dom := [], cod := [.bool m n] }
    | .anyReal m n => { dom := [], cod := [.real m n] }

/-- The generator of an instance. -/
def Inst.erase : Inst → Gen
  | .real c => .real c
  | .bool c => .bool c
  | .and .. => .and
  | .or .. => .or
  | .xor .. => .xor
  | .not .. => .not
  | .le .. => .le
  | .lt .. => .lt
  | .ge .. => .ge
  | .gt .. => .gt
  | .eq .. => .eq
  | .ne .. => .ne
  | .linear A B .. => .linear A B
  | .add .. => .add
  | .sub .. => .sub
  | .relu .. => .relu
  | .argmax .. => .argmax
  | .min .. => .min
  | .max .. => .max
  | .transpose .. => .transpose
  | .ite .. => .ite
  | .id .. => .id
  | .uninterpreted x .. => .uninterpreted x
  | .zero => .zero
  | .realZerograd m n _ => .realZerograd m n
  | .anyBool m n => .anyBool m n
  | .anyReal m n => .anyReal m n

/-- Read the sorts of an instance off the wires; `check` compares the rest. -/
def Gen.infer : Gen → Signature SortsLRA → Option Inst
  | .real c, _ => some (.real c)
  | .bool c, _ => some (.bool c)
  | .and, ⟨_, [.bool m n]⟩ => some (.and m n)
  | .or, ⟨_, [.bool m n]⟩ => some (.or m n)
  | .xor, ⟨_, [.bool m n]⟩ => some (.xor m n)
  | .not, ⟨_, [.bool m n]⟩ => some (.not m n)
  | .le, ⟨[.real m n r, _], _⟩ => some (.le m n r)
  | .lt, ⟨[.real m n r, _], _⟩ => some (.lt m n r)
  | .ge, ⟨[.real m n r, _], _⟩ => some (.ge m n r)
  | .gt, ⟨[.real m n r, _], _⟩ => some (.gt m n r)
  | .eq, ⟨[.real m n r, _], _⟩ => some (.eq m n r)
  | .ne, ⟨[.real m n r, _], _⟩ => some (.ne m n r)
  | .linear A B, ⟨[.real _ b r], _⟩ =>
      if h : linearFits A B then some (.linear A B b r h) else none
  | .add, ⟨_, [.real m n r]⟩ => some (.add m n r)
  | .sub, ⟨_, [.real m n r]⟩ => some (.sub m n r)
  | .relu, ⟨_, [.real m n r]⟩ => some (.relu m n r)
  | .argmax, ⟨[s], [.real m n r]⟩ =>
      if h : m = 1 ∨ n = 1 then some (.argmax s m n r h) else none
  | .min, ⟨[s], [.real m n r]⟩ =>
      if h : m = 1 ∨ n = 1 then some (.min s m n r h) else none
  | .max, ⟨[s], [.real m n r]⟩ =>
      if h : m = 1 ∨ n = 1 then some (.max s m n r h) else none
  | .transpose, ⟨[.real m n r], _⟩ => some (.transpose m n r)
  | .ite, ⟨_, [s]⟩ => some (.ite s)
  | .id, ⟨_, [s]⟩ => some (.id s)
  | .uninterpreted x, ⟨[s], []⟩ => some (.uninterpreted x true s)
  | .uninterpreted x, ⟨[], [s]⟩ => some (.uninterpreted x false s)
  | .zero, _ => some .zero
  | .realZerograd m n, ⟨_, [.real _ _ (r + 1)]⟩ => some (.realZerograd m n r)
  | .anyBool m n, _ => some (.anyBool m n)
  | .anyReal m n, _ => some (.anyReal m n)
  | _, _ => none

instance : Elab SortsLRA Gen Inst where
  infer := Gen.infer
  erase := Inst.erase
  infer_erase i := by
    cases i <;> try rfl
    all_goals first
      | (next h => simp [Gen.infer, Inst.erase, HasSignature.signature, h])
      | (next r _ => cases r <;> rfl)
  erase_infer g σ i h := by
    unfold Gen.infer at h
    split at h <;> (try split at h) <;> cases h <;> rfl

instance : Sequential SortsLRA Gen where
  skip _ := .id

instance : Combinatorial SortsLRA Gen where
  havoc
    | .bool m n => .anyBool m n
    | .real m n _ => .anyReal m n
    -- havoc over a singleton is the singleton
    | .zero => .zero

instance : Differential SortsLRA Gen where
  zero
    | .real m n _ => .realZerograd m n
    | .bool .. | .zero => .zero

end LRA

/-- The theory of LRA. -/
abbrev thrLRA : Theory SortsLRA := { gen := LRA.Gen, inst := LRA.Inst }

end Zrth
