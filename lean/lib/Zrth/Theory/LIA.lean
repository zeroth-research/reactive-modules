import Zrth.Theory

/-!
# Linear integer arithmetic

Sorts and signature of linear integer arithmetic over matrices, mixing integer
and boolean matrices (mirrors `theory::lia` in the `theory` crate).

`LIA.Gen` are the generators of `theory::lia::LIA`, polymorphic in the shapes;
`LIA.Inst` are their instances at concrete shapes, with signatures. All sorts
are constant: their tangent is the trivial sort `zero`, whose only writer is
the `zero` generator.

Where the typing differs from `LIA::check`:
- comparisons must read exactly two values and write one (`check_cmp` ignores
  surplus wires);
- a literal writes the sort of its tensor (`Int(t)` with a boolean `t` writes
  a `Bool` wire in Rust).
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

/-- The generators of LIA (mirrors `theory::lia::LIA`). -/
inductive Gen where
  -- constant matrix literals
  | int (c : Tensor Int)
  | bool (c : Tensor Bool)
  -- boolean operations
  | and | or | xor | not
  -- pointwise integer comparisons
  | le | lt | ge | gt | eq | ne
  /-- `X ↦ A·X + B`; `B` is a column added to every column of `A·X`, and
  `none` stands for the empty tensor. -/
  | linear (A : Tensor Int) (B : Option (Tensor Int))
  | add | sub | relu
  /-- reductions of a matrix to a vector -/
  | argmax | min | max
  | transpose
  -- control flow
  | ite | id
  /-- an uninterpreted source (writes one value) or sink (reads one value) -/
  | uninterpreted (name : String)
  -- havoc: an arbitrary value
  | anyInt (m n : Nat)
  | anyBool (m n : Nat)
  /-- the unique inhabitant of `zero` -/
  | zero

/-- `A` is not empty, and `B` is empty or a column with as many rows as `A`. -/
def linearFits (A : Tensor Int) (B : Option (Tensor Int)) : Prop :=
  A.rows ≠ 0 ∧ ∀ b ∈ B, b.rows = A.rows ∧ b.cols = 1

instance {A : Tensor Int} : ∀ {B}, Decidable (linearFits A B)
  | none => decidable_of_iff (A.rows ≠ 0) (by simp [linearFits])
  | some b => decidable_of_iff (A.rows ≠ 0 ∧ b.rows = A.rows ∧ b.cols = 1) (by simp [linearFits])

/-- The generators of LIA at concrete shapes. -/
inductive Inst where
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
  /-- A reduction of `s` to an `m × n` vector. `check` leaves the input
  unconstrained (FIXME in the Rust `check_mat_ops`). -/
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

/-- The signatures of the LIA instances. -/
instance : HasSignature SortsLIA Inst where
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

/-- The generator of an instance. -/
def Inst.erase : Inst → Gen
  | .int c => .int c
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
  | .anyInt m n => .anyInt m n
  | .anyBool m n => .anyBool m n
  | .zero => .zero

/-- Read the shapes of an instance off the wires; `check` compares the rest. -/
def Gen.infer : Gen → Signature SortsLIA → Option Inst
  | .int c, _ => some (.int c)
  | .bool c, _ => some (.bool c)
  | .and, ⟨_, [.bool m n]⟩ => some (.and m n)
  | .or, ⟨_, [.bool m n]⟩ => some (.or m n)
  | .xor, ⟨_, [.bool m n]⟩ => some (.xor m n)
  | .not, ⟨_, [.bool m n]⟩ => some (.not m n)
  | .le, ⟨[.int m n, _], _⟩ => some (.le m n)
  | .lt, ⟨[.int m n, _], _⟩ => some (.lt m n)
  | .ge, ⟨[.int m n, _], _⟩ => some (.ge m n)
  | .gt, ⟨[.int m n, _], _⟩ => some (.gt m n)
  | .eq, ⟨[.int m n, _], _⟩ => some (.eq m n)
  | .ne, ⟨[.int m n, _], _⟩ => some (.ne m n)
  | .linear A B, ⟨[.int _ b], _⟩ => if h : linearFits A B then some (.linear A B b h) else none
  | .add, ⟨_, [.int m n]⟩ => some (.add m n)
  | .sub, ⟨_, [.int m n]⟩ => some (.sub m n)
  | .relu, ⟨_, [.int m n]⟩ => some (.relu m n)
  | .argmax, ⟨[s], [.int m n]⟩ => if h : m = 1 ∨ n = 1 then some (.argmax s m n h) else none
  | .min, ⟨[s], [.int m n]⟩ => if h : m = 1 ∨ n = 1 then some (.min s m n h) else none
  | .max, ⟨[s], [.int m n]⟩ => if h : m = 1 ∨ n = 1 then some (.max s m n h) else none
  | .transpose, ⟨[.int m n], _⟩ => some (.transpose m n)
  | .ite, ⟨_, [s]⟩ => some (.ite s)
  | .id, ⟨_, [s]⟩ => some (.id s)
  | .uninterpreted x, ⟨[s], []⟩ => some (.uninterpreted x true s)
  | .uninterpreted x, ⟨[], [s]⟩ => some (.uninterpreted x false s)
  | .anyInt m n, _ => some (.anyInt m n)
  | .anyBool m n, _ => some (.anyBool m n)
  | .zero, _ => some .zero
  | _, _ => none

instance : Elab SortsLIA Gen Inst where
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

instance : Sequential SortsLIA Gen where
  skip _ := .id

instance : Combinatorial SortsLIA Gen where
  havoc
    | .int m n => .anyInt m n
    | .bool m n => .anyBool m n
    -- havoc over a singleton is the singleton
    | .zero => .zero

end LIA

/-- The theory of LIA. -/
abbrev thrLIA : Theory SortsLIA := { gen := LIA.Gen, inst := LIA.Inst }

end Zrth
