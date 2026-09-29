import Zrth.Theory

/-!
# Bit-vectors

Sorts and signature of matrices of fixed-width bit-vectors (mirrors
`theory::bv` in the `theory` crate). Operations follow the SMT-LIB2 semantics.

`BV.Gen` are the generators of `theory::bv::BV`, polymorphic in the widths and
shapes; `BV.Inst` are their instances at concrete sorts, with signatures. All
sorts are constant: their tangent is the trivial sort `zero`, whose only
writer is the `zero` generator.

As in `BV::check`, the element-wise operations that only compare their operand
sorts (`not`, `id`, `neg`, `abs`, `and`, `or`, `xor`, `add`, `sub`, `mul`) also
accept `zero` operands.
-/

namespace Zrth

/-- Sorts of BV: `m × n` matrices of `w`-bit bit-vectors, and the trivial
tangent `zero`. -/
inductive SortsBV where | bv (w m n : Nat) | zero deriving DecidableEq, Repr

instance: Tangent SortsBV where
  toType := fun (s: SortsBV) =>
    match s with
    | .bv w m n => Mat (BitVec w) m n
    | .zero => Unit
  -- bit-vectors cannot move during delay
  T := fun _ => .zero

namespace BV

/-- The generators of BV (mirrors `theory::bv::BV`). -/
inductive Gen where
  /-- a literal; its width is taken from the write wire -/
  | const (c : Tensor Int)
  -- arithmetic modulo `2^w`
  | add | sub | mul | udiv | sdiv | umod | smod | neg | abs
  /-- matrix multiplication -/
  | matmul
  -- bit-wise operations
  | and | or | xor | not
  -- comparisons, into `1`-bit bit-vectors
  | ule | ult | uge | ugt | sle | slt | sge | sgt | eq | ne
  /-- if-then-else on a `1`-bit guard -/
  | ite
  | id
  /-- `x ≠ 0`, into `1`-bit bit-vectors -/
  | bvToBool
  /-- the bits `[high..=low]` -/
  | bitSelect (high low : Nat)
  /-- zero-extension by `extra` bits -/
  | extend (extra : Nat)
  /-- an uninterpreted source (writes one value) or sink (reads one value) -/
  | uninterpreted (name : String)
  /-- an arbitrary value -/
  | havoc (w m n : Nat)
  /-- the unique inhabitant of `zero` -/
  | zero

/-- A literal fits `w` bits: at most 64 bits, and (below 64) its entries are in
`[0, 2^w)`. -/
def constFits (w : Nat) (c : Tensor Int) : Bool :=
  w ≤ 64 && (w == 64 || c.entries.all fun x => decide (0 ≤ x) && decide (x < 2 ^ w))

/-- The generators of BV at concrete sorts. -/
inductive Inst where
  | const (w : Nat) (c : Tensor Int) (h : constFits w c = true)
  -- element-wise operations on operands of sort `s`
  | add (s : SortsBV)
  | sub (s : SortsBV)
  | mul (s : SortsBV)
  | neg (s : SortsBV)
  | abs (s : SortsBV)
  | and (s : SortsBV)
  | or (s : SortsBV)
  | xor (s : SortsBV)
  | not (s : SortsBV)
  | id (s : SortsBV)
  -- element-wise operations on `w`-bit `m × n` matrices
  | udiv (w m n : Nat)
  | sdiv (w m n : Nat)
  | umod (w m n : Nat)
  | smod (w m n : Nat)
  /-- `(m × k) · (k × n)` -/
  | matmul (w m k n : Nat)
  | ule (w m n : Nat)
  | ult (w m n : Nat)
  | uge (w m n : Nat)
  | ugt (w m n : Nat)
  | sle (w m n : Nat)
  | slt (w m n : Nat)
  | sge (w m n : Nat)
  | sgt (w m n : Nat)
  | eq (w m n : Nat)
  | ne (w m n : Nat)
  /-- `ite` with a guard of shape `gm × gn` (the shape is not checked) -/
  | ite (gm gn : Nat) (s : SortsBV)
  | bvToBool (w m n : Nat)
  | bitSelect (high low w m n : Nat) (h : low ≤ high ∧ high < w)
  | extend (extra w m n : Nat)
  /-- a sink (`read = true`) or a source (`read = false`) of sort `s` -/
  | uninterpreted (name : String) (read : Bool) (s : SortsBV)
  | havoc (w m n : Nat)
  | zero

/-- The signatures of the BV instances. -/
instance : HasSignature SortsBV Inst where
  signature := fun g => match g with
    | .const w c _ => { dom := [], cod := [.bv w c.rows c.cols] }
    | .add s | .sub s | .mul s | .and s | .or s | .xor s =>
        { dom := [s, s], cod := [s] }
    | .neg s | .abs s | .not s | .id s => { dom := [s], cod := [s] }
    | .udiv w m n | .sdiv w m n | .umod w m n | .smod w m n =>
        { dom := [.bv w m n, .bv w m n], cod := [.bv w m n] }
    | .matmul w m k n => { dom := [.bv w m k, .bv w k n], cod := [.bv w m n] }
    | .ule w m n | .ult w m n | .uge w m n | .ugt w m n | .sle w m n | .slt w m n
    | .sge w m n | .sgt w m n | .eq w m n | .ne w m n =>
        { dom := [.bv w m n, .bv w m n], cod := [.bv 1 m n] }
    | .ite gm gn s => { dom := [.bv 1 gm gn, s, s], cod := [s] }
    | .bvToBool w m n => { dom := [.bv w m n], cod := [.bv 1 m n] }
    | .bitSelect high low w m n _ => { dom := [.bv w m n], cod := [.bv (high - low + 1) m n] }
    | .extend extra w m n => { dom := [.bv w m n], cod := [.bv (w + extra) m n] }
    | .uninterpreted _ true s => { dom := [s], cod := [] }
    | .uninterpreted _ false s => { dom := [], cod := [s] }
    | .havoc w m n => { dom := [], cod := [.bv w m n] }
    | .zero => { dom := [], cod := [.zero] }

/-- The generator of an instance. -/
def Inst.erase : Inst → Gen
  | .const _ c _ => .const c
  | .add .. => .add
  | .sub .. => .sub
  | .mul .. => .mul
  | .neg .. => .neg
  | .abs .. => .abs
  | .and .. => .and
  | .or .. => .or
  | .xor .. => .xor
  | .not .. => .not
  | .id .. => .id
  | .udiv .. => .udiv
  | .sdiv .. => .sdiv
  | .umod .. => .umod
  | .smod .. => .smod
  | .matmul .. => .matmul
  | .ule .. => .ule
  | .ult .. => .ult
  | .uge .. => .uge
  | .ugt .. => .ugt
  | .sle .. => .sle
  | .slt .. => .slt
  | .sge .. => .sge
  | .sgt .. => .sgt
  | .eq .. => .eq
  | .ne .. => .ne
  | .ite .. => .ite
  | .bvToBool .. => .bvToBool
  | .bitSelect high low .. => .bitSelect high low
  | .extend extra .. => .extend extra
  | .uninterpreted x .. => .uninterpreted x
  | .havoc w m n => .havoc w m n
  | .zero => .zero

/-- Read the sorts of an instance off the wires; `check` compares the rest. -/
def Gen.infer : Gen → Signature SortsBV → Option Inst
  | .const c, ⟨_, [.bv w _ _]⟩ => if h : constFits w c = true then some (.const w c h) else none
  | .add, ⟨_, [s]⟩ => some (.add s)
  | .sub, ⟨_, [s]⟩ => some (.sub s)
  | .mul, ⟨_, [s]⟩ => some (.mul s)
  | .neg, ⟨_, [s]⟩ => some (.neg s)
  | .abs, ⟨_, [s]⟩ => some (.abs s)
  | .and, ⟨_, [s]⟩ => some (.and s)
  | .or, ⟨_, [s]⟩ => some (.or s)
  | .xor, ⟨_, [s]⟩ => some (.xor s)
  | .not, ⟨_, [s]⟩ => some (.not s)
  | .id, ⟨_, [s]⟩ => some (.id s)
  | .udiv, ⟨_, [.bv w m n]⟩ => some (.udiv w m n)
  | .sdiv, ⟨_, [.bv w m n]⟩ => some (.sdiv w m n)
  | .umod, ⟨_, [.bv w m n]⟩ => some (.umod w m n)
  | .smod, ⟨_, [.bv w m n]⟩ => some (.smod w m n)
  | .matmul, ⟨[.bv w m k, .bv _ _ n], _⟩ => some (.matmul w m k n)
  | .ule, ⟨[.bv w m n, _], _⟩ => some (.ule w m n)
  | .ult, ⟨[.bv w m n, _], _⟩ => some (.ult w m n)
  | .uge, ⟨[.bv w m n, _], _⟩ => some (.uge w m n)
  | .ugt, ⟨[.bv w m n, _], _⟩ => some (.ugt w m n)
  | .sle, ⟨[.bv w m n, _], _⟩ => some (.sle w m n)
  | .slt, ⟨[.bv w m n, _], _⟩ => some (.slt w m n)
  | .sge, ⟨[.bv w m n, _], _⟩ => some (.sge w m n)
  | .sgt, ⟨[.bv w m n, _], _⟩ => some (.sgt w m n)
  | .eq, ⟨[.bv w m n, _], _⟩ => some (.eq w m n)
  | .ne, ⟨[.bv w m n, _], _⟩ => some (.ne w m n)
  | .ite, ⟨[.bv _ gm gn, _, _], [s]⟩ => some (.ite gm gn s)
  | .bvToBool, ⟨[.bv w m n], _⟩ => some (.bvToBool w m n)
  | .bitSelect high low, ⟨[.bv w m n], _⟩ =>
      if h : low ≤ high ∧ high < w then some (.bitSelect high low w m n h) else none
  | .extend extra, ⟨[.bv w m n], _⟩ => some (.extend extra w m n)
  | .uninterpreted x, ⟨[s], []⟩ => some (.uninterpreted x true s)
  | .uninterpreted x, ⟨[], [s]⟩ => some (.uninterpreted x false s)
  | .havoc w m n, _ => some (.havoc w m n)
  | .zero, _ => some .zero
  | _, _ => none

instance : Elab SortsBV Gen Inst where
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

instance : Sequential SortsBV Gen where
  skip _ := .id

instance : Combinatorial SortsBV Gen where
  havoc
    | .bv w m n => .havoc w m n
    -- havoc over a singleton is the singleton
    | .zero => .zero

end BV

/-- The theory of BV. -/
abbrev thrBV : Theory SortsBV := { gen := BV.Gen, inst := BV.Inst }

end Zrth
