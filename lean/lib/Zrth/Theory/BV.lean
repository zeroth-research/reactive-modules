import Zrth.Theory

/-!
# Bit-vectors

Sorts and signature of matrices of fixed-width bit-vectors (mirrors
`theory::bv` in the `theory` crate). Operations follow the SMT-LIB2 semantics.

The generators are those of `theory::bv::BV` at concrete sorts, so each has a
fixed signature. All sorts are constant: their tangent is the trivial sort
`zero`, whose only writer is the `zero` generator.

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

/-- A literal fits `w` bits: at most 64 bits, and (below 64) its entries are in
`[0, 2^w)`. -/
def constFits (w : Nat) (c : Tensor Int) : Bool :=
  w ≤ 64 && (w == 64 || c.entries.all fun x => decide (0 ≤ x) && decide (x < 2 ^ w))

/-- The generators of BV (mirrors `theory::bv::BV`), at concrete sorts.
Operations follow the SMT-LIB2 semantics. -/
inductive Gen where
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

/-- The signatures of the BV generators. -/
instance : HasSignature SortsBV Gen where
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

instance : Sequential SortsBV Gen where
  skip s := .id s
  skip_sig _ := rfl

instance : Combinatorial SortsBV Gen where
  havoc
    | .bv w m n => .havoc w m n
    -- havoc over a singleton is the singleton
    | .zero => .zero
  havoc_sig s := by cases s <;> rfl

end BV

/-- The theory of BV. -/
abbrev thrBV : Theory SortsBV := { gen := BV.Gen }

end Zrth
