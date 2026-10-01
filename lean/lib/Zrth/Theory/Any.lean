import Zrth.Theory.LRA
import Zrth.Theory.LIA
import Zrth.Theory.BV

/-!
# Any theory

The union of LRA, LIA and BV over a common multi-sort, with the structural
generators `havoc`, `skip` and `zero` (mirrors `theory::any` in the `theory`
crate). A generator of a base theory has the signature it has there, embedded.

The catch-alls `Combinatorial`, `Sequential` and `Differential` of
`theory::any` are the sub-signatures selecting the generators allowed in the
init, next and flow of an atom.
-/

namespace Zrth

/-- Sorts of all theories: boolean, real, integer and bit-vector matrices, the
`(k+1)`-th derivatives of real matrices, and the trivial tangent `zero`. -/
inductive SortsAny where
  | bool (m n : Nat)
  | real (m n : Nat)
  | δreal (m n k : Nat)
  | int (m n : Nat)
  | bitVec (w m n : Nat)
  | zero
deriving DecidableEq, Repr

instance: Tangent SortsAny where
  toType := fun (s: SortsAny) =>
    match s with
    | .bool m n => Mat Bool m n
    | .real m n | .δreal m n _ => Mat Float m n
    | .int m n => Mat Int m n
    | .bitVec w m n => Mat (BitVec w) m n
    | .zero => Unit
  T := fun (s: SortsAny) =>
    match s with
    -- reals are differentiated, the constant sorts collapse to `zero`
    | .real m n => .δreal m n 0
    | .δreal m n k => .δreal m n (k + 1)
    | .bool .. | .int .. | .bitVec .. | .zero => .zero

/-! ## Embeddings of the sorts of the base theories -/

def SortsLRA.toAny : SortsLRA → SortsAny
  | .real m n => .real m n
  | .δreal m n k => .δreal m n k
  | .bool m n => .bool m n
  | .zero => .zero

def SortsLIA.toAny : SortsLIA → SortsAny
  | .int m n => .int m n
  | .bool m n => .bool m n
  | .zero => .zero

def SortsBV.toAny : SortsBV → SortsAny
  | .bv w m n => .bitVec w m n
  | .zero => .zero

section Embedding

variable {S : Type} [MultiSort S]

/-- The image of a signature along an embedding of sorts. -/
def Signature.map (f : S → SortsAny) (σ : Signature S) : Signature SortsAny :=
  ⟨σ.dom.map f, σ.cod.map f⟩

end Embedding

namespace Any

/-- The generators of all theories (mirrors `theory::any::Any`). -/
inductive Gen where
  /-- an arbitrary value of sort `s` -/
  | havoc (s : SortsAny)
  /-- no change: copies a value of sort `s` -/
  | skip (s : SortsAny)
  /-- zero rate of change: writes the zero of the tangent sort `s` -/
  | zero (s : SortsAny)
  | lra (g : LRA.Gen)
  | lia (g : LIA.Gen)
  | bv (g : BV.Gen)

/-- The signatures of the generators: those of the base theories, embedded. -/
instance : HasSignature SortsAny Gen where
  signature := fun g => match g with
    | .havoc s => { dom := [], cod := [s] }
    | .skip s => { dom := [s], cod := [s] }
    | .zero s => { dom := [], cod := [s] }
    | .lra i => (HasSignature.signature i).map SortsLRA.toAny
    | .lia i => (HasSignature.signature i).map SortsLIA.toAny
    | .bv i => (HasSignature.signature i).map SortsBV.toAny

/-! ## Sub-signatures -/

/-- Generators of the init of an atom (`theory::any::Combinatorial`). -/
def Gen.isCombinatorial : Gen → Bool
  | .havoc _ | .lra _ | .lia _ | .bv _ => true
  | .skip _ | .zero _ => false

/-- Generators of the next of an atom (`theory::any::Sequential`). -/
def Gen.isSequential : Gen → Bool
  | .havoc _ | .skip _ | .lra _ | .lia _ | .bv _ => true
  | .zero _ => false

/-- Generators of the flow of an atom (`theory::any::Differential`). -/
def Gen.isDifferential : Gen → Bool
  | .zero _ | .lra _ => true
  | .havoc _ | .skip _ | .lia _ | .bv _ => false

-- `Combinatorial -> Sequential`, as in the cast lattice of `theory::any`
theorem Gen.isSequential_of_isCombinatorial {g : Gen} (h : g.isCombinatorial) :
    g.isSequential := by
  cases g <;> simp_all [isCombinatorial, isSequential]

instance : Combinatorial SortsAny Gen where
  havoc := .havoc
  havoc_sig _ := rfl
instance : Sequential SortsAny Gen where
  skip := .skip
  skip_sig _ := rfl
instance : Differential SortsAny Gen where
  zero := .zero
  zero_sig _ := rfl

instance : Combinatorial SortsAny {g : Gen // g.isCombinatorial} where
  havoc s := ⟨.havoc s, rfl⟩
  havoc_sig _ := rfl
instance : Combinatorial SortsAny {g : Gen // g.isSequential} where
  havoc s := ⟨.havoc s, rfl⟩
  havoc_sig _ := rfl
instance : Sequential SortsAny {g : Gen // g.isSequential} where
  skip s := ⟨.skip s, rfl⟩
  skip_sig _ := rfl
instance : Differential SortsAny {g : Gen // g.isDifferential} where
  zero s := ⟨.zero s, rfl⟩
  zero_sig _ := rfl

end Any

/-- The theory of all theories. -/
abbrev thrAny : Theory SortsAny := { gen := Any.Gen }

/-- The theory of the init of an atom. -/
abbrev thrCombinatorial : Theory SortsAny :=
  { gen := {g : Any.Gen // g.isCombinatorial} }

/-- The theory of the next of an atom. -/
abbrev thrSequential : Theory SortsAny :=
  { gen := {g : Any.Gen // g.isSequential} }

/-- The theory of the flow of an atom. -/
abbrev thrDifferential : Theory SortsAny :=
  { gen := {g : Any.Gen // g.isDifferential} }

end Zrth
