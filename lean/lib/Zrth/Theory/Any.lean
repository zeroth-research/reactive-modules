import Zrth.Theory.LRA
import Zrth.Theory.LIA
import Zrth.Theory.BV

/-!
# Any theory

The union of LRA, LIA and BV over a common multi-sort, with the structural
generators `havoc`, `skip` and `zero` (mirrors `theory::any` in the `theory`
crate). A generator of a base theory type-checks against wires whose sorts all
belong to that theory.

The catch-alls `Combinatorial`, `Sequential` and `Differential` of
`theory::any` are the sub-signatures selecting the generators allowed in the
init, next and flow of an atom.
-/

namespace Zrth

/-- Sorts of all theories: boolean, real (of a differential grade), integer
and bit-vector matrices, and the trivial tangent `zero`. -/
inductive SortsAny where
  | bool (m n : Nat)
  | real (m n : Nat) (rank : Nat := 0)
  | int (m n : Nat)
  | bitVec (w m n : Nat)
  | zero
deriving DecidableEq, Repr

instance: Tangent SortsAny where
  toType := fun (s: SortsAny) =>
    match s with
    | .bool m n => Mat Bool m n
    | .real m n _ => Mat Float m n
    | .int m n => Mat Int m n
    | .bitVec w m n => Mat (BitVec w) m n
    | .zero => Unit
  T := fun (s: SortsAny) =>
    match s with
    -- reals grade up, the constant sorts collapse to `zero`
    | .real m n r => .real m n (r + 1)
    | .bool .. | .int .. | .bitVec .. | .zero => .zero

/-! ## Embeddings of the sorts of the base theories -/

def SortsLRA.toAny : SortsLRA → SortsAny
  | .real m n r => .real m n r
  | .bool m n => .bool m n
  | .zero => .zero

def SortsLIA.toAny : SortsLIA → SortsAny
  | .int m n => .int m n
  | .bool m n => .bool m n
  | .zero => .zero

def SortsBV.toAny : SortsBV → SortsAny
  | .bv w m n => .bitVec w m n
  | .zero => .zero

def SortsAny.toLRA? : SortsAny → Option SortsLRA
  | .real m n r => some (.real m n r)
  | .bool m n => some (.bool m n)
  | .zero => some .zero
  | _ => none

def SortsAny.toLIA? : SortsAny → Option SortsLIA
  | .int m n => some (.int m n)
  | .bool m n => some (.bool m n)
  | .zero => some .zero
  | _ => none

def SortsAny.toBV? : SortsAny → Option SortsBV
  | .bitVec w m n => some (.bv w m n)
  | .zero => some .zero
  | _ => none

section Embedding

variable {S : Type} [MultiSort S]

/-- The image of a signature along an embedding of sorts. -/
def Signature.map (f : S → SortsAny) (σ : Signature S) : Signature SortsAny :=
  ⟨σ.dom.map f, σ.cod.map f⟩

/-- The preimage of a signature along an embedding of sorts, if all its sorts
are in the image. -/
def Signature.pull (f : SortsAny → Option S) (σ : Signature SortsAny) : Option (Signature S) :=
  return ⟨← σ.dom.mapM f, ← σ.cod.mapM f⟩

omit [MultiSort S] in
theorem List.mapM_map_of_retract {f : S → SortsAny} {g : SortsAny → Option S}
    (h : ∀ s, g (f s) = some s) (l : List S) : (l.map f).mapM g = some l := by
  induction l with
  | nil => rfl
  | cons s l ih => simp [List.mapM_cons, h, ih]

theorem Signature.pull_map {f : S → SortsAny} {g : SortsAny → Option S}
    (h : ∀ s, g (f s) = some s) (σ : Signature S) : (σ.map f).pull g = some σ := by
  simp [Signature.map, Signature.pull, List.mapM_map_of_retract h]

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

/-- The generators of all theories at concrete sorts. -/
inductive Inst where
  | havoc (s : SortsAny)
  | skip (s : SortsAny)
  | zero (s : SortsAny)
  | lra (i : LRA.Inst)
  | lia (i : LIA.Inst)
  | bv (i : BV.Inst)

/-- The signatures of the instances: those of the base theories, embedded. -/
instance : HasSignature SortsAny Inst where
  signature := fun g => match g with
    | .havoc s => { dom := [], cod := [s] }
    | .skip s => { dom := [s], cod := [s] }
    | .zero s => { dom := [], cod := [s] }
    | .lra i => (HasSignature.signature i).map SortsLRA.toAny
    | .lia i => (HasSignature.signature i).map SortsLIA.toAny
    | .bv i => (HasSignature.signature i).map SortsBV.toAny

/-- The generator of an instance. -/
def Inst.erase : Inst → Gen
  | .havoc s => .havoc s
  | .skip s => .skip s
  | .zero s => .zero s
  | .lra i => .lra (Elab.erase i)
  | .lia i => .lia (Elab.erase i)
  | .bv i => .bv (Elab.erase i)

/-- Infer in the base theory, on the wires cast to its sorts. -/
def Gen.infer : Gen → Signature SortsAny → Option Inst
  | .havoc s, _ => some (.havoc s)
  | .skip s, _ => some (.skip s)
  | .zero s, _ => some (.zero s)
  | .lra g, σ => do .lra <$> Elab.infer g (← σ.pull SortsAny.toLRA?)
  | .lia g, σ => do .lia <$> Elab.infer g (← σ.pull SortsAny.toLIA?)
  | .bv g, σ => do .bv <$> Elab.infer g (← σ.pull SortsAny.toBV?)

theorem toLRA?_toAny (s : SortsLRA) : s.toAny.toLRA? = some s := by cases s <;> rfl
theorem toLIA?_toAny (s : SortsLIA) : s.toAny.toLIA? = some s := by cases s <;> rfl
theorem toBV?_toAny (s : SortsBV) : s.toAny.toBV? = some s := by cases s <;> rfl

instance : Elab SortsAny Gen Inst where
  infer := Gen.infer
  erase := Inst.erase
  infer_erase i := by
    cases i with
    | havoc | skip | zero => rfl
    | lra i =>
      show Gen.infer _ ((HasSignature.signature i).map _) = _
      simp only [Inst.erase, Gen.infer, Signature.pull_map toLRA?_toAny]
      simp [Elab.infer_erase]
    | lia i =>
      show Gen.infer _ ((HasSignature.signature i).map _) = _
      simp only [Inst.erase, Gen.infer, Signature.pull_map toLIA?_toAny]
      simp [Elab.infer_erase]
    | bv i =>
      show Gen.infer _ ((HasSignature.signature i).map _) = _
      simp only [Inst.erase, Gen.infer, Signature.pull_map toBV?_toAny]
      simp [Elab.infer_erase]
  erase_infer g σ i h := by
    cases g with
    | havoc | skip | zero => cases h; rfl
    | lra g =>
      simp only [Gen.infer] at h
      cases hp : σ.pull SortsAny.toLRA? with
      | none => simp [hp] at h
      | some σ' =>
        cases hi : Elab.infer g σ' with
        | none => simp [hp, hi] at h
        | some i' =>
          simp [hp, hi] at h; subst h
          simp [Inst.erase, Elab.erase_infer _ _ _ hi]
    | lia g =>
      simp only [Gen.infer] at h
      cases hp : σ.pull SortsAny.toLIA? with
      | none => simp [hp] at h
      | some σ' =>
        cases hi : Elab.infer g σ' with
        | none => simp [hp, hi] at h
        | some i' =>
          simp [hp, hi] at h; subst h
          simp [Inst.erase, Elab.erase_infer _ _ _ hi]
    | bv g =>
      simp only [Gen.infer] at h
      cases hp : σ.pull SortsAny.toBV? with
      | none => simp [hp] at h
      | some σ' =>
        cases hi : Elab.infer g σ' with
        | none => simp [hp, hi] at h
        | some i' =>
          simp [hp, hi] at h; subst h
          simp [Inst.erase, Elab.erase_infer _ _ _ hi]

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

instance : Combinatorial SortsAny Gen := ⟨.havoc⟩
instance : Sequential SortsAny Gen := ⟨.skip⟩
instance : Differential SortsAny Gen := ⟨.zero⟩

instance : Combinatorial SortsAny {g : Gen // g.isCombinatorial} := ⟨fun s => ⟨.havoc s, rfl⟩⟩
instance : Combinatorial SortsAny {g : Gen // g.isSequential} := ⟨fun s => ⟨.havoc s, rfl⟩⟩
instance : Sequential SortsAny {g : Gen // g.isSequential} := ⟨fun s => ⟨.skip s, rfl⟩⟩
instance : Differential SortsAny {g : Gen // g.isDifferential} := ⟨fun s => ⟨.zero s, rfl⟩⟩

end Any

/-- The theory of all theories. -/
abbrev thrAny : Theory SortsAny := { gen := Any.Gen, inst := Any.Inst }

/-- The theory of the init of an atom. -/
abbrev thrCombinatorial : Theory SortsAny :=
  { gen := {g : Any.Gen // g.isCombinatorial}
    inst := {i : Any.Inst // Any.Gen.isCombinatorial (Elab.erase i)} }

/-- The theory of the next of an atom. -/
abbrev thrSequential : Theory SortsAny :=
  { gen := {g : Any.Gen // g.isSequential}
    inst := {i : Any.Inst // Any.Gen.isSequential (Elab.erase i)} }

/-- The theory of the flow of an atom. -/
abbrev thrDifferential : Theory SortsAny :=
  { gen := {g : Any.Gen // g.isDifferential}
    inst := {i : Any.Inst // Any.Gen.isDifferential (Elab.erase i)} }

end Zrth
