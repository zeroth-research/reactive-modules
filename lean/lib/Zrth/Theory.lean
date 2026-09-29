import Zrth.Basic
import Zrth.Theory.Sorts

/-!
# Signatures and Theories
-/

namespace Zrth

universe u v w


/-! ----------------------------------------------------------
  ## Signature
- ------------------------------------------------------------/

/-- Signature of generators, operations, boxes, ... -/
structure Signature (S: Type u) [MultiSort S] where
  /-- The domain: a list of elements of the multi-sort S --/
  dom: List S
  /-- The codomain: a list of elements of the multi-sort S --/
  cod: List S
deriving DecidableEq

/-- Anything that has a signature implements this class:
    α HasSignature over the multi-sort S --/
class HasSignature (S : outParam (Type u)) [MultiSort S] (α : Type v) where
  signature : α → Signature S


variable {S : Type u} [MultiSort S]

/-- Map signature domain to a list of Lean types --/
def Signature.domType (sig: Signature S) :=
  sig.dom.map (fun s => (MultiSort.toType s))

/-- Map signature codomain to a list of Lean types --/
def Signature.codType (sig: Signature S) :=
  sig.cod.map (fun s => (MultiSort.toType s))


/-! ----------------------------------------------------------
  ## Elaboration
- ------------------------------------------------------------/

/-- Generators typed against the wires they are used with, like
`theory::Signature::check`. A generator `g : G` is polymorphic (`add` adds
matrices of any shape); its instances `I` are the generators at concrete
sorts (`add` at `2 × 3`), each with a signature.

`infer g σ` reads the sizes off the proposed signature `σ` the way the Rust
`check` reads them off the wires, and proposes an instance of `g`; `check`
then accepts `σ` iff it is the signature of that instance. -/
class Elab (S : outParam (Type u)) [MultiSort S] (G : Type v) (I : outParam (Type w))
    [HasSignature S I] where
  /-- The instance of `g` at the sizes read off `σ`, if any. -/
  infer : G → Signature S → Option I
  /-- The generator an instance instantiates. -/
  erase : I → G
  /-- Every instance is found from its own signature. -/
  infer_erase : ∀ i, infer (erase i) (HasSignature.signature i) = some i
  /-- Inference instantiates the generator it is given. -/
  erase_infer : ∀ g σ i, infer g σ = some i → erase i = g

namespace Elab

variable {G : Type v} {I : Type w} [HasSignature S I] [Elab S G I]

/-- Type-check a generator against a signature (mirrors `Signature::check`). -/
def check (g : G) (σ : Signature S) : Bool :=
  match infer g σ with
  | some i => HasSignature.signature i == σ
  | none => false

/-- `g` can be used at the signature `σ`. -/
abbrev WellTyped (g : G) (σ : Signature S) : Prop := check g σ = true

theorem infer_of_wellTyped {g : G} {σ : Signature S} (h : WellTyped g σ) :
    ∃ i, infer g σ = some i ∧ HasSignature.signature i = σ := by
  unfold WellTyped check at h
  split at h
  · next i hi => exact ⟨i, hi, by simpa using h⟩
  · contradiction

/-- The instance of a well-typed generator. -/
def elaborate (g : G) (σ : Signature S) (h : WellTyped g σ) : I :=
  match hi : infer g σ with
  | some i => i
  | none => absurd h (by simp [WellTyped, check, hi])

theorem infer_elaborate {g : G} {σ : Signature S} (h : WellTyped g σ) :
    infer g σ = some (elaborate g σ h) := by
  unfold elaborate; split
  · next hi => exact hi
  · next hi => simp [WellTyped, check, hi] at h

/-- The elaborated instance has the signature it was checked against. -/
theorem elaborate_sig {g : G} {σ : Signature S} (h : WellTyped g σ) :
    HasSignature.signature (elaborate g σ h) = σ := by
  obtain ⟨i, hi, hs⟩ := infer_of_wellTyped h
  rw [infer_elaborate h] at hi; cases hi; exact hs

/-- The elaborated instance instantiates the checked generator. -/
theorem erase_elaborate {g : G} {σ : Signature S} (h : WellTyped g σ) :
    erase (elaborate g σ h) = g :=
  erase_infer g σ _ (infer_elaborate h)

/-- Every instance type-checks at its signature. -/
theorem wellTyped_erase (i : I) : WellTyped (erase (G := G) i) (HasSignature.signature i) := by
  simp [WellTyped, check, infer_erase]

/-- Restrict to the generators satisfying `p` (the sub-signatures of
`theory::any`); the instances are restricted accordingly. -/
instance restrict (p : G → Bool) : HasSignature S {i : I // p (erase i)} where
  signature i := HasSignature.signature i.1

instance restrictElab (p : G → Bool) : Elab S {g : G // p g} {i : I // p (erase i)} where
  infer g σ :=
    match h : infer g.1 σ with
    | some i => some ⟨i, by rw [erase_infer _ _ _ h]; exact g.2⟩
    | none => none
  erase i := ⟨erase i.1, i.2⟩
  infer_erase i := by
    obtain ⟨i, hi⟩ := i
    split
    · next j h => simp [HasSignature.signature, infer_erase] at h; subst h; rfl
    · next h => simp [HasSignature.signature, infer_erase] at h
  erase_infer g σ i h := by
    obtain ⟨g, hg⟩ := g
    split at h
    · next j hj => cases h; exact Subtype.ext (erase_infer _ _ _ hj)
    · contradiction

end Elab


/-! ----------------------------------------------------------
  ## Theory
- ------------------------------------------------------------/

/-- A theory over a multi-sort `S`: polymorphic generators (as in the `theory`
crate), their instances at concrete sorts with their signatures, and the
elaboration from the former to the latter. -/
structure Theory (S: Type u) [MultiSort S] where
  /-- The generators --/
  gen     : Type v
  /-- The generators instantiated at concrete sorts --/
  inst    : Type w
  [hasSignature: HasSignature S inst]
  [hasElab: Elab S gen inst]

  /-- The multi-sort of the theory --/
  sort := S

attribute [instance] Theory.hasSignature Theory.hasElab

end Zrth
