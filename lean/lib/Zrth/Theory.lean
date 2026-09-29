import Zrth.Basic
import Zrth.Theory.Sorts

/-!
# Signatures and Theories
-/

namespace Zrth

universe u v


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


/-- A sub-collection of generators has the signatures of the generators. -/
instance [HasSignature S α] {p : α → Bool} : HasSignature S {a : α // p a} where
  signature a := HasSignature.signature a.1


/-! ----------------------------------------------------------
  ## Structural generators
- ------------------------------------------------------------/

/-- Generators with an arbitrary value of every sort (mirrors
`theory::Combinatorial`). -/
class Combinatorial (S : outParam (Type u)) [MultiSort S] (G : Type v) [HasSignature S G] where
  /-- the generator choosing a value of sort `range` -/
  havoc : (range : S) → G

/-- Generators with a copy of every sort (mirrors `theory::Sequential`). -/
class Sequential (S : outParam (Type u)) [MultiSort S] (G : Type v) [HasSignature S G] where
  /-- the generator leaving a value of sort `range` unchanged -/
  skip : (range : S) → G
  skip_sig : ∀ s, HasSignature.signature (skip s) = ⟨[s], [s]⟩

/-- Generators with the zero of every tangent sort (mirrors
`theory::Differential`). -/
class Differential (S : outParam (Type u)) [Tangent S] (G : Type v) [HasSignature S G] where
  /-- the generator writing the zero rate of change; `range` is the tangent
  sort it writes -/
  zero : (range : S) → G
  /-- `zero` is only asked for tangent sorts (the derivative wires) -/
  zero_sig : ∀ s, HasSignature.signature (zero (Tangent.T s)) = ⟨[], [Tangent.T s]⟩


/-! ----------------------------------------------------------
  ## Theory
- ------------------------------------------------------------/

/-- A theory over a multi-sort `S`: generators together with their signatures -/
structure Theory (S: Type u) [MultiSort S] where
  /-- The generators --/
  gen     : Type v
  [hasSignature: HasSignature S gen]

  /-- The multi-sort of the theory --/
  sort := S

attribute [instance] Theory.hasSignature

end Zrth
