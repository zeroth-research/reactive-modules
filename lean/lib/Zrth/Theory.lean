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

/-- Anything that has a signature implements this class:
    α HasSignature over the multi-sort S --/
class HasSignature {S : outParam (Type u)} [MultiSort S] (α : Type v) where
  signature : α → Signature S


variable {S : Type u} [MultiSort S]

/-- Map signature domain to a list of Lean types --/
def Signature.domType (sig: Signature S) :=
  sig.dom.map (fun s => (MultiSort.toType s))

/-- Map signature codomain to a list of Lean types --/
def Signature.codType (sig: Signature S) :=
  sig.cod.map (fun s => (MultiSort.toType s))


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
