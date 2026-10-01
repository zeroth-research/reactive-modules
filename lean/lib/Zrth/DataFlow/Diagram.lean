import Zrth.Theory

/-!
# Data-flow diagrams

Diagrams of boxes (instances of generators of a signature) connected by wires.
-/

namespace Zrth

/-- A wire carrying values of the sort `sort`; wires are identified by `id`. -/
structure Wire (S : Type) [MultiSort S] where
  id    : Nat
  sort  : S
deriving Repr

/-- An instance of the generator `gen` reading and writing the given wires. -/
structure Box [MultiSort S] (thr: Theory S)  where
  gen: thr.gen
  read: List (Wire S)
  write: List (Wire S)

  -- the wires must fit the signature of the generator
  read_sig: read.map (fun w => w.sort) = (HasSignature.signature gen).dom := by rfl
  write_sig: write.map (fun w => w.sort) = (HasSignature.signature gen).cod := by rfl

/-- A box has the signature of its generator. -/
instance [MultiSort S] {thr : Theory S} : HasSignature S (Box thr) where
  signature := fun b => HasSignature.signature b.gen

/-- A diagram of boxes with its input (`read`), output (`write`) and all wires. -/
structure Diagram {S: Type} [MultiSort S] (thr : Theory S)  where
  boxes : List (Box thr)
  read   : List (Wire S)
  write  : List (Wire S)
  wires : List (Wire S)

end Zrth
