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

/-- A use of the generator `gen` reading and writing the given wires. -/
structure Box [MultiSort S] (thr: Theory S)  where
  gen: thr.gen
  read: List (Wire S)
  write: List (Wire S)

  -- the generator must type-check against the wires
  typed: Elab.WellTyped gen ⟨read.map (fun w => w.sort), write.map (fun w => w.sort)⟩ := by decide

/-- A box has the signature of its wires. -/
instance [MultiSort S] {thr : Theory S} : HasSignature S (Box thr) where
  signature := fun b => ⟨b.read.map (fun w => w.sort), b.write.map (fun w => w.sort)⟩

/-- The instance of the generator at the sorts of the wires. -/
def Box.inst [MultiSort S] {thr : Theory S} (b : Box thr) : thr.inst :=
  Elab.elaborate b.gen _ b.typed

/-- The instance has the signature of the box. -/
theorem Box.inst_sig [MultiSort S] {thr : Theory S} (b : Box thr) :
    HasSignature.signature b.inst = HasSignature.signature b :=
  Elab.elaborate_sig b.typed

/-- A diagram of boxes with its input (`read`), output (`write`) and all wires. -/
structure Diagram {S: Type} [MultiSort S] (thr : Theory S)  where
  boxes : List (Box thr)
  read   : List (Wire S)
  write  : List (Wire S)
  wires : List (Wire S)

end Zrth
