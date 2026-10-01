import Zrth.Atom
import Zrth.Hybrid.Basic

/-!
# Closed atoms as hybrid systems

A single atom is a hybrid system as soon as it depends on nothing outside
itself: it awaits nothing and reads only the variables it controls. Its state
is the valuation of its controlled variables: its initialisation gives the
initial distributions, its update the jumps (defined where the update is),
and its flow the flow.
-/

namespace Zrth

variable {V : Type*}
  {E : V → Type*} [∀ v, NormedAddCommGroup (E v)] [∀ v, NormedSpace ℝ (E v)]
  {H : V → Type*} [∀ v, TopologicalSpace (H v)]
  {I : ∀ v, ModelWithCorners ℝ (E v) (H v)}
  {M : V → Type*} [∀ v, TopologicalSpace (M v)] [∀ v, ChartedSpace (H v) (M v)]

/-- The hybrid system of a closed atom: it awaits nothing and reads only the
    variables it controls, whose current values it reads. It may jump wherever
    its update is defined, and that set is its only jump region. -/
def Atom.toHybrid (a : Atom I M) (hr : a.read ⊆ a.ctrl) (hw : a.wait = ∅) :
    Hybrid (Val.model I a.ctrl) (Val M a.ctrl) where
  init := FinDist.toPMF '' a.init (Val.empty hw)
  jump s := FinDist.toPMF '' a.update (Val.restrict hr s, Val.empty hw)
  flow s := a.flow s (Val.restrict hr s, Val.emptyTangent hw)
  regions := {{s | (a.update (Val.restrict hr s, Val.empty hw)).Nonempty}}

end Zrth
