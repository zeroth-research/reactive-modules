import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Probability.Process.Filtration

/-!
# Evolutions and their filtrations

An *evolution* is a function `ι → S` of a time `ι`: discrete for the steps of a
run and the atoms of a round, continuous for a flow. Its information grows with
time: `Evolution.filtration ι S` is the filtration whose σ-algebra at `i` is
that of the evolution up to `i`, and the state at `i` is known at `i`
(`Evolution.measurable_state`).

The states carry any σ-algebra in which points are measurable — for real
valuations, the Borel one. The distributions of the system being discrete,
only countably many states ever matter at a step, which keeps the events of
the proofs measurable.

A flow is an evolution in continuous time (`Evolution.flowMeasure`):
deterministic, once its trajectory is chosen.
-/

namespace Zrth

open MeasureTheory Set Preorder

namespace Evolution

variable {S : Type*} [MeasurableSpace S]

/-- The filtration of the evolutions `ι → S`: at `i`, the evolution up to `i`. -/
def filtration (ι S : Type*) [Preorder ι] [MeasurableSpace S] :
    Filtration ι (MeasurableSpace.pi : MeasurableSpace (ι → S)) :=
  Filtration.piLE

/-- An evolution is adapted to its filtration: its state at `i` is known at `i`. -/
theorem measurable_state {ι : Type*} [Preorder ι] (i : ι) :
    Measurable[filtration ι S i] fun ω : ι → S => ω i :=
  (measurable_pi_apply (⟨i, Set.mem_Iic.2 le_rfl⟩ : Set.Iic i)).comp
    (comap_measurable (restrictLe i))

/-- The evolution of a flow along the trajectory `γ`, in continuous time: a
    (deterministic) process on the trajectories, adapted to the filtration
    `filtration ℝ S` of the trajectory up to each time. -/
noncomputable def flowMeasure (γ : ℝ → S) : Measure (ℝ → S) := Measure.dirac γ

/-- A flow surely follows its trajectory. -/
instance (γ : ℝ → S) : IsProbabilityMeasure (flowMeasure γ) := by
  unfold flowMeasure; infer_instance

/-- A flow ends as its move: the state at its duration `d` is the next state. -/
theorem flowMeasure_map [MeasurableSingletonClass S] (γ : ℝ → S) (d : ℝ) :
    (flowMeasure γ).map (fun ω => ω d) = (PMF.pure (γ d)).toMeasure := by
  rw [flowMeasure, Measure.map_dirac' (measurable_pi_apply d), PMF.toMeasure_pure]

end Evolution

end Zrth
