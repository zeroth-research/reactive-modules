import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Probability.Process.Filtration

/-!
# Discrete σ-algebras and evolutions

The stochastic semantics measures sets of states and of their evolutions. The
states are given the *discrete* σ-algebra (`Disc S`, every set is measurable),
so that any choice made from a state — by a scheduler, an atom — is
measurable, whatever the state space. The distributions being discrete, only
countably many states ever matter.

An *evolution* is a function `ι → S` of a time `ι`: discrete for runs (steps)
and rounds (atoms), continuous for flows. Its information grows with time:
`Disc.filtration ι S` is the filtration whose σ-algebra at `i` is that of the
evolution up to `i`, and the state at `i` is known at `i`
(`Disc.measurable_state`).

A flow is an evolution in continuous time (`Disc.flowMeasure`): deterministic,
once its trajectory is chosen.
-/

namespace Zrth

open MeasureTheory ProbabilityTheory Filter Set Preorder

/-- A type with the discrete σ-algebra. -/
def Disc (S : Type*) : Type _ := S

instance {S : Type*} : MeasurableSpace (Disc S) := ⊤

instance {S : Type*} : DiscreteMeasurableSpace (Disc S) :=
  ⟨fun _ => MeasurableSpace.measurableSet_top⟩

/-- A distribution on `S` as a measure for the discrete σ-algebra. -/
noncomputable def Disc.measure {S : Type*} (μ : PMF S) : Measure (Disc S) :=
  PMF.toMeasure (α := Disc S) μ

instance {S : Type*} (μ : PMF S) : IsProbabilityMeasure (Disc.measure μ) :=
  PMF.toMeasure.isProbabilityMeasure (α := Disc S) μ

theorem Disc.measure_eq_zero_iff {S : Type*} (μ : PMF S) (A : Set S) :
    Disc.measure μ (A : Set (Disc S)) = 0 ↔ Disjoint μ.support A :=
  PMF.toMeasure_apply_eq_zero_iff (α := Disc S) μ MeasurableSet.of_discrete

instance {S : Type*} : MeasurableSingletonClass (Disc S) := ⟨fun _ => MeasurableSet.of_discrete⟩

/-- The filtration of the evolutions `ι → S` (of the runs in discrete time,
    of the trajectories in continuous time): the σ-algebra at `i` is that of
    the evolution up to `i`. -/
def Disc.filtration (ι S : Type*) [Preorder ι] :
    Filtration ι (MeasurableSpace.pi : MeasurableSpace (ι → Disc S)) :=
  Filtration.piLE

/-- An evolution is adapted to the filtration: its state at `i` is known at `i`. -/
theorem Disc.measurable_state {ι S : Type*} [Preorder ι] (i : ι) :
    Measurable[Disc.filtration ι S i] fun ω : ι → Disc S => ω i :=
  (measurable_pi_apply (⟨i, Set.mem_Iic.2 le_rfl⟩ : Set.Iic i)).comp
    (comap_measurable (restrictLe i))

theorem Disc.measure_pure {S : Type*} (a : S) :
    Disc.measure (PMF.pure a) = Measure.dirac (α := Disc S) a :=
  PMF.toMeasure_pure (α := Disc S) a

/-- The evolution of a flow along the trajectory `γ`, in continuous time: a
    (deterministic) process on the trajectories, adapted to the filtration
    `Disc.filtration ℝ S` of the trajectory up to each time. -/
noncomputable def Disc.flowMeasure {S : Type*} (γ : ℝ → S) : Measure (ℝ → Disc S) :=
  Measure.dirac (α := ℝ → Disc S) γ

instance {S : Type*} (γ : ℝ → S) : IsProbabilityMeasure (Disc.flowMeasure γ) :=
  Measure.dirac.isProbabilityMeasure (α := ℝ → Disc S)

/-- A flow ends as its move: the state at its duration `d` is the next state. -/
theorem Disc.flowMeasure_map {S : Type*} (γ : ℝ → S) (d : ℝ) :
    (Disc.flowMeasure γ).map (fun ω => ω d) = Disc.measure (PMF.pure (γ d)) :=
  (Measure.map_dirac' (measurable_pi_apply (X := fun _ : ℝ => Disc S) d) γ).trans
    (Disc.measure_pure _).symm

end Zrth
