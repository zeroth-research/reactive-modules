import Zrth.Hybrid.Round
import Zrth.Stochastic.Evolution

/-!
# Rounds, atom by atom

A round of a module is itself a small stochastic process: the atoms draw one
after the other. `Module.roundMeasure` is the measure on the traces of a round
(the valuations before each atom and after the last one), adapted to the
filtration `Evolution.filtration ℕ`, whose σ-algebra at `i` is the information of
the draws of the first `i` atoms. After the last atom, the valuation is
distributed as the round (`roundMeasure_map`): the refinement is faithful.
-/

namespace Zrth

open MeasureTheory ProbabilityTheory Filter Set Preorder

variable {V : Type*} [DecidableEq V]
  {E : V → Type*} [∀ v, NormedAddCommGroup (E v)] [∀ v, NormedSpace ℝ (E v)]
  {H : V → Type*} [∀ v, TopologicalSpace (H v)]
  {I : ∀ v, ModelWithCorners ℝ (E v) (H v)}
  {M : V → Type*} [∀ v, TopologicalSpace (M v)] [∀ v, ChartedSpace (H v) (M v)]
  [∀ v, MeasurableSpace (M v)] [∀ v, MeasurableSingletonClass (M v)]

namespace Module

variable {m : Module I M}

/-- The evaluation of a round from `s`, atom by atom: the probability measure
    on its traces, adapted to the filtration `Evolution.filtration ℕ`, whose
    σ-algebra at `i` is that of the draws of the first `i` atoms. -/
noncomputable def roundMeasure (κ : m.Draws) (s : Val M m.ctrl) : Measure (ℕ → Val M m.ctrl) :=
  (m.trace κ m.atoms (fun _ h => h) s).toMeasure

/-- The evaluation of a round is a probability. -/
instance (κ : m.Draws) (s : Val M m.ctrl) : IsProbabilityMeasure (roundMeasure κ s) := by
  unfold roundMeasure; infer_instance

omit [∀ v, MeasurableSingletonClass (M v)] in
/-- The evaluation ends as the round: after the last atom, the valuation is
    distributed as the round. -/
theorem roundMeasure_map (κ : m.Draws) (s : Val M m.ctrl) :
    (roundMeasure κ s).map (fun ω => ω m.atoms.length) =
      (m.round κ m.atoms (fun _ h => h) s).toMeasure := by
  rw [roundMeasure, PMF.toMeasure_map _ _ (measurable_pi_apply _), trace_map_length]

/-- Almost surely, the evaluation starts from `s`, and every atom in turn
    overrides the valuation drawn so far with its draw, given it. -/
theorem ae_roundMeasure (κ : m.Draws) (s : Val M m.ctrl) :
    ∀ᵐ ω ∂roundMeasure κ s, ω 0 = s ∧ ∀ i (hi : i < m.atoms.length),
      ∃ c : Val M m.atoms[i].ctrl, c ∈ (κ m.atoms[i] (List.getElem_mem hi) (ω i)).support ∧
        ω (i + 1) = Val.override (ω i) c := by
  -- Every trace in the support does so (`trace_step`), and the support has
  -- probability one.
  rw [ae_iff]
  apply measure_mono_null (t := (m.trace κ m.atoms (fun _ h => h) s).supportᶜ)
  · intro ω hω hs
    exact hω (trace_step hs)
  · exact (PMF.toMeasure_apply_eq_zero_iff _
      (PMF.support_countable _).measurableSet.compl).2 disjoint_compl_right

end Module

end Zrth
