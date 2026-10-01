import Zrth.Hybrid.Round
import Zrth.Stochastic.Disc

/-!
# Rounds, atom by atom

A round of a module is itself a small stochastic process: the atoms draw one
after the other. `Module.roundMeasure` is the measure on the traces of a round
(the valuations before each atom and after the last one), adapted to the
filtration `Disc.filtration ℕ`, whose σ-algebra at `i` is the information of
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

namespace Module

variable {m : Module I M}

/-- The evaluation of a round from `s`, atom by atom: the probability measure
    on its traces, adapted to the filtration `Disc.filtration ℕ (Val M m.ctrl)`,
    whose σ-algebra at `i` is that of the draws of the first `i` atoms. -/
noncomputable def roundMeasure (κ : m.Draws) (s : Val M m.ctrl) :
    Measure (ℕ → Disc (Val M m.ctrl)) :=
  PMF.toMeasure (α := ℕ → Disc (Val M m.ctrl)) (m.trace κ m.atoms (fun _ h => h) s)

instance (κ : m.Draws) (s : Val M m.ctrl) : IsProbabilityMeasure (roundMeasure κ s) :=
  PMF.toMeasure.isProbabilityMeasure (α := ℕ → Disc (Val M m.ctrl)) _

/-- The evaluation ends as the round: after the last atom, the valuation is
    distributed as the round. -/
theorem roundMeasure_map (κ : m.Draws) (s : Val M m.ctrl) :
    (roundMeasure κ s).map (fun ω => ω m.atoms.length) =
      Disc.measure (m.round κ m.atoms (fun _ h => h) s) := by
  exact (PMF.toMeasure_map (α := ℕ → Disc (Val M m.ctrl)) (β := Disc (Val M m.ctrl))
    (fun ω => ω m.atoms.length) (m.trace κ m.atoms (fun _ h => h) s)
    (measurable_pi_apply _)).trans
    (congrArg (PMF.toMeasure (α := Disc (Val M m.ctrl))) trace_map_length)

/-- Almost surely, the evaluation starts from `s`, and every atom in turn
    overrides the valuation drawn so far with its draw, given it. -/
theorem ae_roundMeasure (κ : m.Draws) (s : Val M m.ctrl) :
    ∀ᵐ ω ∂roundMeasure κ s, (ω 0 : Val M m.ctrl) = s ∧ ∀ i (hi : i < m.atoms.length),
      ∃ c : Val M m.atoms[i].ctrl, c ∈ (κ m.atoms[i] (List.getElem_mem hi) (ω i)).support ∧
        (ω (i + 1) : Val M m.ctrl) = Val.override (ω i) c := by
  rw [ae_iff]
  apply measure_mono_null (t := ((m.trace κ m.atoms (fun _ h => h) s).support : Set (ℕ → Val M m.ctrl))ᶜ)
  · intro ω hω hs
    exact hω (trace_step hs)
  · exact (PMF.toMeasure_apply_eq_zero_iff (α := ℕ → Disc (Val M m.ctrl)) _
      (PMF.support_countable (α := ℕ → Disc (Val M m.ctrl)) _).measurableSet.compl).2
      disjoint_compl_right

end Module

end Zrth
