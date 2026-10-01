import Zrth.Stochastic.Scheduler

/-!
# The filtration of a whole run

A run is observed in *hybrid time*: a point `(n, τ)` is the local time `τ`
within the step `n`, and the points are ordered lexicographically — all of a
step comes before the next one (`HybridTime`). This is the usual
parametrisation of hybrid time domains, with the step numbering the jumps and
the local time running along the flows.

* The filtration of a run in hybrid time (`Disc.runFiltration`): the
  σ-algebra at `(n, τ)` is that of the run up to its step `n`. It is constant
  in the local time: a flow brings no information, being determined, by the
  scheduler, from the state it starts from. Information only arrives with the
  steps, randomly at jumps.
* The state of the run at hybrid time (`Scheduler.state`): within the step `n`,
  the evolution the scheduler chooses from `ω n`, at local time `τ` (clamped to
  the duration of the step). It is adapted to the filtration
  (`Scheduler.measurable_state`), as is the time at which a step starts
  (`Scheduler.measurable_start`).
* Almost surely, the evolution in hybrid time is that of the system
  (`Scheduler.ae_evolution`): every step is a jump taking no time, to a state
  of positive probability, or a flow, whose state at local time `τ` follows a
  trajectory of the system, and ends at the next state of the run.

The rounds of a module refine the jumps further, atom by atom
(`Zrth.Stochastic.Round`); they are not spliced into the hybrid time here.
-/

namespace Zrth

open MeasureTheory ProbabilityTheory Filter Set Preorder

/-- Hybrid time: a step, and a local time within it, ordered lexicographically. -/
abbrev HybridTime := ℕ ×ₗ ℝ

/-- The step of a hybrid time. -/
abbrev HybridTime.step (p : HybridTime) : ℕ := (ofLex p).1

/-- The local time of a hybrid time, within its step. -/
abbrev HybridTime.elapsed (p : HybridTime) : ℝ := (ofLex p).2

theorem HybridTime.step_mono {p q : HybridTime} (h : p ≤ q) : p.step ≤ q.step :=
  (Prod.Lex.le_iff.1 h).elim le_of_lt fun h => h.1.le

/-- The filtration of a run in hybrid time: at `(n, τ)`, the run up to its step `n`. -/
def Disc.runFiltration (S : Type*) :
    Filtration HybridTime (MeasurableSpace.pi : MeasurableSpace (ℕ → Disc S)) where
  seq p := Disc.filtration ℕ S p.step
  mono' _ _ h := (Disc.filtration ℕ S).mono (HybridTime.step_mono h)
  le' p := (Disc.filtration ℕ S).le p.step

variable {E H : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [TopologicalSpace H]
  {I : ModelWithCorners ℝ E H} {S : Type*} [TopologicalSpace S] [ChartedSpace H S]

namespace Hybrid.Scheduler

variable {h : Hybrid I S} (σ : h.Scheduler)

/-- The local time `τ` clamped to the duration `d` of a step. -/
def clamp (d τ : ℝ) : ℝ := max 0 (min τ d)

theorem clamp_of_mem {d τ : ℝ} (hτ : τ ∈ Icc 0 d) : clamp d τ = τ := by
  rw [clamp, min_eq_left hτ.2, max_eq_right hτ.1]

theorem clamp_zero (d : ℝ) : clamp d 0 = 0 := max_eq_left (min_le_left 0 d)

/-- The state of the run `ω` at hybrid time `p`: within its step, the
    evolution the scheduler chooses, at the local time of `p`. -/
def state (p : HybridTime) (ω : ℕ → Disc S) : S :=
  σ.path p.step (ω p.step) (clamp (σ.time p.step (ω p.step)) p.elapsed)

/-- The state at hybrid time is adapted to the filtration of the run. -/
theorem measurable_state (p : HybridTime) :
    Measurable[Disc.runFiltration S p, (inferInstance : MeasurableSpace (Disc S))] (σ.state p) :=
  (Measurable.of_discrete (α := Disc S) (β := Disc S) (f := fun s =>
    σ.path p.step s (clamp (σ.time p.step s) p.elapsed))).comp
    (Disc.measurable_state p.step)

/-- A step starts where the run is at that step. -/
theorem state_start (n : ℕ) (ω : ℕ → Disc S) : σ.state (toLex (n, 0)) ω = ω n := by
  simp only [state, HybridTime.step, HybridTime.elapsed, ofLex_toLex, clamp_zero]
  exact σ.path_zero n (ω n)

/-- The time at which the step `n` starts: the durations of the steps before. -/
def start (n : ℕ) (ω : ℕ → Disc S) : ℝ := ∑ i ∈ Finset.range n, σ.time i (ω i)

/-- The start of a step is known when the step starts. -/
theorem measurable_start (n : ℕ) : Measurable[Disc.filtration ℕ S n] (σ.start n) :=
  Finset.measurable_sum _ fun i hi =>
    (Measurable.of_discrete (α := Disc S) (β := ℝ) (f := fun s => σ.time i s)).comp
      ((Disc.measurable_state i).mono ((Disc.filtration ℕ S).mono
        (Finset.mem_range.1 hi).le) le_rfl)

/-- Almost surely, the evolution of the run in hybrid time is that of the
    system: every step is a jump, taking no time, to a state of positive
    probability, or a flow along a trajectory of the system, which the state
    follows in local time, and which ends at the next state. -/
theorem ae_evolution : ∀ᵐ ω ∂σ.measure, ∀ n,
    (σ.time n (ω n) = 0 ∧ ∃ μ ∈ h.jump (ω n), (ω (n + 1) : S) ∈ μ.support) ∨
      (h.IsTrajectory (σ.path n (ω n)) (σ.time n (ω n)) ∧
        (∀ τ ∈ Icc 0 (σ.time n (ω n)), σ.state (toLex (n, τ)) ω = σ.path n (ω n) τ) ∧
        σ.path n (ω n) (σ.time n (ω n)) = ω (n + 1)) := by
  have hs : ∀ᵐ ω ∂σ.measure, ∀ n, (ω n : S) ∈ σ.reach n ∧
      (ω (n + 1) : S) ∈ (σ.next n (ω n)).support :=
    ae_all_iff.2 fun n => (σ.ae_mem_reach n).and (σ.ae_step n (σ.ae_mem_reach n))
  filter_upwards [hs] with ω hs n
  rcases σ.path_move n _ (σ.reachable_of_mem_reach n (hs n).1) with ⟨ht, hj⟩ | ⟨hγ, hn⟩
  · exact .inl ⟨ht, _, hj, (hs n).2⟩
  · have hend : (ω (n + 1) : S) = σ.path n (ω n) (σ.time n (ω n)) :=
      (PMF.mem_support_pure_iff _ _).1 (hn ▸ (hs n).2)
    refine .inr ⟨hγ, fun τ hτ => ?_, hend.symm⟩
    simp only [state, HybridTime.step, HybridTime.elapsed, ofLex_toLex, clamp_of_mem hτ]

end Hybrid.Scheduler

end Zrth
