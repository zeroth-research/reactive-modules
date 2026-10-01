import Zrth.Hybrid.Basic
import Zrth.Stochastic.Evolution
import Mathlib.Probability.Kernel.IonescuTulcea.Traj

/-!
# Schedulers and the chain of steps

A hybrid system chooses nondeterministically (which distribution, which
trajectory, how long to flow) and at random (from the distribution). A
*scheduler* resolves the nondeterminism: at every step, from the current state,
it chooses a move — a jump distribution, or a flow along a trajectory for some
duration — and initially an initial distribution. What remains is random: a
Markov chain of the states at the steps, the *embedded chain* of the process
(as in Davis' construction of piecewise-deterministic Markov processes). Its
runs are measured by Ionescu-Tulcea (`Scheduler.measure` on `ℕ → S`), and its
run is adapted to the filtration of the first steps (`Evolution.filtration ℕ S`).
The process in continuous time is built from it (`Zrth.Stochastic.Continuous`).

The states carry a σ-algebra in which points are measurable (for real
valuations, the Borel one), and the scheduler chooses measurably: from the
step number and the current state only (it is *memoryless*), its distributions,
durations and trajectories depending measurably on the state.

The main result is that sampling a run gives, almost surely, a possible run of
the system (`ae_isRun`): whatever holds of every possible run holds almost
surely under every scheduler (`ae_of_isRun`, `ae_forall_of_reachable`). The
proof follows the states reached with positive probability, a countable set at
every step (`reach`), which makes the events of the proof measurable.
-/

namespace Zrth

open MeasureTheory ProbabilityTheory Filter Set Preorder

variable {E H : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [TopologicalSpace H]
  {I : ModelWithCorners ℝ E H} {S : Type*} [TopologicalSpace S] [ChartedSpace H S]
  [MeasurableSpace S] [MeasurableSingletonClass S]

namespace Hybrid

/-- A memoryless, measurable scheduler of `h`: an initial distribution, and, at
    every step and (reachable) state, a move — a jump distribution, or a flow
    along a trajectory (`path`) for some duration (`time`). -/
structure Scheduler (h : Hybrid I S) where
  /-- The initial distribution. -/
  init : PMF S
  init_mem : init ∈ h.init
  /-- The distribution of the next state. -/
  next : ℕ → S → PMF S
  /-- The duration of the step. -/
  time : ℕ → S → ℝ
  /-- The evolution during the step, in its local time: the trajectory of a flow
      (for a jump, only its start matters). -/
  path : ℕ → S → ℝ → S
  path_zero : ∀ n s, path n s 0 = s
  /-- Every step is a jump, taking no time, or a flow along its path, reaching
      its end deterministically. -/
  path_move : ∀ n s, h.Reachable s →
    (time n s = 0 ∧ next n s ∈ h.jump s) ∨
      (h.IsTrajectory (path n s) (time n s) ∧ next n s = PMF.pure (path n s (time n s)))
  /-- Time does not go backwards. -/
  time_nonneg : ∀ n s, 0 ≤ time n s
  measurable_next : ∀ n, Measurable fun s => (next n s).toMeasure
  measurable_time : ∀ n, Measurable (time n)
  /-- The evolution depends measurably on the state and the local time. -/
  measurable_path : ∀ n, Measurable fun p : S × ℝ => path n p.1 p.2

namespace Scheduler

variable {h : Hybrid I S} (σ : h.Scheduler)

omit [MeasurableSingletonClass S] in
/-- Every step is a move of `h` from every reachable state. -/
theorem move (n : ℕ) (s : S) (hs : h.Reachable s) : h.Move s (σ.time n s) (σ.next n s) := by
  rcases σ.path_move n s hs with ⟨ht, hj⟩ | ⟨hγ, hn⟩
  · rw [ht]; exact .jump hj
  · have := Move.flow hγ
    rwa [σ.path_zero, ← hn] at this

/-- The distribution of the next state at step `n`, as a Markov kernel. -/
noncomputable def step (n : ℕ) : Kernel S S :=
  ⟨fun s => (σ.next n s).toMeasure, σ.measurable_next n⟩

/-- Every step goes somewhere: its kernel is a probability measure everywhere. -/
instance (n : ℕ) : IsMarkovKernel (σ.step n) :=
  ⟨fun s => by simp only [step, Kernel.coe_mk]; infer_instance⟩

/-- The step kernel on the runs so far: from their last state. -/
noncomputable def kernel (n : ℕ) : Kernel (Π _ : Finset.Iic n, S) S :=
  (σ.step n).comap (fun x => x ⟨n, Finset.mem_Iic.2 le_rfl⟩) (measurable_pi_apply _)

/-- So is the step kernel on the runs so far. -/
instance (n : ℕ) : IsMarkovKernel (σ.kernel n) := by
  unfold kernel; infer_instance

/-- The probability measure on the runs of the embedded chain. -/
noncomputable def measure : Measure (ℕ → S) :=
  Kernel.trajMeasure (X := fun _ => S) σ.init.toMeasure σ.kernel

/-- The runs are measured by a probability. -/
instance : IsProbabilityMeasure σ.measure := by
  unfold measure; infer_instance

/-- The states reached with positive probability at step `n`. -/
def reach : ℕ → Set S
  | 0 => σ.init.support
  | n + 1 => ⋃ s ∈ reach n, (σ.next n s).support

omit [MeasurableSingletonClass S] in
/-- Finitely many states are reached in a step from each state (distributions
    are countably supported), so countably many at every step. -/
theorem reach_countable : ∀ n, (σ.reach n).Countable
  | 0 => σ.init.support_countable
  | n + 1 => (reach_countable n).biUnion fun s _ => (σ.next n s).support_countable

omit [MeasurableSingletonClass S] in
/-- What is reached with positive probability is reachable in the system. -/
theorem reachable_of_mem_reach : ∀ n {s : S}, s ∈ σ.reach n → h.Reachable s
  | 0, _, hs => .init σ.init_mem hs
  | n + 1, _, hs => by
    obtain ⟨s₀, hs₀, hs⟩ := mem_iUnion₂.1 hs
    exact .step (reachable_of_mem_reach n hs₀)
      ((σ.move n s₀ (reachable_of_mem_reach n hs₀)).step hs)

omit [MeasurableSingletonClass S] in
/-- The initial state is distributed as the initial distribution. -/
theorem map_zero : σ.measure.map (fun ω => ω 0) = σ.init.toMeasure := by
  -- The run up to step `0` is a single state; Ionescu-Tulcea leaves it as drawn.
  let e := MeasurableEquiv.piUnique (fun _ : Finset.Iic 0 => S)
  have h₁ : (fun ω : ℕ → S => ω 0) = e ∘ frestrictLe 0 := rfl
  have h₂ : σ.measure.map (frestrictLe 0) = σ.init.toMeasure.map e.symm := by
    rw [measure, Kernel.trajMeasure, Measure.map_comp _ _ (by fun_prop),
      Kernel.traj_map_frestrictLe, Kernel.partialTraj_self, Measure.id_comp]
  rw [h₁, ← Measure.map_map e.measurable (by fun_prop), h₂, MeasurableEquiv.map_map_symm]

/-- Almost surely, the run starts in the support of the initial distribution. -/
theorem ae_init : ∀ᵐ ω ∂σ.measure, ω 0 ∈ σ.init.support := by
  -- The complement of the support is measurable (it is co-countable) and has
  -- probability zero under the initial distribution.
  rw [ae_iff]
  change σ.measure ((fun ω : ℕ → S => ω 0) ⁻¹' σ.init.supportᶜ) = 0
  rw [← Measure.map_apply (by fun_prop) σ.init.support_countable.measurableSet.compl, map_zero,
    PMF.toMeasure_apply_eq_zero_iff _ σ.init.support_countable.measurableSet.compl]
  exact disjoint_compl_right

/-- Almost surely, a step reaches a state in the support of its distribution. -/
theorem ae_step (n : ℕ) (hn : ∀ᵐ ω ∂σ.measure, ω n ∈ σ.reach n) :
    ∀ᵐ ω ∂σ.measure, ω (n + 1) ∈ (σ.next n (ω n)).support := by
  -- A bad run either left the reached states (a null event, by `hn`), or it is
  -- at one of the countably many reached states `s` and then misses the
  -- support of `σ.next n s`. Each of the latter events is measurable, and null
  -- by splitting the measure into the first `n` steps and the next one.
  rw [ae_iff] at hn ⊢
  refine measure_mono_null (t := {ω | ω n ∉ σ.reach n} ∪ ⋃ s ∈ σ.reach n,
      (fun ω : ℕ → S => (frestrictLe n ω, ω (n + 1))) ⁻¹'
        (((fun p : Π _ : Finset.Iic n, S => p ⟨n, Finset.mem_Iic.2 le_rfl⟩) ⁻¹' {s}) ×ˢ
          (σ.next n s).supportᶜ))
    (fun ω hω => ?_) (measure_union_null hn ((measure_biUnion_null_iff (σ.reach_countable n)).2
      fun s _ => ?_))
  · by_cases hr : ω n ∈ σ.reach n
    · exact .inr (mem_iUnion₂.2 ⟨_, hr, rfl, hω⟩)
    · exact .inl hr
  · -- The event: at `s` at step `n`, then outside the support of `σ.next n s`.
    have hA : MeasurableSet ((fun p : Π _ : Finset.Iic n, S =>
        p ⟨n, Finset.mem_Iic.2 le_rfl⟩) ⁻¹' {s}) :=
      measurable_pi_apply _ (measurableSet_singleton s)
    have hB : MeasurableSet (σ.next n s).supportᶜ :=
      (σ.next n s).support_countable.measurableSet.compl
    rw [← Measure.map_apply ((measurable_frestrictLe n).prodMk (measurable_pi_apply (n + 1)))
      (hA.prod hB)]
    -- The first `n` steps, then the next one drawn from the kernel.
    rw [measure, ← Kernel.map_frestrictLe_trajMeasure_compProd_eq_map_trajMeasure,
      Measure.compProd_apply_prod hA hB]
    refine setLIntegral_eq_zero hA fun p hp => ?_
    simp only [kernel, step, Kernel.comap_apply, Kernel.coe_mk, Pi.zero_apply]
    rw [Set.mem_preimage, Set.mem_singleton_iff] at hp
    rw [hp, PMF.toMeasure_apply_eq_zero_iff _ hB]
    exact disjoint_compl_right

/-- Almost surely, the run stays among the states reached with positive probability. -/
theorem ae_mem_reach : ∀ n, ∀ᵐ ω ∂σ.measure, ω n ∈ σ.reach n
  | 0 => σ.ae_init
  | n + 1 => by
    filter_upwards [ae_mem_reach n, σ.ae_step n (ae_mem_reach n)] with ω h₁ h₂
    exact mem_iUnion₂.2 ⟨_, h₁, h₂⟩

/-- Almost surely, every step of the run is from a state reached with positive
    probability, to a state in the support of its distribution. -/
theorem ae_steps : ∀ᵐ ω ∂σ.measure, ∀ n, ω n ∈ σ.reach n ∧ ω (n + 1) ∈ (σ.next n (ω n)).support :=
  ae_all_iff.2 fun n => (σ.ae_mem_reach n).and (σ.ae_step n (σ.ae_mem_reach n))

/-- Almost surely, the run of the process is a possible run of the system,
    each step taking the time the scheduler chooses. -/
theorem ae_isRun : ∀ᵐ ω ∂σ.measure, h.IsRun ω (fun n => σ.time n (ω n)) := by
  filter_upwards [σ.ae_init, σ.ae_steps] with ω h₀ hs
  exact ⟨⟨σ.init, σ.init_mem, h₀⟩,
    fun n => (σ.move n _ (σ.reachable_of_mem_reach n (hs n).1)).step (hs n).2⟩

/-- A property of all possible runs holds almost surely. -/
theorem ae_of_isRun {P : (ℕ → S) → (ℕ → ℝ) → Prop}
    (hP : ∀ ρ δ, h.IsRun ρ δ → P ρ δ) :
    ∀ᵐ ω ∂σ.measure, P ω (fun n => σ.time n (ω n)) :=
  σ.ae_isRun.mono fun _ hω => hP _ _ hω

/-- An invariant of all reachable states holds almost surely at every step. -/
theorem ae_forall_of_reachable {P : S → Prop} (hP : ∀ s, h.Reachable s → P s) :
    ∀ᵐ ω ∂σ.measure, ∀ n, P (ω n) :=
  σ.ae_isRun.mono fun _ hω n => hP _ (hω.reachable n)

end Scheduler

end Hybrid

end Zrth
