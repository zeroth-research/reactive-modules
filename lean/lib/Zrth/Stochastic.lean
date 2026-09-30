import Zrth.Hybrid
import Mathlib.Probability.Kernel.IonescuTulcea.Traj
import Mathlib.Probability.Process.Filtration

/-!
# Stochastic semantics

A probabilistic hybrid system is nondeterministic and probabilistic: a
*scheduler* resolves the nondeterminism, choosing at each step, from the
current state, a move — a flow along a trajectory of some duration, or a jump
distribution — and an initial distribution. The scheduler turns the system
into a stochastic process:

* the sample space is that of the runs, `ℕ → S` (`Ω`), with the product
  σ-algebra of the discrete σ-algebras on the states;
* the probability measure is that of the trajectories of the Markov chain the
  scheduler defines (Ionescu-Tulcea, `Scheduler.measure`);
* the run is the coordinate process `n ↦ ω n`, adapted to the filtration
  `Scheduler.filtration` of the first steps (`piLE`, `measurable_state`).

The schedulers are *memoryless*: they choose from the current state and the
number of the step only. This makes the step kernels measurable for the
discrete σ-algebra on the states, whatever the state space. (The states
reached with positive probability form a countable set, as the distributions
are discrete, so the events of interest are measurable.)

The possible runs of the system are its runs with positive probability:
almost surely, the run of the process is a possible run (`ae_isRun`). Any
property of all possible runs is thus an almost sure property under every
scheduler.
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

/-- The filtration of the first steps of the runs in `S`. -/
def Disc.filtration (S : Type*) :
    Filtration ℕ (MeasurableSpace.pi : MeasurableSpace (ℕ → Disc S)) :=
  Filtration.piLE

/-- The run is adapted to the filtration: the state at step `n` is known after `n` steps. -/
theorem Disc.measurable_state {S : Type*} (n : ℕ) :
    Measurable[Disc.filtration S n] fun ω : ℕ → Disc S => ω n :=
  (measurable_pi_apply (⟨n, Set.mem_Iic.2 le_rfl⟩ : Set.Iic n)).comp
    (comap_measurable (restrictLe n))

variable {E H : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [TopologicalSpace H]
  {I : ModelWithCorners ℝ E H} {S : Type*} [TopologicalSpace S] [ChartedSpace H S]

namespace Hybrid

/-- A memoryless scheduler of `h`: an initial distribution, and, at every step
    and (reachable) state, a move, its duration and its distribution. -/
structure Scheduler (h : Hybrid I S) where
  /-- The initial distribution. -/
  init : PMF S
  init_mem : init ∈ h.init
  /-- The distribution of the next state. -/
  next : ℕ → S → PMF S
  /-- The duration of the step. -/
  time : ℕ → S → ℝ
  /-- Every step is a move of `h` from every reachable state. -/
  move : ∀ n s, h.Reachable s → h.Move s (time n s) (next n s)

namespace Scheduler

variable {h : Hybrid I S} (σ : h.Scheduler)

/-- The distribution of the next state at step `n`, as a Markov kernel. -/
noncomputable def step (n : ℕ) : Kernel (Disc S) (Disc S) :=
  ⟨fun s => Disc.measure (σ.next n s), Measurable.of_discrete⟩

instance (n : ℕ) : IsMarkovKernel (σ.step n) :=
  ⟨fun s => by simp only [step, Kernel.coe_mk]; infer_instance⟩

/-- The step kernel on the runs so far: from their last state. -/
noncomputable def kernel (n : ℕ) : Kernel (Π _ : Finset.Iic n, Disc S) (Disc S) :=
  (σ.step n).comap (fun x => x ⟨n, Finset.mem_Iic.2 le_rfl⟩) (measurable_pi_apply _)

instance (n : ℕ) : IsMarkovKernel (σ.kernel n) := by
  unfold kernel; infer_instance

/-- The initial distribution, as a measure. -/
noncomputable def initMeasure : Measure (Disc S) := Disc.measure σ.init

instance : IsProbabilityMeasure σ.initMeasure := by
  unfold initMeasure; infer_instance

/-- The probability measure on the runs. -/
noncomputable def measure : Measure (ℕ → Disc S) :=
  Kernel.trajMeasure (X := fun _ => Disc S) σ.initMeasure σ.kernel

instance : IsProbabilityMeasure σ.measure := by
  unfold measure; infer_instance

/-- The states reached with positive probability at step `n`. -/
def reach : ℕ → Set S
  | 0 => σ.init.support
  | n + 1 => ⋃ s ∈ reach n, (σ.next n s).support

theorem reach_countable : ∀ n, (σ.reach n).Countable
  | 0 => σ.init.support_countable
  | n + 1 => (reach_countable n).biUnion fun s _ => (σ.next n s).support_countable

theorem reachable_of_mem_reach : ∀ n {s : S}, s ∈ σ.reach n → h.Reachable s
  | 0, _, hs => .init σ.init_mem hs
  | n + 1, _, hs => by
    obtain ⟨s₀, hs₀, hs⟩ := mem_iUnion₂.1 hs
    exact .step (reachable_of_mem_reach n hs₀)
      ((σ.move n s₀ (reachable_of_mem_reach n hs₀)).step hs)

/-- The initial state is distributed as the initial distribution. -/
theorem map_zero : σ.measure.map (fun ω => ω 0) = σ.initMeasure := by
  let e := MeasurableEquiv.piUnique (fun _ : Finset.Iic 0 => Disc S)
  have h₁ : (fun ω : ℕ → Disc S => ω 0) = e ∘ frestrictLe 0 := rfl
  have h₂ : σ.measure.map (frestrictLe 0) = σ.initMeasure.map e.symm := by
    rw [measure, Kernel.trajMeasure, Measure.map_comp _ _ (by fun_prop),
      Kernel.traj_map_frestrictLe, Kernel.partialTraj_self, Measure.id_comp]
  rw [h₁, ← Measure.map_map e.measurable (by fun_prop), h₂, MeasurableEquiv.map_map_symm]

theorem ae_init : ∀ᵐ ω ∂σ.measure, (ω 0 : S) ∈ σ.init.support := by
  rw [ae_iff]
  change σ.measure ((fun ω : ℕ → Disc S => ω 0) ⁻¹' ((σ.init.support : Set S)ᶜ : Set (Disc S))) = 0
  rw [← Measure.map_apply (by fun_prop) MeasurableSet.of_discrete, map_zero, initMeasure]
  exact (Disc.measure_eq_zero_iff _ _).2 disjoint_compl_right

/-- Almost surely, a step reaches a state in the support of its distribution. -/
theorem ae_step (n : ℕ) (hn : ∀ᵐ ω ∂σ.measure, (ω n : S) ∈ σ.reach n) :
    ∀ᵐ ω ∂σ.measure, (ω (n + 1) : S) ∈ (σ.next n (ω n)).support := by
  rw [ae_iff] at hn ⊢
  refine measure_mono_null (t := {ω | (ω n : S) ∉ σ.reach n} ∪ ⋃ s ∈ σ.reach n,
      (fun ω : ℕ → Disc S => (frestrictLe n ω, ω (n + 1))) ⁻¹'
        (((fun p : Π _ : Finset.Iic n, Disc S => (p ⟨n, Finset.mem_Iic.2 le_rfl⟩ : S)) ⁻¹' {s}) ×ˢ
          ((σ.next n s).supportᶜ : Set (Disc S))))
    (fun ω hω => ?_) (measure_union_null hn ((measure_biUnion_null_iff (σ.reach_countable n)).2
      fun s _ => ?_))
  · by_cases hr : (ω n : S) ∈ σ.reach n
    · exact .inr (mem_iUnion₂.2 ⟨_, hr, rfl, hω⟩)
    · exact .inl hr
  · have hA : MeasurableSet ((fun p : Π _ : Finset.Iic n, Disc S =>
        (p ⟨n, Finset.mem_Iic.2 le_rfl⟩ : S)) ⁻¹' {s}) :=
      measurable_pi_apply _ (MeasurableSet.of_discrete (s := ({s} : Set (Disc S))))
    rw [← Measure.map_apply ((measurable_frestrictLe n).prodMk (measurable_pi_apply (n + 1)))
      (hA.prod MeasurableSet.of_discrete)]
    rw [measure, ← Kernel.map_frestrictLe_trajMeasure_compProd_eq_map_trajMeasure,
      Measure.compProd_apply_prod hA MeasurableSet.of_discrete]
    refine setLIntegral_eq_zero hA fun p hp => ?_
    simp only [kernel, step, Kernel.comap_apply, Kernel.coe_mk, Pi.zero_apply]
    change (p ⟨n, _⟩ : S) = s at hp
    rw [hp]
    exact (Disc.measure_eq_zero_iff _ _).2 disjoint_compl_right

/-- Almost surely, the run stays among the states reached with positive probability. -/
theorem ae_mem_reach : ∀ n, ∀ᵐ ω ∂σ.measure, (ω n : S) ∈ σ.reach n
  | 0 => σ.ae_init
  | n + 1 => by
    filter_upwards [ae_mem_reach n, σ.ae_step n (ae_mem_reach n)] with ω h₁ h₂
    exact mem_iUnion₂.2 ⟨_, h₁, h₂⟩

/-- Almost surely, the run of the process is a possible run of the system,
    each step taking the time the scheduler chooses. -/
theorem ae_isRun : ∀ᵐ ω ∂σ.measure, h.IsRun (fun n => (ω n : S)) (fun n => σ.time n (ω n)) := by
  have hs : ∀ᵐ ω ∂σ.measure, ∀ n, (ω n : S) ∈ σ.reach n ∧
      (ω (n + 1) : S) ∈ (σ.next n (ω n)).support :=
    ae_all_iff.2 fun n => (σ.ae_mem_reach n).and (σ.ae_step n (σ.ae_mem_reach n))
  filter_upwards [σ.ae_init, hs] with ω h₀ hs
  exact ⟨⟨σ.init, σ.init_mem, h₀⟩,
    fun n => (σ.move n _ (σ.reachable_of_mem_reach n (hs n).1)).step (hs n).2⟩

/-- A property of all possible runs holds almost surely. -/
theorem ae_of_isRun {P : (ℕ → S) → (ℕ → ℝ) → Prop}
    (hP : ∀ ρ δ, h.IsRun ρ δ → P ρ δ) :
    ∀ᵐ ω ∂σ.measure, P (fun n => (ω n : S)) (fun n => σ.time n (ω n)) :=
  σ.ae_isRun.mono fun _ hω => hP _ _ hω

/-- An invariant of all reachable states holds almost surely at every step. -/
theorem ae_forall_of_reachable {P : S → Prop} (hP : ∀ s, h.Reachable s → P s) :
    ∀ᵐ ω ∂σ.measure, ∀ n, P (ω n) :=
  σ.ae_isRun.mono fun _ hω n => hP _ (hω.reachable n)

end Scheduler

end Hybrid

end Zrth
