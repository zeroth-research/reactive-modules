import Zrth.Stochastic.Run

/-!
# The process in continuous time

The semantics as a single process in continuous time, as for jump processes
(Davis' piecewise-deterministic Markov processes): built from the embedded
chain of the steps (`Scheduler.measure`) by a *time change*. A step `n` starts
at the time `start n`, the sum of the durations of the steps before it, and at
a time `t` the run is in the step it has started last: the *active* step, with
`start n ≤ t < start (n + 1)`.

* `Scheduler.process t ω`: the marked state at time `t`, the active step and
  the state along its evolution — `none` before time `0`, or when the run has
  exploded (its time does not diverge: infinitely many steps before `t`). The
  steps are marked, as the jumps of a jump process are counted: a step may
  change nothing visible (e.g., a flow split in two), and still be one.
* `Scheduler.filtration`: the natural filtration of the process, a single
  filtration in continuous time; the process is adapted to it
  (`adapted_process`), and the start of every step is a stopping time
  (`isStoppingTime_start`).
* A divergent run never explodes (`process_ne_none`).

How it relates to the other levels:

* *Hybrid time* projects onto continuous time: during a step of positive
  duration, the process is the state in hybrid time, at the local time
  `t - start n` (`process_of_active`, `state_eq_process`).
* Steps taking no time — jumps — are never active (`time_pos_of_active`): at a
  time `t`, the process shows the state after all the jumps at `t`, so that its
  paths are right-continuous at jumps. A burst of jumps at the same instant —
  and the atoms of each round — collapses into one jump of the process: their
  information sits between the σ-algebras just before and at the time of the
  jump, where hybrid time (`Zrth.Stochastic.Run`) and the rounds
  (`Zrth.Stochastic.Round`) refine it.
* Almost surely, the process follows the system (`ae_follows`): in every active
  step, it moves along a trajectory of the system.
-/

namespace Zrth

open MeasureTheory ProbabilityTheory Filter Set Preorder

/-- The σ-algebra on `Option α`: that of `α` on the values, the extra point
    `none` (a cemetery) being free. -/
instance Option.measurableSpace {α : Type*} [MeasurableSpace α] :
    MeasurableSpace (Option α) :=
  MeasurableSpace.map some ‹_›

/-- Wrapping a value in `some` is measurable. -/
theorem measurable_some {α : Type*} [MeasurableSpace α] : Measurable (some : α → Option α) :=
  fun _ h => h

variable {E H : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [TopologicalSpace H]
  {I : ModelWithCorners ℝ E H} {S : Type*} [TopologicalSpace S] [ChartedSpace H S]
  [MeasurableSpace S]

namespace Hybrid.Scheduler

variable {h : Hybrid I S} (σ : h.Scheduler)

/-! ## Time change -/

/-- The first step starts at time `0`. -/
theorem start_zero (ω : ℕ → S) : σ.start 0 ω = 0 := Finset.sum_range_zero _

/-- The next step starts when this one ends. -/
theorem start_succ (n : ℕ) (ω : ℕ → S) : σ.start (n + 1) ω = σ.start n ω + σ.time n (ω n) :=
  Finset.sum_range_succ _ _

/-- Steps start one after the other. -/
theorem start_mono (ω : ℕ → S) : Monotone fun n => σ.start n ω :=
  monotone_nat_of_le_succ fun n => by
    rw [start_succ]; linarith [σ.time_nonneg n (ω n)]

/-- Steps start from time `0` on. -/
theorem start_nonneg (n : ℕ) (ω : ℕ → S) : 0 ≤ σ.start n ω :=
  σ.start_zero ω ▸ σ.start_mono ω (Nat.zero_le n)

/-- The start of a step is measurable (forgetting when it is known). -/
theorem measurable_start' (n : ℕ) : Measurable (σ.start n) :=
  (σ.measurable_start n).mono ((Evolution.filtration ℕ S).le n) le_rfl

/-- The step `n` is active at time `t`: it has started, and the next has not. -/
def Active (t : ℝ) (ω : ℕ → S) (n : ℕ) : Prop := σ.start n ω ≤ t ∧ t < σ.start (n + 1) ω

/-- At most one step is active at a time. -/
theorem Active.unique {t : ℝ} {ω : ℕ → S} {n n' : ℕ} (h : σ.Active t ω n) (h' : σ.Active t ω n') :
    n = n' := by
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
  · exact absurd (h'.1.trans_lt h.2) (not_lt.2 (σ.start_mono ω hlt))
  · exact absurd (h.1.trans_lt h'.2) (not_lt.2 (σ.start_mono ω hlt))

/-- An active step takes time: jumps are never active. -/
theorem time_pos_of_active {t : ℝ} {ω : ℕ → S} {n : ℕ} (h : σ.Active t ω n) :
    0 < σ.time n (ω n) := by
  have := h.1.trans_lt h.2
  rw [start_succ] at this
  linarith

/-- The run has exploded by time `t`: infinitely many steps start before `t`. -/
def Explodes (t : ℝ) (ω : ℕ → S) : Prop := ∀ n, σ.start n ω ≤ t

/-- From time `0` on, a run is in an active step, unless it has exploded. -/
theorem active_or_explodes {t : ℝ} (ht : 0 ≤ t) (ω : ℕ → S) :
    (∃ n, σ.Active t ω n) ∨ σ.Explodes t ω := by
  by_contra hne
  push Not at hne
  apply hne.2
  intro n
  induction n with
  | zero => rw [start_zero]; exact ht
  | succ n ih => exact not_lt.1 fun hlt => hne.1 n ⟨ih, hlt⟩

/-- A run whose time diverges never explodes. -/
theorem not_explodes {ω : ℕ → S} (hdiv : Hybrid.Divergent fun n => σ.time n (ω n)) (t : ℝ) :
    ¬σ.Explodes t ω := fun h => by
  obtain ⟨n, hn⟩ := (hdiv.eventually_gt_atTop t).exists
  exact absurd (h n) (not_le.2 hn)

/-! ## The process -/

open Classical in
/-- The marked state of the run at time `t`: the active step, and the state
    along its evolution; `none` before time `0`, or after an explosion. -/
noncomputable def process (t : ℝ) (ω : ℕ → S) : Option (ℕ × S) :=
  if h : ∃ n, σ.Active t ω n then
    some (Nat.find h, σ.path (Nat.find h) (ω (Nat.find h)) (t - σ.start (Nat.find h) ω))
  else none

open Classical in
/-- In its active step, the run is along the evolution of the step. -/
theorem process_of_active {t : ℝ} {ω : ℕ → S} {n : ℕ} (h : σ.Active t ω n) :
    σ.process t ω = some (n, σ.path n (ω n) (t - σ.start n ω)) := by
  have hex : ∃ n, σ.Active t ω n := ⟨n, h⟩
  have hn : Nat.find hex = n := Active.unique σ (Nat.find_spec hex) h
  rw [process, dite_eq_left_of_eq_true (eq_true hex), hn]

/-- Without an active step, the run is nowhere (`none`). -/
theorem process_of_not_active {t : ℝ} {ω : ℕ → S} (h : ¬∃ n, σ.Active t ω n) :
    σ.process t ω = none := by
  rw [process, dite_eq_right_of_eq_false (eq_false h)]

/-- The process is `none` exactly before time `0` and after an explosion. -/
theorem process_eq_none_iff {t : ℝ} {ω : ℕ → S} :
    σ.process t ω = none ↔ t < 0 ∨ σ.Explodes t ω := by
  constructor
  · intro hp
    by_contra hne
    push Not at hne
    rcases σ.active_or_explodes hne.1 ω with ⟨n, hn⟩ | hx
    · rw [σ.process_of_active hn] at hp; exact Option.some_ne_none _ hp
    · exact hne.2 hx
  · intro h
    refine σ.process_of_not_active fun ⟨n, hn⟩ => ?_
    rcases h with ht | hx
    · exact absurd (σ.start_nonneg n ω) (not_le.2 (hn.1.trans_lt ht))
    · exact absurd (hx (n + 1)) (not_le.2 hn.2)

/-- A divergent run is never `none` from time `0` on. -/
theorem process_ne_none {ω : ℕ → S} (hdiv : Hybrid.Divergent fun n => σ.time n (ω n))
    {t : ℝ} (ht : 0 ≤ t) : σ.process t ω ≠ none := fun hp =>
  ((σ.process_eq_none_iff.1 hp).elim (fun h => absurd ht (not_le.2 h))
    (σ.not_explodes hdiv t))

/-- Whether a step is active is an event: a comparison of measurable times. -/
theorem measurableSet_active (t : ℝ) (n : ℕ) : MeasurableSet {ω | σ.Active t ω n} :=
  (measurableSet_le (σ.measurable_start' n) measurable_const).inter
    (measurableSet_lt measurable_const (σ.measurable_start' (n + 1)))

/-- Whether the run has exploded is an event: countably many comparisons. -/
theorem measurableSet_explodes (t : ℝ) : MeasurableSet {ω | σ.Explodes t ω} := by
  have : {ω | σ.Explodes t ω} = ⋂ n, {ω | σ.start n ω ≤ t} := by ext; simp [Explodes]
  rw [this]
  exact MeasurableSet.iInter fun n => measurableSet_le (σ.measurable_start' n) measurable_const

/-- The marked state of the run in its step `n`, at time `t`. -/
noncomputable def along (t : ℝ) (n : ℕ) (ω : ℕ → S) : ℕ × S :=
  (n, σ.path n (ω n) (t - σ.start n ω))

/-- The state along a fixed step is measurable: the scheduler's evolution,
    at a measurable local time. -/
theorem measurable_along (t : ℝ) (n : ℕ) : Measurable (σ.along t n) :=
  measurable_const.prodMk ((σ.measurable_path n).comp
    ((measurable_pi_apply n).prodMk (measurable_const.sub (σ.measurable_start' n))))

/-- The process is measurable at every time. -/
theorem measurable_process (t : ℝ) : Measurable (σ.process t) := by
  -- The preimage of `B` splits by cases: the run is nowhere (and `none ∈ B`),
  -- or it is in the step `n`, along which it lands in `B`. Each piece is an
  -- event, and there are countably many.
  intro B hB
  have : σ.process t ⁻¹' B =
      ({ω | (t < 0 ∨ σ.Explodes t ω)} ∩ {_ω | none ∈ B}) ∪
        ⋃ n, ({ω | σ.Active t ω n} ∩ σ.along t n ⁻¹' (some ⁻¹' B)) := by
    ext ω
    simp only [mem_preimage, mem_union, mem_inter_iff, Set.mem_ofPred_eq, mem_iUnion]
    by_cases hact : ∃ n, σ.Active t ω n
    · obtain ⟨n, hn⟩ := hact
      rw [σ.process_of_active hn]
      refine ⟨fun hB => .inr ⟨n, hn, hB⟩, fun h' => ?_⟩
      rcases h' with ⟨hn', _⟩ | ⟨m, hm, hB⟩
      · exact absurd (σ.process_eq_none_iff.2 hn') (by rw [σ.process_of_active hn]; simp)
      · rwa [Active.unique σ hn hm]
    · rw [σ.process_of_not_active hact]
      have hnone := σ.process_eq_none_iff.1 (σ.process_of_not_active hact)
      exact ⟨fun hB => .inl ⟨hnone, hB⟩, fun h' => h'.elim (·.2)
        fun ⟨m, hm, _⟩ => absurd ⟨m, hm⟩ hact⟩
  rw [this]
  refine (((MeasurableSet.const (t < 0)).union
    (σ.measurableSet_explodes t)).inter (MeasurableSet.const _)).union
    (MeasurableSet.iUnion fun n => (σ.measurableSet_active t n).inter
      (σ.measurable_along t n hB))

/-! ## The filtration -/

/-- The natural filtration of the process: at `t`, the process up to `t`. -/
noncomputable def filtration : Filtration ℝ (MeasurableSpace.pi : MeasurableSpace (ℕ → S)) where
  seq t := ⨆ s ≤ t, MeasurableSpace.comap (σ.process s) inferInstance
  mono' _ _ hst := iSup₂_mono' fun s hs => ⟨s, hs.trans hst, le_rfl⟩
  le' _ := iSup₂_le fun s _ => (σ.measurable_process s).comap_le

/-- The process is adapted to its filtration. -/
theorem adapted_process (t : ℝ) : Measurable[σ.filtration t] (σ.process t) :=
  (comap_measurable _).mono
    (le_iSup₂ (f := fun s (_ : s ≤ t) => MeasurableSpace.comap (σ.process s) inferInstance)
      t le_rfl) le_rfl

/-- A step has started by time `t ≥ 0` iff the run is at it or later, or has exploded. -/
theorem start_le_iff {t : ℝ} (ht : 0 ≤ t) (n : ℕ) (ω : ℕ → S) :
    σ.start n ω ≤ t ↔ σ.process t ω ∈ {none} ∪ some '' {p | n ≤ p.1} := by
  -- The marks count the steps: the active step is the last one started.
  rcases σ.active_or_explodes ht ω with ⟨k, hk⟩ | hx
  · rw [σ.process_of_active hk]
    simp only [mem_union, mem_singleton_iff, reduceCtorEq, mem_image, Set.mem_ofPred_eq,
      Option.some.injEq, false_or]
    constructor
    · intro hn
      refine ⟨_, ?_, rfl⟩
      by_contra hlt
      push Not at hlt
      exact absurd (hn.trans_lt hk.2) (not_lt.2 (σ.start_mono ω (Nat.succ_le_of_lt hlt)))
    · rintro ⟨_, hle, rfl⟩
      exact (σ.start_mono ω hle).trans hk.1
  · rw [(σ.process_eq_none_iff).2 (.inr hx)]
    exact ⟨fun _ => .inl rfl, fun _ => hx n⟩

/-- The start of every step is a stopping time. -/
theorem isStoppingTime_start (n : ℕ) :
    IsStoppingTime σ.filtration fun ω => (σ.start n ω : WithTop ℝ) := by
  -- Whether the step has started by `t` is read off the process at `t`, by
  -- its mark (`start_le_iff`); before time `0`, no step has started.
  intro t
  simp only [WithTop.coe_le_coe]
  by_cases ht : 0 ≤ t
  · have : {ω | σ.start n ω ≤ t} = σ.process t ⁻¹' ({none} ∪ some '' {p | n ≤ p.1}) := by
      ext ω; exact σ.start_le_iff ht n ω
    have hB : MeasurableSet ({none} ∪ some '' {p : ℕ × S | n ≤ p.1}) := by
      show MeasurableSet (some ⁻¹' ({none} ∪ some '' {p : ℕ × S | n ≤ p.1}))
      rw [Set.preimage_union, Set.preimage_image_eq _ (Option.some_injective _)]
      refine MeasurableSet.union ?_ (measurable_fst (measurableSet_le measurable_const measurable_id))
      convert MeasurableSet.empty
      ext; simp
    rw [this]
    exact σ.adapted_process t hB
  · have : {ω | σ.start n ω ≤ t} = ∅ := by
      ext ω
      simp only [Set.mem_ofPred_eq, mem_empty_iff_false, iff_false, not_le]
      exact (not_le.1 ht).trans_le (σ.start_nonneg n ω)
    rw [this]
    exact @MeasurableSet.empty _ (σ.filtration t)

/-! ## Hybrid time and continuous time -/

/-- During a step of positive duration, the process is the state in hybrid
    time, at the local time within the step. -/
theorem state_eq_process {t : ℝ} {ω : ℕ → S} {n : ℕ} (h : σ.Active t ω n) :
    σ.process t ω = some (n, σ.state (toLex (n, t - σ.start n ω)) ω) := by
  rw [σ.process_of_active h]
  congr
  have hτ : t - σ.start n ω ∈ Icc 0 (σ.time n (ω n)) := by
    have := h.2
    rw [start_succ] at this
    exact ⟨by linarith [h.1], by linarith⟩
  simp only [HybridTime.step, HybridTime.elapsed, ofLex_toLex, clamp_of_mem hτ]

/-- Almost surely, the process follows the system: in every active step, it
    moves along a trajectory of the system. -/
theorem ae_follows [MeasurableSingletonClass S] : ∀ᵐ ω ∂σ.measure, ∀ t n, σ.Active t ω n →
    h.IsTrajectory (σ.path n (ω n)) (σ.time n (ω n)) ∧
      σ.process t ω = some (n, σ.path n (ω n) (t - σ.start n ω)) := by
  filter_upwards [σ.ae_evolution] with ω hev t n hn
  refine ⟨?_, σ.process_of_active hn⟩
  rcases hev n with ⟨h0, _⟩ | ⟨hγ, _⟩
  · exact absurd h0 (σ.time_pos_of_active hn).ne'
  · exact hγ

end Hybrid.Scheduler

end Zrth
