import Zrth
import Zrth.Examples.Clocks

/-!
# Proofs about the clocks

For the hybrid system of a clock atom (`Atom.toHybrid`), where jumps are not
urgent but a flow cannot get out of a jump region — here the single point `5`,
which a flow at rate `1` cannot stay at, so it has to stop there:

* the flow moves the clock at rate `1`: along a trajectory, the value grows by
  the time elapsed (`val_eq_of_trajectory`);
* *invariant*: every reachable value lies in `[0, 5]` (`reachable_bounds`);
* *ranking function*: `rank = 5 - value` is nonnegative on reachable states and
  every step other than a reset from `5` decreases it by exactly the time the
  step takes (`step_rank`), so time cannot advance by more than `5` between
  resets;
* *liveness*: in every run whose time diverges, the clock is `0` infinitely
  many times (`infinite_zero`);
* *non-vacuity*: such a run exists (`run`, `run_divergent`).

Then the same for the composed module (`msys`), where the invariant also
shows that the clocks stay synchronised (`mreachable_bounds`), and both are
`0` at once infinitely many times (`minfinite_zero`).

The clocks are deterministic: their initial and update distributions are
Dirac. The properties of all possible runs hold almost surely under every
scheduler of the stochastic semantics (`ae_mbounds`, `ae_minfinite_zero`).
-/

namespace Zrth.Examples.Clocks

open Manifold Set Filter

/-- The hybrid system of the clock `k`. -/
noncomputable def sys (k : Clock) :
    Hybrid (Val.model I (clock k).ctrl) (Val M (clock k).ctrl) :=
  (clock k).toHybrid (Finset.Subset.refl _) rfl

variable {k : Clock}

/-- The value of the clock `k`. -/
def val (k : Clock) (s : Val M (clock k).ctrl) : ℝ := s ⟨k, Finset.mem_singleton_self k⟩

/-- The ranking function of the clock `k`: the time until its next reset. -/
def rank (k : Clock) (s : Val M (clock k).ctrl) : ℝ := 5 - val k s

/-- The clock starts at `0`. -/
theorem of_mem_init {μ : PMF (Val M (clock k).ctrl)} {s : Val M (clock k).ctrl}
    (hμ : μ ∈ (sys k).init) (hs : s ∈ μ.support) : s = 0 := by
  obtain ⟨_, rfl, rfl⟩ := hμ
  exact (PMF.mem_support_pure_iff _ _).1 hs

theorem pure_mem_init : PMF.pure 0 ∈ (sys k).init := ⟨_, rfl, rfl⟩

/-- The clock jumps from `5` to `0`. -/
theorem of_mem_jump {μ : PMF (Val M (clock k).ctrl)} {s s' : Val M (clock k).ctrl}
    (hμ : μ ∈ (sys k).jump s) (hs' : s' ∈ μ.support) : val k s = 5 ∧ s' = 0 := by
  obtain ⟨_, ⟨h5, rfl⟩, rfl⟩ := hμ
  exact ⟨h5, (PMF.mem_support_pure_iff _ _).1 hs'⟩

theorem pure_mem_jump {s : Val M (clock k).ctrl} (h5 : val k s = 5) :
    PMF.pure 0 ∈ (sys k).jump s := ⟨_, ⟨h5, rfl⟩, rfl⟩

/-- The jump region of the clock: the single point `5`. -/
theorem regions_eq : (sys k).regions = {{s | val k s = 5}} := by
  show {{s | _}} = _
  congr 1
  ext s
  exact ⟨fun ⟨_, h, _⟩ => h, fun h => ⟨_, h, rfl⟩⟩

/-- Along a trajectory, the clock grows by the time elapsed. -/
theorem val_eq_of_trajectory {γ : ℝ → Val M (clock k).ctrl} {d : ℝ}
    (hγ : (sys k).IsTrajectory γ d) : ∀ t ∈ Icc 0 d, val k (γ t) = val k (γ 0) + t :=
  Val.eq_add_of_rate_one _ fun t ht => by
    obtain ⟨v, hv, hγt⟩ := hγ.follows t ht
    exact ⟨v, congrFun (hv : v = fun _ => 1) _, hγt⟩

/-- A flow cannot get out of the jump region, a single point it cannot stay
    at, so it stops at `5`. -/
theorem val_le_of_trajectory {γ : ℝ → Val M (clock k).ctrl} {d : ℝ}
    (hγ : (sys k).IsTrajectory γ d) (h5 : val k (γ 0) ≤ 5) : val k (γ 0) + d ≤ 5 := by
  by_contra hlt
  have ht : 5 - val k (γ 0) ∈ Icc 0 d := ⟨by linarith, by linarith⟩
  have hin : γ (5 - val k (γ 0)) ∈ {s | val k s = 5} := by
    show val k _ = 5
    rw [val_eq_of_trajectory hγ _ ht]; ring
  have : val k (γ d) = 5 :=
    hγ.stays _ (by rw [regions_eq]; rfl) _ ht d ⟨ht.2, le_rfl⟩ hin
  rw [val_eq_of_trajectory hγ _ ⟨hγ.nonneg, le_rfl⟩] at this
  linarith

/-- A step from a state within bounds is a reset from `5`, or a flow that
    advances the clock by the time elapsed, staying within bounds. -/
theorem step_cases {s s' : Val M (clock k).ctrl} {d : ℝ} (hs : (sys k).Step s d s')
    (h5 : val k s ≤ 5) :
    (val k s = 5 ∧ s' = 0 ∧ d = 0) ∨ (0 ≤ d ∧ val k s' = val k s + d ∧ val k s' ≤ 5) := by
  cases hs with
  | jump hμ hs' => exact .inl ⟨(of_mem_jump hμ hs').1, (of_mem_jump hμ hs').2, rfl⟩
  | flow hγ =>
    have := val_eq_of_trajectory hγ _ ⟨hγ.nonneg, le_rfl⟩
    exact .inr ⟨hγ.nonneg, this, this ▸ val_le_of_trajectory hγ h5⟩

/-- Invariant: the clock stays within `[0, 5]`. -/
theorem reachable_bounds {s : Val M (clock k).ctrl} (hs : (sys k).Reachable s) :
    0 ≤ val k s ∧ val k s ≤ 5 := by
  induction hs with
  | init hμ hs => rw [of_mem_init hμ hs]; norm_num [val]
  | step _ hst ih =>
    rcases step_cases hst ih.2 with ⟨_, rfl, _⟩ | ⟨hd, heq, hle⟩
    · norm_num [val]
    · exact ⟨by linarith, hle⟩

/-- The ranking function is nonnegative on reachable states. -/
theorem rank_nonneg {s : Val M (clock k).ctrl} (hs : (sys k).Reachable s) : 0 ≤ rank k s := by
  have := reachable_bounds hs
  simp only [rank]
  linarith

/-- Every step other than a reset from `5` decreases the ranking function by
    exactly the time it takes. -/
theorem step_rank {s s' : Val M (clock k).ctrl} {d : ℝ} (hs : (sys k).Reachable s)
    (hst : (sys k).Step s d s') : (rank k s = 0 ∧ s' = 0) ∨ rank k s' = rank k s - d := by
  rcases step_cases hst (reachable_bounds hs).2 with ⟨h5, rfl, _⟩ | ⟨_, heq, _⟩
  · exact .inl ⟨by simp [rank, h5], rfl⟩
  · exact .inr (by simp only [rank, heq]; ring)

/-- Liveness: in a run whose time diverges, the clock is `0` infinitely many times. -/
theorem frequently_zero {ρ : ℕ → Val M (clock k).ctrl} {δ : ℕ → ℝ} (hρ : (sys k).IsRun ρ δ)
    (hdiv : Hybrid.Divergent δ) : ∀ N, ∃ n ≥ N, val k (ρ n) = 0 := by
  intro N
  by_contra hne
  push Not at hne
  -- without resets, the clock advances by the time elapsed
  have key : ∀ m, val k (ρ (N + m)) = val k (ρ N) + ∑ i ∈ Finset.range m, δ (N + i) := by
    intro m
    induction m with
    | zero => simp
    | succ m ih =>
      rcases step_cases (hρ.step (N + m)) (reachable_bounds (hρ.reachable _)).2 with
        ⟨_, h0, _⟩ | ⟨_, heq, _⟩
      · exact absurd (by rw [h0]; rfl) (hne (N + m + 1) (by omega))
      · rw [← add_assoc, heq, ih, Finset.sum_range_succ]
        ring
  -- so, by the ranking function, the time elapsed after `N` is at most `5`
  have bound : ∀ m, ∑ i ∈ Finset.range (N + m), δ i ≤ ∑ i ∈ Finset.range N, δ i + 5 := by
    intro m
    have h₁ := reachable_bounds (hρ.reachable (N + m))
    have h₂ := reachable_bounds (hρ.reachable N)
    have := key m
    rw [Finset.sum_range_add]
    linarith
  obtain ⟨n, hn⟩ := (hdiv.eventually_gt_atTop (∑ i ∈ Finset.range N, δ i + 5)).exists_forall_of_atTop
  exact absurd (bound n) (not_le.2 (hn (N + n) (by omega)))

theorem infinite_zero {ρ : ℕ → Val M (clock k).ctrl} {δ : ℕ → ℝ} (hρ : (sys k).IsRun ρ δ)
    (hdiv : Hybrid.Divergent δ) : {n | val k (ρ n) = 0}.Infinite :=
  Nat.frequently_atTop_iff_infinite.1 (frequently_atTop.2 (frequently_zero hρ hdiv))

/-! ## A divergent run -/

/-- The run that flows from `0` to `5` and resets, forever. -/
noncomputable def run (k : Clock) : ℕ → Val M (clock k).ctrl :=
  fun n => if n % 2 = 0 then 0 else fun _ => 5

/-- The durations of the steps of `run`. -/
def dur : ℕ → ℝ := fun n => if n % 2 = 0 then 5 else 0

/-- The trajectory from `0` to `5`. -/
theorem isTrajectory_ramp : (sys k).IsTrajectory (fun t _ => t) 5 where
  nonneg := by norm_num
  follows t _ := by
    have : HasDerivAt (fun t (_ : (clock k).ctrl) => t) (fun _ => (1 : ℝ)) t :=
      hasDerivAt_pi.2 fun _ => hasDerivAt_id t
    exact ⟨fun _ => 1, rfl, Val.hasFDerivAt_iff.2 this.hasFDerivAt⟩
  stays G hG t₁ _ t₂ ht₂ hin := by
    rw [regions_eq, Set.mem_singleton_iff] at hG
    subst hG
    have h5 : t₁ = 5 := hin
    rwa [show t₂ = t₁ by linarith [ht₂.1, ht₂.2]]

theorem run_isRun : (sys k).IsRun (run k) dur where
  init := ⟨_, pure_mem_init, by simp [run]⟩
  step n := by
    rcases Nat.mod_two_eq_zero_or_one n with h | h
    · have h' : (n + 1) % 2 = 1 := by omega
      simp only [run, dur, h, h', one_ne_zero, ↓reduceIte]
      exact Hybrid.Step.flow isTrajectory_ramp
    · have h' : (n + 1) % 2 = 0 := by omega
      simp only [run, dur, h, h', one_ne_zero, ↓reduceIte]
      exact Hybrid.Step.jump (pure_mem_jump rfl) (by simp)

theorem run_divergent : Hybrid.Divergent dur := by
  have hnn : ∀ i, 0 ≤ dur i := fun i => by unfold dur; split_ifs <;> norm_num
  have heven : ∀ m, ∑ i ∈ Finset.range (2 * m), dur i = 5 * m := by
    intro m
    induction m with
    | zero => simp
    | succ m ih =>
      rw [show 2 * (m + 1) = 2 * m + 1 + 1 by ring, Finset.sum_range_succ,
        Finset.sum_range_succ, ih]
      have h₁ : (2 * m) % 2 = 0 := by omega
      have h₂ : (2 * m + 1) % 2 = 1 := by omega
      simp only [dur, h₁, h₂, one_ne_zero, ↓reduceIte]
      push_cast
      ring
  refine tendsto_atTop_atTop.2 fun b => ?_
  obtain ⟨m, hm⟩ := exists_nat_ge (b / 5)
  refine ⟨2 * m, fun n hn => ?_⟩
  calc b ≤ 5 * m := by linarith [div_le_iff₀ (by norm_num : (0 : ℝ) < 5) |>.1 hm]
    _ = ∑ i ∈ Finset.range (2 * m), dur i := (heven m).symm
    _ ≤ ∑ i ∈ Finset.range n, dur i :=
      Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hn) fun i _ _ => hnn i

/-- Both clocks are `0` infinitely many times along `run`. -/
example (k : Clock) : {n | val k (run k n) = 0}.Infinite :=
  infinite_zero run_isRun run_divergent

/-! ## The composed module

The same properties for `composed`, the parallel composition of the modules of
either clock, under the module semantics: the clocks flow together, racing to
their jump regions at `5`, which they cannot get out of; when one of them
fires there, every clock at `5` resets while the others stutter. The invariant additionally shows that the clocks stay
synchronised, so they always win the race together. -/

theorem composed_isClosed : composed.IsClosed := show composed.extl = ∅ by decide

/-- The hybrid system of the composed clocks. -/
noncomputable def msys : Hybrid (Val.model I composed.ctrl) (Val M composed.ctrl) :=
  composed.toHybrid composed_isClosed

theorem mem_ctrl (k : Clock) : k ∈ composed.ctrl := by cases k <;> decide

theorem atoms_composed : composed.atoms = [clock .a, clock .b] := rfl

theorem mem_atoms (k : Clock) : clock k ∈ composed.atoms := by
  rw [atoms_composed]; cases k <;> simp

/-- The value of the clock `k` in the composed module. -/
def mval (k : Clock) (s : Val M composed.ctrl) : ℝ := s ⟨k, mem_ctrl k⟩

/-- The ranking function of the composed module: the time until the next reset. -/
def mrank (s : Val M composed.ctrl) : ℝ := 5 - mval .a s

/-- A function on the controlled variables of a clock atom is constant iff it
    is so at the clock. -/
theorem restrict_eq_const (k : Clock) {X : Finset Clock} (h : (clock k).ctrl ⊆ X)
    (f : (v : X) → ℝ) (c : ℝ) :
    (fun v : (clock k).ctrl => f ⟨v, h v.2⟩) = (fun _ => c) ↔
      f ⟨k, h (Finset.mem_singleton_self k)⟩ = c := by
  refine ⟨fun e => congrFun e ⟨k, Finset.mem_singleton_self k⟩, fun e => ?_⟩
  funext ⟨v, hv⟩
  obtain rfl := Finset.mem_singleton.1 (hv : v ∈ ({k} : Finset Clock))
  exact e

/-- A function on the controlled variables of a clock atom is determined by
    its value at the clock. -/
theorem restrict_eq_iff (k : Clock) {X : Finset Clock} (h : (clock k).ctrl ⊆ X)
    (f g : (v : X) → ℝ) :
    (fun v : (clock k).ctrl => f ⟨v, h v.2⟩) = (fun v : (clock k).ctrl => g ⟨v, h v.2⟩) ↔
      f ⟨k, h (Finset.mem_singleton_self k)⟩ = g ⟨k, h (Finset.mem_singleton_self k)⟩ := by
  refine ⟨fun e => congrFun e ⟨k, Finset.mem_singleton_self k⟩, fun e => ?_⟩
  funext ⟨v, hv⟩
  obtain rfl := Finset.mem_singleton.1 (hv : v ∈ ({k} : Finset Clock))
  exact e

/-- Every atom of the composed module is a clock. -/
theorem forall_atoms {P : ∀ a, a ∈ composed.atoms → Prop} :
    (∀ a ha, P a ha) ↔ P (clock .a) (mem_atoms .a) ∧ P (clock .b) (mem_atoms .b) := by
  refine ⟨fun h => ⟨h _ _, h _ _⟩, fun ⟨ha, hb⟩ a hmem => ?_⟩
  have hmem' := hmem
  rw [atoms_composed] at hmem'
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem'
  rcases hmem' with rfl | rfl
  exacts [ha, hb]

theorem exists_atoms {P : ∀ a, a ∈ composed.atoms → Prop} :
    (∃ a, ∃ ha, P a ha) ↔ P (clock .a) (mem_atoms .a) ∨ P (clock .b) (mem_atoms .b) := by
  refine ⟨fun ⟨a, hmem, h⟩ => ?_, fun h => h.elim (⟨_, _, ·⟩) (⟨_, _, ·⟩)⟩
  have hmem' := hmem
  rw [atoms_composed] at hmem'
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem'
  rcases hmem' with rfl | rfl
  exacts [.inl h, .inr h]

/-- The update of a clock is enabled iff the clock is at `5`. -/
theorem enabled_iff (k : Clock) {s x : Val M composed.ctrl} :
    composed.Enabled composed_isClosed (mem_atoms k) s x ↔ mval k s = 5 :=
  ⟨fun ⟨_, h, _⟩ => h, fun h => ⟨_, h, rfl⟩⟩

/-- Initially, every clock is at `0`. -/
theorem init_clock (k : Clock) {s : Val M composed.ctrl}
    (h : ∃ ν ∈ composed.initDraws composed_isClosed (mem_atoms k) s,
      Val.restrict (composed.ctrl_subset (mem_atoms k)) s ∈ ν.support) : mval k s = 0 := by
  obtain ⟨_, ⟨_, rfl, rfl⟩, hs⟩ := h
  exact (restrict_eq_const k (composed.ctrl_subset (mem_atoms k)) s 0).1
    ((PMF.mem_support_pure_iff _ _).1 hs)

theorem of_mem_minit {μ : PMF (Val M composed.ctrl)} {s : Val M composed.ctrl}
    (hμ : μ ∈ msys.init) (hs : s ∈ μ.support) : mval .a s = 0 ∧ mval .b s = 0 :=
  ⟨init_clock .a (Module.draw_of_mem_init hμ hs _ _),
    init_clock .b (Module.draw_of_mem_init hμ hs _ _)⟩

/-- The deterministic draws of the clocks of a valuation `t`. -/
noncomputable def pureDraws (t : Val M composed.ctrl) : composed.Draws :=
  fun _ ha _ => PMF.pure (Val.restrict (composed.ctrl_subset ha) t)

theorem pure_mem_minit : PMF.pure 0 ∈ msys.init :=
  ⟨pureDraws 0, 0, forall_atoms.2 ⟨fun _ => ⟨_, rfl, rfl⟩, fun _ => ⟨_, rfl, rfl⟩⟩,
    (Module.round_pure_atoms (fun _ _ _ => rfl) 0).symm⟩

/-- In a jump, the clock `k` resets if it is at `5`, and keeps its value otherwise. -/
def ResetOrKeep (k : Clock) (s s' : Val M composed.ctrl) : Prop :=
  (mval k s = 5 → mval k s' = 0) ∧ (mval k s ≠ 5 → mval k s' = mval k s)

theorem jump_clock (k : Clock) {s s' : Val M composed.ctrl}
    (h : ∃ ν ∈ composed.draws composed_isClosed (mem_atoms k) s s',
      Val.restrict (composed.ctrl_subset (mem_atoms k)) s' ∈ ν.support) : ResetOrKeep k s s' := by
  obtain ⟨_, ⟨_, _, ⟨h5, rfl⟩, rfl⟩ | ⟨he, rfl⟩, hs'⟩ := h
  · have := (restrict_eq_const k (composed.ctrl_subset (mem_atoms k)) s' 0).1
      ((PMF.mem_support_pure_iff _ _).1 hs')
    exact ⟨fun _ => this, fun h => absurd h5 h⟩
  · have := (restrict_eq_iff k (composed.ctrl_subset (mem_atoms k)) s' s).1
      ((PMF.mem_support_pure_iff _ _).1 hs')
    exact ⟨fun h => absurd ((enabled_iff k).2 h) he, fun _ => this⟩

/-- The jump region of a clock atom: the clock at `5`. -/
theorem region_iff (k : Clock) {s : Val M composed.ctrl} :
    s ∈ composed.Region composed_isClosed (mem_atoms k) ↔ mval k s = 5 :=
  ⟨fun ⟨_, h⟩ => (enabled_iff k).1 h, fun h => ⟨s, (enabled_iff k).2 h⟩⟩

/-- A jump happens when some clock is at `5`; then every clock resets if it is
    at `5`, and keeps its value otherwise. -/
theorem of_mem_mjump {μ : PMF (Val M composed.ctrl)} {s s' : Val M composed.ctrl}
    (hμ : μ ∈ msys.jump s) (hs' : s' ∈ μ.support) :
    (mval .a s = 5 ∨ mval .b s = 5) ∧ ResetOrKeep .a s s' ∧ ResetOrKeep .b s s' :=
  ⟨(exists_atoms.1 hμ.1).imp (region_iff .a).1 (region_iff .b).1,
    jump_clock .a (Module.draw_of_mem_jump hμ hs' _ _),
    jump_clock .b (Module.draw_of_mem_jump hμ hs' _ _)⟩

/-- Both clocks at `5` reset together. -/
theorem pure_mem_mjump {s : Val M composed.ctrl} (ha : mval .a s = 5) (hb : mval .b s = 5) :
    PMF.pure 0 ∈ msys.jump s :=
  ⟨⟨_, _, (region_iff .a).2 ha⟩, pureDraws 0,
    forall_atoms.2 ⟨fun _ => .inl ⟨(enabled_iff .a).2 ha, _, ⟨ha, rfl⟩, rfl⟩,
      fun _ => .inl ⟨(enabled_iff .b).2 hb, _, ⟨hb, rfl⟩, rfl⟩⟩,
    (Module.round_pure_atoms (fun _ _ _ => rfl) s).symm⟩

theorem mem_mflow {s : Val M composed.ctrl} {w : TangentSpace (Val.model I composed.ctrl) s} :
    w ∈ msys.flow s ↔ w ⟨.a, mem_ctrl .a⟩ = 1 ∧ w ⟨.b, mem_ctrl .b⟩ = 1 :=
  forall_atoms.trans (and_congr
    (restrict_eq_const .a (composed.ctrl_subset (mem_atoms .a)) w 1)
    (restrict_eq_const .b (composed.ctrl_subset (mem_atoms .b)) w 1))

/-- Along a trajectory of the module, both clocks grow by the time elapsed. -/
theorem mval_eq_of_trajectory {γ : ℝ → Val M composed.ctrl} {d : ℝ}
    (hγ : msys.IsTrajectory γ d) (k : Clock) :
    ∀ t ∈ Icc 0 d, mval k (γ t) = mval k (γ 0) + t :=
  Val.eq_add_of_rate_one _ fun t ht => by
    obtain ⟨v, hv, hγt⟩ := hγ.follows t ht
    have := mem_mflow.1 hv
    exact ⟨v, by cases k; exacts [this.1, this.2], hγt⟩

theorem region_mem (k : Clock) :
    composed.Region composed_isClosed (mem_atoms k) ∈ msys.regions := ⟨_, _, rfl⟩

theorem exists_region {a : Atom I M} (ha : a ∈ composed.atoms) :
    ∃ k, ∀ s, s ∈ composed.Region composed_isClosed ha ↔ mval k s = 5 := by
  have hmem' := ha
  rw [atoms_composed] at hmem'
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem'
  rcases hmem' with rfl | rfl
  exacts [⟨.a, fun _ => region_iff .a⟩, ⟨.b, fun _ => region_iff .b⟩]

/-- A flow cannot get out of the jump region of the clock `a`, a single point
    it cannot stay at, so it stops when `a` reaches `5`. -/
theorem mval_le_of_trajectory {γ : ℝ → Val M composed.ctrl} {d : ℝ}
    (hγ : msys.IsTrajectory γ d) (h5 : mval .a (γ 0) ≤ 5) : mval .a (γ 0) + d ≤ 5 := by
  by_contra hlt
  have ht : 5 - mval .a (γ 0) ∈ Icc 0 d := ⟨by linarith, by linarith⟩
  have hin := (region_iff .a (s := γ (5 - mval .a (γ 0)))).2
    (by rw [mval_eq_of_trajectory hγ .a _ ht]; ring)
  have := (region_iff .a).1 (hγ.stays _ (region_mem .a) _ ht d ⟨ht.2, le_rfl⟩ hin)
  rw [mval_eq_of_trajectory hγ .a _ ⟨hγ.nonneg, le_rfl⟩] at this
  linarith

/-- A step from synchronised clocks within bounds is a reset of both from `5`,
    or a flow that advances both by the time elapsed, staying within bounds. -/
theorem mstep_cases {s s' : Val M composed.ctrl} {d : ℝ} (hs : msys.Step s d s')
    (hab : mval .a s = mval .b s) (h5 : mval .a s ≤ 5) :
    (mval .a s = 5 ∧ mval .a s' = 0 ∧ mval .b s' = 0 ∧ d = 0) ∨
      (0 ≤ d ∧ mval .a s' = mval .a s + d ∧ mval .b s' = mval .b s + d ∧ mval .a s' ≤ 5) := by
  cases hs with
  | jump hμ hs' =>
    obtain ⟨h5, ⟨ha, _⟩, ⟨hb, _⟩⟩ := of_mem_mjump hμ hs'
    have h5a : mval .a s = 5 := h5.elim id (hab ▸ ·)
    exact .inl ⟨h5a, ha h5a, hb (hab ▸ h5a), rfl⟩
  | flow hγ =>
    have ha := mval_eq_of_trajectory hγ .a _ ⟨hγ.nonneg, le_rfl⟩
    have hb := mval_eq_of_trajectory hγ .b _ ⟨hγ.nonneg, le_rfl⟩
    exact .inr ⟨hγ.nonneg, ha, hb, ha ▸ mval_le_of_trajectory hγ h5⟩

/-- Invariant: the clocks are synchronised and stay within `[0, 5]`. -/
theorem mreachable_bounds {s : Val M composed.ctrl} (hs : msys.Reachable s) :
    mval .a s = mval .b s ∧ 0 ≤ mval .a s ∧ mval .a s ≤ 5 := by
  induction hs with
  | init hμ hs =>
    have := of_mem_minit hμ hs
    rw [this.1, this.2]
    norm_num
  | step _ hst ih =>
    rcases mstep_cases hst ih.1 ih.2.2 with ⟨_, ha, hb, _⟩ | ⟨hd, ha, hb, hle⟩
    · rw [ha, hb]
      norm_num
    · exact ⟨by rw [ha, hb, ih.1], by linarith, hle⟩

/-- The ranking function is nonnegative on reachable states. -/
theorem mrank_nonneg {s : Val M composed.ctrl} (hs : msys.Reachable s) : 0 ≤ mrank s := by
  have := mreachable_bounds hs
  simp only [mrank]
  linarith

/-- Every step other than a reset of both clocks from `5` decreases the
    ranking function by exactly the time it takes. -/
theorem mstep_rank {s s' : Val M composed.ctrl} {d : ℝ} (hs : msys.Reachable s)
    (hst : msys.Step s d s') :
    (mrank s = 0 ∧ mval .a s' = 0 ∧ mval .b s' = 0) ∨ mrank s' = mrank s - d := by
  have := mreachable_bounds hs
  rcases mstep_cases hst this.1 this.2.2 with ⟨h5, ha, hb, _⟩ | ⟨_, ha, _, _⟩
  · exact .inl ⟨by simp [mrank, h5], ha, hb⟩
  · exact .inr (by simp only [mrank, ha]; ring)

/-- Liveness: in a run of the module whose time diverges, both clocks are `0`
    (at once) infinitely many times. -/
theorem mfrequently_zero {ρ : ℕ → Val M composed.ctrl} {δ : ℕ → ℝ} (hρ : msys.IsRun ρ δ)
    (hdiv : Hybrid.Divergent δ) : ∀ N, ∃ n ≥ N, mval .a (ρ n) = 0 ∧ mval .b (ρ n) = 0 := by
  intro N
  by_contra hne
  push Not at hne
  have inv n := mreachable_bounds (hρ.reachable n)
  -- without resets, the clocks advance by the time elapsed
  have key : ∀ m, mval .a (ρ (N + m)) = mval .a (ρ N) + ∑ i ∈ Finset.range m, δ (N + i) := by
    intro m
    induction m with
    | zero => simp
    | succ m ih =>
      rcases mstep_cases (hρ.step (N + m)) (inv _).1 (inv _).2.2 with
        ⟨_, ha, hb, _⟩ | ⟨_, ha, _, _⟩
      · exact absurd hb (hne (N + m + 1) (by omega) ha)
      · rw [← add_assoc, ha, ih, Finset.sum_range_succ]
        ring
  -- so, by the ranking function, the time elapsed after `N` is at most `5`
  have bound : ∀ m, ∑ i ∈ Finset.range (N + m), δ i ≤ ∑ i ∈ Finset.range N, δ i + 5 := by
    intro m
    have h₁ := inv (N + m)
    have h₂ := inv N
    have := key m
    rw [Finset.sum_range_add]
    linarith
  obtain ⟨n, hn⟩ := (hdiv.eventually_gt_atTop (∑ i ∈ Finset.range N, δ i + 5)).exists_forall_of_atTop
  exact absurd (bound n) (not_le.2 (hn (N + n) (by omega)))

theorem minfinite_zero {ρ : ℕ → Val M composed.ctrl} {δ : ℕ → ℝ} (hρ : msys.IsRun ρ δ)
    (hdiv : Hybrid.Divergent δ) : {n | mval .a (ρ n) = 0 ∧ mval .b (ρ n) = 0}.Infinite :=
  Nat.frequently_atTop_iff_infinite.1 (frequently_atTop.2 (mfrequently_zero hρ hdiv))

/-- The run of the module that flows both clocks from `0` to `5` and resets
    them, forever. -/
noncomputable def mrun : ℕ → Val M composed.ctrl :=
  fun n => if n % 2 = 0 then 0 else fun _ => 5

theorem isTrajectory_mramp : msys.IsTrajectory (fun t _ => t) 5 where
  nonneg := by norm_num
  follows t _ := by
    have : HasDerivAt (fun t (_ : composed.ctrl) => t) (fun _ => (1 : ℝ)) t :=
      hasDerivAt_pi.2 fun _ => hasDerivAt_id t
    exact ⟨fun _ => 1, mem_mflow.2 ⟨rfl, rfl⟩, Val.hasFDerivAt_iff.2 this.hasFDerivAt⟩
  stays G hG t₁ _ t₂ ht₂ hin := by
    obtain ⟨a, ha, rfl⟩ := hG
    obtain ⟨k, hk⟩ := exists_region ha
    have h5 : t₁ = 5 := (hk _).1 hin
    rwa [show t₂ = t₁ by linarith [ht₂.1, ht₂.2]]

theorem mrun_isRun : msys.IsRun mrun dur where
  init := ⟨_, pure_mem_minit, by simp [mrun]⟩
  step n := by
    rcases Nat.mod_two_eq_zero_or_one n with h | h
    · have h' : (n + 1) % 2 = 1 := by omega
      simp only [mrun, dur, h, h', one_ne_zero, ↓reduceIte]
      exact Hybrid.Step.flow isTrajectory_mramp
    · have h' : (n + 1) % 2 = 0 := by omega
      simp only [mrun, dur, h, h', one_ne_zero, ↓reduceIte]
      exact Hybrid.Step.jump (pure_mem_mjump rfl rfl) (by simp)

/-- Both clocks of the composed module are `0` infinitely many times along `mrun`. -/
example : {n | mval .a (mrun n) = 0 ∧ mval .b (mrun n) = 0}.Infinite :=
  minfinite_zero mrun_isRun run_divergent

/-! ## Almost surely -/

/-- Under every scheduler, almost surely, the clocks stay synchronised and within `[0, 5]`. -/
theorem ae_mbounds (σ : msys.Scheduler) :
    ∀ᵐ ω ∂σ.measure, ∀ n, mval .a (ω n) = mval .b (ω n) ∧ 0 ≤ mval .a (ω n) ∧ mval .a (ω n) ≤ 5 :=
  σ.ae_forall_of_reachable fun _ => mreachable_bounds

/-- Under every scheduler, almost surely, if time diverges, both clocks are `0`
    (at once) infinitely many times. -/
theorem ae_minfinite_zero (σ : msys.Scheduler) :
    ∀ᵐ ω ∂σ.measure, Hybrid.Divergent (fun n => σ.time n (ω n)) →
      {n | mval .a (ω n) = 0 ∧ mval .b (ω n) = 0}.Infinite :=
  σ.ae_of_isRun (P := fun ρ δ => Hybrid.Divergent δ →
    {n | mval .a (ρ n) = 0 ∧ mval .b (ρ n) = 0}.Infinite) fun _ _ hρ hdiv => minfinite_zero hρ hdiv

end Zrth.Examples.Clocks
