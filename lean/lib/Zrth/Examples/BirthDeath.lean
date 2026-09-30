import Zrth.Hybrid
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# A continuous birth-death process

Three atoms, one per variable: the population `pop`, and the cumulative
`births` and `deaths`.

* The birth atom reads the current population and flows `births' = 2 pop`.
* The death atom reads the current population and flows `deaths' = pop`.
* The population atom *awaits the tangents* of `births` and `deaths`, but not
  their values, and flows `pop' = births' - deaths'`.

Their flows form the system of differential equations of the process. The
updates skip, so jumps only stutter. The population starts at `100`, the
cumulative counts at `0`.

Proved about the process:

* *conservation*: since the population follows the awaited tangents,
  `pop - births + deaths` is constant along every flow, so every reachable
  state satisfies `pop = 100 + births - deaths` (`reachable_conservation`);
* *ratio*: births happen at twice the rate of deaths, so `births = 2 deaths`
  (`reachable_ratio`), and hence `pop = 100 + deaths` (`reachable_pop_eq`);
* *solution*: along every flow, `pop' = births' - deaths' = pop`, so the
  population grows exponentially, `pop t = pop 0 * eᵗ` (`pop_of_trajectory`);
* *survival*: the population never drops below `100` (`reachable_pop_ge`).

The exponential growth `pop = 100 eᵗ`, `births = 200 (eᵗ - 1)`,
`deaths = 100 (eᵗ - 1)` is a trajectory from the initial state
(`isTrajectory_growth`, `growth_reachable`).
-/

namespace Zrth.Examples.BirthDeath

open Manifold Set Real

/-- The variables of the process. -/
inductive Var
  | pop
  | births
  | deaths
  deriving DecidableEq

/-- Every variable ranges over `ℝ`, modelled on itself. -/
noncomputable abbrev I (_ : Var) := 𝓘(ℝ, ℝ)
abbrev M (_ : Var) := ℝ

/-- The birth atom: `births' = 2 pop`, reading the current population. -/
noncomputable def birth : Atom I M where
  ctrl := {.births}
  wait := ∅
  read := {.births, .pop}
  disjoint_ctrl_wait := Finset.disjoint_empty_right _
  init _ := {0}
  update | (r, _) => {fun _ => r ⟨.births, by simp⟩}
  flow _ | (r, _) => {fun _ => 2 * r ⟨.pop, by simp⟩}

/-- The death atom: `deaths' = pop`, reading the current population. -/
noncomputable def death : Atom I M where
  ctrl := {.deaths}
  wait := ∅
  read := {.deaths, .pop}
  disjoint_ctrl_wait := Finset.disjoint_empty_right _
  init _ := {0}
  update | (r, _) => {fun _ => r ⟨.deaths, by simp⟩}
  flow _ | (r, _) => {fun _ => r ⟨.pop, by simp⟩}

/-- The population atom: `pop' = births' - deaths'`, awaiting the tangents of
    `births` and `deaths`. -/
noncomputable def population : Atom I M where
  ctrl := {.pop}
  wait := {.births, .deaths}
  read := {.pop}
  disjoint_ctrl_wait := by decide
  init _ := {fun _ => 100}
  update | (r, _) => {fun _ => r ⟨.pop, by simp⟩}
  flow _ | (_, w) => {fun _ => w ⟨.births, by simp⟩ - w ⟨.deaths, by simp⟩}

/-- The process: the three atoms, the population last, since it awaits the others. -/
noncomputable def process : Module I M where
  extl := ∅
  intf := {.pop, .births, .deaths}
  prvt := ∅
  atoms := [birth, death, population]
  disjoint_extl_intf := Finset.disjoint_empty_left _
  disjoint_extl_prvt := Finset.disjoint_empty_left _
  disjoint_intf_prvt := Finset.disjoint_empty_right _
  pairwise_disjoint_ctrl := by simp [birth, death, population]
  mem_ctrl_iff v := by cases v <;> simp [birth, death, population]
  subset_vars a ha := by
    change a ∈ [birth, death, population] at ha
    simp only [List.mem_cons, List.not_mem_nil, or_false] at ha
    rcases ha with rfl | rfl | rfl <;> decide
  pairwise_await := by simp [birth, death, population]

theorem process_isClosed : process.IsClosed := rfl

/-- The hybrid system of the process. -/
noncomputable def sys : Hybrid (Val.model I process.ctrl) (Val M process.ctrl) :=
  process.toHybrid process_isClosed

theorem mem_ctrl (k : Var) : k ∈ process.ctrl := by cases k <;> decide

theorem mem_birth : birth ∈ process.atoms := by simp [process]
theorem mem_death : death ∈ process.atoms := by simp [process]
theorem mem_population : population ∈ process.atoms := by simp [process]

theorem forall_atoms {P : ∀ a, a ∈ process.atoms → Prop} (h₁ : P birth mem_birth)
    (h₂ : P death mem_death) (h₃ : P population mem_population) : ∀ a ha, P a ha := by
  intro a ha
  have ha' := ha
  change a ∈ [birth, death, population] at ha'
  simp only [List.mem_cons, List.not_mem_nil, or_false] at ha'
  rcases ha' with rfl | rfl | rfl
  exacts [h₁, h₂, h₃]

/-- The value of the variable `k`. -/
def val (k : Var) (s : Val M process.ctrl) : ℝ := s ⟨k, mem_ctrl k⟩

/-- The conserved quantity. -/
def conserved (s : Val M process.ctrl) : ℝ := val .pop s - val .births s + val .deaths s

/-! ## Flows -/

theorem hasFDerivAt_of_hasMFDerivAt {X : Finset Var} {γ : ℝ → Val M X} {t : ℝ}
    {f' : ℝ →L[ℝ] ((v : X) → ℝ)} (h : HasMFDerivAt 𝓘(ℝ, ℝ) (Val.model I X) γ t f') :
    HasFDerivAt γ f' t := by
  have := h.2
  simp only [writtenInExtChartAt, extChartAt, mfld_simps] at this
  exact hasFDerivWithinAt_univ.1 this

theorem hasMFDerivAt_of_hasFDerivAt {X : Finset Var} {γ : ℝ → Val M X} {t : ℝ}
    {f' : ℝ →L[ℝ] ((v : X) → ℝ)} (h : HasFDerivAt γ f' t) :
    HasMFDerivAt 𝓘(ℝ, ℝ) (Val.model I X) γ t f' := by
  refine ⟨h.continuousAt, ?_⟩
  simp only [writtenInExtChartAt, extChartAt, mfld_simps]
  exact hasFDerivWithinAt_univ.2 h

/-- Along a curve with tangent `w`, every variable moves at the rate `w` gives it. -/
theorem hasDerivAt_val {γ : ℝ → Val M process.ctrl} {t : ℝ} {w : (v : process.ctrl) → ℝ}
    (h : HasMFDerivAt 𝓘(ℝ, ℝ) (Val.model I process.ctrl) γ t ((1 : ℝ →L[ℝ] ℝ).smulRight w))
    (k : Var) : HasDerivAt (fun t => val k (γ t)) (w ⟨k, mem_ctrl k⟩) t := by
  have hp := (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : process.ctrl => ℝ)
    ⟨k, mem_ctrl k⟩).hasFDerivAt.comp t (hasFDerivAt_of_hasMFDerivAt h)
  rw [hasDerivAt_iff_hasFDerivAt]
  convert hp using 1
  · rfl
  · ext
    rw [ContinuousLinearMap.toSpanSingleton_apply]
    rfl

/-- A function with zero derivative on `[0, d]` is constant there. -/
theorem eq_of_hasDerivAt_zero {f : ℝ → ℝ} {d : ℝ} (hd : ∀ t ∈ Icc 0 d, HasDerivAt f 0 t) :
    ∀ t ∈ Icc 0 d, f t = f 0 :=
  constant_of_has_deriv_right_zero (fun t ht => (hd t ht).continuousAt.continuousWithinAt)
    (fun t ht => (hd t (Ico_subset_Icc_self ht)).hasDerivWithinAt)

variable {s : Val M process.ctrl} {w : TangentSpace (Val.model I process.ctrl) s}

/-- Births happen at twice the rate of the population. -/
theorem flow_births (hw : w ∈ sys.flow s) : w ⟨.births, mem_ctrl _⟩ = 2 * val .pop s :=
  congrFun (Set.mem_singleton_iff.1 (hw birth mem_birth)) ⟨.births, Finset.mem_singleton_self _⟩

/-- Deaths happen at the rate of the population. -/
theorem flow_deaths (hw : w ∈ sys.flow s) : w ⟨.deaths, mem_ctrl _⟩ = val .pop s :=
  congrFun (Set.mem_singleton_iff.1 (hw death mem_death)) ⟨.deaths, Finset.mem_singleton_self _⟩

/-- The flow of the population follows the awaited tangents. -/
theorem flow_pop (hw : w ∈ sys.flow s) :
    w ⟨.pop, mem_ctrl _⟩ = w ⟨.births, mem_ctrl _⟩ - w ⟨.deaths, mem_ctrl _⟩ :=
  congrFun (Set.mem_singleton_iff.1 (hw population mem_population)) ⟨.pop, Finset.mem_singleton_self _⟩

/-- So the population grows at its own rate. -/
theorem flow_pop' (hw : w ∈ sys.flow s) : w ⟨.pop, mem_ctrl _⟩ = val .pop s := by
  rw [flow_pop hw, flow_births hw, flow_deaths hw]
  ring

variable {γ : ℝ → Val M process.ctrl} {d : ℝ}

/-- Along a trajectory, `pop - births + deaths` is constant: the population
    follows the awaited tangents of births and deaths. -/
theorem conserved_of_trajectory (hγ : sys.IsTrajectory γ d) :
    ∀ t ∈ Icc 0 d, conserved (γ t) = conserved (γ 0) :=
  eq_of_hasDerivAt_zero fun t ht => by
    obtain ⟨w, hw, hγt⟩ := hγ.follows t ht
    have := ((hasDerivAt_val hγt .pop).sub (hasDerivAt_val hγt .births)).add
      (hasDerivAt_val hγt .deaths)
    rwa [flow_pop hw, sub_sub_cancel_left, neg_add_cancel] at this

/-- Along a trajectory, `births - 2 deaths` is constant. -/
theorem ratio_of_trajectory (hγ : sys.IsTrajectory γ d) :
    ∀ t ∈ Icc 0 d, val .births (γ t) - 2 * val .deaths (γ t) =
      val .births (γ 0) - 2 * val .deaths (γ 0) :=
  eq_of_hasDerivAt_zero (f := fun t => val .births (γ t) - 2 * val .deaths (γ t)) fun t ht => by
    obtain ⟨w, hw, hγt⟩ := hγ.follows t ht
    have := (hasDerivAt_val hγt .births).sub ((hasDerivAt_val hγt .deaths).const_mul 2)
    rwa [flow_births hw, flow_deaths hw, sub_self] at this

/-- Along a trajectory, the population grows exponentially. -/
theorem pop_of_trajectory (hγ : sys.IsTrajectory γ d) :
    ∀ t ∈ Icc 0 d, val .pop (γ t) = val .pop (γ 0) * exp t := by
  have hc := eq_of_hasDerivAt_zero (d := d) (f := fun t => val .pop (γ t) * exp (-t))
    fun t ht => by
      obtain ⟨w, hw, hγt⟩ := hγ.follows t ht
      have := (hasDerivAt_val hγt .pop).mul (hasDerivAt_neg t).exp
      convert this using 1
      rw [flow_pop' hw]
      ring
  intro t ht
  have := congrArg (· * exp t) (hc t ht)
  simp only [neg_zero, exp_zero, mul_one, mul_assoc, ← exp_add, neg_add_cancel] at this
  exact this

/-! ## Invariants -/

/-- A jump stutters: the updates skip. -/
theorem val_of_jump {s s' : Val M process.ctrl} (h : s' ∈ sys.jump s) (k : Var) :
    val k s' = val k s := by
  obtain ⟨_, h⟩ := h
  cases k
  · obtain ⟨hf, hk⟩ := h population mem_population
    by_cases he : process.Enabled process_isClosed mem_population s s'
    · exact congrFun (Set.mem_singleton_iff.1 (hf he)) ⟨.pop, Finset.mem_singleton_self _⟩
    · exact congrFun (hk he) ⟨.pop, Finset.mem_singleton_self _⟩
  · obtain ⟨hf, hk⟩ := h birth mem_birth
    by_cases he : process.Enabled process_isClosed mem_birth s s'
    · exact congrFun (Set.mem_singleton_iff.1 (hf he)) ⟨.births, Finset.mem_singleton_self _⟩
    · exact congrFun (hk he) ⟨.births, Finset.mem_singleton_self _⟩
  · obtain ⟨hf, hk⟩ := h death mem_death
    by_cases he : process.Enabled process_isClosed mem_death s s'
    · exact congrFun (Set.mem_singleton_iff.1 (hf he)) ⟨.deaths, Finset.mem_singleton_self _⟩
    · exact congrFun (hk he) ⟨.deaths, Finset.mem_singleton_self _⟩

theorem val_of_init {s : Val M process.ctrl} (h : s ∈ sys.init) :
    val .pop s = 100 ∧ val .births s = 0 ∧ val .deaths s = 0 :=
  ⟨congrFun (Set.mem_singleton_iff.1 (h population mem_population)) ⟨.pop, Finset.mem_singleton_self _⟩,
    congrFun (Set.mem_singleton_iff.1 (h birth mem_birth)) ⟨.births, Finset.mem_singleton_self _⟩,
    congrFun (Set.mem_singleton_iff.1 (h death mem_death)) ⟨.deaths, Finset.mem_singleton_self _⟩⟩

/-- The invariant of the process: the conservation law, the ratio of births
    to deaths, and the population never dropping below its initial value. -/
theorem reachable_invariant {s : Val M process.ctrl} (hs : sys.Reachable s) :
    conserved s = 100 ∧ val .births s = 2 * val .deaths s ∧ 100 ≤ val .pop s := by
  induction hs with
  | init h =>
    obtain ⟨hp, hb, hd⟩ := val_of_init h
    simp only [conserved, hp, hb, hd]
    norm_num
  | step _ hst ih =>
    cases hst with
    | jump h => simpa only [conserved, val_of_jump h] using ih
    | flow hγ =>
      have hd := right_mem_Icc.2 hγ.nonneg
      refine ⟨by rw [conserved_of_trajectory hγ _ hd, ih.1], ?_, ?_⟩
      · have := ratio_of_trajectory hγ _ hd
        linarith [ih.2.1]
      · rw [pop_of_trajectory hγ _ hd]
        nlinarith [ih.2.2, one_le_exp hγ.nonneg]

/-- Conservation: every reachable state satisfies `pop = 100 + births - deaths`. -/
theorem reachable_conservation {s : Val M process.ctrl} (hs : sys.Reachable s) :
    val .pop s = 100 + val .births s - val .deaths s := by
  have := (reachable_invariant hs).1
  simp only [conserved] at this
  linarith

/-- Twice as many births as deaths. -/
theorem reachable_ratio {s : Val M process.ctrl} (hs : sys.Reachable s) :
    val .births s = 2 * val .deaths s :=
  (reachable_invariant hs).2.1

/-- So the population exceeds its initial value by the number of deaths. -/
theorem reachable_pop_eq {s : Val M process.ctrl} (hs : sys.Reachable s) :
    val .pop s = 100 + val .deaths s := by
  rw [reachable_conservation hs, reachable_ratio hs]
  ring

/-- The population never drops below its initial value, so it never goes extinct. -/
theorem reachable_pop_ge {s : Val M process.ctrl} (hs : sys.Reachable s) : 100 ≤ val .pop s :=
  (reachable_invariant hs).2.2

/-! ## Exponential growth -/

/-- The exponential growth of the process. -/
noncomputable def growth (t : ℝ) : Val M process.ctrl := fun v =>
  match v.1 with
  | .pop => 100 * exp t
  | .births => 200 * (exp t - 1)
  | .deaths => 100 * (exp t - 1)

/-- The tangent of `growth`. -/
noncomputable def growth' (t : ℝ) : (v : process.ctrl) → ℝ := fun v =>
  match v.1 with
  | .pop => 100 * exp t
  | .births => 200 * exp t
  | .deaths => 100 * exp t

theorem hasDerivAt_growth (t : ℝ) : HasDerivAt growth (growth' t) t := by
  refine hasDerivAt_pi.2 fun ⟨k, _⟩ => ?_
  cases k
  · exact (hasDerivAt_exp t).const_mul 100
  · exact ((hasDerivAt_exp t).sub_const 1).const_mul 200
  · exact ((hasDerivAt_exp t).sub_const 1).const_mul 100

/-- Every state is in every jump region: the skipping updates are defined everywhere. -/
theorem mem_region {a : Atom I M} (ha : a ∈ process.atoms) (s : Val M process.ctrl) :
    s ∈ process.Region process_isClosed ha :=
  forall_atoms (P := fun _ ha => s ∈ process.Region process_isClosed ha)
    ⟨s, _, rfl⟩ ⟨s, _, rfl⟩ ⟨s, _, rfl⟩ a ha

theorem isTrajectory_growth {d : ℝ} (hd : 0 ≤ d) : sys.IsTrajectory growth d where
  nonneg := hd
  follows t _ := by
    refine ⟨growth' t, ?_, hasMFDerivAt_of_hasFDerivAt (hasDerivAt_growth t).hasFDerivAt⟩
    refine forall_atoms ?_ ?_ ?_
    · refine Set.mem_singleton_iff.2 (funext fun ⟨v, hv⟩ => ?_)
      obtain rfl := Finset.mem_singleton.1 (hv : v ∈ ({.births} : Finset Var))
      show 200 * exp t = 2 * (100 * exp t)
      ring
    · refine Set.mem_singleton_iff.2 (funext fun ⟨v, hv⟩ => ?_)
      obtain rfl := Finset.mem_singleton.1 (hv : v ∈ ({.deaths} : Finset Var))
      rfl
    · refine Set.mem_singleton_iff.2 (funext fun ⟨v, hv⟩ => ?_)
      obtain rfl := Finset.mem_singleton.1 (hv : v ∈ ({.pop} : Finset Var))
      show 100 * exp t = 200 * exp t - 100 * exp t
      ring
  stays G hG _ _ t₂ _ _ := by
    obtain ⟨a, ha, rfl⟩ := hG
    exact mem_region ha _

theorem growth_init : growth 0 ∈ sys.init := by
  refine forall_atoms ?_ ?_ ?_
  · refine Set.mem_singleton_iff.2 (funext fun ⟨v, hv⟩ => ?_)
    obtain rfl := Finset.mem_singleton.1 (hv : v ∈ ({.births} : Finset Var))
    show 200 * (exp 0 - 1) = 0
    simp
  · refine Set.mem_singleton_iff.2 (funext fun ⟨v, hv⟩ => ?_)
    obtain rfl := Finset.mem_singleton.1 (hv : v ∈ ({.deaths} : Finset Var))
    show 100 * (exp 0 - 1) = 0
    simp
  · refine Set.mem_singleton_iff.2 (funext fun ⟨v, hv⟩ => ?_)
    obtain rfl := Finset.mem_singleton.1 (hv : v ∈ ({.pop} : Finset Var))
    show 100 * exp 0 = 100
    simp

/-- The exponential growth is reachable at every time. -/
theorem growth_reachable {d : ℝ} (hd : 0 ≤ d) : sys.Reachable (growth d) :=
  .step (.init growth_init) (.flow (isTrajectory_growth hd))

/-- ...so it satisfies the conservation law. -/
example (d : ℝ) (hd : 0 ≤ d) :
    val .pop (growth d) = 100 + val .births (growth d) - val .deaths (growth d) :=
  reachable_conservation (growth_reachable hd)

end Zrth.Examples.BirthDeath
