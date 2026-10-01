import Zrth

/-!
# A birth-death process

Four atoms, one per variable: a clock `clk`, the cumulative `births` and
`deaths`, and the population `pop`. Births and deaths are random events at
the ticks of the clock:

* the clock flows at rate `1` and ticks at `1`, resetting to `0`;
* at a tick, the birth atom counts a birth with probability `1/2`;
* at a tick, the death atom counts a death with probability `1/3`, if the
  population is at least `1`;
* at a tick, the population atom *awaits the next values* of `births` and
  `deaths`, drawn before it in the round, and changes by the births and deaths
  of the round; between ticks, it flows along the *awaited tangents* of
  `births` and `deaths`, `pop' = births' - deaths'` (here `0`: the counts only
  change by events).

The population starts at `100`, the counts at `0`. Proved about every
possible run, hence almost surely under every scheduler (`ae_invariant`):

* *conservation*: `pop = 100 + births - deaths` (`reachable_invariant`);
* *nonnegativity*: `births`, `deaths` and `pop` stay nonnegative, as a death
  needs a living individual.

The distributions of the atoms are finitely supported: a coin (`coin`) or a
Dirac. For instance, a birth happens at a tick with probability `1/2`
(`birth_prob`).
-/

namespace Zrth.Examples.BirthDeath

open Manifold Set Real

/-- The variables of the process. -/
inductive Var
  | pop
  | births
  | deaths
  | clk
  deriving DecidableEq

/-- Every variable ranges over `ℝ`, modelled on itself. -/
noncomputable abbrev I (_ : Var) := 𝓘(ℝ, ℝ)
abbrev M (_ : Var) := ℝ

theorem half_le_one : (1 / 2 : NNReal) ≤ 1 := div_le_one_of_le₀ (by norm_num) (by norm_num)
theorem third_le_one : (1 / 3 : NNReal) ≤ 1 := div_le_one_of_le₀ (by norm_num) (by norm_num)

/-- The clock: it flows at rate `1` and ticks at `1`, resetting to `0`. -/
noncomputable def tick : Atom I M where
  ctrl := {.clk}
  wait := ∅
  read := {.clk}
  disjoint_ctrl_wait := Finset.disjoint_empty_right _
  init _ := {.pure 0}
  update := fun (r, _) => {μ | r ⟨.clk, by simp⟩ = 1 ∧ μ = .pure 0}
  flow _ _ := {fun _ => 1}

/-- The birth atom: at a tick, a birth with probability `1/2`. -/
noncomputable def birth : Atom I M where
  ctrl := {.births}
  wait := ∅
  read := {.births, .clk}
  disjoint_ctrl_wait := Finset.disjoint_empty_right _
  init _ := {.pure 0}
  update := fun (r, _) => {μ | r ⟨.clk, by simp⟩ = 1 ∧
    μ = .coin (1 / 2) half_le_one (fun _ => r ⟨.births, by simp⟩ + 1) (fun _ => r ⟨.births, by simp⟩)}
  flow _ _ := {fun _ => 0}

/-- The death atom: at a tick, a death with probability `1/3`, if there is
    someone to die. -/
noncomputable def death : Atom I M where
  ctrl := {.deaths}
  wait := ∅
  read := {.deaths, .pop, .clk}
  disjoint_ctrl_wait := Finset.disjoint_empty_right _
  init _ := {.pure 0}
  update := fun (r, _) => {μ | r ⟨.clk, by simp⟩ = 1 ∧
    μ = if 1 ≤ r ⟨.pop, by simp⟩ then
      .coin (1 / 3) third_le_one (fun _ => r ⟨.deaths, by simp⟩ + 1) (fun _ => r ⟨.deaths, by simp⟩)
    else .pure fun _ => r ⟨.deaths, by simp⟩}
  flow _ _ := {fun _ => 0}

/-- The population atom: at a tick, it changes by the births and deaths of the
    round, awaiting their next values; between ticks, it flows along their
    awaited tangents. -/
noncomputable def population : Atom I M where
  ctrl := {.pop}
  wait := {.births, .deaths}
  read := {.pop, .births, .deaths, .clk}
  disjoint_ctrl_wait := by decide
  init _ := {.pure fun _ => 100}
  update := fun (r, w) => {μ | r ⟨.clk, by simp⟩ = 1 ∧ μ = .pure fun _ =>
    r ⟨.pop, by simp⟩ + (w ⟨.births, by simp⟩ - r ⟨.births, by simp⟩) -
      (w ⟨.deaths, by simp⟩ - r ⟨.deaths, by simp⟩)}
  flow := fun _ (_, w) => {fun _ => w ⟨.births, by simp⟩ - w ⟨.deaths, by simp⟩}

/-- The process: the population last, since it awaits the births and deaths. -/
noncomputable def process : Module I M where
  extl := ∅
  intf := {.pop, .births, .deaths, .clk}
  prvt := ∅
  atoms := [tick, birth, death, population]
  disjoint_extl_intf := Finset.disjoint_empty_left _
  disjoint_extl_prvt := Finset.disjoint_empty_left _
  disjoint_intf_prvt := Finset.disjoint_empty_right _
  pairwise_disjoint_ctrl := by simp [tick, birth, death, population]
  mem_ctrl_iff v := by cases v <;> simp [tick, birth, death, population]
  subset_vars a ha := by
    change a ∈ [tick, birth, death, population] at ha
    simp only [List.mem_cons, List.not_mem_nil, or_false] at ha
    rcases ha with rfl | rfl | rfl | rfl <;> decide
  pairwise_await := by simp [tick, birth, death, population]

theorem process_isClosed : process.IsClosed := rfl

/-- The hybrid system of the process. -/
noncomputable def sys : Hybrid (Val.model I process.ctrl) (Val M process.ctrl) :=
  process.toHybrid process_isClosed

theorem mem_ctrl (k : Var) : k ∈ process.ctrl := by cases k <;> decide

theorem mem_tick : tick ∈ process.atoms := by simp [process]
theorem mem_birth : birth ∈ process.atoms := by simp [process]
theorem mem_death : death ∈ process.atoms := by simp [process]
theorem mem_population : population ∈ process.atoms := by simp [process]

/-- The value of the variable `k`. -/
def val (k : Var) (s : Val M process.ctrl) : ℝ := s ⟨k, mem_ctrl k⟩

/-- The probability of a birth at a tick (the update is only defined at a tick). -/
theorem birth_prob (r : Val M birth.read) (w : Val M birth.wait)
    {μ : FinDist (Val M birth.ctrl)} (hμ : μ ∈ birth.update (r, w)) :
    μ.toPMF (fun _ => r ⟨.births, by simp [birth]⟩ + 1) = 1 / 2 := by
  obtain ⟨_, rfl⟩ := hμ
  refine (FinDist.coin_apply_left fun h => ?_).trans (by norm_num [ENNReal.coe_div])
  have := congrFun h ⟨.births, by simp⟩
  simp at this

/-! ## Rounds -/

variable {s s' : Val M process.ctrl}

/-- What the draw of an atom `a` in a jump from `s` to `s'` gives. -/
abbrev Drawn (a : Atom I M) (ha : a ∈ process.atoms) (s s' : Val M process.ctrl) : Prop :=
  ∃ ν ∈ process.draws process_isClosed ha s s',
    Val.restrict (process.ctrl_subset ha) s' ∈ ν.support

/-- Extracting the value of a variable from equal valuations. -/
theorem val_eq {X : Finset Var} {f g : Val M X} (e : f = g) (k : Var) (hk : k ∈ X) :
    f ⟨k, hk⟩ = g ⟨k, hk⟩ := congrFun e _

theorem tick_draw (h : Drawn tick mem_tick s s') :
    (val .clk s = 1 ∧ val .clk s' = 0) ∨ (val .clk s ≠ 1 ∧ val .clk s' = val .clk s) := by
  obtain ⟨_, ⟨_, _, ⟨h1, rfl⟩, rfl⟩ | ⟨hne, rfl⟩, hs'⟩ := h
  · exact .inl ⟨h1, val_eq ((PMF.mem_support_pure_iff _ _).1 hs') .clk (Finset.mem_singleton_self _)⟩
  · exact .inr ⟨fun h1 => hne ⟨_, h1, rfl⟩,
      val_eq ((PMF.mem_support_pure_iff _ _).1 hs') .clk (Finset.mem_singleton_self _)⟩

theorem birth_draw (h : Drawn birth mem_birth s s') :
    (val .clk s = 1 ∧ (val .births s' = val .births s + 1 ∨ val .births s' = val .births s)) ∨
      (val .clk s ≠ 1 ∧ val .births s' = val .births s) := by
  obtain ⟨_, ⟨_, _, ⟨h1, rfl⟩, rfl⟩ | ⟨hne, rfl⟩, hs'⟩ := h
  · refine .inl ⟨h1, (FinDist.mem_support_coin hs').imp (fun e => ?_) (fun e => ?_)⟩ <;>
      exact val_eq e .births (Finset.mem_singleton_self _)
  · exact .inr ⟨fun h1 => hne ⟨_, h1, rfl⟩,
      val_eq ((PMF.mem_support_pure_iff _ _).1 hs') .births (Finset.mem_singleton_self _)⟩

theorem death_draw (h : Drawn death mem_death s s') :
    (val .clk s = 1 ∧ ((1 ≤ val .pop s ∧ val .deaths s' = val .deaths s + 1) ∨
      val .deaths s' = val .deaths s)) ∨ (val .clk s ≠ 1 ∧ val .deaths s' = val .deaths s) := by
  obtain ⟨_, ⟨_, _, ⟨h1, rfl⟩, rfl⟩ | ⟨hne, rfl⟩, hs'⟩ := h
  · refine .inl ⟨h1, ?_⟩
    split_ifs at hs' with hp
    · refine (FinDist.mem_support_coin hs').imp (fun e => ⟨hp, ?_⟩) (fun e => ?_) <;>
        exact val_eq e .deaths (Finset.mem_singleton_self _)
    · exact .inr (val_eq ((PMF.mem_support_pure_iff _ _).1 hs') .deaths (Finset.mem_singleton_self _))
  · exact .inr ⟨fun h1 => hne ⟨_, h1, rfl⟩,
      val_eq ((PMF.mem_support_pure_iff _ _).1 hs') .deaths (Finset.mem_singleton_self _)⟩

theorem population_draw (h : Drawn population mem_population s s') :
    (val .clk s = 1 ∧ val .pop s' = val .pop s + (val .births s' - val .births s) -
      (val .deaths s' - val .deaths s)) ∨ (val .clk s ≠ 1 ∧ val .pop s' = val .pop s) := by
  obtain ⟨_, ⟨_, _, ⟨h1, rfl⟩, rfl⟩ | ⟨hne, rfl⟩, hs'⟩ := h
  · exact .inl ⟨h1, val_eq ((PMF.mem_support_pure_iff _ _).1 hs') .pop (Finset.mem_singleton_self _)⟩
  · exact .inr ⟨fun h1 => hne ⟨_, h1, rfl⟩,
      val_eq ((PMF.mem_support_pure_iff _ _).1 hs') .pop (Finset.mem_singleton_self _)⟩

/-! ## Flows -/

/-- Along a flow, the counts and the population do not change: the population
    follows the awaited tangents of the counts, which only change by events. -/
theorem flow_rates {w : TangentSpace (Val.model I process.ctrl) s} (hw : w ∈ sys.flow s) :
    w ⟨.births, mem_ctrl _⟩ = 0 ∧ w ⟨.deaths, mem_ctrl _⟩ = 0 ∧ w ⟨.pop, mem_ctrl _⟩ = 0 := by
  have hb : w ⟨.births, mem_ctrl _⟩ = 0 :=
    congrFun (Set.mem_singleton_iff.1 (hw birth mem_birth)) ⟨.births, Finset.mem_singleton_self _⟩
  have hd : w ⟨.deaths, mem_ctrl _⟩ = 0 :=
    congrFun (Set.mem_singleton_iff.1 (hw death mem_death)) ⟨.deaths, Finset.mem_singleton_self _⟩
  have hp : w ⟨.pop, mem_ctrl _⟩ = w ⟨.births, mem_ctrl _⟩ - w ⟨.deaths, mem_ctrl _⟩ :=
    congrFun (Set.mem_singleton_iff.1 (hw population mem_population))
      ⟨.pop, Finset.mem_singleton_self _⟩
  exact ⟨hb, hd, by rw [hp, hb, hd, sub_zero]⟩

theorem val_of_trajectory {γ : ℝ → Val M process.ctrl} {d : ℝ} (hγ : sys.IsTrajectory γ d)
    {k : Var} (hk : k = .births ∨ k = .deaths ∨ k = .pop) :
    val k (γ d) = val k (γ 0) :=
  eq_of_hasDerivAt_zero (f := fun t => val k (γ t)) (fun t ht => by
    obtain ⟨w, hw, hγt⟩ := hγ.follows t ht
    have := Val.hasDerivAt_apply ⟨k, mem_ctrl k⟩ hγt
    obtain ⟨hb, hd, hp⟩ := flow_rates hw
    rcases hk with rfl | rfl | rfl
    exacts [hb ▸ this, hd ▸ this, hp ▸ this]) d (right_mem_Icc.2 hγ.nonneg)

/-! ## Invariants -/

/-- Initially, the population is `100` and the counts are `0`. -/
theorem of_mem_init {μ : PMF (Val M process.ctrl)} (hμ : μ ∈ sys.init) (hs : s ∈ μ.support) :
    val .pop s = 100 ∧ val .births s = 0 ∧ val .deaths s = 0 := by
  have h := Module.draw_of_mem_init hμ hs
  obtain ⟨_, ⟨_, rfl, rfl⟩, hp⟩ := h population mem_population
  obtain ⟨_, ⟨_, rfl, rfl⟩, hb⟩ := h birth mem_birth
  obtain ⟨_, ⟨_, rfl, rfl⟩, hd⟩ := h death mem_death
  exact ⟨val_eq ((PMF.mem_support_pure_iff _ _).1 hp) .pop (Finset.mem_singleton_self _),
    val_eq ((PMF.mem_support_pure_iff _ _).1 hb) .births (Finset.mem_singleton_self _),
    val_eq ((PMF.mem_support_pure_iff _ _).1 hd) .deaths (Finset.mem_singleton_self _)⟩

/-- The invariant of the process. -/
def Inv (s : Val M process.ctrl) : Prop :=
  val .pop s = 100 + val .births s - val .deaths s ∧ 0 ≤ val .pop s ∧
    0 ≤ val .births s ∧ 0 ≤ val .deaths s

/-- A jump preserves the invariant: births and deaths are counted in the
    population, and a death needs a living individual. -/
theorem inv_of_jump {μ : PMF (Val M process.ctrl)} (hμ : μ ∈ sys.jump s) (hs' : s' ∈ μ.support)
    (hinv : Inv s) : Inv s' := by
  have h := Module.draw_of_mem_jump hμ hs'
  obtain ⟨p, p0, b0, d0⟩ := hinv
  rcases population_draw (h _ mem_population) with ⟨h1, hp⟩ | ⟨h1, hp⟩
  · rcases birth_draw (h _ mem_birth) with ⟨_, hb | hb⟩ | ⟨h1', _⟩
    · rcases death_draw (h _ mem_death) with ⟨_, ⟨hl, hd⟩ | hd⟩ | ⟨h1', _⟩
      · refine ⟨by rw [hp, hb, hd, p]; ring, by rw [hp, hb, hd]; linarith, by linarith, by linarith⟩
      · refine ⟨by rw [hp, hb, hd, p]; ring, by rw [hp, hb, hd]; linarith, by linarith, by linarith⟩
      · exact absurd h1 h1'
    · rcases death_draw (h _ mem_death) with ⟨_, ⟨hl, hd⟩ | hd⟩ | ⟨h1', _⟩
      · refine ⟨by rw [hp, hb, hd, p]; ring, by rw [hp, hb, hd]; linarith, by linarith, by linarith⟩
      · refine ⟨by rw [hp, hb, hd, p]; ring, by rw [hp, hb, hd]; linarith, by linarith, by linarith⟩
      · exact absurd h1 h1'
    · exact absurd h1 h1'
  · rcases birth_draw (h _ mem_birth) with ⟨h1', _⟩ | ⟨_, hb⟩
    · exact absurd h1' h1
    rcases death_draw (h _ mem_death) with ⟨h1', _⟩ | ⟨_, hd⟩
    · exact absurd h1' h1
    exact ⟨by rw [hp, hb, hd, p], by rw [hp]; exact p0, by rw [hb]; exact b0, by rw [hd]; exact d0⟩

/-- Invariant: conservation, and the population and the counts are nonnegative. -/
theorem reachable_invariant (hs : sys.Reachable s) : Inv s := by
  induction hs with
  | init hμ hs =>
    obtain ⟨hp, hb, hd⟩ := of_mem_init hμ hs
    refine ⟨by rw [hp, hb, hd]; ring, ?_, ?_, ?_⟩ <;> simp [hp, hb, hd]
  | step _ hst ih =>
    cases hst with
    | jump hμ hs' => exact inv_of_jump hμ hs' ih
    | flow hγ =>
      rw [Inv, val_of_trajectory hγ (.inr (.inr rfl)), val_of_trajectory hγ (.inl rfl),
        val_of_trajectory hγ (.inr (.inl rfl))]
      exact ih

/-- Under every scheduler, almost surely, the invariant holds at every step:
    the population is conserved and nonnegative, whatever the random events. -/
theorem ae_invariant (σ : sys.Scheduler) : ∀ᵐ ω ∂σ.measure, ∀ n, Inv (ω n) :=
  σ.ae_forall_of_reachable fun _ => reachable_invariant

end Zrth.Examples.BirthDeath
