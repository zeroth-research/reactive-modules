import Zrth.Module

/-!
# Hybrid semantics

A (probabilistic) hybrid system on a manifold `S` (modelled by `I`): sets of
initial distributions, a jump giving, at each state, a set of distributions
of the next state, a flow, i.e., a differential inclusion assigning to every
state the tangent vectors it may move along, and *jump regions*. The sets are
nondeterministic choices, the distributions probabilistic ones; a scheduler
resolving the former turns the system into a stochastic process
(`Zrth.Stochastic`).

Its runs alternate discrete *jumps* and continuous *flows* along
trajectories. Jumps are not urgent: a jump may happen at any state where it is
enabled, and a flow may go on through such states. But a flow cannot get out
of a jump region: once a trajectory enters one, it stays in it, so the jump
must happen before the region is left. (A jump region is thus meant to be
convex along the flow: a flow leaving it and coming back is ruled out
altogether.) The possible runs (`IsRun`) are those whose jumps land in the
support of a jump distribution.

A *closed* atom, one that awaits nothing and reads only what it controls, is
such a system on the valuations of its controlled variables (`Atom.toHybrid`):
its initial action gives the initial distributions, its update the jumps and
its flow the flow.

So is a *closed* module, one without external variables (`Module.toHybrid`),
as a jump process:

* first, all atoms initialise, in the order of the module, each drawing its
  controlled variables from one of its initial distributions, given the
  initial values drawn for the variables it awaits;
* then the atoms flow together, each along a tangent vector its flow admits,
  given the current values it reads and the tangents it awaits, *racing*
  against each other: an atom may fire anywhere in its jump region, i.e.,
  where its update is defined, and must fire before getting out of it;
* the atom winning the race triggers a jump, a round in which the atoms, in
  the order of the module, draw the next values of their controlled
  variables: each *enabled* atom from one of its update distributions, given
  the latched values it reads and the next values drawn for the variables it
  awaits, while the other atoms stutter, keeping their values; then the atoms
  flow again.

The order of the atoms, consistent with their await relation, makes the
next values an atom awaits drawn before it draws: a round is a chain of
`PMF.bind`s (`Module.round`).
-/

namespace Zrth

open Manifold Set Filter

variable {E H : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [TopologicalSpace H]
  (I : ModelWithCorners ℝ E H) (S : Type*) [TopologicalSpace S] [ChartedSpace H S]

/-- A probabilistic hybrid system on `S`. -/
structure Hybrid where
  /-- The initial distributions. -/
  init : Set (PMF S)
  /-- The jumps: the distributions of the state after a discrete step. -/
  jump : S → Set (PMF S)
  /-- The flow: the admissible tangent vectors at each state. -/
  flow : (s : S) → Set (TangentSpace I s)
  /-- The jump regions, which a flow cannot get out of. -/
  regions : Set (Set S)

namespace Hybrid

variable {I S} (h : Hybrid I S)

/-- `γ` is a trajectory of `h` of duration `d`: it follows the flow on
    `[0, d]` and does not get out of any jump region it enters. -/
structure IsTrajectory (γ : ℝ → S) (d : ℝ) : Prop where
  nonneg : 0 ≤ d
  follows : ∀ t ∈ Icc 0 d, ∃ v ∈ h.flow (γ t),
    HasMFDerivAt 𝓘(ℝ, ℝ) I γ t ((1 : ℝ →L[ℝ] ℝ).smulRight v)
  stays : ∀ G ∈ h.regions, ∀ t₁ ∈ Icc 0 d, ∀ t₂ ∈ Icc t₁ d, γ t₁ ∈ G → γ t₂ ∈ G

/-- A move from `s` taking the time `d`, with the distribution of the next
    state: a jump (taking no time), or a flow along a trajectory of duration
    `d` (deterministically reaching its end). -/
inductive Move : S → ℝ → PMF S → Prop
  | jump {s : S} {μ : PMF S} : μ ∈ h.jump s → Move s 0 μ
  | flow {γ : ℝ → S} {d : ℝ} : h.IsTrajectory γ d → Move (γ 0) d (PMF.pure (γ d))

/-- A possible step from `s` to `s'` taking the time `d`: a jump to a state in
    the support of a jump distribution, or a flow along a trajectory of
    duration `d`. -/
inductive Step : S → ℝ → S → Prop
  | jump {s s' : S} {μ : PMF S} : μ ∈ h.jump s → s' ∈ μ.support → Step s 0 s'
  | flow {γ : ℝ → S} {d : ℝ} : h.IsTrajectory γ d → Step (γ 0) d (γ d)

theorem Move.step {h : Hybrid I S} {s s' : S} {d : ℝ} {μ : PMF S} (hm : h.Move s d μ)
    (hs' : s' ∈ μ.support) : h.Step s d s' := by
  cases hm with
  | jump hμ => exact .jump hμ hs'
  | flow hγ =>
    rw [PMF.support_pure, Set.mem_singleton_iff] at hs'
    exact hs' ▸ .flow hγ

/-- The states reachable, with positive probability, from an initial state. -/
inductive Reachable : S → Prop
  | init {μ : PMF S} {s : S} : μ ∈ h.init → s ∈ μ.support → Reachable s
  | step {s s' : S} {d : ℝ} : Reachable s → h.Step s d s' → Reachable s'

/-- A possible run: the states `ρ n`, each step `n` taking the time `δ n`. -/
structure IsRun (ρ : ℕ → S) (δ : ℕ → ℝ) : Prop where
  init : ∃ μ ∈ h.init, ρ 0 ∈ μ.support
  step : ∀ n, h.Step (ρ n) (δ n) (ρ (n + 1))

/-- The time of a run diverges: it is not Zeno. -/
def Divergent (δ : ℕ → ℝ) : Prop :=
  Tendsto (fun n => ∑ i ∈ Finset.range n, δ i) atTop atTop

theorem IsRun.reachable {h : Hybrid I S} {ρ : ℕ → S} {δ : ℕ → ℝ} (hρ : h.IsRun ρ δ) :
    ∀ n, h.Reachable (ρ n)
  | 0 => let ⟨_, hμ, hs⟩ := hρ.init; .init hμ hs
  | n + 1 => .step (hρ.reachable n) (hρ.step n)

end Hybrid

/-! ## Valuations -/

variable {V : Type*}
  {E : V → Type*} [∀ v, NormedAddCommGroup (E v)] [∀ v, NormedSpace ℝ (E v)]
  {H : V → Type*} [∀ v, TopologicalSpace (H v)]
  {I : ∀ v, ModelWithCorners ℝ (E v) (H v)}
  {M : V → Type*} [∀ v, TopologicalSpace (M v)] [∀ v, ChartedSpace (H v) (M v)]

/-- The restriction of a valuation of `Y` to `X ⊆ Y`. -/
def Val.restrict {X Y : Finset V} (hXY : X ⊆ Y) (s : Val M Y) : Val M X :=
  fun v => s ⟨v, hXY v.2⟩

/-- The valuation of `Y` overriding `x` with `c` on the variables `X`. -/
def Val.override [DecidableEq V] {X Y : Finset V} (x : Val M Y) (c : Val M X) : Val M Y :=
  fun v => if hv : v.1 ∈ X then c ⟨v.1, hv⟩ else x v

omit [∀ v, TopologicalSpace (M v)] in
theorem Val.override_of_mem [DecidableEq V] {X Y : Finset V} (x : Val M Y) (c : Val M X)
    {v : Y} (hv : v.1 ∈ X) : Val.override x c v = c ⟨v.1, hv⟩ := by
  simp [Val.override, hv]

omit [∀ v, TopologicalSpace (M v)] in
theorem Val.override_of_not_mem [DecidableEq V] {X Y : Finset V} (x : Val M Y) (c : Val M X)
    {v : Y} (hv : v.1 ∉ X) : Val.override x c v = x v := by
  simp [Val.override, hv]

/-- The unique valuation of no variables. -/
def Val.empty {X : Finset V} (hX : X = ∅) : Val M X :=
  fun v => absurd v.2 (by simp [hX])

/-- The unique tangent vector to the valuations of no variables. -/
def Val.emptyTangent {X : Finset V} (hX : X = ∅) : Val.Tangent I X :=
  fun v => absurd v.2 (by simp [hX])

/-- The restriction of a tangent vector to the valuations of `Y` to `X ⊆ Y`. -/
def Val.restrictT {X Y : Finset V} (hXY : X ⊆ Y) {s : Val M Y}
    (w : TangentSpace (Val.model I Y) s) : TangentSpace (Val.model I X) (Val.restrict hXY s) :=
  fun v => w ⟨v, hXY v.2⟩

/-! ## Closed atoms -/

/-- The hybrid system of a closed atom: it awaits nothing and reads only the
    variables it controls, whose current values it reads. -/
def Atom.toHybrid (a : Atom I M) (hr : a.read ⊆ a.ctrl) (hw : a.wait = ∅) :
    Hybrid (Val.model I a.ctrl) (Val M a.ctrl) where
  init := FinDist.toPMF '' a.init (Val.empty hw)
  jump s := FinDist.toPMF '' a.update (Val.restrict hr s, Val.empty hw)
  flow s := a.flow s (Val.restrict hr s, Val.emptyTangent hw)
  regions := {{s | (a.update (Val.restrict hr s, Val.empty hw)).Nonempty}}

/-! ## Closed modules -/

namespace Module

variable [DecidableEq V]

theorem subset_ctrl_of_isClosed {m : Module I M} (hc : m.IsClosed) {a : Atom I M}
    (ha : a ∈ m.atoms) : a.read ∪ a.wait ⊆ m.ctrl := by
  have := m.vars_subset ha
  rw [vars, show m.extl = ∅ from hc, Finset.empty_union] at this
  exact this

theorem read_subset {m : Module I M} (hc : m.IsClosed) {a : Atom I M} (ha : a ∈ m.atoms) :
    a.read ⊆ m.ctrl :=
  Finset.subset_union_left.trans (subset_ctrl_of_isClosed hc ha)

theorem wait_subset {m : Module I M} (hc : m.IsClosed) {a : Atom I M} (ha : a ∈ m.atoms) :
    a.wait ⊆ m.ctrl :=
  Finset.subset_union_right.trans (subset_ctrl_of_isClosed hc ha)

variable (m : Module I M)

/-! ### Rounds -/

/-- The choices of the draws of the atoms in a round: the distribution of the
    values of the controlled variables of each atom, given the valuation `x`
    drawn so far (in which the atom finds the values it awaits). -/
abbrev Draws := ∀ a, a ∈ m.atoms → Val M m.ctrl → PMF (Val M a.ctrl)

/-- A round of the atoms `l` (of `m`), starting from the valuation `x`: in turn,
    every atom draws the values of its controlled variables given the
    valuation drawn so far, which it overrides with them. -/
noncomputable def round (κ : m.Draws) :
    (l : List (Atom I M)) → (∀ a ∈ l, a ∈ m.atoms) → Val M m.ctrl → PMF (Val M m.ctrl)
  | [], _, x => PMF.pure x
  | a :: l, h, x => (κ a (h a List.mem_cons_self) x).bind fun c =>
      round κ l (fun b hb => h b (List.mem_cons_of_mem _ hb)) (Val.override x c)

variable {m}

/-- A round only changes the variables controlled by its atoms. -/
theorem round_frame {κ : m.Draws} {l : List (Atom I M)} {h : ∀ a ∈ l, a ∈ m.atoms}
    {x s' : Val M m.ctrl} (hs' : s' ∈ (m.round κ l h x).support) (v : m.ctrl)
    (hv : ∀ b ∈ l, v.1 ∉ b.ctrl) : s' v = x v := by
  induction l generalizing x with
  | nil =>
    rw [round, PMF.support_pure, Set.mem_singleton_iff] at hs'
    rw [hs']
  | cons a l ih =>
    rw [round, PMF.mem_support_bind_iff] at hs'
    obtain ⟨c, _, hs'⟩ := hs'
    rw [ih hs' fun b hb => hv b (List.mem_cons_of_mem _ hb),
      Val.override_of_not_mem _ _ (hv a List.mem_cons_self)]

/-- In a round, every atom draws the final values of its controlled variables,
    given the final values of the variables it awaits: the order is consistent
    with the await relation. -/
theorem round_draw {κ : m.Draws} {l : List (Atom I M)} {h : ∀ a ∈ l, a ∈ m.atoms}
    {x s' : Val M m.ctrl} (hs' : s' ∈ (m.round κ l h x).support)
    (hctrl : l.Pairwise fun a b => Disjoint a.ctrl b.ctrl)
    (hwait : l.Pairwise fun a b => Disjoint a.wait b.ctrl)
    {X : Atom I M → Finset V} (hX : ∀ b ∈ l, X b ⊆ m.ctrl)
    (hXctrl : ∀ b ∈ l, Disjoint b.ctrl (X b))
    (hXwait : ∀ b ∈ l, X b ⊆ b.wait) :
    ∀ b (hb : b ∈ l), ∃ x', Val.restrict (m.ctrl_subset (h b hb)) s' ∈ (κ b (h b hb) x').support ∧
      Val.restrict (hX b hb) x' = Val.restrict (hX b hb) s' := by
  induction l generalizing x with
  | nil => exact fun _ hb => absurd hb List.not_mem_nil
  | cons a l ih =>
    rw [round, PMF.mem_support_bind_iff] at hs'
    obtain ⟨c, hc, hs'⟩ := hs'
    rw [List.pairwise_cons] at hctrl hwait
    intro b hb
    rcases List.mem_cons.1 hb with rfl | hb'
    · refine ⟨x, ?_, ?_⟩
      · convert hc using 1
        funext ⟨v, hv⟩
        rw [Val.restrict, round_frame hs' ⟨v, _⟩ fun b' hb' =>
          Finset.disjoint_left.1 (hctrl.1 b' hb') hv, Val.override_of_mem _ _ hv]
      · funext ⟨v, hv⟩
        simp only [Val.restrict]
        rw [round_frame hs' ⟨v, _⟩ fun b' hb' =>
          Finset.disjoint_left.1 (hwait.1 b' hb') (hXwait b hb hv),
          Val.override_of_not_mem _ _ fun hv' =>
            Finset.disjoint_left.1 (hXctrl b hb) hv' hv]
    · exact ih hs' hctrl.2 hwait.2 (fun b hb => hX b (List.mem_cons_of_mem _ hb))
        (fun b hb => hXctrl b (List.mem_cons_of_mem _ hb))
        (fun b hb => hXwait b (List.mem_cons_of_mem _ hb)) b hb'

/-- A round of atoms drawing, deterministically, the values of `t`, draws `t`
    on their controlled variables, and keeps the rest. -/
theorem round_pure {κ : m.Draws} {t : Val M m.ctrl}
    (hκ : ∀ a ha x, κ a ha x = PMF.pure (Val.restrict (m.ctrl_subset ha) t)) :
    ∀ (l : List (Atom I M)) (h : ∀ a ∈ l, a ∈ m.atoms) (x : Val M m.ctrl),
      ∃ z, m.round κ l h x = PMF.pure z ∧ (∀ v : m.ctrl, (∃ b ∈ l, v.1 ∈ b.ctrl) → z v = t v) ∧
        (∀ v : m.ctrl, (∀ b ∈ l, v.1 ∉ b.ctrl) → z v = x v)
  | [], _, x => ⟨x, rfl, fun _ ⟨_, hb, _⟩ => absurd hb List.not_mem_nil, fun _ _ => rfl⟩
  | a :: l, h, x => by
    obtain ⟨z, hz, h₁, h₂⟩ := round_pure hκ l (fun b hb => h b (List.mem_cons_of_mem _ hb))
      (Val.override x (Val.restrict (m.ctrl_subset (h a List.mem_cons_self)) t))
    refine ⟨z, by rw [round, hκ, PMF.pure_bind, hz], fun v ⟨b, hb, hv⟩ => ?_, fun v hv => ?_⟩
    · by_cases hl : ∃ b ∈ l, v.1 ∈ b.ctrl
      · exact h₁ v hl
      · push Not at hl
        rw [h₂ v hl]
        rcases List.mem_cons.1 hb with rfl | hb'
        · exact Val.override_of_mem _ _ hv
        · exact absurd hv (hl b hb')
    · rw [h₂ v fun b hb => hv b (List.mem_cons_of_mem _ hb),
        Val.override_of_not_mem _ _ (hv a List.mem_cons_self)]

/-- A full round of atoms drawing, deterministically, the values of `t`, draws `t`. -/
theorem round_pure_atoms {κ : m.Draws} {t : Val M m.ctrl}
    (hκ : ∀ a ha x, κ a ha x = PMF.pure (Val.restrict (m.ctrl_subset ha) t))
    (x : Val M m.ctrl) : m.round κ m.atoms (fun _ h => h) x = PMF.pure t := by
  obtain ⟨z, hz, h₁, _⟩ := round_pure hκ m.atoms (fun _ h => h) x
  rw [hz]
  congr
  funext v
  exact h₁ v ((m.mem_ctrl_iff v.1).1 v.2)

/-! ### The hybrid system -/

variable (m) (hc : m.IsClosed) {a : Atom I M} (ha : a ∈ m.atoms)

/-- The update of the atom `a` is enabled in a round from `s`, given the
    valuation `x` drawn so far: it is defined on the values it reads (latched,
    from `s`) and awaits (next, from `x`). -/
def Enabled (s x : Val M m.ctrl) : Prop :=
  (a.update (Val.restrict (read_subset hc ha) s, Val.restrict (wait_subset hc ha) x)).Nonempty

/-- The possible draws of the atom `a` in a round from `s`, given the valuation
    `x` drawn so far: one of its update distributions if it is enabled, and
    otherwise keeping the values of its controlled variables. -/
def draws (s x : Val M m.ctrl) : Set (PMF (Val M a.ctrl)) :=
  {ν | (m.Enabled hc ha s x ∧ ν ∈ FinDist.toPMF ''
      a.update (Val.restrict (read_subset hc ha) s, Val.restrict (wait_subset hc ha) x)) ∨
    (¬m.Enabled hc ha s x ∧ ν = PMF.pure (Val.restrict (m.ctrl_subset ha) s))}

/-- The possible draws of the atom `a` in the initial round, given the
    valuation `x` drawn so far: one of its initial distributions. -/
def initDraws (x : Val M m.ctrl) : Set (PMF (Val M a.ctrl)) :=
  FinDist.toPMF '' a.init (Val.restrict (wait_subset hc ha) x)

/-- The jump region of the atom `a`: the states on which its update is
    defined, for some next values of the variables it awaits. -/
def Region : Set (Val M m.ctrl) := {s | ∃ x, m.Enabled hc ha s x}

/-- The hybrid system of a closed module, on the valuations of its (controlled)
    variables: the initial distributions are the initial rounds, and a jump,
    possible when the update of some atom is enabled, is a round in which the
    enabled atoms fire, while the others keep their values; the jump regions
    are those of the atoms. -/
noncomputable def toHybrid : Hybrid (Val.model I m.ctrl) (Val M m.ctrl) where
  init := {μ | ∃ (κ : m.Draws) (x : Val M m.ctrl), (∀ a ha y, κ a ha y ∈ m.initDraws hc ha y) ∧
    μ = m.round κ m.atoms (fun _ h => h) x}
  jump s := {μ | (∃ a, ∃ ha : a ∈ m.atoms, s ∈ m.Region hc ha) ∧
    ∃ κ : m.Draws, (∀ a ha y, κ a ha y ∈ m.draws hc ha s y) ∧
      μ = m.round κ m.atoms (fun _ h => h) s}
  flow s := {w | ∀ a (ha : a ∈ m.atoms),
    Val.restrictT (m.ctrl_subset ha) w ∈ a.flow (Val.restrict (m.ctrl_subset ha) s)
      (Val.restrict (read_subset hc ha) s, Val.restrictT (wait_subset hc ha) w)}
  regions := {G | ∃ a, ∃ ha : a ∈ m.atoms, G = m.Region hc ha}

variable {m hc}

theorem draws_congr {s x y : Val M m.ctrl}
    (hxy : Val.restrict (wait_subset hc ha) x = Val.restrict (wait_subset hc ha) y) :
    m.draws hc ha s x = m.draws hc ha s y := by
  simp only [draws, Enabled, hxy]

theorem initDraws_congr {x y : Val M m.ctrl}
    (hxy : Val.restrict (wait_subset hc ha) x = Val.restrict (wait_subset hc ha) y) :
    m.initDraws hc ha x = m.initDraws hc ha y := by
  simp only [initDraws, hxy]

omit ha in
/-- After a jump, every atom holds a value drawn from one of its possible draws,
    given the next values of the variables it awaits. -/
theorem draw_of_mem_jump {s s' : Val M m.ctrl} {μ : PMF (Val M m.ctrl)}
    (hμ : μ ∈ (m.toHybrid hc).jump s) (hs' : s' ∈ μ.support) :
    ∀ a (ha : a ∈ m.atoms), ∃ ν ∈ m.draws hc ha s s',
      Val.restrict (m.ctrl_subset ha) s' ∈ ν.support := by
  obtain ⟨_, κ, hκ, rfl⟩ := hμ
  intro a ha
  obtain ⟨x', hx', hw⟩ := round_draw hs' m.pairwise_disjoint_ctrl m.pairwise_await
    (fun b hb => wait_subset hc hb) (fun b _ => b.disjoint_ctrl_wait) (fun _ _ => subset_rfl) a ha
  exact ⟨_, draws_congr ha hw ▸ hκ a ha x', hx'⟩

omit ha in
/-- Initially, every atom holds a value drawn from one of its initial
    distributions, given the initial values of the variables it awaits. -/
theorem draw_of_mem_init {s' : Val M m.ctrl} {μ : PMF (Val M m.ctrl)}
    (hμ : μ ∈ (m.toHybrid hc).init) (hs' : s' ∈ μ.support) :
    ∀ a (ha : a ∈ m.atoms), ∃ ν ∈ m.initDraws hc ha s',
      Val.restrict (m.ctrl_subset ha) s' ∈ ν.support := by
  obtain ⟨κ, x, hκ, rfl⟩ := hμ
  intro a ha
  obtain ⟨x', hx', hw⟩ := round_draw hs' m.pairwise_disjoint_ctrl m.pairwise_await
    (fun b hb => wait_subset hc hb) (fun b _ => b.disjoint_ctrl_wait) (fun _ _ => subset_rfl) a ha
  exact ⟨_, initDraws_congr ha hw ▸ hκ a ha x', hx'⟩

end Module

end Zrth
