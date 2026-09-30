import Zrth.Module

/-!
# Hybrid semantics

A hybrid system on a manifold `S` (modelled by `I`): initial states, a jump
relation, a flow, i.e., a differential inclusion assigning to every state
the tangent vectors it may move along, and *jump regions*.

Its runs alternate discrete *jumps* and continuous *flows* along
trajectories. Jumps are not urgent: a jump may happen at any state where it is
enabled, and a flow may go on through such states. But a flow cannot get out
of a jump region: once a trajectory enters one, it stays in it, so the jump
must happen before the region is left. (A jump region is thus meant to be
convex along the flow: a flow leaving it and coming back is ruled out
altogether.)

A *closed* atom, one that awaits nothing and reads only what it controls, is
such a system on the valuations of its controlled variables (`Atom.toHybrid`):
its initial action gives the initial states, its update the jumps and its
flow the flow.

So is a *closed* module, one without external variables (`Module.toHybrid`),
as a jump process:

* first, all atoms initialise, each relating its controlled variables to the
  (initial) values it awaits;
* then the atoms flow together, each along a tangent vector its flow admits,
  given the current values it reads and the tangents it awaits, *racing* against each other: an atom may
  fire anywhere in its jump region, i.e., where its update is defined, and
  must fire before getting out of it;
* the atom winning the race triggers a jump, in which the updates of *all*
  enabled atoms fire, while the other atoms stutter, keeping their values;
  then the atoms flow again.

The order of the atoms, consistent with their await relation, makes the
initialisation and a jump executable atom by atom, but the relational
semantics does not depend on it.
-/

namespace Zrth

open Manifold Set Filter

variable {E H : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [TopologicalSpace H]
  (I : ModelWithCorners ℝ E H) (S : Type*) [TopologicalSpace S] [ChartedSpace H S]

/-- A hybrid system on `S`. -/
structure Hybrid where
  /-- The initial states. -/
  init : Set S
  /-- The jumps: the states reachable by a discrete step. -/
  jump : S → Set S
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

/-- A step from `s` to `s'` taking the time `d`: a jump (taking no time) or
    a flow along a trajectory of duration `d`. -/
inductive Step : S → ℝ → S → Prop
  | jump {s s' : S} : s' ∈ h.jump s → Step s 0 s'
  | flow {γ : ℝ → S} {d : ℝ} : h.IsTrajectory γ d → Step (γ 0) d (γ d)

/-- The states reachable from an initial state. -/
inductive Reachable : S → Prop
  | init {s : S} : s ∈ h.init → Reachable s
  | step {s s' : S} {d : ℝ} : Reachable s → h.Step s d s' → Reachable s'

/-- A run: the states `ρ n`, each step `n` taking the time `δ n`. -/
structure IsRun (ρ : ℕ → S) (δ : ℕ → ℝ) : Prop where
  init : ρ 0 ∈ h.init
  step : ∀ n, h.Step (ρ n) (δ n) (ρ (n + 1))

/-- The time of a run diverges: it is not Zeno. -/
def Divergent (δ : ℕ → ℝ) : Prop :=
  Tendsto (fun n => ∑ i ∈ Finset.range n, δ i) atTop atTop

theorem IsRun.reachable {h : Hybrid I S} {ρ : ℕ → S} {δ : ℕ → ℝ} (hρ : h.IsRun ρ δ) :
    ∀ n, h.Reachable (ρ n)
  | 0 => .init hρ.init
  | n + 1 => .step (hρ.reachable n) (hρ.step n)

end Hybrid

/-! ## Closed atoms -/

variable {V : Type*}
  {E : V → Type*} [∀ v, NormedAddCommGroup (E v)] [∀ v, NormedSpace ℝ (E v)]
  {H : V → Type*} [∀ v, TopologicalSpace (H v)]
  {I : ∀ v, ModelWithCorners ℝ (E v) (H v)}
  {M : V → Type*} [∀ v, TopologicalSpace (M v)] [∀ v, ChartedSpace (H v) (M v)]

/-- The restriction of a valuation of `Y` to `X ⊆ Y`. -/
def Val.restrict {X Y : Finset V} (hXY : X ⊆ Y) (s : Val M Y) : Val M X :=
  fun v => s ⟨v, hXY v.2⟩

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

/-- The hybrid system of a closed atom: it awaits nothing and reads only the
    variables it controls, whose current values it reads. -/
def Atom.toHybrid (a : Atom I M) (hr : a.read ⊆ a.ctrl) (hw : a.wait = ∅) :
    Hybrid (Val.model I a.ctrl) (Val M a.ctrl) where
  init := a.init (Val.empty hw)
  jump s := a.update (Val.restrict hr s, Val.empty hw)
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

variable (m : Module I M) (hc : m.IsClosed) {a : Atom I M} (ha : a ∈ m.atoms)

/-- The update of the atom `a` is enabled in a jump from `s` to `s'`: it is
    defined on the values it reads (latched, from `s`) and awaits (next, from `s'`). -/
def Enabled (s s' : Val M m.ctrl) : Prop :=
  (a.update (Val.restrict (read_subset hc ha) s, Val.restrict (wait_subset hc ha) s')).Nonempty

/-- The update of the atom `a` fires in a jump from `s` to `s'`. -/
def Fires (s s' : Val M m.ctrl) : Prop :=
  Val.restrict (m.ctrl_subset ha) s' ∈
    a.update (Val.restrict (read_subset hc ha) s, Val.restrict (wait_subset hc ha) s')

/-- The atom `a` keeps the values of its controlled variables in a jump from `s` to `s'`. -/
def Keeps (s s' : Val M m.ctrl) : Prop :=
  Val.restrict (m.ctrl_subset ha) s' = Val.restrict (m.ctrl_subset ha) s

/-- The jump region of the atom `a`: the states on which its update is
    defined, for some next values of the variables it awaits. -/
def Region : Set (Val M m.ctrl) := {s | ∃ s', m.Enabled hc ha s s'}

/-- The hybrid system of a closed module, on the valuations of its (controlled)
    variables: a jump happens when the update of some atom is enabled, and
    then all the enabled updates fire, while the other atoms keep their
    values; the jump regions are those of the atoms. -/
def toHybrid : Hybrid (Val.model I m.ctrl) (Val M m.ctrl) where
  init := {s | ∀ a (ha : a ∈ m.atoms),
    Val.restrict (m.ctrl_subset ha) s ∈ a.init (Val.restrict (wait_subset hc ha) s)}
  jump s := {s' | (∃ a, ∃ ha : a ∈ m.atoms, m.Enabled hc ha s s') ∧
    ∀ a (ha : a ∈ m.atoms),
      (m.Enabled hc ha s s' → m.Fires hc ha s s') ∧ (¬m.Enabled hc ha s s' → m.Keeps ha s s')}
  flow s := {w | ∀ a (ha : a ∈ m.atoms),
    Val.restrictT (m.ctrl_subset ha) w ∈ a.flow (Val.restrict (m.ctrl_subset ha) s)
      (Val.restrict (read_subset hc ha) s, Val.restrictT (wait_subset hc ha) w)}
  regions := {G | ∃ a, ∃ ha : a ∈ m.atoms, G = m.Region hc ha}

end Module

end Zrth
