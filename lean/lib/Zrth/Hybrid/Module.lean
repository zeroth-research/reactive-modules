import Zrth.Hybrid.Basic
import Zrth.Hybrid.Round

/-!
# Closed modules as hybrid systems

A closed module is a hybrid system on the valuations of its variables, as a
*jump process*:

* first, the atoms initialise in an initial round, each drawing from one of
  its initial distributions;
* then the atoms flow together, each along a tangent vector its flow admits
  (given the current values it reads and the tangents it awaits), racing
  against each other: an atom may fire anywhere in its jump region, where its
  update is defined (`Module.Region`), and must fire before getting out of it;
* the atom winning the race triggers a jump: a round in which every atom whose
  update is enabled (`Module.Enabled`) draws from one of its update
  distributions, while the others stutter, keeping their values
  (`Module.draws`); then the atoms flow again.

`draw_of_mem_jump` and `draw_of_mem_init` are the handles for proofs: in any
state reached by a jump (resp. initially) with positive probability, every
atom holds a value drawn from one of its possible draws, given the final
values of what it awaits.
-/

namespace Zrth

open Manifold

variable {V : Type*} [DecidableEq V]
  {E : V → Type*} [∀ v, NormedAddCommGroup (E v)] [∀ v, NormedSpace ℝ (E v)]
  {H : V → Type*} [∀ v, TopologicalSpace (H v)]
  {I : ∀ v, ModelWithCorners ℝ (E v) (H v)}
  {M : V → Type*} [∀ v, TopologicalSpace (M v)] [∀ v, ChartedSpace (H v) (M v)]

namespace Module

variable (m : Module I M) (hc : m.IsClosed) {a : Atom I M} (ha : a ∈ m.atoms)

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
