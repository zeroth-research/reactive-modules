import Zrth.Module.Basic

/-!
# Parallel composition

The parallel composition of two modules puts their atoms side by side: shared
observable variables *couple* (they are the same variable), so an atom of one
module may read or await what the other controls. The composite is a module
again, provided the two do not fight over a variable (`Composable`).
-/

namespace Zrth

open Manifold

variable {V : Type*} [DecidableEq V]
  {E : V → Type*} [∀ v, NormedAddCommGroup (E v)] [∀ v, NormedSpace ℝ (E v)]
  {H : V → Type*} [∀ v, TopologicalSpace (H v)]
  {I : ∀ v, ModelWithCorners ℝ (E v) (H v)}
  {M : V → Type*} [∀ v, TopologicalSpace (M v)] [∀ v, ChartedSpace (H v) (M v)]

namespace Module

/-- The side conditions of the parallel composition of `m₁` and `m₂`
    (the error conditions of `Module::compose`), together with the order of
    the atoms of the composite: an interleaving of the atoms of `m₁` and `m₂`
    consistent with the await relation. -/
structure Composable (m₁ m₂ : Module I M) where
  /-- No variable is controlled by both modules. -/
  disjoint_ctrl : Disjoint m₁.ctrl m₂.ctrl
  /-- The private variables of either module are not coupled with the other:
      the other module does not know them. -/
  disjoint_prvt_vars : Disjoint m₁.prvt m₂.vars
  disjoint_vars_prvt : Disjoint m₁.vars m₂.prvt
  /-- The atoms of the composite. -/
  atoms : List (Atom I M)
  /-- They are the atoms of both modules, reordered. -/
  perm : atoms.Perm (m₁.atoms ++ m₂.atoms)
  /-- The await relation is acyclic, witnessed by the order of `atoms`. -/
  pairwise_await : atoms.Pairwise fun a b => Disjoint a.wait b.ctrl

/-- Composable modules whose atoms can be ordered as those of `m₁` followed
    by those of `m₂`, i.e., no atom of `m₁` awaits a variable of `m₂`. -/
def Composable.append {m₁ m₂ : Module I M} (hc : Disjoint m₁.ctrl m₂.ctrl)
    (hp₁ : Disjoint m₁.prvt m₂.vars) (hp₂ : Disjoint m₁.vars m₂.prvt)
    (hw : ∀ a ∈ m₁.atoms, Disjoint a.wait m₂.ctrl) : Composable m₁ m₂ where
  disjoint_ctrl := hc
  disjoint_prvt_vars := hp₁
  disjoint_vars_prvt := hp₂
  atoms := m₁.atoms ++ m₂.atoms
  perm := .refl _
  pairwise_await := List.pairwise_append.2
    ⟨m₁.pairwise_await, m₂.pairwise_await,
      fun a ha _ hb => (hw a ha).mono_right (m₂.ctrl_subset hb)⟩

/-- The parallel composition of `m₁` and `m₂`, coupling their shared
    observable variables: a variable is external to the composite if it is
    external to a component and not controlled by the other. -/
def compose (m₁ m₂ : Module I M) (h : Composable m₁ m₂) : Module I M where
  extl := (m₁.extl ∪ m₂.extl) \ (m₁.ctrl ∪ m₂.ctrl)
  intf := m₁.intf ∪ m₂.intf
  prvt := m₁.prvt ∪ m₂.prvt
  atoms := h.atoms
  -- The side conditions make the composite a module: its classes of variables
  -- are disjoint, and its atoms control its interface and private variables.
  disjoint_extl_intf := by
    rw [Finset.disjoint_left]; intro v; simp only [ctrl, Finset.mem_sdiff, Finset.mem_union]
    tauto
  disjoint_extl_prvt := by
    rw [Finset.disjoint_left]; intro v; simp only [ctrl, Finset.mem_sdiff, Finset.mem_union]
    tauto
  disjoint_intf_prvt := by
    have h₁ := Finset.disjoint_left.1 m₁.disjoint_intf_prvt
    have h₂ := Finset.disjoint_left.1 m₂.disjoint_intf_prvt
    have h₃ := Finset.disjoint_left.1 h.disjoint_prvt_vars
    have h₄ := Finset.disjoint_left.1 h.disjoint_vars_prvt
    rw [Finset.disjoint_left]; intro v
    simp only [vars, Finset.mem_union] at h₁ h₂ h₃ h₄ ⊢
    have := @h₁ v; have := @h₂ v; have := @h₃ v; have := @h₄ v
    tauto
  pairwise_disjoint_ctrl := by
    refine (h.perm.pairwise_iff fun hab => Disjoint.symm hab).2 (List.pairwise_append.2
      ⟨m₁.pairwise_disjoint_ctrl, m₂.pairwise_disjoint_ctrl, fun a ha b hb => ?_⟩)
    exact h.disjoint_ctrl.mono (m₁.ctrl_subset ha) (m₂.ctrl_subset hb)
  mem_ctrl_iff v := by
    have h₁ := m₁.mem_ctrl_iff v
    have h₂ := m₂.mem_ctrl_iff v
    simp only [h.perm.mem_iff, List.mem_append, or_and_right, exists_or,
      Finset.mem_union] at h₁ h₂ ⊢
    rw [← h₁, ← h₂]
    tauto
  subset_vars a ha := by
    intro v hv
    rcases List.mem_append.1 (h.perm.mem_iff.1 ha) with ha | ha
    · have := m₁.vars_subset ha hv
      simp only [vars, ctrl, Finset.mem_sdiff, Finset.mem_union] at this ⊢
      tauto
    · have := m₂.vars_subset ha hv
      simp only [vars, ctrl, Finset.mem_sdiff, Finset.mem_union] at this ⊢
      tauto
  pairwise_await := h.pairwise_await

end Module

end Zrth
