import Zrth.Atom

/-!
# Modules

The Lean counterpart of `base::Module`, abstracted from wires and terms: a
module is a partition of its variables by visibility together with its atoms,
kept in a linear order consistent with their await relation.

```text
    *====================*
    | extl | intf | prvt |
    *--------------------*
    |     obs     | prvt |
    *--------------------*
    | extl |    ctrl     |
    *====================*
```

The atoms of a module are coupled through the variables they share.
-/

namespace Zrth

open Manifold

variable {V : Type*} [DecidableEq V]
  {E : V → Type*} [∀ v, NormedAddCommGroup (E v)] [∀ v, NormedSpace ℝ (E v)]
  {H : V → Type*} [∀ v, TopologicalSpace (H v)]
  (I : ∀ v, ModelWithCorners ℝ (E v) (H v))
  (M : V → Type*) [∀ v, TopologicalSpace (M v)] [∀ v, ChartedSpace (H v) (M v)]

/-- A module: its variables partitioned into the external (`extl`), interface
    (`intf`) and private (`prvt`) ones, and its atoms, which control exactly
    the interface and private variables. -/
structure Module where
  /-- The external variables, controlled by the environment. -/
  extl : Finset V
  /-- The interface variables, controlled by the module and observable. -/
  intf : Finset V
  /-- The private variables, controlled by the module and hidden. -/
  prvt : Finset V
  /-- The atoms, in a linear order consistent with the await relation. -/
  atoms : List (Atom I M)
  /-- The three classes of variables are disjoint. -/
  disjoint_extl_intf : Disjoint extl intf
  disjoint_extl_prvt : Disjoint extl prvt
  disjoint_intf_prvt : Disjoint intf prvt
  /-- Every controlled variable is controlled by exactly one atom. -/
  pairwise_disjoint_ctrl : atoms.Pairwise fun a b => Disjoint a.ctrl b.ctrl
  /-- The atoms control exactly the interface and private variables. -/
  mem_ctrl_iff : ∀ v, v ∈ intf ∪ prvt ↔ ∃ a ∈ atoms, v ∈ a.ctrl
  /-- The atoms read and await only the variables of the module. -/
  subset_vars : ∀ a ∈ atoms, a.read ∪ a.wait ⊆ extl ∪ intf ∪ prvt
  /-- The order is consistent with the await relation: no atom awaits a later one. -/
  pairwise_await : atoms.Pairwise fun a b => Disjoint a.wait b.ctrl

namespace Module

variable {I M}

/-- The observable variables. -/
def obs (m : Module I M) : Finset V := m.extl ∪ m.intf

/-- The controlled variables. -/
def ctrl (m : Module I M) : Finset V := m.intf ∪ m.prvt

/-- All the variables. -/
def vars (m : Module I M) : Finset V := m.extl ∪ m.intf ∪ m.prvt

/-- A closed module has no external variables. -/
def IsClosed (m : Module I M) : Prop := m.extl = ∅

/-- An atom controls variables of its module. -/
theorem ctrl_subset (m : Module I M) {a : Atom I M} (ha : a ∈ m.atoms) :
    a.ctrl ⊆ m.ctrl := fun _ hv => (m.mem_ctrl_iff _).2 ⟨a, ha, hv⟩

/-- An atom reads and awaits variables of its module. -/
theorem vars_subset (m : Module I M) {a : Atom I M} (ha : a ∈ m.atoms) :
    a.read ∪ a.wait ⊆ m.vars := m.subset_vars a ha

/-! ## Closed modules

A module is *closed* when it has no external variables: it runs on its own,
without an environment. All its variables are then controlled by its atoms, so
whatever an atom reads or awaits is controlled by some atom of the module. -/

/-- In a closed module, an atom reads and awaits controlled variables only. -/
theorem subset_ctrl_of_isClosed {m : Module I M} (hc : m.IsClosed) {a : Atom I M}
    (ha : a ∈ m.atoms) : a.read ∪ a.wait ⊆ m.ctrl := by
  have := m.vars_subset ha
  rw [vars, show m.extl = ∅ from hc, Finset.empty_union] at this
  exact this

/-- In a closed module, an atom reads controlled variables only. -/
theorem read_subset {m : Module I M} (hc : m.IsClosed) {a : Atom I M} (ha : a ∈ m.atoms) :
    a.read ⊆ m.ctrl :=
  Finset.subset_union_left.trans (subset_ctrl_of_isClosed hc ha)

/-- In a closed module, an atom awaits controlled variables only. -/
theorem wait_subset {m : Module I M} (hc : m.IsClosed) {a : Atom I M} (ha : a ∈ m.atoms) :
    a.wait ⊆ m.ctrl :=
  Finset.subset_union_right.trans (subset_ctrl_of_isClosed hc ha)

end Module

end Zrth
