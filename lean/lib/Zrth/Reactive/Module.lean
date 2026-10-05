import Zrth.Reactive.Atom

namespace Zrth2

/-!
# Modules

A module is a collection of atoms behind a visibility interface: the data
is the atoms and the *observable* variables; everything else is inferred —
controlled is what some atom controls, and

```text
    *====================*
    | extl | intf | prvt |
    *--------------------*
    |     obs     | prvt |
    *--------------------*
    | extl |    ctrl     |
    *====================*
```

arises by set algebra: `intf = ctrl ∩ obs`, `prvt = ctrl \ obs`,
`extl = obs \ ctrl`. The disjointness of the three classes is a theorem,
not a condition.
-/


universe u v

/-- A module over the variables `V`: its atoms, and the variables it
    exposes. -/
structure Module (V : Type u) [Var V] where
  /-- The atoms. -/
  atoms : List (Atom V)
  /-- The observable variables: visible to the environment. -/
  obs : Finset V
  /-- No variable is controlled twice: the concatenation of the atoms'
      controlled variables has no duplicates. -/
  nodup : (atoms.flatMap Atom.ctrl).Nodup
  /-- The atoms are ordered consistently with the awaits relation: no atom
      awaits a later one. -/
  ordered : atoms.Pairwise fun a b => ¬ a.Awaits b

namespace Module

variable {V : Type u} [Var V]

/-- The controlled variables: those some atom controls. -/
def ctrl (m : Module V) : Finset V := (m.atoms.flatMap Atom.ctrl).toFinset

/-- The interface variables: controlled and observable — the module's
    public outputs. -/
def intf (m : Module V) : Finset V := m.ctrl ∩ m.obs

/-- The private variables: controlled and not observable — hidden. -/
def prvt (m : Module V) : Finset V := m.ctrl \ m.obs

/-- The external variables: observable and not controlled — inputs,
    controlled by the environment. -/
def extl (m : Module V) : Finset V := m.obs \ m.ctrl

/-- All the variables of a module: those its atoms mention. -/
def vars (m : Module V) : Finset V :=
  (m.atoms.flatMap fun a => a.ctrl ++ a.read ++ a.wait).toFinset

/-- A list of modules is compatible — they can be composed: no two control
    a common variable, none sees another's private variables, and all their
    atoms together admit an order consistent with the awaits relation.

    The first two conditions are pairwise; the last is irreducibly global —
    pairwise schedulability does not rule out awaits cycles through three
    or more modules. -/
def Compatible (ms : List (Module V)) : Prop :=
  (ms.Pairwise fun m₁ m₂ => Disjoint m₁.ctrl m₂.ctrl) ∧
  (ms.Pairwise fun m₁ m₂ => Disjoint m₁.prvt m₂.vars ∧ Disjoint m₁.vars m₂.prvt) ∧
  ∃ l : List (Atom V), l.Perm (ms.flatMap Module.atoms) ∧
    l.Pairwise fun a b => ¬ a.Awaits b

end Module

end Zrth2
