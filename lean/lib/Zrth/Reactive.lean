--import Init.Data.List.Basic

import Zrth.Theory

/-!
# Reactive modules

Atoms and modules whose computations are given by data-flow diagrams.
-/

namespace Zrth

universe u v n

variable (S: Type u) [MultiSort S]

abbrev Name := Nat

/-- A variable (for now just an identifier). -/
structure Var where
  name: Name
  sort: S
deriving DecidableEq


/-- An atom: the variables it controls, reads (latched) and waits on
(current round), and the diagrams computing its initial, update and flow
behavior. -/
structure Atom where
  read: List (Var S)
  wait: List (Var S)
  -- controlled variables
  ctrl: List (Var S)
  -- controlled flow variables
  ctrl_f   : List (Var S)

  init   : Signature S
  update : Signature S
  flow   : Signature S

  -- controlled and awaited variables must be disjoint
  ctrl_wait_disjoint: ∀ c ∈ ctrl, ∀ w ∈ wait, c ≠ w := by decide
  -- flow variables are subset of controlled variables
  ctrl_f_subst: ∀ c ∈ ctrl_f, c ∈ ctrl := by decide


  -- The signature of init, update and flow must fit the list of variables
  init_sig_dom: init.dom = wait.map (fun v => v.sort) := by rfl
  update_sig_dom: update.dom = (read ++ wait).map (fun v => v.sort) := by rfl
  -- TODO: it can await derivatives
  flow_sig_dom: flow.dom = (read ++ wait).map (fun v => v.sort) := by rfl

  init_sig_cod: init.cod = ctrl.map (fun v => v.sort) := by rfl
  update_sig_cod: update.cod = ctrl.map (fun v => v.sort) := by rfl
  -- TODO: it should write only flow variables
  flow_sig_cod: flow.cod = ctrl_f.map (fun v => v.sort) := by rfl


instance: HasSignature S (Atom S) where
  signature := fun (a: Atom S) => {
    dom := (a.read ++ a.wait).map Var.sort
    cod := a.ctrl.map Var.sort
  }

/-- A module: its external, interface and private variables, and its atoms. -/
structure Module where
  extl  : List (Var S)
  intf  : List (Var S)
  prvt  : List (Var S)

  atoms : List (Atom S)

  /-- variable sets are disjoint --/
  extl_intf_disj : ∀ v ∈ extl, v ∉ intf := by decide
  extl_prvt_disj : ∀ v ∈ extl, v ∉ prvt := by decide
  intf_prvt_disj : ∀ v ∈ intf, v ∉ prvt := by decide
  /-- variables have unique control --/
  unique_ctrl : (atoms.flatMap Atom.ctrl).Nodup := by decide


instance: HasSignature S (Module S) where
  signature := fun (m: Module S) => {
    dom := m.extl.map Var.sort
    cod := m.intf.map Var.sort
  }


end Zrth
