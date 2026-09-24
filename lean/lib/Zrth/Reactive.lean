--import Init.Data.List.Basic

import Zrth.Theory

/-!
# Reactive modules

Atoms and modules whose computations are given by data-flow diagrams.
-/

namespace Zrth

universe u v n

variable {S: Type u} [MultiSort S]

abbrev Name := Nat

/-- A variable (for now just an identifier). -/
structure Var where
  name: Name
  sort: S


/-- An atom: the variables it controls, reads (latched) and waits on
(current round), and the diagrams computing its initial, update and flow
behavior. -/
structure Atom where
  read: List Var
  wait: List Var
  -- controlled variables
  ctrl: List Var
  -- controlled flow variables
  ctrl_f   : List Var

  init   : Signature S
  update : Signature S
  flow   : Signature S

  -- controlled and awaited variables must be disjoint
  ctrl_wait_disjoint: ∀ c ∈ ctrl, ∀ w ∈ wait, c ≠ w
  -- flow variables are subset of controlled variables
  ctrl_f_subst: ∀ c ∈ ctrl_f, c ∈ ctrl


  -- The signature of init, update and flow must fit the list of variables
  init_sig_dom: init.dom = wait.map (fun v => v.sort)
  update_sig_dom: update.dom = (read ++ wait).map (fun v : Var => v.sort)
  -- TODO: it can await derivatives
  flow_sig_dom: flow.dom = (read ++ wait).map (fun v => v.sort)

  init_sig_cod: init.cod = ctrl.map (fun v => v.sort)
  update_sig_cod: update.cod = ctrl.map (fun v => v.sort)
  -- TODO: it should write only flow variables
  flow_sig_cod: flow.cod = ctrl_f.map (fun v => v.sort)


/-- A module: its external, interface and private variables, and its atoms. -/
structure Module where
  extl  : List (@Var S)
  intf  : List (@Var S)
  prvt  : List (@Var S)

  atoms : List (Atom (S:= S))

  -- TODO: : well-formdness predicates

end Zrth
