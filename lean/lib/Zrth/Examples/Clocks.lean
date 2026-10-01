import Zrth.Module

/-!
# Two clocks

Two atoms, each controlling its own clock: the clock flows at rate `1` and the
update resets it to `0`, but only when it reads `5` — the update is a partial
function defined on `5` only. Together, they form a closed module, both
directly (`clocks`) and as the parallel composition of the modules of either
clock (`composed`).
-/

namespace Zrth.Examples.Clocks

open Manifold

/-- The variables: one per clock. -/
inductive Clock
  | a
  | b
  deriving DecidableEq

/-- Every clock is modelled on `ℝ`… -/
noncomputable abbrev I (_ : Clock) := 𝓘(ℝ, ℝ)
/-- …and ranges over `ℝ`. -/
abbrev M (_ : Clock) := ℝ

/-- The atom of the clock `k`: it controls and reads `k` and awaits nothing. -/
noncomputable def clock (k : Clock) : Atom I M where
  ctrl := {k}
  wait := ∅
  read := {k}
  disjoint_ctrl_wait := Finset.disjoint_empty_right _
  -- starts at `0`
  init _ := {.pure 0}
  -- defined on `5` only: no next value otherwise
  update | (r, _) => {μ | r ⟨k, Finset.mem_singleton_self k⟩ = 5 ∧ μ = .pure 0}
  -- moves at rate `1`
  flow _ _ := {fun _ => (1 : ℝ)}

/-- The update of a clock resets it from `5` (deterministically)... -/
example (k : Clock) (w : Val M ∅) :
    (clock k).update (fun _ => 5, w) = {.pure 0} := by
  ext μ
  simp [clock]
  exact Iff.rfl

/-- ...and is undefined elsewhere. -/
example (k : Clock) (w : Val M ∅) (x : ℝ) (hx : x ≠ 5) :
    (clock k).update (fun _ => x, w) = ∅ := by
  ext μ
  simp [clock, hx]
  exact Iff.rfl

/-- The two clocks as a closed module, both observable. -/
noncomputable def clocks : Module I M where
  extl := ∅
  intf := {.a, .b}
  prvt := ∅
  atoms := [clock .a, clock .b]
  disjoint_extl_intf := Finset.disjoint_empty_left _
  disjoint_extl_prvt := Finset.disjoint_empty_left _
  disjoint_intf_prvt := Finset.disjoint_empty_right _
  pairwise_disjoint_ctrl := by simp [clock]
  mem_ctrl_iff v := by cases v <;> simp [clock]
  subset_vars := by simp [clock]
  pairwise_await := by simp [clock]

-- No environment: the module is closed.
example : clocks.IsClosed := rfl

/-- The module of the clock `k` alone, observable. -/
noncomputable def single (k : Clock) : Module I M where
  extl := ∅
  intf := {k}
  prvt := ∅
  atoms := [clock k]
  disjoint_extl_intf := Finset.disjoint_empty_left _
  disjoint_extl_prvt := Finset.disjoint_empty_left _
  disjoint_intf_prvt := Finset.disjoint_empty_right _
  pairwise_disjoint_ctrl := by simp
  mem_ctrl_iff v := by simp [clock]
  subset_vars := by simp [clock]
  pairwise_await := by simp

/-- The two clocks as the parallel composition of their single modules: the
    clocks share no variable, so the atoms of `a` can precede those of `b`. -/
noncomputable def composed : Module I M :=
  (single .a).compose (single .b) <| .append
    (by simp [Module.ctrl, single])
    (Finset.disjoint_empty_left _)
    (Finset.disjoint_empty_right _)
    (fun _ ha => List.mem_singleton.1 ha ▸ Finset.disjoint_empty_left _)

/-- The composite has the atoms and the variables of `clocks`. -/
example : composed.atoms = clocks.atoms := rfl
example : composed.intf = clocks.intf := rfl
example : composed.prvt = clocks.prvt := rfl
example : composed.extl = clocks.extl := by decide
example : composed.IsClosed := show composed.extl = ∅ by decide

end Zrth.Examples.Clocks
