import Zrth.Reactive.Var
import Mathlib.Probability.Kernel.Basic

namespace Zrth2

/-!
# Atoms

An atom controls, reads and awaits variables, and acts through three
behaviours: a point (`init`) and two fields — the discrete step (`next`,
a transition kernel) and the continuous evolution (`flow`, into the
tangent bundle).
-/


open Manifold MeasureTheory ProbabilityTheory

universe u v

/-- An atom over the variables `V`. -/
structure Atom (V : Type u) [Var V] where
  /-- The controlled variables. -/
  ctrl : List V
  /-- The read variables. -/
  read : List V
  /-- The awaited variables. -/
  wait : List V

  /-- Each variable is controlled once: no duplicates. -/
  nodup_ctrl : ctrl.Nodup
  /-- An atom does not await the variables it controls. -/
  disjoint_ctrl_wait : ctrl.Disjoint wait

  /-- The initial action: the distribution of the initial values of the
      controlled variables, given those of the awaited ones. -/
  init : M wait → Measure (M ctrl)
  /-- The next action: the distribution of the next values of the
      controlled variables, given the values of the read and awaited ones. -/
  next : M read × M wait → Measure (M ctrl)
  /-- The flow: a point of the tangent bundle of the controlled values —
      the current controlled values and their rates of change — given the
      values of the read variables and the tangents of the awaited ones. -/
  flow : M read × T wait → T ctrl

  /-- The initial action is a transition kernel: a measurable family of
      measures. -/
  init_measurable : Measurable init
  /-- The next action is a transition kernel: a measurable family of
      measures. -/
  next_measurable : Measurable next

namespace Atom

variable {V : Type u} [Var V]

/-- `a` awaits `b`: `a` reads, within the current round, a variable that
    `b` controls. The awaits relation orders the atoms of a module. -/
def Awaits (a b : Atom V) : Prop := ∃ v ∈ a.wait, v ∈ b.ctrl

end Atom

end Zrth2
