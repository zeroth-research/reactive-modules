import Zrth.Reactive.Var
import Mathlib.Probability.Kernel.Basic
import Mathlib.Analysis.Convex.Basic

namespace Zrth2

/-!
# Atoms

An atom controls, reads and awaits variables, and acts through three
behaviours: a point (`init`) and two fields — the discrete step (`next`)
and the continuous evolution (`flow`).

The behaviours are nondeterministic and probabilistic, in the imprecise
probability sense: `init` and `next` yield *credal sets* — convex sets of
probability measures — and `flow` is a *differential inclusion with convex
values* — a convex set of tangent vectors at the current value. A
non-singleton set is demonic choice, the empty set a guard. Convexity is
closure under the environment's resolution strategies: a randomising
scheduler realises exactly the convex combinations of the offered
distributions, and chattering between tangent directions realises the
convex combinations of the offered velocities (Filippov–Ważewski).
-/

open Manifold MeasureTheory NNReal ProbabilityTheory

universe u v

/-- An atom over the variables `V`. -/
structure Atom (V : Type u) [Var V] where
  /-- The arities: inferred from `ctrl`, `read` and `wait`. -/
  {n m k : ℕ}
  /-- The controlled variables. -/
  ctrl : Fin n → V
  /-- The read variables. -/
  read : Fin m → V
  /-- The awaited variables. -/
  wait : Fin k → V

  /-- Each variable is controlled once: no duplicates. -/
  nodup_ctrl : Function.Injective ctrl
  /-- An atom does not await the variables it controls. -/
  disjoint_ctrl_wait : ∀ i j, ctrl i ≠ wait j

  /-- The initial action: the credal set of the distributions of the
      initial values of the controlled variables, given those of the
      awaited ones. -/
  init : M wait → Set (Measure (M ctrl))
  /-- The next action: the credal set of the distributions of the next
      values of the controlled variables, given the values of the read and
      awaited ones. -/
  next : M read × M wait → Set (Measure (M ctrl))
  /-- The flow: the tangent vectors the controlled variables may move
      along at their current value `c`, given the values of the read
      variables and the tangents of the awaited ones — a differential
      inclusion. -/
  flow : M read × TangentBundle (I wait) (M wait) →
    (c : M ctrl) → Set (TangentSpace (I ctrl) c)


  /-- The initial action chooses among probability measures. -/
  init_prob : ∀ w, ∀ μ ∈ init w, IsProbabilityMeasure μ

  /-- The next action chooses among probability measures. -/
  next_prob : ∀ p, ∀ μ ∈ next p, IsProbabilityMeasure μ

  /-- The initial credal set is convex: closed under randomised resolution
      of the choice. -/
  init_convex : ∀ w, Convex ℝ≥0 (init w)
  /-- The next credal set is convex: closed under randomised resolution of
      the choice. -/
  next_convex : ∀ p, Convex ℝ≥0 (next p)
  /-- The flow's values are convex: closed under chattering between the
      offered directions. -/
  flow_convex : ∀ p c, Convex ℝ (flow p c)

namespace Atom

variable {V : Type u} [Var V]

/-- `a` awaits `b`: `a` reads, within the current round, a variable that
    `b` controls. The awaits relation orders the atoms of a module. -/
def Awaits (a b : Atom V) : Prop := ∃ i j, a.wait i = b.ctrl j

end Atom

end Zrth2
