import Zrth.Plain.Var
import Mathlib.Probability.Kernel.Basic

namespace Zrth2.Plain

/-!
# Atoms

An atom controls, reads and awaits variables, and acts through three
behaviours: a point (`init`) and two fields — the discrete step (`next`)
and the continuous evolution (`flow`).

The behaviours are purely probabilistic and deterministic: `init` and
`next` are Markov kernels, and `flow` is an ordinary vector field: one
tangent vector at the current value.
-/

open Manifold MeasureTheory ProbabilityTheory

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

  /-- The initial action: the distribution of the initial values of the
      controlled variables, given those of the awaited ones. -/
  init : Kernel (M wait) (M ctrl)
  /-- The next action: the distribution of the next values of the
      controlled variables, given the values of the read and awaited
      ones. -/
  next : Kernel (M read × M wait) (M ctrl)
  /-- The flow: the tangent vector the controlled variables move along at
      their current value `c`, given the values of the read variables and
      the tangents of the awaited ones — a vector field. The anchor `c` is
      not an input the dynamics may depend on, only where the vector
      sits. -/
  flow : M read × TangentBundle (I wait) (M wait) →
    (c : M ctrl) → TangentSpace (I ctrl) c

  /-- The initial action is a Markov kernel: probability-valued. -/
  init_markov : IsMarkovKernel init
  /-- The next action is a Markov kernel: probability-valued. -/
  next_markov : IsMarkovKernel next

namespace Atom

variable {V : Type u} [Var V]

/-- `a` awaits `b`: `a` reads, within the current round, a variable that
    `b` controls. The awaits relation orders the atoms of a module. -/
def Awaits (a b : Atom V) : Prop := ∃ i j, a.wait i = b.ctrl j

end Atom

end Zrth2.Plain
