import Mathlib.Geometry.Manifold.VectorBundle.Tangent
import Mathlib.Probability.Kernel.Basic

namespace Zrth2

open Manifold MeasureTheory ProbabilityTheory

universe u v

/-- A variable is anything with decidable equality and a type — a *manifold*:
    `M v` is the manifold of the values the variable `v` ranges over,
    modelled on the vector space `E v` through the model space `H v` by the
    model with corners `I v`. A real variable is `ℝ` modelled on itself; a
    discrete one is a manifold of dimension `0`. -/
class Variable (V : Type u) where
  [deq : DecidableEq V]
  /-- The model vector space of a variable. -/
  E : V → Type v
  [normed : ∀ v, NormedAddCommGroup (E v)]
  [normedSpace : ∀ v, NormedSpace ℝ (E v)]
  /-- The model space of a variable. -/
  H : V → Type v
  [topH : ∀ v, TopologicalSpace (H v)]
  /-- The model with corners of a variable. -/
  I : (v : V) → ModelWithCorners ℝ (E v) (H v)
  /-- The manifold of the values of a variable. -/
  M : V → Type v
  [topM : ∀ v, TopologicalSpace (M v)]
  [charted : ∀ v, ChartedSpace (H v) (M v)]
  [meas : ∀ v, MeasurableSpace (M v)]

attribute [instance_reducible, instance] Variable.deq Variable.normed
  Variable.normedSpace Variable.topH Variable.topM Variable.charted
  Variable.meas

variable {V : Type u} [Variable V]

/-- The values of a list of variables: a value of the right type for each
    entry, positionally. A manifold, modelled by `Monomial.model l`. -/
abbrev M (l : List V) : Type v :=
  (i : Fin l.length) → Variable.M (l.get i)

/-- The model of the values of a list of variables: the product of the
    models of its entries. -/
noncomputable abbrev M.model (l : List V) :
    ModelWithCorners ℝ ((i : Fin l.length) → Variable.E (l.get i))
      (ModelPi fun i : Fin l.length => Variable.H (l.get i)) :=
  ModelWithCorners.pi fun i : Fin l.length => Variable.I (l.get i)

/-- The tangent bundle of the values of a list of variables: a value of
    each entry together with a rate of change over it. -/
noncomputable abbrev T (l : List V) : Type v :=
  TangentBundle (M.model l) (M l)

/-- An atom over the variables `V`. -/
structure Atom (V : Type u) [Variable V] where
   /-- The controlled variables. -/
   ctrl : List V
   /-- The read variables. -/
   read : List V
   /-- The awaited variables. -/
   wait : List V
   /-- The initial action: the distribution of the initial values of the
         controlled variables, given those of the awaited ones. -/
   init : M wait → Measure (M ctrl)
   /-- The update action: the distribution of the next values of the
         controlled variables, given the values of the read and awaited ones. -/
   update : M read × M wait → Measure (M ctrl)
   /-- The flow: a point of the tangent bundle of the controlled values —
         the current controlled values and their rates of change — given the
         values of the read variables. -/
   flow : M read × T wait → T ctrl

   /-- The initial action is a transition kernel: a measurable family of
      measures. -/
   init_measurable : Measurable init
   /-- The update action is a transition kernel: a measurable family of
         measures. -/
   update_measurable : Measurable update

end Zrth2
