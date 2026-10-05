import Mathlib.Geometry.Manifold.VectorBundle.Tangent
import Mathlib.MeasureTheory.MeasurableSpace.Defs

namespace Zrth2

/-!
# Variables and their values

A variable is anything with decidable equality and a type — a manifold. The
values of a list of variables form a manifold (`M`), with a model (`M.model`)
and a tangent bundle (`T`).
-/


open Manifold

universe u v

/-- A variable is anything with decidable equality and a type — a *manifold*:
    `M v` is the manifold of the values the variable `v` ranges over,
    modelled on the vector space `E v` through the model space `H v` by the
    model with corners `I v`. A real variable is `ℝ` modelled on itself; a
    discrete one is a manifold of dimension `0`. -/
class Var (V : Type u) where
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

attribute [instance_reducible, instance] Var.deq Var.normed
  Var.normedSpace Var.topH Var.topM Var.charted
  Var.meas

variable {V : Type u} [Var V]

/-- The values of a list of variables: a value of the right type for each
    entry, positionally. A manifold, modelled by `M.model l`. -/
abbrev M (l : List V) : Type v :=
  (i : Fin l.length) → Var.M (l.get i)

/-- The model of the values of a list of variables: the product of the
    models of its entries. -/
noncomputable abbrev M.model (l : List V) :
    ModelWithCorners ℝ ((i : Fin l.length) → Var.E (l.get i))
      (ModelPi fun i : Fin l.length => Var.H (l.get i)) :=
  ModelWithCorners.pi fun i : Fin l.length => Var.I (l.get i)

/-- The tangent bundle of the values of a list of variables: a value of
    each entry together with a rate of change over it. -/
noncomputable abbrev T (l : List V) : Type v :=
  TangentBundle (M.model l) (M l)

end Zrth2
