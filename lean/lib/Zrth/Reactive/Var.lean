import Mathlib.Geometry.Manifold.VectorBundle.Tangent
import Mathlib.MeasureTheory.MeasurableSpace.Defs

namespace Zrth2

/-!
# Variables and their values

A variable is anything with decidable equality and a type — a manifold. The
values of a list of variables form a manifold (`Val`), with a model
(`Val.model`).
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

/-- The values of a tuple of variables: a value of the right type at each
    position. A manifold, modelled by `I t`. -/
abbrev M {n : ℕ} (t : Fin n → V) : Type v :=
  (i : Fin n) → Var.M (t i)

/-- The model of the values of a tuple of variables: the product of the
    models of its variables. -/
noncomputable abbrev I {n : ℕ} (t : Fin n → V) :
    ModelWithCorners ℝ ((i : Fin n) → Var.E (t i))
      (ModelPi fun i : Fin n => Var.H (t i)) :=
  ModelWithCorners.pi fun i : Fin n => Var.I (t i)

end Zrth2
