import Mathlib.Geometry.Manifold.VectorBundle.Tangent

/-!
# Valuations

The state of a reactive module is a *valuation*: a value for each of its
variables. Here a variable `v : V` is a name with a sort, the manifold `M v`
it ranges over (modelled by `I v`), so that its values can flow: a real
variable is `ℝ` modelled on itself, a discrete one is a discrete manifold of
dimension `0` (see `Zrth.Examples.Discrete`).

The valuations of a finite set `X` of variables, `Val M X`, are then the
product of the manifolds of its variables, itself a manifold, modelled by the
product `Val.model I X`: a flow of the valuation moves each variable along its
own tangent space.

Most of the plumbing of the semantics is moving valuations between sets of
variables: an atom sees only its variables (`Val.restrict`), and writes back
only those it controls (`Val.override`).
-/

namespace Zrth

open Manifold

/- The variables `V`: each variable `v` ranges over the manifold `M v`
   modelled by `I v`. -/
variable {V : Type*}
  {E : V → Type*} [∀ v, NormedAddCommGroup (E v)] [∀ v, NormedSpace ℝ (E v)]
  {H : V → Type*} [∀ v, TopologicalSpace (H v)]
  (I : ∀ v, ModelWithCorners ℝ (E v) (H v))
  (M : V → Type*) [∀ v, TopologicalSpace (M v)] [∀ v, ChartedSpace (H v) (M v)]

/-- The valuations of the variables `X`: a manifold modelled by `Val.model I X`. -/
abbrev Val (X : Finset V) : Type _ := (v : X) → M v

/-- The model of the valuations of `X`: the product of the models of its variables. -/
noncomputable abbrev Val.model (X : Finset V) :
    ModelWithCorners ℝ ((v : X) → E v) (ModelPi fun v : X => H v) :=
  ModelWithCorners.pi fun v : X => I v

/-- The tangent vectors to the valuations of `X`, at any point: the model vector
    space of `Val.model I X`, which is every tangent space `TangentSpace (Val.model I X) s`. -/
abbrev Val.Tangent (_ : ∀ v, ModelWithCorners ℝ (E v) (H v)) (X : Finset V) : Type _ :=
  (v : X) → E v


section Operations

variable {V : Type*}
  {E : V → Type*} [∀ v, NormedAddCommGroup (E v)] [∀ v, NormedSpace ℝ (E v)]
  {H : V → Type*} [∀ v, TopologicalSpace (H v)]
  {I : ∀ v, ModelWithCorners ℝ (E v) (H v)}
  {M : V → Type*} [∀ v, TopologicalSpace (M v)] [∀ v, ChartedSpace (H v) (M v)]

/-- The restriction of a valuation of `Y` to `X ⊆ Y`. -/
def Val.restrict {X Y : Finset V} (hXY : X ⊆ Y) (s : Val M Y) : Val M X :=
  fun v => s ⟨v, hXY v.2⟩

/-- The valuation of `Y` overriding `x` with `c` on the variables `X`. -/
def Val.override [DecidableEq V] {X Y : Finset V} (x : Val M Y) (c : Val M X) : Val M Y :=
  fun v => if hv : v.1 ∈ X then c ⟨v.1, hv⟩ else x v

omit [∀ v, TopologicalSpace (M v)] in
/-- Overriding writes the new values on the overridden variables… -/
theorem Val.override_of_mem [DecidableEq V] {X Y : Finset V} (x : Val M Y) (c : Val M X)
    {v : Y} (hv : v.1 ∈ X) : Val.override x c v = c ⟨v.1, hv⟩ := by
  simp [Val.override, hv]

omit [∀ v, TopologicalSpace (M v)] in
/-- …and keeps the old values elsewhere. -/
theorem Val.override_of_not_mem [DecidableEq V] {X Y : Finset V} (x : Val M Y) (c : Val M X)
    {v : Y} (hv : v.1 ∉ X) : Val.override x c v = x v := by
  simp [Val.override, hv]

/-- The unique valuation of no variables. -/
def Val.empty {X : Finset V} (hX : X = ∅) : Val M X :=
  fun v => absurd v.2 (by simp [hX])

/-- The unique tangent vector to the valuations of no variables. -/
def Val.emptyTangent {X : Finset V} (hX : X = ∅) : Val.Tangent I X :=
  fun v => absurd v.2 (by simp [hX])

/-- The restriction of a tangent vector to the valuations of `Y` to `X ⊆ Y`. -/
def Val.restrictT {X Y : Finset V} (hXY : X ⊆ Y) {s : Val M Y}
    (w : TangentSpace (Val.model I Y) s) : TangentSpace (Val.model I X) (Val.restrict hXY s) :=
  fun v => w ⟨v, hXY v.2⟩

end Operations

end Zrth
