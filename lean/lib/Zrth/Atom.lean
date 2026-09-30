import Mathlib.Geometry.Manifold.VectorBundle.Tangent
import Mathlib.Probability.ProbabilityMassFunction.Monad

/-!
# Atoms

The Lean counterpart of `base::Atom`, abstracted from wires and terms: an atom
is its three classes of variables and its three blocks, typed by the
valuations of those classes.

* `ctrl`: the controlled variables, which every block writes;
* `wait`: the awaited variables, read in the current round, i.e., their
  next values (in `init` and `update`) or their tangents (in `flow`);
* `read`: the read variables, read as latched from the previous round (in
  `update`) or with their current values (in `flow`).

The variables are the reactive identities (`Var`), not wires: atoms sharing
a variable are coupled through it. Every variable `v` ranges over a manifold
`M v` modelled by `I v`, so the valuations `Val M X` of a set `X` of variables
form a manifold, modelled by the product `Val.model I X`.

Writing `C`, `W` and `R` for the valuations of `ctrl`, `wait` and `read`, and
following the checks of `Atom::new_unchecked`, the blocks have the signatures

* `init   : W → 𝒫 D(C)`        (only next values of awaited variables),
* `update : R × W → 𝒫 D(C)`    (latched reads and next values of awaited variables),
* `flow   : C × R × T_W → T C` (current values of the controlled and read
  variables, and tangents of awaited variables),

where `𝒫 D(C)` are the sets of finitely supported distributions over `C`,
`T` is the tangent bundle (`theory::Tangent`) and `T_W` the tangent vectors to
`W`, `Val.Tangent I wait`.

The flows of the atoms of a module form a system of differential equations:
a flow sees its own current value, and those of the variables it reads, which
may be moving along the flows of other atoms. It may also await the tangents of
variables, but not their values: an awaited value could jump in the current
round, while along a flow values move continuously. As every tangent space of
a manifold modelled on a vector space `E` is `E` itself, awaited tangents
without their base points are the elements of `Val.Tangent I wait`. (Unlike
`Atom::new_unchecked`, which lets the delay read awaited derivatives only,
the flow reads the current values of the read variables.)

The derivative of the controlled variables is a tangent vector *at* their
current value `c`, a vector in the fiber `T_c C` over it: the output of
`flow`, as an element of `T C`, lies over `c` by construction (see
`Atom.flowBundle`).

The discrete blocks are probabilistic and nondeterministic: `init` and
`update` yield *sets* of finitely supported distributions (`FinDist`) over the
controlled variables. A set that is not a singleton is a nondeterministic
choice (e.g., the `HAVOC` of a jump atom's initialisation, the set of all
Dirac distributions), and an empty set leaves the block undefined (a guard).
The flow is nondeterministic, but not probabilistic: a differential inclusion.
-/

namespace Zrth

open Manifold

/-- A finitely supported probability distribution. -/
structure FinDist (α : Type*) where
  /-- The distribution. -/
  toPMF : PMF α
  /-- Its support is finite. -/
  finite : toPMF.support.Finite

instance {α : Type*} : CoeOut (FinDist α) (PMF α) := ⟨FinDist.toPMF⟩

/-- The Dirac distribution at `a`. -/
noncomputable def FinDist.pure {α : Type*} (a : α) : FinDist α :=
  ⟨PMF.pure a, by simp⟩

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

/-- An atom controlling `ctrl`, awaiting `wait` and reading `read`. -/
structure Atom where
  /-- The controlled variables. -/
  ctrl : Finset V
  /-- The awaited variables. -/
  wait : Finset V
  /-- The read variables. -/
  read : Finset V
  /-- An atom does not await the variables it controls. -/
  disjoint_ctrl_wait : Disjoint ctrl wait
  /-- The initial action: the distributions of the initial values of the
      controlled variables, given the initial values of the awaited ones. -/
  init : Val M wait → Set (FinDist (Val M ctrl))
  /-- The update action: the distributions of the next values of the controlled
      variables, given the latched values of the read variables and the next
      values of the awaited ones. -/
  update : Val M read × Val M wait → Set (FinDist (Val M ctrl))
  /-- The flow (the delay activity): the derivatives of the controlled variables
      at their current value `c`, given the current values of the read variables
      and the tangents of the awaited ones. -/
  flow : (c : Val M ctrl) → Val M read × Val.Tangent I wait →
    Set (TangentSpace (Val.model I ctrl) c)

namespace Atom

variable {I M}

/-- The flow as a relation into the tangent bundle, `C × R × T_W → T C`. -/
def flowBundle (a : Atom I M) (c : Val M a.ctrl) (p : Val M a.read × Val.Tangent I a.wait) :
    Set (TangentBundle (Val.model I a.ctrl) (Val M a.ctrl)) :=
  Bundle.TotalSpace.mk c '' a.flow c p

/-- Every output of the flow lies over the controlled value. -/
theorem proj_of_mem_flowBundle (a : Atom I M) {c : Val M a.ctrl}
    {p : Val M a.read × Val.Tangent I a.wait}
    {v : TangentBundle (Val.model I a.ctrl) (Val M a.ctrl)} (h : v ∈ a.flowBundle c p) :
    v.proj = c := by
  obtain ⟨_, _, rfl⟩ := h
  rfl

end Atom

end Zrth
