import Mathlib.Geometry.Manifold.VectorBundle.Tangent

/-!
# Trivial tangent bundles of discrete sorts

`ℕ`, `ℤ` and `Bool` are discrete, hence 0-dimensional manifolds over the
one-point model space `Unit`. Their tangent bundles are trivial: every fiber
is the singleton `Unit`, so `T M ≃ M` via the projection and the zero section.
-/

namespace Zrth.Examples.Discrete

open Manifold Bundle

/- A discrete space is a 0-dimensional manifold: charted over the one-point
   model space `Unit`, the trivial real normed space. -/
attribute [local instance] ChartedSpace.ofDiscreteTopology

instance : IsManifold 𝓘(ℝ, Unit) ⊤ ℕ := .of_discreteTopology _
instance : IsManifold 𝓘(ℝ, Unit) ⊤ ℤ := .of_discreteTopology _
instance : IsManifold 𝓘(ℝ, Unit) ⊤ Bool := .of_discreteTopology _

abbrev Tℕ := TangentBundle 𝓘(ℝ, Unit) ℕ
abbrev Tℤ := TangentBundle 𝓘(ℝ, Unit) ℤ
abbrev TBool := TangentBundle 𝓘(ℝ, Unit) Bool

-- Every fiber is the singleton `Unit`: the only tangent vector is zero.
example (n : ℕ) : TangentSpace 𝓘(ℝ, Unit) n = Unit := rfl
example (v : Tℕ) : v = ⟨v.proj, 0⟩ := rfl

/-- `T M ≃ M` for discrete `M`, via the zero section and the projection. -/
def zeroSection : ℤ ≃ Tℤ where
  toFun z := ⟨z, ()⟩
  invFun := TotalSpace.proj
  left_inv _ := rfl
  right_inv _ := rfl

-- The bundle is smooth as a manifold too.
example : IsManifold 𝓘(ℝ, Unit).tangent ⊤ TBool := inferInstance

end Zrth.Examples.Discrete
