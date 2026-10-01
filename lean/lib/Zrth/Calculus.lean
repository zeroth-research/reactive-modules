import Zrth.Val
import Mathlib.Geometry.Manifold.MFDeriv.FDeriv
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.Analysis.Calculus.MeanValue

/-!
# Calculus on real valuations

Trajectories are curves on manifolds, differentiated as such (`HasMFDerivAt`).
When every variable is real, modelled on itself, the valuations are just the
vector space `X → ℝ`, and a manifold derivative is an ordinary one: these
lemmas bring trajectories back to real analysis, where the proofs of the
examples happen.

* `Val.hasFDerivAt_iff`: manifold and ordinary derivatives agree;
* `Val.hasDerivAt_apply`: along a curve with tangent `w`, a variable moves at
  its rate in `w`;
* `eq_of_hasDerivAt_zero`, `Val.eq_add_of_rate_one`: what does not move stays,
  and what moves at rate `1` advances by the time elapsed.
-/

namespace Zrth

open Manifold Set

variable {V : Type*} {X : Finset V}

/-- On real valuations, a manifold derivative is an ordinary one. -/
theorem Val.hasFDerivAt_iff {γ : ℝ → Val (fun _ : V => ℝ) X} {t : ℝ}
    {f' : ℝ →L[ℝ] ((v : X) → ℝ)} :
    HasMFDerivAt 𝓘(ℝ, ℝ) (Val.model (fun _ : V => 𝓘(ℝ, ℝ)) X) γ t f' ↔ HasFDerivAt γ f' t := by
  refine ⟨fun h => ?_, fun h => ⟨h.continuousAt, ?_⟩⟩
  · have := h.2
    simp only [writtenInExtChartAt, extChartAt, mfld_simps] at this
    exact hasFDerivWithinAt_univ.1 this
  · simp only [writtenInExtChartAt, extChartAt, mfld_simps]
    exact hasFDerivWithinAt_univ.2 h

/-- Along a curve with tangent `w`, the variable `i` moves at the rate `w i`. -/
theorem Val.hasDerivAt_apply (i : X) {γ : ℝ → Val (fun _ : V => ℝ) X} {t : ℝ}
    {w : (v : X) → ℝ}
    (h : HasMFDerivAt 𝓘(ℝ, ℝ) (Val.model (fun _ : V => 𝓘(ℝ, ℝ)) X) γ t
      ((1 : ℝ →L[ℝ] ℝ).smulRight w)) :
    HasDerivAt (fun t => γ t i) (w i) t := by
  have hp := (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : X => ℝ) i).hasFDerivAt.comp t
    (Val.hasFDerivAt_iff.1 h)
  rw [hasDerivAt_iff_hasFDerivAt]
  convert hp using 1
  · rfl
  · ext
    rw [ContinuousLinearMap.toSpanSingleton_apply]
    rfl

/-- A function with zero derivative on `[0, d]` stays constant there. -/
theorem eq_of_hasDerivAt_zero {f : ℝ → ℝ} {d : ℝ} (hd : ∀ t ∈ Icc 0 d, HasDerivAt f 0 t) :
    ∀ t ∈ Icc 0 d, f t = f 0 :=
  constant_of_has_deriv_right_zero (fun t ht => (hd t ht).continuousAt.continuousWithinAt)
    (fun t ht => (hd t (Ico_subset_Icc_self ht)).hasDerivWithinAt)

/-- A variable moving at rate `1` advances by the time elapsed. -/
theorem Val.eq_add_of_rate_one (i : X) {γ : ℝ → Val (fun _ : V => ℝ) X} {d : ℝ}
    (h : ∀ t ∈ Icc 0 d, ∃ w : (v : X) → ℝ, w i = 1 ∧
      HasMFDerivAt 𝓘(ℝ, ℝ) (Val.model (fun _ : V => 𝓘(ℝ, ℝ)) X) γ t
        ((1 : ℝ →L[ℝ] ℝ).smulRight w)) :
    ∀ t ∈ Icc 0 d, γ t i = γ 0 i + t := by
  have hc := eq_of_hasDerivAt_zero (f := fun t => γ t i - t) fun t ht => by
    obtain ⟨w, hw, hγt⟩ := h t ht
    have := (Val.hasDerivAt_apply i hγt).sub (hasDerivAt_id' t)
    rwa [hw, sub_self] at this
  intro t ht
  have := hc t ht
  simp only [sub_zero] at this
  linarith

end Zrth
