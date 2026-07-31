import Coverage
import Net

/-!
The mask lower bound on `V` — one half of a cell's decrease obligation.

A mask underestimates a ReLU layer whatever pattern it names: on a coordinate the
mask keeps, `v i ≤ relu (v i)`; on one it zeroes, `0 ≤ relu (v i)`. Pushing that
through a non-negative-weight output layer gives `affine_mask_le`, for *any*
pattern `p` and with no hypothesis on it.

That is what lets a cell leave the pre-state activations unconstrained. The
successor still needs `V` evaluated exactly, via `reluᵥ_eq_mask`, so a cell
constrains the successor's signs alone:

  `mask` bound at `s`  (no signs)   +   `reluᵥ_eq_mask` at `s'` (signs)
    ⟹ `V s - V s' ≥ <affine in s>`, then Farkas on that affine form.

The pattern the mask names is a free parameter of the certificate: any choice is
sound, and the choice `p j = (0 < pre-activation j)` makes the bound an equality,
so nothing is lost where the pre-state signs happen to be known.
-/

namespace Matrix
open Fin

/-- ReLU dominates its input. -/
theorem le_relu (x : Int) : x ≤ relu x := by
  unfold relu; split <;> omega

/-- A masked coordinate is below the ReLU whichever way the pattern goes. -/
theorem mask_le_reluᵥ {n : Nat} (p : Vector n Bool) (v : Vector n Int) (i : Fin n) :
    mask p v i ≤ reluᵥ v i := by
  simp only [mask, reluᵥ]
  split
  · exact le_relu (v i)
  · exact relu_nonneg (v i)

/-- A non-negative-weight affine layer preserves the mask bound. No condition on
    `p`: this is the lower bound a cell uses at the pre-state. -/
theorem affine_mask_le {m n : Nat} (W : Matrix m n Int) (b : Vector m Int)
    (p : Vector n Bool) (v : Vector n Int) (hW : ∀ i j, 0 ≤ W i j) (i : Fin m) :
    affine W b (mask p v) i ≤ affine W b (reluᵥ v) i := by
  simp only [affine, addᵥ, mulVec_apply]
  have h : sum (fun j => mask p v j * W i j) ≤ sum (fun j => reluᵥ v j * W i j) :=
    sum_le_sum fun j => Int.mul_le_mul_of_nonneg_right (mask_le_reluᵥ p v j) (hW i j)
  omega

end Matrix
