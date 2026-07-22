import Matrix

/-!
Integer ReLU networks over the `Matrix` substrate — the proof-side network the
V-lift reasons about.

Unlike a `Float` model, every value is an exact `Int`, and the linear algebra is
`Matrix.lean`'s own (`*ᵥ`, `sum`), so a network's forward pass and a Farkas
certificate speak the *same* vocabulary and `simp`/`omega` can reason about the
network. A layer is `s ↦ relu? (W *ᵥ s + b)`; a net is a composition of layers.
The key facts are the two ReLU-resolution lemmas: on a fixed sign of its input a
`relu` is the identity or zero, which is what collapses a network to its affine
piece on an activation cell (the V-lift).
-/

namespace Matrix
open Fin

/-- ReLU on an integer. -/
def relu (x : Int) : Int := if 0 ≤ x then x else 0

/-- On a non-negative input ReLU is the identity. -/
theorem relu_of_nonneg {x : Int} (h : 0 ≤ x) : relu x = x := by
  unfold relu; split <;> omega

/-- On a non-positive input ReLU is zero. -/
theorem relu_of_nonpos {x : Int} (h : x ≤ 0) : relu x = 0 := by
  unfold relu; split <;> omega

/-- ReLU is always non-negative (used for the bounded-below obligation). -/
theorem relu_nonneg (x : Int) : 0 ≤ relu x := by
  unfold relu; split <;> omega

/-- Elementwise ReLU on a vector. -/
def reluᵥ {n : Nat} (v : Vector n Int) : Vector n Int := fun i => relu (v i)

/-- Keep the coordinates a pattern marks active; zero the rest. -/
def mask {n : Nat} (p : Vector n Bool) (v : Vector n Int) : Vector n Int :=
  fun i => if p i then v i else 0

/-- The V-lift core: on an activation pattern `p` (active ⟹ pre-activation ≥ 0,
    inactive ⟹ ≤ 0), a ReLU layer is exactly the mask. Proven once; a network
    collapses to its affine piece on a cell by rewriting with this, after which
    the pattern is concrete and the result is linear (closed by `omega`). -/
theorem reluᵥ_eq_mask {n : Nat} (p : Vector n Bool) (v : Vector n Int)
    (hp : ∀ i, (p i = true → 0 ≤ v i) ∧ (p i = false → v i ≤ 0)) :
    reluᵥ v = mask p v := by
  funext i
  simp only [reluᵥ, mask]
  by_cases h : p i = true
  · rw [if_pos h]; exact relu_of_nonneg ((hp i).1 h)
  · rw [if_neg h]; exact relu_of_nonpos ((hp i).2 (by simp_all))

/-- Vector addition — a layer's bias. -/
def addᵥ {n : Nat} (x y : Vector n Int) : Vector n Int := fun i => x i + y i

infixl:65 " +ᵥ " => addᵥ

/-- An affine (fully-connected) layer: `s ↦ W *ᵥ s + b`. -/
def affine {m n : Nat} (W : Matrix m n Int) (b : Vector m Int) (s : Vector n Int) : Vector m Int :=
  (W *ᵥ s) +ᵥ b

/-! Bounded-below: a ranking network whose output layer has non-negative
    weights and bias is non-negative, because its input is post-ReLU (so
    non-negative) and a non-negative combination of non-negatives is
    non-negative. These are the structural facts the emitted `V_nonneg` uses;
    no activation cell is needed (this holds on all of input space). -/

/-- `reluᵥ` is non-negative in every coordinate. -/
theorem reluᵥ_nonneg {n : Nat} (v : Vector n Int) (i : Fin n) : 0 ≤ reluᵥ v i :=
  relu_nonneg (v i)

/-- The sum of a coordinatewise-non-negative vector is non-negative. -/
theorem sum_nonneg : ∀ {n : Nat} (x : Vector n Int), (∀ i, 0 ≤ x i) → 0 ≤ sum x
  | 0, _, _ => Int.le_refl 0
  | Nat.succ _, x, h => by
      show 0 ≤ x fzero + sum (x ∘ fsucc)
      have h0 := h fzero
      have hr := sum_nonneg (x ∘ fsucc) (fun i => h (fsucc i))
      omega

/-- A non-negative-weight affine layer applied to a non-negative vector is
    non-negative in every output coordinate. -/
theorem affine_nonneg {m n : Nat} (W : Matrix m n Int) (b : Vector m Int)
    (v : Vector n Int) (hW : ∀ i j, 0 ≤ W i j) (hb : ∀ i, 0 ≤ b i)
    (hv : ∀ j, 0 ≤ v j) (i : Fin m) : 0 ≤ affine W b v i := by
  simp only [affine, addᵥ]
  have hmv : (W *ᵥ v) i = sum (fun j => v j * W i j) := rfl
  rw [hmv]
  have hs := sum_nonneg (fun j => v j * W i j)
    (fun j => Int.mul_nonneg (hv j) (hW i j))
  have hbi := hb i
  omega

end Matrix
