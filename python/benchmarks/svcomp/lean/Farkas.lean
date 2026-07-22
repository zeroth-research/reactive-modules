import Matrix

open Nat
open Int

namespace Matrix
open Fin

/- ### Farkas soundness

A generated proof file checks a certificate's three side conditions by
computation: `Aᵀy = 𝟎ᵥ`, `𝟎ᵥ ≤ᵥ y`, and `b ·ᵥ y < 0`. `farkas_sound`
is the once-proved bridge from those conditions to the conclusion that
matters: the system `A x ≤ᵥ b` has no solution. It is proved here a
single time; each generated cell merely *applies* it to its own data.

The argument: if some `x` satisfied `A x ≤ᵥ b`, then weighting the
inequalities by `𝟎ᵥ ≤ᵥ y` and summing gives `y ·ᵥ (A x) ≤ y ·ᵥ b`. But
`y ·ᵥ (A x) = (Aᵀ y) ·ᵥ x = 𝟎ᵥ ·ᵥ x = 0`, so `0 ≤ b ·ᵥ y < 0`.

The vocabulary (`Vector`, `Matrix`, `·ᵥ`, `*ᵥ`, `≤ᵥ`, …) lives in
`Matrix.lean`; this file holds only the theorems. The sum/dot lemmas
below are scaffolding for `farkas_sound`; `adjoint` and `sum_swap` carry
the real content. -/

/-- Unfold one step of `sum`, with the tail in canonical lambda form. -/
theorem sum_succ {k : Nat} (x : Vector (k + 1) Int) :
    sum x = x fzero + sum (fun i => x (fsucc i)) :=
  rfl

theorem sum_congr {n : Nat} {x y : Vector n Int} (h : ∀ i, x i = y i) :
    sum x = sum y :=
  congrArg sum (funext h)

theorem sum_zero : ∀ {n : Nat}, sum (fun _ : Fin n => (0 : Int)) = 0 := by
  intro n
  induction n with
  | zero      => rfl
  | succ k ih => simp [sum_succ, ih]

theorem sum_add : ∀ {n : Nat} (x y : Vector n Int),
    sum (fun i => x i + y i) = sum x + sum y := by
  intro n
  induction n with
  | zero      => intro x y; rfl
  | succ k ih =>
      intro x y
      simp only [sum_succ]
      rw [ih (fun i => x (fsucc i)) (fun i => y (fsucc i))]
      omega

theorem sum_le_sum : ∀ {n : Nat} {x y : Vector n Int},
    (∀ i, x i ≤ y i) → sum x ≤ sum y := by
  intro n
  induction n with
  | zero      => intro x y _; exact Int.le_refl _
  | succ k ih =>
      intro x y h
      simp only [sum_succ]
      have h₀ := h fzero
      have hs := ih (x := fun i => x (fsucc i)) (y := fun i => y (fsucc i))
                    (fun i => h (fsucc i))
      omega

theorem sum_mul_left : ∀ {n : Nat} (a : Int) (x : Vector n Int),
    sum (fun i => a * x i) = a * sum x := by
  intro n
  induction n with
  | zero      => intro a x; simp [sum]
  | succ k ih =>
      intro a x
      simp only [sum_succ]
      rw [ih a (fun i => x (fsucc i)), Int.mul_add]

theorem sum_mul_right {n : Nat} (x : Vector n Int) (a : Int) :
    sum (fun i => x i * a) = sum x * a := by
  calc sum (fun i => x i * a)
      = sum (fun i => a * x i) := sum_congr fun i => Int.mul_comm ..
    _ = a * sum x := sum_mul_left ..
    _ = sum x * a := Int.mul_comm ..

theorem sum_swap : ∀ {m n : Nat} (f : Fin m → Fin n → Int),
    sum (fun i => sum (fun j => f i j)) = sum (fun j => sum (fun i => f i j)) := by
  intro m
  induction m with
  | zero =>
      intro n f
      rw [show (fun j : Fin n => sum (fun i : Fin 0 => f i j))
            = (fun _ : Fin n => (0 : Int)) from funext fun _ => rfl,
          sum_zero]
      rfl
  | succ k ih =>
      intro n f
      simp only [sum_succ]
      rw [ih (fun i j => f (fsucc i) j)]
      exact (sum_add (fun j => f fzero j)
                     (fun j => sum (fun i => f (fsucc i) j))).symm

theorem zero_dot {n : Nat} (x : Vector n Int) : 𝟎ᵥ ·ᵥ x = 0 :=
  (sum_congr fun i => Int.zero_mul (x i)).trans sum_zero

theorem dot_comm {n : Nat} (x y : Vector n Int) : x ·ᵥ y = y ·ᵥ x :=
  sum_congr fun i => Int.mul_comm (x i) (y i)

theorem dot_mono {n : Nat} {w u v : Vector n Int}
    (hw : 𝟎ᵥ ≤ᵥ w) (huv : u ≤ᵥ v) : w ·ᵥ u ≤ w ·ᵥ v :=
  sum_le_sum fun i => Int.mul_le_mul_of_nonneg_left (huv i) (hw i)

theorem adjoint {m n : Nat} (A : Matrix m n Int) (y : Vector m Int) (x : Vector n Int) :
    (A ᵀ *ᵥ y) ·ᵥ x = y ·ᵥ (A *ᵥ x) := by
  show sum (fun j => sum (fun i => y i * A i j) * x j)
     = sum (fun i => y i * sum (fun j => x j * A i j))
  calc sum (fun j => sum (fun i => y i * A i j) * x j)
      = sum (fun j => sum (fun i => y i * A i j * x j)) :=
        sum_congr fun j => (sum_mul_right (fun i => y i * A i j) (x j)).symm
    _ = sum (fun i => sum (fun j => y i * A i j * x j)) :=
        (sum_swap (fun i j => y i * A i j * x j)).symm
    _ = sum (fun i => y i * sum (fun j => x j * A i j)) :=
        sum_congr fun i => by
          rw [← sum_mul_left]
          exact sum_congr fun j => by
            rw [Int.mul_comm (x j) (A i j), ← Int.mul_assoc]

theorem farkas_sound {m n : Nat} (A : Matrix m n Int) (b : Vector m Int) (y : Vector m Int)
    (h₁ : A ᵀ *ᵥ y = 𝟎ᵥ) (h₂ : 𝟎ᵥ ≤ᵥ y) (h₃ : b ·ᵥ y < 0) :
    ∀ x : Vector n Int, ¬ (A *ᵥ x ≤ᵥ b) := by
  intro x hx
  have e₁ : y ·ᵥ (A *ᵥ x) = 0 := by rw [← adjoint, h₁]; exact zero_dot x
  have e₂ : y ·ᵥ (A *ᵥ x) ≤ y ·ᵥ b := dot_mono h₂ hx
  have e₃ : y ·ᵥ b = b ·ᵥ y := dot_comm ..
  omega

end Matrix
