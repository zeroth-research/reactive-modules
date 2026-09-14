open Nat
open Int
open Prod

namespace Matrix

-- The standard inductive definition of finite types.
inductive Fin : Nat → Type where
  | fzero : {n : Nat} → Fin (succ n)
  | fsucc : {n : Nat} → (i : Fin n) → Fin (succ n)

open Fin

def Vector (n : Nat) (X : Type) : Type :=
  Fin n → X

def Matrix (m  : Nat) (n : Nat) (X : Type) : Type :=
  Vector m (Vector n X)

def transpose {X : Type} {m n : Nat} (A : Matrix m n X) : Matrix n m X :=
  fun i j ↦ A j i

postfix:75 " ᵀ" => transpose

variable {A B C : Type}
variable {m n : Nat}

def zipWith (f : A → B → C) (x : Vector n A) (y : Vector n B) : Vector n C :=
  λ i ↦ f (x i) (y i)

def zip : Vector n A → Vector n B → Vector n (A × B) :=
  zipWith (· , ·)

-- Scalar multiplication of a vector of integers.
def scale (a : Int) (x : Vector n Int) : Vector n Int :=
  λ i ↦ a * (x i)

-- Sum of a vector of integers.
def sum {n : Nat} (x : Vector n Int) : Int :=
  match n with
  | zero   => Int.ofNat Nat.zero
  | succ _ => x fzero + sum (x ∘ fsucc)

def zeroVector {n : Nat} : Vector n Int :=
  λ _ ↦ Int.ofNat Nat.zero

notation "𝟎ᵥ" => zeroVector

def lessThanᵥ {n : Nat} (x : Vector n Int) (y : Vector n Int) : Prop :=
  ∀ (i : Fin n) , x i < y i

def lessOrEqualᵥ {n : Nat} (x : Vector n Int) (y : Vector n Int) : Prop :=
  ∀ (i : Fin n) , x i ≤ y i

infixl:65 " ≤ᵥ " => lessOrEqualᵥ

def rowSum (A : Matrix m n Int) : Vector m Int :=
  sum ∘ A

def columnSum (A : Matrix m n Int) : Vector n Int :=
  sum ∘ (transpose A)

-- Hadamard product on vectors.
def hadamard : Vector n Int → Vector n Int → Vector n Int :=
  zipWith (· * ·)

infixl:70 " ⊙ " => hadamard

def dot : Vector n Int → Vector n Int → Int :=
  λ x y ↦ sum (x ⊙ y)

infixl:70 " ·ᵥ " => dot

def matrixVectorProduct : Matrix m n Int → Vector n Int → Vector m Int :=
  λ A b ↦ columnSum (zipWith scale b (transpose A))

infixl:70 " *ᵥ " => matrixVectorProduct

end Matrix
