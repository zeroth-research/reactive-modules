import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.List.OfFn
import Mathlib.Data.List.Zip
import Mathlib.Tactic.FinCases
import Core.Basic
import Core.Box

/-!
Prototype: Option 2 (reflection with a dense `List (List Int)`).

The constant matrix is emitted as a plain list-of-lists literal; a computable
`matVecAffine` contracts it against the state vector, and a single generic lemma
proves it equals `affineLinear (matrixOf A) x (colOf b)`.

Two things to check:
  1. does `matVecAffine [literal] x i 0` REDUCE cleanly to a linear expr (omega)?
  2. does the generic correspondence lemma prove?
-/

@[simp] def dotL : List (Int × Int) → Int
  | [] => 0
  | p :: ps => p.1 * p.2 + dotL ps

/-- Contract a dense `List (List Int)` matrix against a column vector. -/
def matVecAffine (m : Nat) (A : List (List Int)) (b : List Int)
    {n : Nat} (x : Mat Int n 1) : Mat Int m 1 :=
  fun i _ => dotL ((A.getD i.val []).zip (List.ofFn (fun l : Fin n => x l 0))) + b.getD i.val 0

/-- Dense matrix denoted by a `List (List Int)`. -/
def matrixOf (m n : Nat) (A : List (List Int)) : Mat Int m n :=
  fun i j => (A.getD i.val []).getD j.val 0

/-- Column vector denoted by a `List Int`. -/
def colOf (m : Nat) (b : List Int) : Mat Int m 1 :=
  fun i _ => b.getD i.val 0

-- (1) Reduction test: a concrete literal reduces to the linear form omega wants.
-- A = [[1,0,2],[0,3,0]], b = [5,0]. Row 0 = x0 + 2·x2 + 5.
example (x : Mat Int 3 1) :
    matVecAffine 2 [[1, 0, 2], [0, 3, 0]] [5, 0] x 0 0 = x 0 0 + 2 * x 2 0 + 5 := by
  simp [matVecAffine, dotL, List.ofFn_succ, List.ofFn_zero]

-- (2) Core bridge: the list contraction equals the Finset.sum contraction.
theorem dotL_zip_ofFn {n : Nat} (row : List Int) (g : Fin n → Int) (hlen : row.length = n) :
    dotL (row.zip (List.ofFn g)) = ∑ l : Fin n, row.getD l.val 0 * g l := by
  induction n generalizing row with
  | zero => simp
  | succ k ih =>
    match row with
    | [] => simp at hlen
    | a :: as =>
      rw [List.ofFn_succ, List.zip_cons_cons, dotL, Fin.sum_univ_succ,
          ih as (fun i => g i.succ) (by simpa using hlen)]
      simp

-- (3) The generic correspondence: for any well-formed dense literal, the reflected
-- contraction equals affineLinear of the matrix it denotes. Proven ONCE.
theorem matVecAffine_eq (m n : Nat) (A : List (List Int)) (b : List Int) (x : Mat Int n 1)
    (hwf : ∀ i : Fin m, (A.getD i.val []).length = n) :
    matVecAffine m A b x = affineLinear (matrixOf m n A) x (colOf m b) := by
  funext i j
  fin_cases j
  simp only [matVecAffine, affineLinear_apply, matrixOf, colOf]
  rw [dotL_zip_ofFn (A.getD i.val []) (fun l => x l 0) (hwf i)]
  rfl

-- (4) Payoff: a per-instance correspondence for a concrete matrix is now CHEAP —
-- just instantiate the generic lemma; the only side-condition (each row has the
-- right length) is `by decide`, O(m), independent of the entries. This is what
-- the direct `precontracted = affineLinear A x b` proof could not do at scale.
example (x : Mat Int 3 1) :
    matVecAffine 2 [[1, 0, 2], [0, 3, 0]] [5, 0] x
      = affineLinear (matrixOf 2 3 [[1, 0, 2], [0, 3, 0]]) x (colOf 2 [5, 0]) :=
  matVecAffine_eq 2 3 _ _ x (by decide)
