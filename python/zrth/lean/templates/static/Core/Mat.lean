import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.List.OfFn
import Mathlib.Data.List.Zip
import Mathlib.Tactic.FinCases

/-!
  Mat — matrix type, operations, and lemmas.
  Core `Box` combinators that consume/produce `Mat` live in `Core.Box`.
-/

abbrev Mat (t: Type) (m n: Nat)  := Fin m → Fin n → t

/-  Matrix operations as plain functions -/

def MatAdd [HAdd t t t] (a b : Fin m → Fin n → t) : Fin m → Fin n → t :=
  fun i j => a i j + b i j

def MatMul [HMul t t t] [AddCommMonoid t](a : Mat t m k) (b : Mat t k n) : Mat t m n :=
  fun i j => Finset.sum Finset.univ (fun l => a i l * b l j)

instance [HAdd t t t]: HAdd (Mat t m n) (Mat t m n) (Mat t m n) where
  hAdd := MatAdd

instance [HMul t t t][AddCommMonoid t]: HMul (Mat t m k) (Mat t k n) (Mat t m n) where
  hMul := MatMul

def MatZero [OfNat t 0]: Mat t m n := fun _ _ => (0 : t)

def MatTranspose (A : Mat t m n) : Mat t n m := fun i j => A j i


/-- Affine map (math convention): `affineLinear A x b = A * x + b`, i.e. the
    matrix `A` acts on the column vector(s) `x` from the left. This matches the
    theory's `LIA.Linear`/`LRA.Linear` semantics `Y = A·X + B`. -/
def affineLinear [HMul t t t] [AddCommMonoid t] [HAdd t t t]
    (A : Mat t m k) (x : Mat t k n) (b : Mat t m n) : Mat t m n :=
  MatMul A x + b

/-- Element-wise ReLU: `relu x i j = max 0 (x i j)` -/
def ReLu [Max t] [OfNat t 0] (x : Mat t m n) : Mat t m n :=
  fun i j => Max.max 0 (x i j)

/-- 1-dimensional argmax: returns the column index of the maximum element
    of a `Mat t 1 n`, packed as `Mat Nat 1 1`. -/
def argmax_1d {t : Type} [LE t] [DecidableRel ((· ≤ ·) : t → t → Prop)] [Inhabited t]
    {n : Nat} (x : Mat t 1 n) : Mat Nat 1 1 :=
  fun _ _ =>
    ((List.finRange n).foldl
      (fun (best : Nat × t) j =>
        let v := x 0 j
        if best.2 ≤ v then (j.val, v) else best)
      (0, default)).1

/-- 2-dimensional argmax: returns `[i, j]`, the position of the maximum
    element of a `Mat t m n`, packed as `Mat Nat 1 2`. -/
def argmax {t : Type} [LE t] [DecidableRel ((· ≤ ·) : t → t → Prop)] [Inhabited t]
    {m n : Nat} (x : Mat t m n) : Mat Nat 1 2 :=
  let pos :=
    ((List.finRange m).foldl
      (fun (bestI : Nat × Nat × t) i =>
        (List.finRange n).foldl
          (fun (best : Nat × Nat × t) j =>
            let v := x i j
            if best.2.2 ≤ v then (i.val, j.val, v) else best)
          bestI)
      (0, 0, default))
  fun _ k => if k = 0 then pos.1 else pos.2.1


/-! Helper lemmas for simp -/
@[simp] theorem affineLinear_apply [HMul t t t] [AddCommMonoid t] [HAdd t t t]
    (A : Mat t m k) (x : Mat t k n) (b : Mat t m n) (i : Fin m) (j : Fin n) :
    affineLinear A x b i j = Finset.sum Finset.univ (fun l => A i l * x l j) + b i j := by
  simp [affineLinear]
  unfold MatMul
  exact (congrArg (((fun i j => ∑ l, A i l * x l j) + b) i) ∘ fun a => a) rfl


@[simp] theorem relu_apply [Max t] [OfNat t 0]
    (x : Mat t m n) (i : Fin m) (j : Fin n) :
    ReLu x i j = Max.max 0 (x i j) := rfl

  @[simp] theorem ReLu_eq_sup (x : Mat t m n) [Max t] [OfNat t 0] :
      ReLu x = 0 ⊔ x := by
    ext i j; simp [ReLu]


@[simp] theorem MatAdd_apply {m n : Nat} (a b : Fin m → Fin n → Int) (i : Fin m) (j : Fin n) :
    MatAdd a b i j = a i j + b i j := rfl

@[simp] theorem MatMul_apply {m k n : Nat} (a : Fin m → Fin k → Int) (b : Fin k → Fin n → Int) (i : Fin m) (j : Fin n) :
    MatMul a b i j = Finset.sum Finset.univ (fun l => a i l * b l j) := rfl

@[simp] theorem MatZero_apply {t : Type} [OfNat t 0] {m n : Nat} (i : Fin m) (j : Fin n) :
    (MatZero : Mat t m n) i j = 0 := rfl

@[simp] theorem MatAdd_eq_add {m n : Nat} (a b : Fin m → Fin n → Int) :
    MatAdd a b = a + b := rfl

@[simp] theorem MatMul_eq_mul {m n : Nat} [HMul t t t] [AddCommMonoid t]
  (a : Mat t m k) (b : Mat t k n) :
    MatMul a b = a * b := rfl

/-- Reduce `(a * b) i j` directly (needed because `MatMul_eq_mul` rewrites
    `MatMul` away before `MatMul_apply` can fire). -/
@[simp] theorem mul_Mat_apply [HMul t t t] [AddCommMonoid t]
    (a : Mat t m k) (b : Mat t k n) (i : Fin m) (j : Fin n) :
    (a * b) i j = Finset.sum Finset.univ (fun l => a i l * b l j) := rfl

@[simp] theorem add_Mat_apply [HAdd t t t]
    (a b : Mat t m n) (i : Fin m) (j : Fin n) :
    (a + b) i j = a i j + b i j := rfl

-- This complicates the proofs..
/- coercion between single-element matrix and the element -/
-- instance {t: Type} : Coe t (Mat t 1 1) where
--   coe x := fun _ _ => x

-- instance : Coe (Mat Bool 1 1) Bool where
--   coe m := m 0 0

-- instance : Coe (Mat Real 1 1) Real where
--   coe m := m 0 0

-- instance : Coe (Mat Int 1 1) Int where
--   coe m := m 0 0

/- break ValTuple...
instance {t: Type} : Coe (Mat t 1 1) t where
  coe m := m 0 0
-/

instance : Coe Bool (Mat Bool 1 1) where
  coe x := fun _  _ => x

instance : Coe (Mat Bool 1 1) Bool where
  coe m := m 0 0



/- this might be necessary to automate the proofs -/
@[simp] theorem coe_bool_mat (b : Bool) :
  Coe.coe b = ((fun _ _ => b): Mat Bool 1 1) := by rfl

@[simp] theorem coe_mat_bool (m : Mat Bool 1 1) :
  Coe.coe m = m 0 0 := by rfl

theorem Mat_1_1_eq (f : Mat t 1 1) : f = fun _ _ => f 0 0 := by
  ext i j;
  have hi: i = ⟨ 0, by simp ⟩ := by exact Fin.fin_one_eq_zero i
  have hj: j = ⟨ 0, by simp ⟩ := by exact Fin.fin_one_eq_zero j
  rw [hi, hj]
  rfl

@[simp] theorem Mat_1_1_eq_iff {t : Type} (a b : Mat t 1 1) :
    (a = b) ↔ (a 0 0 = b 0 0) := by
  constructor
  · intro h; rw [h]
  · intro h
    ext i j
    have hi : i = ⟨0, by simp⟩ := Fin.fin_one_eq_zero i
    have hj : j = ⟨0, by simp⟩ := Fin.fin_one_eq_zero j
    rw [hi, hj]; exact h

/- theorem ite_Mat_1_1 (p : Prop) [Decidable p] (a b : Mat t 1 1) : -/
/-       (if p then a else b) = fun _ _ => if p then a 0 0 else b 0 0 := by -/
/-     ext i j; fin_cases i; fin_cases j; split <;> rfl -/

  instance [LT t] : LT (Mat t 1 1) where
    lt a b := a 0 0 < b 0 0

  instance [LE t] : LE (Mat t 1 1) where
    le a b := a 0 0 ≤ b 0 0

@[simp] theorem Mat_1_1_lt_iff {t : Type} [LT t] (a b : Mat t 1 1) :
    a < b ↔ a 0 0 < b 0 0 := Iff.rfl

@[simp] theorem Mat_1_1_le_iff {t : Type} [LE t] (a b : Mat t 1 1) :
    a ≤ b ↔ a 0 0 ≤ b 0 0 := Iff.rfl

  instance [LT t] [DecidableRel (· < · : t → t → Prop)] :
      DecidableRel (· < · : Mat t 1 1 → Mat t 1 1 → Prop) :=
    fun a b => inferInstanceAs (Decidable (a 0 0 < b 0 0))

  instance [LE t] [DecidableRel (· ≤ · : t → t → Prop)] :
      DecidableRel (· ≤ · : Mat t 1 1 → Mat t 1 1 → Prop) :=
    fun a b => inferInstanceAs (Decidable (a 0 0 ≤ b 0 0))

/-! ### Scalar collapse lemmas for zeroth_hammer -/

/-- Collapse `(fun _ _ => v) i j` to `v`. Fires after `ite_fun_apply` peels
    ite through Mat 1 1 constructors. -/
theorem fun_const_1_1_apply {t : Type} (v : t) (i : Fin 1) (j : Fin 1) :
    (fun (_ : Fin 1) (_ : Fin 1) => v) i j = v := rfl

/-- Reduce `a ≠ b` for `Mat t 1 1` to pointwise inequality. -/
theorem Mat_1_1_ne_iff {t : Type} (a b : Mat t 1 1) :
    (a ≠ b) ↔ (a 0 0 ≠ b 0 0) := by
  simp [Mat_1_1_eq_iff]


/-! ### Reflected affine map over list literals

A constant `Linear` op is emitted as a `matVecAffine` over plain `List` literals
(cheap to elaborate; reduces to a linear expression under `simp` without a dense
matrix `match` or a symbolic `Finset.sum`). `matVecAffine_eq` proves — once,
generically — that it agrees with `affineLinear` of the matrix the lists denote,
so the codegen's contraction is machine-checked rather than trusted. -/

@[simp] def dotL [Mul t] [Add t] [Zero t] : List (t × t) → t
  | [] => 0
  | p :: ps => p.1 * p.2 + dotL ps

/-- Contract a dense `List (List t)` matrix and `List t` (column) bias against a
    `Mat t n batch`: `Y i j = Σ_l A[i][l] · x[l][j] + b[i]` (bias broadcast over
    the `batch` columns), matching the theory's `Y = A·X + B`. -/
def matVecAffine [Mul t] [Add t] [Zero t] (m : Nat) (A : List (List t)) (b : List t)
    {n batch : Nat} (x : Mat t n batch) : Mat t m batch :=
  fun i j => dotL ((A.getD i.val []).zip (List.ofFn (fun l : Fin n => x l j))) + b.getD i.val 0

/-- Dense matrix denoted by a `List (List t)` (out-of-range entries are `0`). -/
def matrixOf [Zero t] (m n : Nat) (A : List (List t)) : Mat t m n :=
  fun i j => (A.getD i.val []).getD j.val 0

/-- Column bias `List t` broadcast to a `Mat t m batch` (every column is `b`). -/
def colOf [Zero t] (m batch : Nat) (b : List t) : Mat t m batch :=
  fun i _ => b.getD i.val 0

/-- Core bridge: the list fold equals the `Finset.sum` contraction. -/
theorem dotL_zip_ofFn [Mul t] [AddCommMonoid t] {n : Nat}
    (row : List t) (g : Fin n → t) (hlen : row.length = n) :
    dotL (row.zip (List.ofFn g)) = ∑ l : Fin n, row.getD l.val 0 * g l := by
  induction n generalizing row with
  | zero => simp [dotL]
  | succ k ih =>
    match row with
    | [] => simp at hlen
    | a :: as =>
      rw [List.ofFn_succ, List.zip_cons_cons, dotL, Fin.sum_univ_succ,
          ih as (fun i => g i.succ) (by simpa using hlen)]
      simp

/-- Generic correspondence: the reflected form equals `affineLinear` of the
    matrix/bias the literals denote, given each row has the expected length.
    Holds for any batch width. -/
theorem matVecAffine_eq [Mul t] [AddCommMonoid t] (m n batch : Nat)
    (A : List (List t)) (b : List t) (x : Mat t n batch)
    (hwf : ∀ i : Fin m, (A.getD i.val []).length = n) :
    matVecAffine m A b x = affineLinear (matrixOf m n A) x (colOf m batch b) := by
  funext i j
  simp only [matVecAffine, affineLinear_apply, matrixOf, colOf]
  rw [dotL_zip_ofFn (A.getD i.val []) (fun l => x l j) (hwf i)]
