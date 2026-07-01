import Mathlib.Algebra.BigOperators.Fin
import Core.Basic
import Core.Box
import ZerothHammer

/-!
# Playground

A scratchpad for experimenting with proof tactics used in generated certificates.

This file is also a regression test: every `example` / `theorem` here must compile
without `sorry`.  Add new experiments freely; remove or fix anything that breaks.

## Context

Generated certificate files (`Certs/*.lean`) define module-specific macros:
- `simp_mat`  — unfolds RM/init/update/inv/ranking + reduces Mat arithmetic
- `simp_defs` — unfolds definitions everywhere (`at *`), cheaply
- `mat_collapse` — collapses `Mat 1 1` comparisons to scalar form

`zeroth_hammer` is defined in `ZerothHammer.lean` and calls these macros by name.
**Important**: `zeroth_hammer` resolves `simp_mat`/`simp_defs`/`mat_collapse` against
the macro environment of `ZerothHammer.lean` (compile-time), NOT the caller's
environment.  Certificate proofs therefore call the cert macros *directly* rather
than going through `zeroth_hammer`.
-/

open Lean Elab Tactic

-- ──────────────────────────────────────────────────────────────────────────────
-- Countdown module
-- ──────────────────────────────────────────────────────────────────────────────

@[simp] def cd_init (_ : Unit) : Mat Int 1 1 := fun _ _ => 100

@[simp] def cd_update (ctrl : Mat Int 1 1) (_ _ : Unit) : Mat Int 1 1 :=
  if decide (ctrl 0 0 = 0) then fun _ _ => 100 else fun _ _ => ctrl 0 0 - 1

def cd_inv    : Mat Int 1 1 → Prop := fun s => s 0 0 ≥ 0 ∧ s 0 0 ≤ 100
def cd_P      : Mat Int 1 1 → Prop := fun s => s 0 0 = 0
def cd_rank   : Mat Int 1 1 → Nat  := fun s => (if s 0 0 = 0 then (0 : Int) else s 0 0).toNat

macro "cd_simp_mat" : tactic =>
  `(tactic| simp [cd_init, cd_update, cd_inv, cd_P, cd_rank,
      Bool.or_eq_true, decide_eq_true_eq, ite_fun_apply, Pi.sub_apply,
      Fin.isValue, Mat_1_1_le_iff, Mat_1_1_eq_iff])

macro "cd_simp_defs" : tactic =>
  `(tactic| (simp only [cd_init, cd_update, cd_inv, cd_P, cd_rank] at *; try dsimp at *))

-- inv holds after init (trivially: 100 ∈ [0,100])
theorem cd_init_inv (e : Unit × Unit) : cd_inv (cd_init e.2) := by
  cd_simp_mat

-- inv is preserved by update
theorem cd_step_inv (s : Mat Int 1 1) (e : Unit × Unit) (h : cd_inv s) :
    cd_inv (cd_update s e.1 e.2) := by
  cd_simp_defs; cd_simp_mat; split_ifs <;> omega

-- ranking decreases when ¬P
theorem cd_hrank (s : Mat Int 1 1) (e : Unit × Unit) (hi : cd_inv s) (hP : ¬cd_P s) :
    cd_rank (cd_update s e.1 e.2) < cd_rank s := by
  cd_simp_defs; cd_simp_mat; split_ifs <;> first | omega | (norm_cast; omega)

-- ──────────────────────────────────────────────────────────────────────────────
-- Collatz module (bounded: state ∈ [1,8])
-- ──────────────────────────────────────────────────────────────────────────────

@[simp] def col_update (ctrl : Mat Int 1 1) (_ _ : Unit) : Mat Int 1 1 :=
  if ctrl 0 0 = 1 then fun _ _ => 7
  else if ctrl 0 0 > 4 then fun _ _ => ctrl 0 0 - 3
  else if ctrl 0 0 > 1 then fun _ _ => ctrl 0 0 - 1
  else ctrl

def col_inv  : Mat Int 1 1 → Prop := fun s => s 0 0 ≥ 1 ∧ s 0 0 ≤ 8
def col_P    : Mat Int 1 1 → Prop := fun s => s 0 0 = 1
def col_rank : Mat Int 1 1 → Nat  := fun s => (if s 0 0 = 1 then (0 : Int) else s 0 0 - 1).toNat

macro "col_simp_mat" : tactic =>
  `(tactic| simp [col_update, col_inv, col_P, col_rank,
      Bool.or_eq_true, decide_eq_true_eq, ite_fun_apply, Pi.sub_apply, Fin.isValue])

macro "col_simp_defs" : tactic =>
  `(tactic| (simp only [col_update, col_inv, col_P, col_rank] at *; try dsimp at *))

theorem col_step_inv (s : Mat Int 1 1) (e : Unit × Unit) (h : col_inv s) :
    col_inv (col_update s e.1 e.2) := by
  col_simp_defs; col_simp_mat; split_ifs <;> omega

theorem col_hrank (s : Mat Int 1 1) (e : Unit × Unit) (hi : col_inv s) (hP : ¬col_P s) :
    col_rank (col_update s e.1 e.2) < col_rank s := by
  col_simp_defs; col_simp_mat; split_ifs <;> first | omega | (norm_cast; omega)

-- ──────────────────────────────────────────────────────────────────────────────
-- TwoVars module: two Mat Int 1 1 components as product type
-- ──────────────────────────────────────────────────────────────────────────────

@[simp] def tv_update (ctrl : Mat Int 1 1 × Mat Int 1 1) (_ _ : Unit) :
    Mat Int 1 1 × Mat Int 1 1 :=
  if ctrl.1 0 0 < ctrl.2 0 0
  then (fun _ _ => ctrl.1 0 0 + 1, ctrl.2)
  else (fun _ _ => 0, fun _ _ => 10)

def tv_inv  : Mat Int 1 1 × Mat Int 1 1 → Prop :=
  fun s => s.1 0 0 ≥ 0 ∧ s.1 0 0 ≤ s.2 0 0 ∧ s.2 0 0 = 10
def tv_P    : Mat Int 1 1 × Mat Int 1 1 → Prop := fun s => s.1 0 0 = s.2 0 0
def tv_rank : Mat Int 1 1 × Mat Int 1 1 → Nat  :=
  fun s => (if s.1 0 0 = s.2 0 0 then (0 : Int) else s.2 0 0 - s.1 0 0).toNat

macro "tv_simp_mat" : tactic =>
  `(tactic| simp [tv_update, tv_inv, tv_P, tv_rank,
      Bool.or_eq_true, decide_eq_true_eq, ite_fun_apply, Pi.add_apply, Fin.isValue])

macro "tv_simp_defs" : tactic =>
  `(tactic| (simp only [tv_update, tv_inv, tv_P, tv_rank] at *; try dsimp at *))

theorem tv_step_inv (s : Mat Int 1 1 × Mat Int 1 1) (e : Unit × Unit) (h : tv_inv s) :
    tv_inv (tv_update s e.1 e.2) := by
  tv_simp_defs; tv_simp_mat; split_ifs <;> omega

theorem tv_hrank (s : Mat Int 1 1 × Mat Int 1 1) (e : Unit × Unit)
    (hi : tv_inv s) (hP : ¬tv_P s) :
    tv_rank (tv_update s e.1 e.2) < tv_rank s := by
  tv_simp_defs; tv_simp_mat; split_ifs <;> first | omega | (norm_cast; omega)

-- ──────────────────────────────────────────────────────────────────────────────
-- affineLinear convention: `A · x + b` (matrix acts on the column from the LEFT)
-- ──────────────────────────────────────────────────────────────────────────────
-- Regression guard for the theory-aligned convention (Y = A·X + B). The state
-- `acx` is a 2×1 column, so this only type-checks under left-multiplication —
-- the old ML form `x · A` would be ill-typed here — and the pinned value
-- distinguishes `A · x` (= 13) from any transposed variant.

def acA : Mat Int 2 2 := fun i j =>
  if i = 0 then (if j = 0 then 1 else 2) else (if j = 0 then 0 else 1)
def acx : Mat Int 2 1 := fun i _ => if i = 0 then 3 else 5

theorem affineLinear_is_left_mul :
    affineLinear acA acx MatZero = (fun i _ => if i = 0 then (13 : Int) else 5) := by
  funext i j
  fin_cases i <;> fin_cases j <;>
    simp [affineLinear, MatMul, MatZero, acA, acx, Fin.sum_univ_two] <;> decide
