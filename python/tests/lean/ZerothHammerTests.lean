/-!
# ZerothHammerTests

Illustrative proofs for each phase of `zeroth_hammer`.

Each `example` shows a goal shape that arises in generated certificates and the
specific tactic — or combination — that zeroth_hammer invokes at that phase.
Goals use bare `Int`/`Nat`/`Bool` because after `simp_mat` reduces matrix
arithmetic the remaining obligations are plain integer arithmetic.

## How to run

**Via pytest (recommended)** — syncs `Core/` from templates automatically:

```
just pytest tests/lean/                   # fast: only checks Core/ sync
just pytest tests/lean/ -m slow           # slow: also runs lake build
```

**Directly with lake** — first sync Core/ from templates, then build:

```
just pytest tests/lean/ && lake build ZerothHammerTests
```

The first `lake build` downloads Mathlib and may take several minutes.
Subsequent runs use the cache.

To check interactively, open this file in VS Code / Emacs with the lean4
extension — it will type-check inside the `tests/lean/` lake project.
-/
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic
import Core.Mat

-- ============================================================
-- Phase 0 — `simp_mat` alone closes trivially-True goals
--
-- Arises in `init_inv` / `step_inv` when `inv = fun _ => True`.
-- The old Phase 0 (`simp_mat; omega`) silently failed here:
-- `simp` closed the `True` goal, then `omega` threw "No goals"
-- which was caught by the outer `try`, so the whole phase was
-- skipped and the tactic fell through to `sorry`.
-- Fix: try `simp_mat` alone first (Phase 0), then `simp_mat <;> omega`
-- (Phase 1c) using `<;>` so `omega` never sees an empty goal list.
-- ============================================================

-- trivially True invariant — simp alone, no omega needed
example (s : Int) : True := by trivial

example (s₀ s₁ : Int) : True := by trivial

-- ============================================================
-- Phase 1a — `omega` alone
--
-- Arises after `simp_mat` has already unfolded everything and
-- the remaining goal is already bare linear integer arithmetic.
-- Trying omega *before* simp_mat avoids the simp overhead.
-- ============================================================

example (x : Int) (h : x ≥ 0) : x + 1 ≥ 1 := by omega

-- typical inv-preservation shape: counter bounded from below
example (x : Int) (hinv : x ≥ 0) (hstep : x' = x - 1) (hP : x > 0) :
    x' ≥ 0 := by omega

-- ============================================================
-- Phase 1b — `norm_cast <;> omega`  (NEW)
--
-- The canonical `hrank` goal when `ranking := fun s => (s 0 0).toNat`:
--   hinv : 0 ≤ s
--   hP   : ¬(s ≤ 0)   i.e. s > 0
--   goal : (s - 1).toNat < s.toNat
--
-- `omega` alone fails because the goal is at `Nat` level after
-- `toNat`; `norm_cast` pushes casts through and exposes the
-- underlying `Int` inequality, then `omega` closes it.
-- ============================================================

-- countdown: ranking = s.toNat, update = s - 1
example (s : Int) (hinv : 0 ≤ s) (hP : ¬(s ≤ 0)) :
    (s - 1).toNat < s.toNat := by
  norm_cast
  omega

-- general decrement toward target
example (s target : Int) (hinv : target ≤ s) (hP : s ≠ target) :
    (s - 1 - target).toNat < (s - target).toNat := by
  norm_cast
  omega

-- ranking via conditional: `ite (s = 0) 0 s`
example (s : Int) (hinv : 0 ≤ s) (hP : s ≠ 0) :
    (if s - 1 = 0 then (0 : Int) else s - 1).toNat <
    (if s = 0 then (0 : Int) else s).toNat := by
  norm_cast
  split_ifs <;> omega

-- ============================================================
-- Phase 1d — `simp_mat <;> linarith`  (NEW)
--
-- Fallback for ordered-ring arithmetic that `omega` cannot
-- handle (e.g. real-valued state, or nonlinear products in
-- hypotheses that `linarith` can use as witnesses).
-- In Int/Nat modules this rarely fires; it is cheap to try.
-- ============================================================

-- linarith can combine multiple linear inequalities
example (a b c : Int) (h1 : 2 * a ≤ b + 3) (h2 : b ≤ a + c) (h3 : c ≤ 1) :
    a ≤ 4 := by linarith

-- ============================================================
-- Phase 2 — `push_neg; simp_mat <;> omega`  (NEW)
--
-- Arises when `inv` or `P` contains negated comparisons, e.g.
-- `inv s = ¬(s 0 0 > 10)`.  After `intro` the hypothesis is
-- `hinv : ¬(s > 10)`.  `push_neg` rewrites it to `s ≤ 10`
-- so that `omega` can use it.
-- ============================================================

-- negated upper bound in invariant
example (x : Int) (h : ¬(x > 10)) : x ≤ 10 := by
  push_neg at h; exact h

-- property as negated equality: ¬P means s ≠ target
example (s : Int) (hP : ¬(s = 0)) : s ≠ 0 := by
  push_neg at hP; exact hP

-- compound negation across two variables
example (x y : Int) (h : ¬(x > 5 ∧ y > 5)) (hx : x > 5) : y ≤ 5 := by
  push_neg at h
  have := h hx
  omega

-- ============================================================
-- Phase 3 — `simp_mat <;> deep case-split`  (extended)
--
-- Handles modules whose `update` has multiple nested
-- `if-then-else` branches.  Now tries four levels of `split`
-- (was three) and adds `linarith` / `norm_cast; omega` at
-- each leaf so toNat rankings and ordered-ring goals are covered.
-- ============================================================

-- two-branch update; inv must hold after each branch
example (x : Int) (b : Bool) (h : x ≥ 0) :
    (if b then x + 1 else x) ≥ 0 := by
  split <;> omega

-- three-branch (two nested flags)
example (x : Int) (b₁ b₂ : Bool) (h : x ≥ 2) :
    (if b₁ then if b₂ then x - 2 else x - 1 else x) ≥ 0 := by
  split <;> split <;> omega

-- four-level nest — new fourth split level
example (x : Int) (a b c d : Bool) (h : x ≥ 4) :
    (if a then if b then if c then if d then x - 4 else x - 3
                             else x - 2
                   else x - 1
          else x) ≥ 0 := by
  split <;> split <;> split <;> split <;> omega

-- toNat ranking inside a split (norm_cast at the leaf)
example (x : Int) (b : Bool) (hx : 0 < x) :
    (if b then (x - 1).toNat else x.toNat) ≤ x.toNat := by
  split
  · norm_cast; omega   -- decreasing branch
  · omega              -- staying branch

-- ============================================================
-- Phase 4 — `simp_defs <;> case-split`  (NEW)
--
-- When `inv` / `ranking` are still opaque (not yet unfolded by
-- `simp_mat`) the `split` heuristic cannot see the branches.
-- `simp_defs` unfolds all module and certificate definitions
-- first; the subsequent split then finds the `ite` structure.
-- ============================================================

-- ranking is a piecewise function; needs unfolding before split
private def ex_ranking (s : Int) : Nat := if s > 0 then s.toNat else 0

example (s : Int) (h : s ≥ 0) :
    ex_ranking (s - 1) ≤ ex_ranking s := by
  simp only [ex_ranking]      -- corresponds to simp_defs in the hammer
  split <;> split <;> norm_cast <;> omega

-- ============================================================
-- Phase 5 — full pipeline: `split_ifs at *` + extended `first`
--
-- After simp_defs + simp_mat + mat_collapse, `split_ifs at *`
-- case-splits every residual `ite` in the *hypotheses* too.
-- The extended `first` list now includes `linarith`,
-- `norm_cast; omega`, `norm_cast; linarith`, and `positivity`.
-- ============================================================

-- linarith closes a goal left open by split_ifs + omega
example (x y : Int) (h1 : x ≤ y) (h2 : y ≤ x + 2) : y - x ≤ 2 := by
  linarith

-- norm_cast + linarith: cast-heavy inequality
example (n : Nat) (h : n > 0) : (n : Int) - 1 ≥ 0 := by
  norm_cast; omega

-- positivity: `0 ≤ a * b` when both sides are known non-negative
example (a b : Int) (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a * b := by
  positivity

-- ============================================================
-- Phase 6 — `aesop`  (NEW, before SMT)
--
-- General-purpose proof search for goals with structured `Prop`
-- reasoning that none of the arithmetic tactics see.
-- ============================================================

example (p q : Prop) (hp : p) (hpq : p → q) : q := by aesop

example (p q r : Prop) (h : p ∧ q) (hqr : q → r) : p ∧ r := by aesop

-- ============================================================
-- Mat Int n m examples
--
-- Generated certificates use `Mat Int n m` as the state type.
-- `simp_mat` reduces matrix arithmetic to bare scalar `Int`
-- arithmetic using `MatAdd_apply`, `MatMul_apply`, `Fin.sum_univ_*`
-- etc., leaving goals that the arithmetic tactics above can close.
--
-- These examples show the reduction and what the hammer sees
-- after `simp_mat` fires.
-- ============================================================

-- Phase 0 — Mat: trivially True invariant on a matrix state
-- inv = fun (_ : Mat Int 1 1) => True
example (s : Mat Int 1 1) : True := by trivial

-- Phase 1a — Mat: after simp_mat fires, goal is bare Int omega
-- inv = fun s => s 0 0 ≥ 0, step decrements only when positive
-- After simp_mat: x 0 0 - 1 ≥ 0 given x 0 0 > 0
example (x : Mat Int 1 1) (hinv : x 0 0 ≥ 0) (hP : ¬(x 0 0 ≤ 0)) :
    x 0 0 - 1 ≥ 0 := by omega

-- Phase 1b — Mat: toNat ranking on a 1×1 matrix state
-- ranking = fun s => (s 0 0).toNat
-- After simp_mat reduces update (x - 1) to a scalar expression,
-- hrank becomes: (x 0 0 - 1).toNat < (x 0 0).toNat
example (x : Mat Int 1 1) (hinv : x 0 0 ≥ 0) (hP : ¬(x 0 0 ≤ 0)) :
    (x 0 0 - 1).toNat < (x 0 0).toNat := by
  norm_cast   -- Phase 1b
  omega

-- Phase 1b — Mat 2×1: multi-component state, toNat ranking
-- ranking = fun s => (s 0 0 + s 1 0).toNat  (sum of components)
example (x : Mat Int 2 1)
    (hinv : x 0 0 ≥ 0 ∧ x 1 0 ≥ 0)
    (hP   : ¬(x 0 0 = 0 ∧ x 1 0 = 0)) :
    -- update: decrement the first positive component
    let x' : Mat Int 2 1 := fun i j =>
      if i = 0 then x 0 0 - 1 else x 1 0
    (x' 0 0 + x' 1 0).toNat < (x 0 0 + x 1 0).toNat := by
  simp only []
  norm_cast
  obtain ⟨h0, h1⟩ := hinv
  push_neg at hP
  omega

-- Phase 2 — Mat: push_neg on a negated matrix predicate
-- inv has shape ¬(s 0 0 > upper_bound)
-- After intro, hypothesis is ¬(x 0 0 > 10); push_neg rewrites it
example (x : Mat Int 1 1) (h : ¬(x 0 0 > 10)) : x 0 0 ≤ 10 := by
  push_neg at h; exact h

-- Phase 3 — Mat: conditional update with matrix state
-- update = if x 0 0 > 0 then x 0 0 - 1 else x 0 0
-- inv = fun s => s 0 0 ≥ 0
example (x : Mat Int 1 1) (hinv : x 0 0 ≥ 0) :
    (if x 0 0 > 0 then x 0 0 - 1 else x 0 0) ≥ 0 := by
  split <;> omega   -- Phase 3: split + omega at each leaf

-- Phase 3 — Mat: two-variable state, branching invariant
-- s 0 0 counts up, s 1 0 counts down; inv = 0 ≤ s 0 0 ∧ 0 ≤ s 1 0
example (x : Mat Int 2 1) (hinv : x 0 0 ≥ 0 ∧ x 1 0 ≥ 0) (b : Bool) :
    let x' := fun (i : Fin 2) (_ : Fin 1) =>
      if b then (if i = 0 then x 0 0 + 1 else x 1 0 - 1)
           else x i 0
    x' 0 0 ≥ 0 ∧ x' 1 0 ≥ 0 := by
  simp only []
  obtain ⟨h0, h1⟩ := hinv
  split <;> simp_all <;> omega

-- Phase 5 — Mat: Mat_1_1 collapse + split_ifs on matrix comparison
-- mat_collapse rewrites `a < b : Mat Int 1 1` to `a 0 0 < b 0 0`
-- then split_ifs + omega closes the ranking obligation
example (x y : Mat Int 1 1) (h : x 0 0 > y 0 0) : y 0 0 < x 0 0 := by
  -- mat_collapse step: Mat_1_1_lt_iff rewrites the iff, then omega
  rw [show x 0 0 > y 0 0 ↔ y 0 0 < x 0 0 from Iff.rfl] at h
  exact h

-- Phase 5 — Mat: toNat ranking with conditional, full pipeline
-- ranking = fun s => (if s 0 0 > 0 then s 0 0 else 0).toNat
-- After mat_collapse + split_ifs, each branch is Int arithmetic
example (x : Mat Int 1 1) (hinv : x 0 0 ≥ 0) (hP : x 0 0 ≠ 0) :
    (if x 0 0 - 1 > 0 then x 0 0 - 1 else (0 : Int)).toNat <
    (if x 0 0 > 0 then x 0 0 else (0 : Int)).toNat := by
  split_ifs <;> norm_cast <;> omega
