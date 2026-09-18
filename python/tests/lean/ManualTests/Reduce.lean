import Core.Mat

/-!
# ManualTests.Reduce

Pins `Core.Mat.matMin` and `Core.Mat.matMax`, the two reductions no
certificate in `Certs` exercises. `ManualTests.Argmax` does the same for
their two siblings, against `torch.argmax`; these two answer to
`torch.amin` / `torch.amax`, and the SMT encoder follows the same reference
(`tests/test_lean_smt_encode.py`).

Both halves are here on purpose, because they fail independently:

* **what they compute** — `decide` runs the kernel, which will evaluate any
  enumeration that is computable at all. The definitions were rewritten to
  enumerate with `List.ofFn` (KNOWN_ISSUES #26, #30) and nothing pinned the
  answers, so this half is what says the rewrite preserved them.

* **that they reduce under `simp`** — which is the defect that rewrite
  fixed, and the half `decide` cannot see: the old `List.finRange` fold was
  perfectly computable and `decide` was happy with it, while every
  certificate closer failed on it. So the `simp only` examples below pass
  exactly the lemmas the generated `simp_mat` carries, and no others. If a
  reduction stops reducing under them, these fail here rather than as
  `linarith failed` in someone's certificate.

Run via pytest:
    just py-test tests/lean/       # fast: only checks generated files exist
    just test-lean                 # slow: runs lake build
-/

private def row1 (a : Int) : Mat Int 1 1 :=
  fun _ _ => a

private def row2 (a b : Int) : Mat Int 1 2 :=
  fun _ j => if j = 0 then a else b

private def row3 (a b c : Int) : Mat Int 1 3 :=
  fun _ j => if j = 0 then a else if j = 1 then b else c

private def mat23 (a b c d e f : Int) : Mat Int 2 3 :=
  fun i j =>
    if i = 0 then (if j = 0 then a else if j = 1 then b else c)
    else (if j = 0 then d else if j = 1 then e else f)

/-- No columns, so nothing to reduce: the documented answer is `default`. -/
private def wide0 : Mat Int 1 0 :=
  fun _ j => j.elim0

/-- No rows either, which is the other way to reach the empty fold. -/
private def tall0 : Mat Int 0 2 :=
  fun i _ => i.elim0


-- ──────────────────────────────────────────────────────────────
-- What they compute
-- ──────────────────────────────────────────────────────────────

-- The extremum in each position of a row.
example : matMax (row3 1 5 3) 0 0 = 5 := by decide
example : matMax (row3 5 1 3) 0 0 = 5 := by decide
example : matMax (row3 1 3 5) 0 0 = 5 := by decide
example : matMin (row3 5 1 3) 0 0 = 1 := by decide
example : matMin (row3 1 5 3) 0 0 = 1 := by decide
example : matMin (row3 3 5 1) 0 0 = 1 := by decide

-- A single element is its own minimum and maximum.
example : matMax (row1 7) 0 0 = 7 := by decide
example : matMin (row1 (-7)) 0 0 = -7 := by decide

-- All-negative: the fold is seeded with element 0, not with a neutral
-- value, so 0 must not be a candidate. This is the bug `argmax_1d` had.
example : matMax (row2 (-5) (-3)) 0 0 = -3 := by decide
example : matMax (row3 (-5) (-3) (-9)) 0 0 = -3 := by decide
example : matMin (row2 (-5) (-3)) 0 0 = -5 := by decide

-- All-positive, for the same reason in the other direction.
example : matMin (row2 5 3) 0 0 = 3 := by decide
example : matMin (row3 9 4 7) 0 0 = 4 := by decide

-- Ties are a value, not an index, so either one will do -- but the answer
-- still has to come out.
example : matMax (row3 3 3 3) 0 0 = 3 := by decide
example : matMin (row2 3 3) 0 0 = 3 := by decide

-- Two dimensions: the fold is over every element, row-major, not over the
-- first row or the first column.
example : matMax (mat23 1 9 3 4 5 6) 0 0 = 9 := by decide
example : matMax (mat23 1 2 3 4 5 9) 0 0 = 9 := by decide
example : matMax (mat23 9 2 3 4 5 6) 0 0 = 9 := by decide
example : matMin (mat23 4 9 3 1 5 6) 0 0 = 1 := by decide
example : matMin (mat23 9 8 7 6 5 0) 0 0 = 0 := by decide
example : matMin (mat23 0 8 7 6 5 4) 0 0 = 0 := by decide
example : matMin (mat23 (-9) (-2) (-3) (-4) (-5) (-6)) 0 0 = -9 := by decide
example : matMax (mat23 (-9) (-2) (-3) (-4) (-5) (-6)) 0 0 = -2 := by decide

-- An empty matrix has no extremum and yields `default`, which for `Int`
-- is 0 -- the one case where 0 is the answer rather than a bug.
example : matMax wide0 0 0 = (default : Int) := by decide
example : matMin wide0 0 0 = (default : Int) := by decide
example : matMax tall0 0 0 = (default : Int) := by decide
example : matMin tall0 0 0 = (default : Int) := by decide

-- The maximum is the element `argmax` points at, which is what makes the
-- two consistent for a caller that asks for both.
example : matMax (row3 1 5 3) 0 0 = row3 1 5 3 0 1 := by decide
example : matMax (mat23 1 2 3 4 5 9) 0 0 = mat23 1 2 3 4 5 9 1 2 := by decide


-- ──────────────────────────────────────────────────────────────
-- That they reduce, under what `simp_mat` carries
-- ──────────────────────────────────────────────────────────────

/-- The lemmas the generated `simp_mat` has for an enumeration:
    `Certificate.lean.j2` lists the two `ofFn` unfoldings and the four
    reductions themselves. Nothing here may be added to close a goal that
    a certificate would meet with this much and no more. -/
example (a b c : Int) :
    matMax (row3 a b c) 0 0 = max (max a b) c := by
  simp [matMax, row3, List.ofFn_succ, List.ofFn_zero]

example (a b c : Int) :
    matMin (row3 a b c) 0 0 = min (min a b) c := by
  simp [matMin, row3, List.ofFn_succ, List.ofFn_zero]

example (a b : Int) :
    matMax (row2 a b) 0 0 = max a b := by
  simp [matMax, row2, List.ofFn_succ, List.ofFn_zero]

/-- Two dimensions, where the enumeration is a `flatten` of two `ofFn`s --
    the shape that stayed stuck. -/
example (a b c d e f : Int) :
    matMax (mat23 a b c d e f) 0 0 = max (max (max (max (max a b) c) d) e) f := by
  simp [matMax, mat23, List.ofFn_succ, List.ofFn_zero]

/-- Reducing to a *comparison* rather than to a normal form, which is what
    a certificate obligation actually asks: `linarith` gets nowhere while
    the fold is opaque, and needs nothing else once it is not. -/
example (a b : Int) (h : a ≤ 0) (h' : b ≤ 0) :
    matMax (row2 a b) 0 0 ≤ 0 := by
  simp only [matMax, row2, List.ofFn_succ, List.ofFn_zero, List.flatten]
  simp only [Fin.isValue, reduceIte]
  exact max_le h h'
