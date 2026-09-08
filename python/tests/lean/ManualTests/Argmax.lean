import Core.Mat

/-!
# ManualTests.Argmax

Pins the semantics of `Core.Mat.argmax_1d` and `Core.Mat.argmax` to
`torch.argmax`, the runtime reference that the SMT encoder in
`zrth/lean/smt_encode.py` also follows. The Python-side half of this is
`tests/test_lean_argmax.py`.

Two conventions matter, and both definitions used to get them wrong:

* a tie resolves to the **first** maximal index — it folded with a
  non-strict `≤`, which kept the last one;
* the search is seeded with element 0, not with a neutral value — it seeded
  `(0, default)`, entering `0` as a candidate, so every all-negative row
  reported index 0.

`argmax` additionally returns the wrong *kind* of answer: it reported the
`[i, j]` pair of the maximum, packed as `Mat Nat 1 2`, where torch flattens
row-major and returns the single index `i * n + j`.

Run via pytest:
    just py-test tests/lean/       # fast: only checks generated files exist
    just test-lean                 # slow: runs lake build
-/

private def row2 (a b : Int) : Mat Int 1 2 :=
  fun _ j => if j = 0 then a else b

private def row3 (a b c : Int) : Mat Int 1 3 :=
  fun _ j => if j = 0 then a else if j = 1 then b else c

-- Plain maxima, in each position.
example : argmax_1d (row3 1 5 3) 0 0 = 1 := by decide
example : argmax_1d (row3 5 1 3) 0 0 = 0 := by decide
example : argmax_1d (row3 1 3 5) 0 0 = 2 := by decide

-- Ties resolve to the first maximal index.
example : argmax_1d (row2 3 3) 0 0 = 0 := by decide
example : argmax_1d (row3 3 3 3) 0 0 = 0 := by decide
example : argmax_1d (row3 (-1) 0 0) 0 0 = 1 := by decide

-- All-negative rows: 0 must not be a candidate.
example : argmax_1d (row2 (-5) (-3)) 0 0 = 1 := by decide
example : argmax_1d (row3 (-5) (-3) (-9)) 0 0 = 1 := by decide
example : argmax_1d (row2 (-1) (-1)) 0 0 = 0 := by decide

-- A zero that is genuinely the maximum still wins by position.
example : argmax_1d (row2 0 (-1)) 0 0 = 0 := by decide
example : argmax_1d (row2 (-1) 0) 0 0 = 1 := by decide


-- ──────────────────────────────────────────────────────────────
-- 2-D argmax: a single row-major flat index, as torch returns.
-- ──────────────────────────────────────────────────────────────

private def mat23 (a b c d e f : Int) : Mat Int 2 3 :=
  fun i j =>
    if i = 0 then (if j = 0 then a else if j = 1 then b else c)
    else (if j = 0 then d else if j = 1 then e else f)

-- torch.argmax on [[1,9,3],[4,5,6]] is 1: flat index of the 9.
example : argmax (mat23 1 9 3 4 5 6) 0 0 = 1 := by decide
-- Maximum in the second row: flat index 1 * 3 + 2 = 5.
example : argmax (mat23 1 2 3 4 5 9) 0 0 = 5 := by decide
-- First element wins.
example : argmax (mat23 9 2 3 4 5 6) 0 0 = 0 := by decide
-- Last element wins: flat index 5.
example : argmax (mat23 1 2 3 4 5 6) 0 0 = 5 := by decide

-- Ties go to the lowest flat index, across rows as well as within one.
example : argmax (mat23 9 9 3 4 5 6) 0 0 = 0 := by decide
example : argmax (mat23 1 2 3 9 5 9) 0 0 = 3 := by decide
example : argmax (mat23 5 5 5 5 5 5) 0 0 = 0 := by decide

-- All-negative: 0 must not be a candidate here either.
example : argmax (mat23 (-9) (-2) (-3) (-4) (-5) (-6)) 0 0 = 1 := by decide
example : argmax (mat23 (-9) (-8) (-7) (-6) (-5) (-4)) 0 0 = 5 := by decide

-- On a single row the flat index is the column index, so `argmax` and
-- `argmax_1d` must agree — the property that makes the two consistent.
example : argmax (row3 1 5 3) 0 0 = argmax_1d (row3 1 5 3) 0 0 := by decide
example : argmax (row2 3 3) 0 0 = argmax_1d (row2 3 3) 0 0 := by decide
example : argmax (row3 (-5) (-3) (-9)) 0 0 = argmax_1d (row3 (-5) (-3) (-9)) 0 0 := by decide
