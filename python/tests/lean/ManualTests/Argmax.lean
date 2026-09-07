import Core.Mat

/-!
# ManualTests.Argmax

Pins the semantics of `Core.Mat.argmax_1d` to `torch.argmax`, the runtime
reference that the SMT encoder in `zrth/lean/smt_encode.py` also follows.
The Python-side half of this is `tests/test_lean_argmax.py`.

Two conventions matter, and `argmax_1d` used to get both wrong:

* a tie resolves to the **first** maximal index — it folded with a
  non-strict `≤`, which kept the last one;
* the search is seeded with element 0, not with a neutral value — it seeded
  `(0, default)`, entering `0` as a candidate, so every all-negative row
  reported index 0.

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
