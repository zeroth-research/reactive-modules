# `lean-smt`: kernel rejects a reconstructed `ARITH_SUM_UB` proof

`smt` closes the goal, but the proof term it hands the kernel is ill-typed:
a `sum_ub` step produces an **equality** where the enclosing application
expects a **`≤`**.

Filed against [ufmg-smite/lean-smt](https://github.com/ufmg-smite/lean-smt).
Found by the `--fbk-proveit` sweep (`tests/lean/fbk/`), probe `NiS2Odd3`;
this file is the minimised form, and depends on nothing but `Smt`.

## Versions

| | |
|---|---|
| Lean | `leanprover/lean4:v4.28.0` |
| `smt` | `f58d19d5d0803fcccb5ccb1b4473774dd2ae9f9a` |
| `cvc5` | `ef0efbf437ae79124c65557c13aa5bfcee948f80` |
| platform | macOS arm64 (Darwin 25.2.0) |

## Reproducer

```lean
import Smt

example (x x' : Int) (h1 : 2 <= x ∨ x = 0) (htr : x' = x + 2) : ¬ (x' = 3) := by
  revert h1 htr
  smt
```

The statement is true — `x' = 3` forces `x = 1`, which contradicts `h1` —
and `rcases h1 with h | h <;> omega` proves it in the same file. Failure is
deterministic across runs.

## Error

```
error: (kernel) application type mismatch
  Smt.Reconstruct.Prop.eqResolve
    (Smt.Reconstruct.Int.sum_ub₉
      (Smt.Reconstruct.Int.sum_ub₉ a.20
        (Smt.Reconstruct.Prop.modusPonens … Smt.Reconstruct.Int.mul_neg_eq))
      (Smt.Reconstruct.Prop.modusPonens … Smt.Reconstruct.Int.mul_neg_eq))
    …
argument has type
  x + -1 * x' + -1 * (x + -1 * x') = 0 + -1 * 3 + -1 * -2
but function has type
  x + -1 * x' + -1 * (x + -1 * x') ≤ 0 + -1 * 3 + -1 * -2 →
    (x + -1 * x' + -1 * (x + -1 * x') ≤ 0 + -1 * 3 + -1 * -2) = False → False
```

## Diagnosis

The two types differ only in their relation symbol: `=` supplied, `≤`
required. `Int.sum_ubᵢ` is a family indexed by the relation kinds of its two
summands (`<`, `≤`, `=` — nine combinations), and each variant concludes in
the relation appropriate to that pair. Here the *inner* `sum_ub₉` concludes
an equality, while the term consuming it was built expecting the `≤`
variant's conclusion, so whichever side picks the index is picking it from
the wrong information. Reconstruction of cvc5's `ARITH_SUM_UB` looks like
the place to check.

## What is and is not needed to trigger it

Reduced from the original obligation, one ingredient at a time. All of the
following are **required** — removing any one makes the example compile:

| ingredient | drop it | result |
|---|---|---|
| disjunctive hypothesis | `h1 : 2 <= x` | compiles |
| disequality goal | goal `0 <= x + 2` | compiles |
| the second variable | `¬ (x + 2 = 3)`, no `x'`/`htr` | compiles |

And these do **not** matter — the bug survives them:

- the `ite` in the original transition relation (`x' = if x = 10 then 0 else x + 2`)
- the extra `¬ (x = 3)` hypothesis
- splitting a conjunctive hypothesis into two hypotheses
- widening the goal to `(2 <= x' ∨ x' = 0) ∧ ¬ (x' = 3)`

So the shape is: **two variables linked by a linear equation, a disjunctive
hypothesis, and a disequality goal.**

## Why it matters here

This is the shape every *parity-like* safety obligation takes. `m_step2`
steps `x` by 2 through `0,2,…,10`; proving `x ≠ 3` needs an invariant that
excludes the odd values, ic3ia finds `(x ≥ 2 ∨ x = 0) ∧ x ≠ 3`, and the
inductiveness check is then exactly the reproducer above. The certificate
`vmt2lean.py` emits is correct — only its `smt` call fails to reconstruct.

`bv_decide_light` is unaffected: it is the SAT-mode (`vmt2lean -m sat`) path
and does not go through this reconstruction.
