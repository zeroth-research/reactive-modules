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

### Narrowing the two halves

"A disjunctive hypothesis and a disequality goal" is looser than what the
bug actually needs. Each half was varied while the other three ingredients
were held fixed; every variant below is a true statement, and the ones
marked *compiles* close cleanly through `smt`.

| variant | hypothesis | goal | result |
|---|---|---|---|
| baseline | `2 <= x ∨ x = 0` | `¬ (x' = 3)` | **kernel error** |
| negation on an inequality | `2 <= x ∨ x = 0`, `x <= 2` | `¬ (5 <= x')` | compiles |
| positive equalities in the goal | `x = 4 ∨ x = 6 ∨ x = 8` | `x' = 6 ∨ x' = 8 ∨ x' = 10` | compiles |
| no equality in the disjunction | `2 <= x ∨ x <= -2` | `¬ (x' = 1)` | compiles |
| disequality on the *other* variable | `2 <= x ∨ x = 0` | `¬ (x = 1)` | compiles |

So it is not "a negation" and not "equalities" that matter, but one of each,
on opposite sides and in opposite polarities:

**an equality *disjunct* in the hypothesis, a *disequality* goal, and that
disequality on the variable the linear equation defines** — `x'`, not `x`.
Negating an inequality instead of an equality is fine; so is a goal that is
a positive disjunction of equalities; so is a disjunction of inequalities in
the hypothesis. The `sum_ub` index is picked from the relation kinds of the
two summands, and this is exactly the case where a linear combination has to
mix an equality coming from one side with a disequality coming from the
other.

### Corroboration from the sweep

Two probes over the *same* module (`m_step2`, parity by construction) land
on opposite sides of this, which is what prompted the narrowing:

| probe | property | ic3ia's invariant | verdict |
|---|---|---|---|
| `NiS2Odd3` | `(not (= s0 3))` | `(x ≥ 2 ∨ x = 0) ∧ ¬(x = 3)` | `lean-fail` |
| `Step2Odd` | `(=> (>= s0 5) (or (= s0 6) (= s0 8) (= s0 10)))` | `(¬(5 ≤ x) ∨ x = 10 ∨ x = 6 ∨ x = 8) ∧ (x ≥ 2 ∨ x = 0) ∧ (x ≥ 4 ∨ x ≤ 2)` | `certified` |

Both need the parity argument and both get an invariant carrying a negation.
The one that reconstructs negates an *inequality* and keeps its equalities
positive; the one that fails carries `¬(x = 3)`. So "parity-like obligation"
is not the predicate — the disequality is.

## A second, unrelated symptom: `Max`

Probe `ReluTrans`. A transition containing a ReLU reaches Lean as
`Max.max 0 (x - 1)` — that is how the scalar encoding spells it — and the
goal carrying it fails at `smt` with

```
error: incorrect number of universe levels Max
```

Reproducer, again nothing but `Smt`:

```lean
import Smt

example (x x' : Int) (h : x' = Max.max 0 (x - 1)) (hx : 0 <= x) : 0 <= x' := by
  revert h hx
  smt
```

Different failure mode from the `sum_ub` one above — this is universe-level
bookkeeping on the `Max` constant, not a mis-typed arithmetic lemma — so it
is probably a separate issue. Worth noting that the VMT and the certificate
are both correct here; only the tactic fails. `omega` closes the same goal
after `simp [Max.max]`.

**This route no longer reaches it.** `Max.max 0 (x - 1)` was how the *scalar
Lean* printer spelled a ReLU, and the NA encoding stopped using that printer:
its transition now comes from `smt_encode`, where a ReLU is an `ite`, so no
`Max` constant reaches Lean at all. `ReluTrans` and `ReluImplies` certify end
to end as of that change. The reproducer above still fails and is still worth
fixing upstream — a hand-written model, or any other producer of
`Max.max`, walks into it.

## Why it matters here

It is the shape a safety obligation takes whenever the property *is* a
disequality. `m_step2` steps `x` by 2 through `0,2,…,10`; proving `x ≠ 3`
needs an invariant that excludes the odd values, ic3ia finds
`(x ≥ 2 ∨ x = 0) ∧ x ≠ 3`, and the inductiveness check is then exactly the
reproducer above. The certificate `vmt2lean.py` emits is correct — only its
`smt` call fails to reconstruct.

The workaround that falls out of the narrowing: state the property so that
its invariant does not need a disequality. `Step2Odd` above asks the same
parity question as an implication into a disjunction of equalities and
certifies end to end.

`bv_decide_light` is unaffected: it is the SAT-mode (`vmt2lean -m sat`) path
and does not go through this reconstruction.
