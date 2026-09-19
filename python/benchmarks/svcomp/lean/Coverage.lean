import Farkas

namespace Matrix
open Fin

/- ### Coverage bridge

The per-cell certificate proofs state infeasibility in *matrix* form
(`¬ (A *ᵥ s ≤ᵥ b)`, via `farkas_sound`). The coverage proof, that the
certified cells exhaust the loop guard, is discharged by `omega`, which
reasons over *scalar* linear arithmetic. These lemmas are the one-time
bridge between the two: they let a matrix inequality `A *ᵥ s ≤ᵥ b` be
unfolded into an explicit conjunction of scalar row inequalities that
`omega` can see.

- `mulVec_apply` rewrites the i-th entry of a matrix-vector product into
  the i-th row dotted with the vector (it holds by `rfl`).
- `forall_fin_succ` / `forall_fin_zero` expand a `∀ i : Fin m` over the
  inductive `Fin` into an explicit finite conjunction.

A generated proof uses them as a fixed `simp` set; see the emitted
`cellK_decrease` lemmas and per-path coverage theorems. -/

/-- The i-th entry of `A *ᵥ s` is the i-th row of `A` dotted with `s`.
True by definition (`matrixVectorProduct` unfolds to this). -/
theorem mulVec_apply {m n : Nat} (A : Matrix m n Int) (s : Vector n Int)
    (i : Fin m) : (A *ᵥ s) i = sum (fun j => s j * A i j) :=
  rfl

/-- Peel the first index off a `∀` over `Fin (n+1)`. -/
theorem forall_fin_succ {n : Nat} {P : Fin (n + 1) → Prop} :
    (∀ i, P i) ↔ P fzero ∧ ∀ i : Fin n, P (fsucc i) := by
  constructor
  · intro h
    exact ⟨h fzero, fun i => h (fsucc i)⟩
  · intro ⟨h0, hs⟩ i
    match i with
    | fzero => exact h0
    | fsucc j => exact hs j

/-- `Fin 0` is empty, so a `∀` over it is vacuously true. -/
theorem forall_fin_zero {P : Fin 0 → Prop} : (∀ i, P i) ↔ True := by
  constructor
  · intro _
    trivial
  · intro _ i
    nomatch i

/- ### The per-disjunct refutation bridge, as one tactic

Every generated `cellRdD_refute` lemma has the same shape: from the region's
`farkas_sound` infeasibility `hinf : ¬ (A *ᵥ s ≤ᵥ b)` and the guard/invariant
and sign hypotheses, conclude `¬ (row₁ ∧ … ∧ rowₖ)` for the rows of one disjunct
of the rule's negation. The proof is mechanical and identical save for the names
involved, so it lives here once instead of being re-emitted per disjunct:

  - assume the disjunct's rows;
  - feed `hinf` the full row system; `simp only` unfolds the matrix product
    (`mulVec_apply`/`sum`/`forall_fin_*`) and the guard/invariant/sign
    predicates into scalar rows;
  - `omega` closes: the sign/guard/invariant rows hold by hypothesis and the
    disjunct's rows by assumption, contradicting `hinf`.

`hinf` is passed already applied to the state (`cellRdD_infeasible s`); the
`with` clause lists the definitions to unfold: `trans`, `invariants`, the
region's signs, and the cell's `cellRdD_A`/`cellRdD_b`. -/
syntax "refute_bridge" term " with " Lean.Parser.Tactic.simpLemma,* : tactic
macro_rules
  | `(tactic| refute_bridge $inf with $us,*) => `(tactic| (
      intro _hrows
      apply $inf
      simp only [lessOrEqualᵥ, forall_fin_succ, forall_fin_zero, mulVec_apply, sum,
                 Int.negOfNat_eq, $us,*] at *
      omega))

end Matrix
