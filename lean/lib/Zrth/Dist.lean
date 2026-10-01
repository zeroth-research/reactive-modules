import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-!
# Finitely supported distributions

The discrete blocks of an atom (its initialisation and update) are
probabilistic: they choose the next values of the controlled variables at
random. As in the Rust implementation, the randomness is *finite*: every
distribution is a `PMF` whose support is finite (`FinDist`).

Two distributions are enough for the examples:

* the Dirac distribution `FinDist.pure a`, a deterministic choice of `a`;
* the coin `FinDist.coin p hp x y`, choosing `x` with probability `p` and `y`
  otherwise.
-/

namespace Zrth

/-- A finitely supported probability distribution. -/
structure FinDist (α : Type*) where
  /-- The distribution. -/
  toPMF : PMF α
  /-- Its support is finite. -/
  finite : toPMF.support.Finite

/-- A finite distribution is used as a distribution. -/
instance {α : Type*} : CoeOut (FinDist α) (PMF α) := ⟨FinDist.toPMF⟩

/-- The Dirac distribution at `a`. -/
noncomputable def FinDist.pure {α : Type*} (a : α) : FinDist α :=
  ⟨PMF.pure a, by simp⟩

/-- The coin landing on `true` with probability `p`. -/
noncomputable def FinDist.flip (p : NNReal) (hp : p ≤ 1) : PMF Bool :=
  PMF.ofFintype (fun b => if b then (p : ENNReal) else ((1 - p : NNReal) : ENNReal)) (by
    simp only [Fintype.sum_bool, ↓reduceIte, Bool.false_eq_true, ← ENNReal.coe_add,
      add_tsub_cancel_of_le hp, ENNReal.coe_one])

/-- The distribution of `x` with probability `p`, and of `y` otherwise. -/
noncomputable def FinDist.coin {α : Type*} (p : NNReal) (hp : p ≤ 1) (x y : α) : FinDist α :=
  ⟨(FinDist.flip p hp).map fun b => if b then x else y, by
    rw [PMF.support_map]; exact (Set.toFinite _).image _⟩

/-- A coin lands on one of its two faces. -/
theorem FinDist.mem_support_coin {α : Type*} {p : NNReal} {hp : p ≤ 1} {x y z : α}
    (h : z ∈ (FinDist.coin p hp x y).toPMF.support) : z = x ∨ z = y := by
  simp only [FinDist.coin, PMF.support_map] at h
  obtain ⟨b, _, rfl⟩ := h
  cases b <;> simp

/-- A coin with distinct faces lands on the first one with probability `p`. -/
theorem FinDist.coin_apply_left {α : Type*} {p : NNReal} {hp : p ≤ 1} {x y : α} (hxy : x ≠ y) :
    (FinDist.coin p hp x y).toPMF x = p := by
  simp only [FinDist.coin, FinDist.flip, PMF.map_apply, tsum_fintype, Fintype.sum_bool,
    PMF.ofFintype_apply, ↓reduceIte, Bool.false_eq_true, hxy, add_zero]

end Zrth
