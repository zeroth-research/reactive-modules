import Zrth.Module.Basic

/-!
# Rounds

A *round* is how the atoms of a module take a discrete step together. They go
in the order of the module, each drawing the values of its controlled
variables at random, given the valuation drawn so far — in which it finds the
next values of the variables it awaits, drawn by earlier atoms (the order is
consistent with the await relation) — and overriding it with its draw. A round
is thus a chain of `PMF.bind`s (`Module.round`), resolving, for each atom, the
choice of its distribution by a `Module.Draws`.

* `round_draw`: after a round, every atom holds a value drawn given the *final*
  values of what it awaits, as no later atom overrides those;
* `round_pure_atoms`: a round of deterministic draws is deterministic;
* `Module.trace`: the round recorded atom by atom, the valuations before each
  atom and after the last one (the basis of the round's filtration, see
  `Zrth.Stochastic.Round`).
-/

namespace Zrth

open Manifold

variable {V : Type*} [DecidableEq V]
  {E : V → Type*} [∀ v, NormedAddCommGroup (E v)] [∀ v, NormedSpace ℝ (E v)]
  {H : V → Type*} [∀ v, TopologicalSpace (H v)]
  {I : ∀ v, ModelWithCorners ℝ (E v) (H v)}
  {M : V → Type*} [∀ v, TopologicalSpace (M v)] [∀ v, ChartedSpace (H v) (M v)]

namespace Module

variable (m : Module I M)

/-- The choices of the draws of the atoms in a round: the distribution of the
    values of the controlled variables of each atom, given the valuation `x`
    drawn so far (in which the atom finds the values it awaits). -/
abbrev Draws := ∀ a, a ∈ m.atoms → Val M m.ctrl → PMF (Val M a.ctrl)

/-- A round of the atoms `l` (of `m`), starting from the valuation `x`: in turn,
    every atom draws the values of its controlled variables given the
    valuation drawn so far, which it overrides with them. -/
noncomputable def round (κ : m.Draws) :
    (l : List (Atom I M)) → (∀ a ∈ l, a ∈ m.atoms) → Val M m.ctrl → PMF (Val M m.ctrl)
  | [], _, x => PMF.pure x
  | a :: l, h, x => (κ a (h a List.mem_cons_self) x).bind fun c =>
      round κ l (fun b hb => h b (List.mem_cons_of_mem _ hb)) (Val.override x c)

variable {m}

/-- A round only changes the variables controlled by its atoms. -/
theorem round_frame {κ : m.Draws} {l : List (Atom I M)} {h : ∀ a ∈ l, a ∈ m.atoms}
    {x s' : Val M m.ctrl} (hs' : s' ∈ (m.round κ l h x).support) (v : m.ctrl)
    (hv : ∀ b ∈ l, v.1 ∉ b.ctrl) : s' v = x v := by
  induction l generalizing x with
  | nil =>
    rw [round, PMF.support_pure, Set.mem_singleton_iff] at hs'
    rw [hs']
  | cons a l ih =>
    rw [round, PMF.mem_support_bind_iff] at hs'
    obtain ⟨c, _, hs'⟩ := hs'
    rw [ih hs' fun b hb => hv b (List.mem_cons_of_mem _ hb),
      Val.override_of_not_mem _ _ (hv a List.mem_cons_self)]

/-- In a round, every atom draws the final values of its controlled variables,
    given the final values of the variables it awaits: the order is consistent
    with the await relation.

    The variables `X b` are some of those `b` awaits; they are controlled in
    `m` but not by `b`. (For a module, `X b = b.wait`.) -/
theorem round_draw {κ : m.Draws} {l : List (Atom I M)} {h : ∀ a ∈ l, a ∈ m.atoms}
    {x s' : Val M m.ctrl} (hs' : s' ∈ (m.round κ l h x).support)
    (hctrl : l.Pairwise fun a b => Disjoint a.ctrl b.ctrl)
    (hwait : l.Pairwise fun a b => Disjoint a.wait b.ctrl)
    {X : Atom I M → Finset V} (hX : ∀ b ∈ l, X b ⊆ m.ctrl)
    (hXctrl : ∀ b ∈ l, Disjoint b.ctrl (X b))
    (hXwait : ∀ b ∈ l, X b ⊆ b.wait) :
    ∀ b (hb : b ∈ l), ∃ x', Val.restrict (m.ctrl_subset (h b hb)) s' ∈ (κ b (h b hb) x').support ∧
      Val.restrict (hX b hb) x' = Val.restrict (hX b hb) s' := by
  induction l generalizing x with
  | nil => exact fun _ hb => absurd hb List.not_mem_nil
  | cons a l ih =>
    rw [round, PMF.mem_support_bind_iff] at hs'
    obtain ⟨c, hc, hs'⟩ := hs'
    rw [List.pairwise_cons] at hctrl hwait
    intro b hb
    rcases List.mem_cons.1 hb with rfl | hb'
    · -- The first atom drew `c` given `x`. Later atoms neither control what it
      -- controls (so `c` survives to the end) nor what it awaits (so the
      -- awaited values in `x` are still the final ones).
      refine ⟨x, ?_, ?_⟩
      · convert hc using 1
        funext ⟨v, hv⟩
        rw [Val.restrict, round_frame hs' ⟨v, _⟩ fun b' hb' =>
          Finset.disjoint_left.1 (hctrl.1 b' hb') hv, Val.override_of_mem _ _ hv]
      · funext ⟨v, hv⟩
        simp only [Val.restrict]
        rw [round_frame hs' ⟨v, _⟩ fun b' hb' =>
          Finset.disjoint_left.1 (hwait.1 b' hb') (hXwait b hb hv),
          Val.override_of_not_mem _ _ fun hv' =>
            Finset.disjoint_left.1 (hXctrl b hb) hv' hv]
    · -- A later atom: by induction on the rest of the round.
      exact ih hs' hctrl.2 hwait.2 (fun b hb => hX b (List.mem_cons_of_mem _ hb))
        (fun b hb => hXctrl b (List.mem_cons_of_mem _ hb))
        (fun b hb => hXwait b (List.mem_cons_of_mem _ hb)) b hb'

/-- A round of atoms drawing, deterministically, the values of `t`, draws `t`
    on their controlled variables, and keeps the rest. -/
theorem round_pure {κ : m.Draws} {t : Val M m.ctrl}
    (hκ : ∀ a ha x, κ a ha x = PMF.pure (Val.restrict (m.ctrl_subset ha) t)) :
    ∀ (l : List (Atom I M)) (h : ∀ a ∈ l, a ∈ m.atoms) (x : Val M m.ctrl),
      ∃ z, m.round κ l h x = PMF.pure z ∧ (∀ v : m.ctrl, (∃ b ∈ l, v.1 ∈ b.ctrl) → z v = t v) ∧
        (∀ v : m.ctrl, (∀ b ∈ l, v.1 ∉ b.ctrl) → z v = x v)
  | [], _, x => ⟨x, rfl, fun _ ⟨_, hb, _⟩ => absurd hb List.not_mem_nil, fun _ _ => rfl⟩
  | a :: l, h, x => by
    obtain ⟨z, hz, h₁, h₂⟩ := round_pure hκ l (fun b hb => h b (List.mem_cons_of_mem _ hb))
      (Val.override x (Val.restrict (m.ctrl_subset (h a List.mem_cons_self)) t))
    refine ⟨z, by rw [round, hκ, PMF.pure_bind, hz], fun v ⟨b, hb, hv⟩ => ?_, fun v hv => ?_⟩
    · -- A variable controlled in the round: by a later atom, or else by `a`,
      -- which wrote `t` on it, and nobody later touched it.
      by_cases hl : ∃ b ∈ l, v.1 ∈ b.ctrl
      · exact h₁ v hl
      · push Not at hl
        rw [h₂ v hl]
        rcases List.mem_cons.1 hb with rfl | hb'
        · exact Val.override_of_mem _ _ hv
        · exact absurd hv (hl b hb')
    · rw [h₂ v fun b hb => hv b (List.mem_cons_of_mem _ hb),
        Val.override_of_not_mem _ _ (hv a List.mem_cons_self)]

/-- A full round of atoms drawing, deterministically, the values of `t`, draws `t`. -/
theorem round_pure_atoms {κ : m.Draws} {t : Val M m.ctrl}
    (hκ : ∀ a ha x, κ a ha x = PMF.pure (Val.restrict (m.ctrl_subset ha) t))
    (x : Val M m.ctrl) : m.round κ m.atoms (fun _ h => h) x = PMF.pure t := by
  obtain ⟨z, hz, h₁, _⟩ := round_pure hκ m.atoms (fun _ h => h) x
  rw [hz]
  congr
  funext v
  exact h₁ v ((m.mem_ctrl_iff v.1).1 v.2)

/-! ### Traces of rounds -/

/-- Prepending `x` to the sequence `ω`. -/
def seqCons {α : Type*} (x : α) (ω : ℕ → α) : ℕ → α
  | 0 => x
  | n + 1 => ω n

variable (m) in
/-- The trace of a round of the atoms `l` (of `m`), starting from `x`: the
    valuations drawn so far, before each atom in turn and after the last one
    (then repeated). The evaluation of the round, atom by atom. -/
noncomputable def trace (κ : m.Draws) :
    (l : List (Atom I M)) → (∀ a ∈ l, a ∈ m.atoms) → Val M m.ctrl → PMF (ℕ → Val M m.ctrl)
  | [], _, x => PMF.pure fun _ => x
  | a :: l, h, x => (κ a (h a List.mem_cons_self) x).bind fun c =>
      PMF.map (seqCons x) (trace κ l (fun b hb => h b (List.mem_cons_of_mem _ hb)) (Val.override x c))

/-- The end of the trace of a round is distributed as the round. -/
theorem trace_map_length {κ : m.Draws} {l : List (Atom I M)} {h : ∀ a ∈ l, a ∈ m.atoms}
    {x : Val M m.ctrl} : PMF.map (fun ω => ω l.length) (m.trace κ l h x) = m.round κ l h x := by
  induction l generalizing x with
  | nil => rw [trace, PMF.pure_map]; rfl
  | cons a l ih =>
    rw [trace, round, PMF.map_bind]
    congr 1
    funext c
    rw [PMF.map_comp]
    exact ih

/-- The trace of a round starts from its initial valuation, and every atom in
    turn overrides the valuation drawn so far with its draw, given it. -/
theorem trace_step {κ : m.Draws} {l : List (Atom I M)} {h : ∀ a ∈ l, a ∈ m.atoms}
    {x : Val M m.ctrl} {ω : ℕ → Val M m.ctrl} (hω : ω ∈ (m.trace κ l h x).support) :
    ω 0 = x ∧ ∀ i (hi : i < l.length),
      ∃ c : Val M l[i].ctrl, c ∈ (κ l[i] (h _ (List.getElem_mem hi)) (ω i)).support ∧
        ω (i + 1) = Val.override (ω i) c := by
  induction l generalizing x ω with
  | nil =>
    rw [trace, PMF.support_pure, Set.mem_singleton_iff] at hω
    exact ⟨by rw [hω], fun _ hi => absurd hi (Nat.not_lt_zero _)⟩
  | cons a l ih =>
    rw [trace, PMF.mem_support_bind_iff] at hω
    obtain ⟨c, hc, hω⟩ := hω
    rw [PMF.mem_support_map_iff] at hω
    obtain ⟨ω', hω', rfl⟩ := hω
    obtain ⟨h₀, hs⟩ := ih hω'
    refine ⟨rfl, fun i hi => ?_⟩
    cases i with
    | zero => exact ⟨c, hc, h₀⟩
    | succ i => exact hs i (Nat.lt_of_succ_lt_succ hi)

end Module

end Zrth
