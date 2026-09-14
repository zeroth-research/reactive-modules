/-!
A minimal linear temporal logic over traces. A trace is a plain function
`Nat → State` and the suffix of a trace is `drop`.

  * `LTLFormula` — syntax. `AP` is an atomic proposition over a state; `APₛ` is
    one over a *step* `(tr i, tr (i+1))`, which is what the domain of a
    transition relation is.
  * `sem` — the standard semantics; `Finally`/`Globally` are the usual
    abbreviations `F φ = True U φ`, `G φ = ¬ F ¬ φ`.
  * `Globally.explicit`, `Finally.explicit`, `globally_AP`, `globally_APₛ`,
    `globally_finally_not_APₛ` — the unfolded forms that the proof rules in
    `ReactiveModule.lean` consume.

Lean core only.
-/

inductive LTLFormula {State : Type} where
  | AP    : (State → Prop) → LTLFormula            -- atomic proposition over a state
  | APₛ   : (State → State → Prop) → LTLFormula    -- atomic proposition over a step (our addition)
  | Next  : LTLFormula → LTLFormula
  | Until : LTLFormula → LTLFormula → LTLFormula
  | Or    : LTLFormula → LTLFormula → LTLFormula
  | And   : LTLFormula → LTLFormula → LTLFormula
  | Not   : LTLFormula → LTLFormula

namespace LTLFormula

/-- The suffix of a trace starting at position `n`: `drop n tr i = tr (n + i)`. -/
def drop {State : Type} (n : Nat) (tr : Nat → State) : Nat → State := fun i => tr (n + i)

def sem {State : Type} (F : @LTLFormula State) (tr : Nat → State) : Prop :=
  match F with
  | AP p      => p (tr 0)
  | APₛ p     => p (tr 0) (tr 1)
  | Next F'   => F'.sem (drop 1 tr)
  | Until A B => ∃ n : Nat, B.sem (drop n tr) ∧ ∀ j, j < n → A.sem (drop j tr)
  | Or A B    => A.sem tr ∨ B.sem tr
  | And A B   => A.sem tr ∧ B.sem tr
  | Not F'    => ¬ F'.sem tr

abbrev Finally {State : Type} (φ : @LTLFormula State) : @LTLFormula State := Until (AP fun _ => True) φ
abbrev Globally {State : Type} (φ : @LTLFormula State) : @LTLFormula State := Not (Finally (Not φ))

scoped prefix:75 "F" => Finally
scoped prefix:75 "G" => Globally
scoped notation:30 tr " ⊧ " φ => LTLFormula.sem φ tr

/-! ### Explicit forms

`G ψ` unfolds to `¬ ∃ n, ¬ ψ.sem (drop n tr) ∧ ∀ j < n, True`, and `F ψ` to
`∃ n, ψ.sem (drop n tr) ∧ ∀ j < n, True`; the lemmas below strip the trivial
side conditions and the double negation. `drop n tr 0` and `drop n tr 1` are
`tr n` and `tr (n + 1)` by `rfl`, and `drop m (drop n tr) i = tr (n + (m + i))`.
-/

theorem Globally.explicit {State : Type} (ψ : @LTLFormula State) (tr : Nat → State) :
    (tr ⊧ G ψ) ↔ ∀ i, ψ.sem (drop i tr) := by
  constructor
  · intro h i
    exact Classical.byContradiction fun hn => h ⟨i, hn, fun _ _ => True.intro⟩
  · intro h hex
    obtain ⟨n, hn, -⟩ := hex
    exact hn (h n)

theorem Finally.explicit {State : Type} (ψ : @LTLFormula State) (tr : Nat → State) :
    (tr ⊧ F ψ) ↔ ∃ i, ψ.sem (drop i tr) := by
  constructor
  · intro h
    obtain ⟨n, hn, -⟩ := h
    exact ⟨n, hn⟩
  · intro h
    obtain ⟨n, hn⟩ := h
    exact ⟨n, hn, fun _ _ => True.intro⟩

theorem globally_AP {State : Type} (p : State → Prop) (tr : Nat → State) :
    (tr ⊧ G (AP p)) ↔ ∀ i, p (tr i) :=
  Globally.explicit (AP p) tr

theorem globally_APₛ {State : Type} (p : State → State → Prop) (tr : Nat → State) :
    (tr ⊧ G (APₛ p)) ↔ ∀ i, p (tr i) (tr (i + 1)) :=
  Globally.explicit (APₛ p) tr

theorem globally_finally_not_APₛ {State : Type} (p : State → State → Prop) (tr : Nat → State) :
    (tr ⊧ G (F (Not (APₛ p)))) ↔ ∀ n, ∃ i, n ≤ i ∧ ¬ p (tr i) (tr (i + 1)) := by
  refine Iff.trans (Globally.explicit _ _) ?_
  constructor
  · intro h n
    obtain ⟨m, hm⟩ := (Finally.explicit _ _).mp (h n)
    have hm' : ¬ p (tr (n + (m + 0))) (tr (n + (m + 1))) := hm
    have e1 : n + (m + 0) = n + m := by omega
    have e2 : n + (m + 1) = n + m + 1 := by omega
    rw [e1, e2] at hm'
    exact ⟨n + m, Nat.le_add_right n m, hm'⟩
  · intro h n
    obtain ⟨i, hni, hi⟩ := h n
    refine (Finally.explicit _ _).mpr ⟨i - n, ?_⟩
    show ¬ p (tr (n + (i - n + 0))) (tr (n + (i - n + 1)))
    have e1 : n + (i - n + 0) = i := by omega
    have e2 : n + (i - n + 1) = i + 1 := by omega
    rw [e1, e2]
    exact hi

end LTLFormula
