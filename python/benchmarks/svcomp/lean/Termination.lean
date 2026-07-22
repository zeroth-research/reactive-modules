/-!
Well-foundedness principles for ranking functions, proven once and instantiated
per program — the `farkas_sound` of the termination argument.

  * `peel` — the recursive engine: a bounded-below rank `V` that on every step
    either strictly drops, or is non-increasing and hands off to a sub-relation
    `Step'`, is well-founded provided `Step'` is. `V` can drop only finitely
    often, so on any tail it is constant and the run is a `Step'`-run.
  * `no_infinite_run_lex` — a *list* of bounded-below ranks decreasing
    lexicographically (`lexDec`) admits no infinite run, by iterating `peel`
    down to the empty base. This is the lex order on `ℤ≥0ⁿ` — nested loops /
    vector ranking functions, with the scalar single loop as the length-1
    instance (`[V]`), so one theorem covers every case.

Only strict decrease is needed — no per-program margin `δ`, since an integer
rank cannot strictly decrease by less than one. (Over the rationals this fails:
`1, ½, ¼, …`; integer-valuedness is what makes the order well-founded.)
-/

/-- The recursive engine. If `V` is bounded below and every `Step` either drops
    `V` or is non-increasing-and-`Step'`, then `Step` is well-founded whenever
    `Step'` is. (Strong induction on `V (g 0)`: a drop shrinks the bound; if `V`
    never drops, the run is entirely a `Step'`-run.) -/
theorem peel {S : Type} (Step Step' : S → S → Prop) (V : S → Int)
    (hpos : ∀ s, 0 ≤ V s)
    (hstep : ∀ s s', Step s s' → V s' < V s ∨ (V s' ≤ V s ∧ Step' s s'))
    (hsub : ¬ ∃ f : Nat → S, ∀ n, Step' (f n) (f (n + 1))) :
    ¬ ∃ f : Nat → S, ∀ n, Step (f n) (f (n + 1)) := by
  suffices h : ∀ b : Nat, ∀ g : Nat → S, (∀ n, Step (g n) (g (n + 1))) →
      (V (g 0)).toNat ≤ b → False by
    rintro ⟨f, hf⟩
    exact h (V (f 0)).toNat f hf (Nat.le_refl _)
  intro b
  induction b using Nat.strongRecOn with
  | ind b ih =>
    intro g hg hb
    have hmono : ∀ m, V (g (m + 1)) ≤ V (g m) := fun m => by
      rcases hstep _ _ (hg m) with h | ⟨h, _⟩ <;> omega
    have hle0 : ∀ m, V (g m) ≤ V (g 0) := by
      intro m
      induction m with
      | zero => omega
      | succ k ih2 => have := hmono k; omega
    by_cases hdrop : ∃ n, V (g (n + 1)) < V (g n)
    · obtain ⟨n, hn⟩ := hdrop
      refine ih (V (g (n + 1))).toNat ?_ (fun k => g (n + 1 + k))
        (fun k => hg (n + 1 + k)) (Nat.le_refl _)
      have := hle0 n; have := hpos (g (n + 1)); have := hpos (g 0); omega
    · exact hsub ⟨g, fun n => by
        rcases hstep _ _ (hg n) with h | ⟨_, h⟩
        · exact absurd ⟨n, h⟩ hdrop
        · exact h⟩

/-- Lexicographic decrease across one step: the first rank drops, or it is
    non-increasing and the remaining ranks decrease lexicographically. -/
def lexDec {S : Type} (Vs : List (S → Int)) (s s' : S) : Prop :=
  match Vs with
  | [] => False
  | V :: rest => V s' < V s ∨ (V s' ≤ V s ∧ lexDec rest s s')

/-- A list of bounded-below ranks decreasing lexicographically on every step
    admits no infinite run — well-foundedness of the lex order on `ℤ≥0ⁿ`, by
    iterating `peel` (each level peels off the leading rank). -/
theorem no_infinite_run_lex {S : Type} (Vs : List (S → Int)) :
    ∀ (Step : S → S → Prop),
      (∀ V ∈ Vs, ∀ s, 0 ≤ V s) →
      (∀ s s', Step s s' → lexDec Vs s s') →
      ¬ ∃ f : Nat → S, ∀ n, Step (f n) (f (n + 1)) := by
  induction Vs with
  | nil =>
    rintro Step - hdec ⟨f, hf⟩
    exact hdec _ _ (hf 0)
  | cons V rest ih =>
    intro Step hpos hdec
    refine peel Step (fun s s' => lexDec rest s s') V (hpos V (by simp)) ?_ ?_
    · intro s s' h; exact hdec s s' h
    · exact ih (fun s s' => lexDec rest s s')
        (fun W hW s => hpos W (by simp [hW]) s) (fun _ _ h => h)
