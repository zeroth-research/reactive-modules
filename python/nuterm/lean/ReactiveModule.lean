import LTL
import Termination

/-!
Reactive modules and their proof rules. A `ReactiveModule` has an
initial-state map `init` and a step map `update`, each guarded by a precondition
on the external input; `traces` is the trace semantics over plain `Nat → State`
sequences; an invariant is a set of states containing every reachable state
(`StateSet_isInvariant`), and a set closed under `init`/`update`
(`StateSet_isInductiveInitial`) is one.

Proof rules, stated in the LTL of `LTL.lean`:
  * `rule_globally`:      an invariant holds `G`lobally on every trace.
  * `rule_globally_step`: a step property implied by the invariant holds
                           `G`lobally on every trace (over steps, not states).
  * `rule_buchi_lex`:     a Büchi-style liveness rule. If inside the step
                           domain `D` a lexicographic list of bounded-below
                           integer ranks strictly decreases, every trace leaves
                           `D` infinitely often (`G F ¬D`). Discharged by
                           `no_infinite_run_lex` (Termination.lean) applied to
                           the tail of the trace.

Lean core only.
-/

structure ReactiveModule (Extl : Type) (State : Type) where
  init       : Extl → State
  update     : State → Extl → State
  init_pre   : Extl → Prop
  update_pre : Extl → Prop

namespace ReactiveModule
open LTLFormula
variable {Extl State : Type}

/-- The runs of the module: a state sequence whose first state `init` produces
    from an input satisfying `init_pre`, and each of whose steps `update`
    produces from an input satisfying `update_pre`. -/
def traces (M : ReactiveModule Extl State) (ss : Nat → State) : Prop :=
  (∃ l, M.init_pre l ∧ ss 0 = M.init l) ∧
  ∀ i : Nat, ∃ l, M.update_pre l ∧ ss (i + 1) = M.update (ss i) l

def reachable (M : ReactiveModule Extl State) (s : State) : Prop :=
  ∃ ss : Nat → State, ∃ n : Nat, M.traces ss ∧ ss n = s

def StateSet_isInvariant (M : ReactiveModule Extl State) (S : State → Prop) : Prop :=
  ∀ s, M.reachable s → S s

def StateSet_isInductiveInitial (M : ReactiveModule Extl State) (S : State → Prop) : Prop :=
  (∀ l, M.init_pre l → S (M.init l)) ∧ (∀ s l, S s → M.update_pre l → S (M.update s l))

/-- An inductive-initial set contains every reachable state: induction along the
    trace that reaches it. -/
theorem StateSet_ind_init_is_inv (M : ReactiveModule Extl State) (S : State → Prop) :
    M.StateSet_isInductiveInitial S → M.StateSet_isInvariant S := by
  rintro ⟨hinit, hstep⟩ s ⟨ss, n, ⟨⟨l0, hl0, h0⟩, hnext⟩, hsn⟩
  subst hsn
  induction n with
  | zero => rw [h0]; exact hinit l0 hl0
  | succ k ih =>
    obtain ⟨l, hl, hk⟩ := hnext k
    rw [hk]; exact hstep _ _ ih hl

theorem trace_states_in_invariant (M : ReactiveModule Extl State) (S : State → Prop)
    (hinv : M.StateSet_isInvariant S) (ss : Nat → State) (htr : M.traces ss) : ∀ i, S (ss i) :=
  fun i => hinv (ss i) ⟨ss, i, htr, rfl⟩

theorem StateSet_isInvariant_true (M : ReactiveModule Extl State) :
    M.StateSet_isInvariant (fun _ => True) :=
  fun _ _ => True.intro

/-- An invariant holds globally on every trace. -/
theorem rule_globally (M : ReactiveModule Extl State) (S : State → Prop) (hinv : M.StateSet_isInvariant S) :
    ∀ ss, M.traces ss → ss ⊧ G (AP S) := by
  intro ss htr
  exact (globally_AP S ss).mpr (trace_states_in_invariant M S hinv ss htr)

/-- The same for a step property: it holds on every step from an invariant state. -/
theorem rule_globally_step (M : ReactiveModule Extl State) (P : State → State → Prop) (I : State → Prop)
    (hinv : M.StateSet_isInvariant I)
    (hstep : ∀ s l s', I s → M.update_pre l → s' = M.update s l → P s s') :
    ∀ ss, M.traces ss → ss ⊧ G (APₛ P) := by
  intro ss htr
  refine (globally_APₛ P ss).mpr fun i => ?_
  obtain ⟨l, hl, hi⟩ := htr.2 i
  exact hstep _ l _ (trace_states_in_invariant M I hinv ss htr i) hl hi

/-- A step domain `D` and a lexicographic list of integer ranks bounded below:
    whenever the module is in the domain and inside the invariant `I`, the ranks drop
    lexicographically, so every trace leaves `D` infinitely often. Proved from
    `no_infinite_run_lex` on the tail of the trace. -/
theorem rule_buchi_lex (M : ReactiveModule Extl State) (D : State → State → Prop) (I : State → Prop)
    (hinv : M.StateSet_isInvariant I)
    (Vs : List (State → Int)) (hpos : ∀ V ∈ Vs, ∀ s, 0 ≤ V s)
    (hrank : ∀ s l s', I s → M.update_pre l → s' = M.update s l → D s s' → lexDec Vs s s') :
    ∀ ss, M.traces ss → ss ⊧ G (F (Not (APₛ D))) := by
  intro ss htr
  refine (globally_finally_not_APₛ D ss).mpr fun N => ?_
  apply Classical.byContradiction
  intro hcon
  -- From position `N` on, every step of the trace stays inside `D`.
  have hnot : ∀ i, N ≤ i → D (ss i) (ss (i + 1)) := by
    intro i hNi
    apply Classical.byContradiction
    intro hD
    exact hcon ⟨i, hNi, hD⟩
  have hI := trace_states_in_invariant M I hinv ss htr
  -- So the tail `fun k => ss (N + k)` is an infinite run of the guarded step
  -- relation, on which the ranks decrease lexicographically, which is impossible.
  refine no_infinite_run_lex Vs
    (fun s s' => I s ∧ D s s' ∧ ∃ l, M.update_pre l ∧ s' = M.update s l) hpos
    (by rintro s s' ⟨hIs, hD, l, hpre, heq⟩; exact hrank s l s' hIs hpre heq hD)
    ⟨fun k => ss (N + k), fun k => ?_⟩
  obtain ⟨l, hl, heq⟩ := htr.2 (N + k)
  exact ⟨hI (N + k), hnot (N + k) (Nat.le_add_right N k), l, hl, heq⟩

/-- Assume-guarantee invariance: `J` initial and inductive relative to an
    invariant `I` makes `I ∧ J` invariant. `StateSet_ind_init_is_inv` is the
    case `I := fun _ => True`. -/
theorem StateSet_isInvariant_relative (M : ReactiveModule Extl State) (I J : State → Prop)
    (hI : M.StateSet_isInvariant I)
    (hinit : ∀ l, M.init_pre l → J (M.init l))
    (hstep : ∀ s l, I s → J s → M.update_pre l → J (M.update s l)) :
    M.StateSet_isInvariant (fun s => I s ∧ J s) := by
  rintro s ⟨ss, n, htr, rfl⟩
  refine ⟨hI _ ⟨ss, n, htr, rfl⟩, ?_⟩
  induction n with
  | zero =>
    obtain ⟨l, hpre, h0⟩ := htr.1
    rw [h0]; exact hinit l hpre
  | succ k ih =>
    obtain ⟨l, hpre, hk⟩ := htr.2 k
    rw [hk]; exact hstep _ l (hI _ ⟨ss, k, htr, rfl⟩) ih hpre

end ReactiveModule

-- Smoke test: a trivial module on `Int` with `rule_globally` and the trivial invariant.
open LTLFormula in
example :
    ∀ ss, ({ init := id, update := fun s _ => s + 0, init_pre := fun _ => True,
             update_pre := fun _ => True } : ReactiveModule Int Int).traces ss →
      ss ⊧ G (AP (fun _ => True)) :=
  ReactiveModule.rule_globally _ (fun _ => True) (ReactiveModule.StateSet_isInvariant_true _)
