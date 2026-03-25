import Cslib.Foundations.Semantics.LTS.Basic
import Core.LTL

/-  ----------------------------------------------------------------
 Sets of states
 ----------------------------------------------------------------- -/
universe u v

def StateSet (State: Type u) := State → Prop

instance {State: Type u} : Membership State (StateSet State) where
  mem S s := S s

instance {State: Type u} : EmptyCollection (StateSet State) where
  emptyCollection := fun _ => False

instance {State: Type u} : Union (StateSet State) where
  union A B := fun s => s ∈ A ∨ s ∈ B

instance {State: Type u} : Inter (StateSet State) where
  inter A B := fun s => s ∈ A ∧ s ∈ B

instance {State: Type u} : HasSubset (StateSet State) where
  Subset A B := ∀ s, s ∈ A → s ∈ B



/-  ----------------------------------------------------------------
 LTS with initial states and reachability
 ----------------------------------------------------------------- -/



/-- LTS with initial states -/
structure LTS' (State: Type u) (Label: Type v) extends Cslib.LTS State Label where
  init: State → Prop

namespace LTS'

--variable {State : Type u} {Label : Type v}

/- initial infinite trace of an LTS' -/
def ωTrace (lts : LTS' State Label)
    (ss : Cslib.ωSequence State) (μs : Cslib.ωSequence Label) : Prop :=
  lts.init (ss.get 0) ∧ lts.ωTr ss μs

/- is state reachable? -/
def reachable [DecidableEq State] (lts : LTS' State Label) (s : State) : Prop :=
  ∃ ss μs, lts.ωTrace ss μs ∧ ∃ n: Nat, ss.get n = s

/- set of reachable states -/
def reachableSet [DecidableEq State] (lts : LTS' State Label) : StateSet State :=
  fun s => lts.reachable s

/- a set of state is inductive (closed under transition relation) -/
def StateSet_isInductive (lts : LTS' State Label) (S : StateSet State) : Prop :=
  ∀ s s' : State, s ∈ S ∧ (∃ l, lts.Tr s l s') → s' ∈ S

/- a set of states is invariant, i.e., a superset of reachable states -/
def StateSet_isInvariant [DecidableEq State] (lts : LTS' State Label)
    (S : StateSet State) : Prop :=
  lts.reachableSet ⊆ S

/- a set of states is inductive and contains initial states -/
def StateSet_isInductiveInitial (lts : LTS' State Label)
    (S : StateSet State) : Prop :=
  (∀ s, lts.init s → s ∈ S) ∧ lts.StateSet_isInductive S

/- every inductive initial state is invariant -/
theorem StateSet_ind_init_is_inv [DecidableEq State]
    (lts : LTS' State Label) :
    ∀ S, lts.StateSet_isInductiveInitial S →
      lts.StateSet_isInvariant S := by
  intro S ⟨hinit, hind⟩ s hs
  obtain ⟨ss, μs, ⟨hsinit, htr⟩, n, rfl⟩ := hs
  induction n with
  | zero => exact hinit _ hsinit
  | succ n ih => exact hind _ _ ⟨ih, ⟨μs.get n, htr n⟩⟩

/- every state on an arbitrary trace is contained in arbitrary invariant -/
lemma trace_states_in_invariant [DecidableEq State]
    (lts : LTS' State Label)
    (S : StateSet State) (hinv : lts.StateSet_isInvariant S)
    (htr : lts.ωTrace ss μs) :
    ∀ x: Nat, S (ss.get x) := by
  intro x
  apply hinv
  exact ⟨ss, μs, htr, x, rfl⟩

end LTS'

/-  ----------------------------------------------------------------
 Proof rules
 ----------------------------------------------------------------- -/
section proof_rules

open LTLFormula

/-- Proof rule saying that if a set of states S is invariant,
    then `Globally S` holds on every trace
    (S is state of sets which is the same as atomic proposition) -/
theorem rule_globally {State: Type u} {Label : Type v} [DecidableEq State]
    (lts : LTS' State Label)
    (S : StateSet State)
    (hinv : lts.StateSet_isInvariant S) :
    ∀ ss μs, lts.ωTrace ss μs → ss ⊧ G (AP S) := by
  intro ss μs htr
  simp [Globally]
  simp [Finally]
  simp [sem]
  intro x
  rw [(drop_get_zero ss x).symm]
  exact lts.trace_states_in_invariant S hinv htr x


/-- Proof rule saying that if we have a ranking function
    on `¬B` states (states where `B` does not hold),
    then every trace in the transition system visits
    `B` states infinitely often. (The existence of such
    function proofs that `¬B` can be visited only finitely
    often on any trace).

    To make the statement more powerful, we work relative to
    states `I` that overapproximate all states in the system
    (`I` is an invariant). This is important, because this
    makes `V` easier to find (without `I`, `V` would have to work
    for *all* possible states, not only reachable states).
    At the same time, we work with `I` instead of reachable states,
    because reachable states might be hard/impossible to represent
    or even compute exactly. Obtaining a superset might be much
    easier.
-/
theorem rule_buchi {State: Type u} {Label : Type v} [DecidableEq State]
    (lts : LTS' State Label)             -- the LTS
    (B : State → Prop) [DecidablePred B]      -- proposition `B`
    (I : StateSet State)                      -- invariant `I`
    (hinv : lts.StateSet_isInvariant I)
    (V : State → Nat)                         -- ranking function on
                                              -- (I ∧ ¬B)-states
    -- assume `V` is a function such that whenever `B` does not hold,
    -- then in next steps the value of `V` decreases
    (hrank : ∀ s s', I s → ¬(B s) → (∃ l, lts.Tr s l s') → V s' < V s) :
    -- Then then `B` holds infinitely many times
    ∀ ss μs, lts.ωTrace ss μs → ss ⊧ G (F (AP B)) := by
  intro ss μs htr
  simp [LTLFormula.sem]
  simp [<- drop_get_zero]
  have hI := lts.trace_states_in_invariant I hinv htr
  have hstep : ∀ j, ∃ l, lts.Tr (ss.get j) l (ss.get (j + 1)) := by
    intro j; exists μs.get j; exact htr.2 j
  intro i
  suffices ∀ k n, V (ss.get n) ≤ k → ∃ x, B (ss.get (n + x)) from
    this (V (ss.get i)) i (Nat.le_refl _)
  intro k
  induction k with
  | zero =>
    intro n hv
    exists 0
    simp
    if hb : B (ss.get n) then
      exact hb
    else
      have : V (ss.get n) ≤ V (ss.get (n + 1)) := by simp_all
      exact absurd (hrank _ _ (hI n) hb (hstep n)) (Nat.not_lt.mpr this)
  | succ k ih =>
    intro n hv
    if hb : B (ss.get n) then
      exact ⟨0, by simp; exact hb⟩
    else
      have hdec := hrank _ _ (hI n) hb (hstep n)
      obtain ⟨x, hbx⟩ := ih (n + 1) (by omega)
      exists x + 1
      have : n + (x + 1) = n + 1 + x := by omega
      rw [this]
      exact hbx


/-- a stronger version of rule_buchi (stronger in `hrank`) -/
theorem rule_buchi' {State: Type u} {Label : Type v} [DecidableEq State]
    (lts : LTS' State Label)             -- the LTS
    (B : State → Prop) [DecidablePred B]      -- proposition `B`
    (I : StateSet State)                      -- invariant `I`
    (hinv : lts.StateSet_isInvariant I)
    (V : State → Nat)                         -- ranking function on
                                              -- (I ∧ ¬B)-states
    -- assume `V` is a function such that whenever `B` does not hold,
    -- then in next steps the value of `V` decreases
    (hrank : ∀ s s', I s ∧ ¬(B s) ∧ (∃ l, lts.Tr s l s') → V s' < V s) :
    -- Then then `B` holds infinitely many times
    ∀ ss μs, lts.ωTrace ss μs → ss ⊧ G (F (AP B)) := by
  intro ss μs htr
  simp [LTLFormula.sem]
  simp [<- drop_get_zero]
  have hI := lts.trace_states_in_invariant I hinv htr
  have hstep : ∀ j, ∃ l, lts.Tr (ss.get j) l (ss.get (j + 1)) := by
    intro j; exists μs.get j; exact htr.2 j
  intro i
  suffices ∀ k n, V (ss.get n) ≤ k → ∃ x, B (ss.get (n + x)) from
    this (V (ss.get i)) i (Nat.le_refl _)
  intro k
  induction k with
  | zero =>
    intro n hv
    exists 0
    simp
    if hb : B (ss.get n) then
      exact hb
    else
      have : V (ss.get n) ≤ V (ss.get (n + 1)) := by simp_all
      exact absurd (hrank _ _ ⟨ (hI n), hb, (hstep n)⟩) (Nat.not_lt.mpr this)
  | succ k ih =>
    intro n hv
    if hb : B (ss.get n) then
      exact ⟨0, by simp; exact hb⟩
    else
      have hdec := hrank _ _ ⟨ hI n, hb, (hstep n)⟩
      obtain ⟨x, hbx⟩ := ih (n + 1) (by omega)
      exists x + 1
      have : n + (x + 1) = n + 1 + x := by omega
      rw [this]
      exact hbx



end proof_rules

structure ReactiveModule (Extl: Type u) (State: Type u) where
  init   : Extl → State
  update : State → Extl → State
  init_pre: Extl → Prop
  update_pre: Extl → Prop

namespace ReactiveModule

/-- describe traces of RM (this basically defines the semantics) -/
def traces (module: ReactiveModule Extl State)
  (ss: Cslib.ωSequence State): Prop :=
  (∃l, module.init_pre l ∧         -- initial inputs satisfy precondition
   ss 0 = module.init l) ∧      -- initial state is defined by `init`
  ∀ i: Nat, ∃ l: Extl,
    module.update_pre l ∧
    -- don't forget that inputs are shifted by 1, because the first inputs
    -- are used for initial states
    (ss (i + 1)) = module.update (ss i) l

/-- describe a trace of a RM generated by a sequence of inputs  -/
def traces'
  (module: ReactiveModule Extl State)
  (ss: Cslib.ωSequence State)
  (inps: Cslib.ωSequence Extl): Prop :=
  (module.init_pre inps.head ∧          -- initial inputs satisfy precondition
   ss 0 = module.init (inps 0)) ∧       -- initial state is defined by `init`
  ∀ i: Nat,
    module.update_pre (inps i) ∧
    -- don't forget that inputs are shifted by 1, because the first inputs
    -- are used for initial states
    (ss (i + 1)) = module.update (ss i) (inps (i + 1))

theorem traces'_impl_traces (module: ReactiveModule Extl State):
  module.traces' ss inps → module.traces ss := by
  unfold traces traces'
  simp_all
  intro h hs hi
  constructor
  . refine ⟨ inps.head, ?_⟩
    constructor
    . exact h
    . have : inps 0 = inps.head := by
        unfold Cslib.ωSequence.head
        rfl
      rw [this]
  . intro i
    exact ⟨inps (i + 1), (hi (i + 1)).1, rfl⟩


def LTS_init (M: ReactiveModule Extl State)
  := fun x => ∃ l, M.init_pre l ∧ M.init l = x

def LTS_update (M: ReactiveModule Extl State)
  := fun x l x' => M.update_pre l ∧ M.update x l = x'

def toLTS' (module: ReactiveModule Extl State): LTS' State Extl :=
  {
    Tr   := module.LTS_update
    init := module.LTS_init
  }

/-- The LTS for a reactive module defines exactly the traces of the
    reactive module -/
theorem LTS_traces (M: ReactiveModule Extl State):
  ∀ ss μs, M.toLTS'.ωTrace ss μs → M.traces ss := by
  intro ss μs h
  unfold traces
  simp_all [toLTS', LTS'.ωTrace, Cslib.LTS.ωTr]
  constructor
  case left =>
    have hi := h.left
    simp [LTS_init] at hi
    have : ss 0 = ss.get 0 := by rfl
    simp [this, Eq.comm]
    exact hi
  case right =>
    intro i
    have hi := h.right i
    simp [LTS_update] at hi
    refine ⟨ (μs i), ?_⟩
    rw [Eq.comm]
    exact hi

theorem traces_to_LTS (M: ReactiveModule Extl State):
  ∀ ss, M.traces ss → (∃ μs, M.toLTS'.ωTrace ss μs) := by
  intro ss
  simp [traces]
  have : ss 0 = ss.get 0 := rfl
  simp_all [toLTS', Cslib.LTS.ωTr, LTS_init, LTS_update, LTS'.ωTrace]
  intro x hx hs hi
  constructor
  . refine ⟨ x, ?_⟩
    exact ⟨ hx, rfl ⟩
  . exact ⟨fun i => (hi i).choose, fun i =>
    let ⟨hpre, hupd⟩ := (hi i).choose_spec
    ⟨hpre, hupd.symm⟩⟩


end ReactiveModule
