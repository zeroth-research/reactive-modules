import Mathlib.Geometry.Manifold.MFDeriv.Basic
import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-!
# Hybrid systems

The semantic core, independent of atoms and modules: a *probabilistic hybrid
system* on a state manifold `S` has

* initial distributions (`init`), and jump distributions (`jump`): a *set* of
  distributions is a nondeterministic choice between random choices;
* a flow (`flow`), assigning to every state the tangent vectors it may move
  along: a differential inclusion, nondeterministic but not random;
* jump regions (`regions`), which a flow cannot get out of.

Time is *hybrid*: a run alternates discrete jumps, taking no time, and
continuous flows along trajectories (`IsTrajectory`), taking their duration.
Jumps are not urgent — a jump may happen anywhere it is enabled, and a flow may
go on through such states — but a trajectory that enters a jump region stays
in it, so the jump must happen before the region is left. (A region is thus
meant to be convex along the flow.)

* `Move`: a step as a scheduler sees it, with the distribution of the next state;
* `Step`: a step as a run sees it, to a state of positive probability;
* `Reachable`, `IsRun`: the states and runs of positive probability.

The probabilities themselves enter with a scheduler (`Zrth.Stochastic`).
-/

namespace Zrth

open Manifold Set Filter

variable {E H : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [TopologicalSpace H]
  (I : ModelWithCorners ℝ E H) (S : Type*) [TopologicalSpace S] [ChartedSpace H S]

/-- A probabilistic hybrid system on `S`. -/
structure Hybrid where
  /-- The initial distributions. -/
  init : Set (PMF S)
  /-- The jumps: the distributions of the state after a discrete step. -/
  jump : S → Set (PMF S)
  /-- The flow: the admissible tangent vectors at each state. -/
  flow : (s : S) → Set (TangentSpace I s)
  /-- The jump regions, which a flow cannot get out of. -/
  regions : Set (Set S)

namespace Hybrid

variable {I S} (h : Hybrid I S)

/-- `γ` is a trajectory of `h` of duration `d`: it follows the flow on
    `[0, d]` and does not get out of any jump region it enters. -/
structure IsTrajectory (γ : ℝ → S) (d : ℝ) : Prop where
  /-- Time does not go backwards. -/
  nonneg : 0 ≤ d
  /-- At every instant, the velocity of `γ` is a tangent vector the flow allows. -/
  follows : ∀ t ∈ Icc 0 d, ∃ v ∈ h.flow (γ t),
    HasMFDerivAt 𝓘(ℝ, ℝ) I γ t ((1 : ℝ →L[ℝ] ℝ).smulRight v)
  /-- Once in a jump region, `γ` stays in it until the end. -/
  stays : ∀ G ∈ h.regions, ∀ t₁ ∈ Icc 0 d, ∀ t₂ ∈ Icc t₁ d, γ t₁ ∈ G → γ t₂ ∈ G

/-- A move from `s` taking the time `d`, with the distribution of the next
    state: a jump (taking no time), or a flow along a trajectory of duration
    `d` (deterministically reaching its end). -/
inductive Move : S → ℝ → PMF S → Prop
  | jump {s : S} {μ : PMF S} : μ ∈ h.jump s → Move s 0 μ
  | flow {γ : ℝ → S} {d : ℝ} : h.IsTrajectory γ d → Move (γ 0) d (PMF.pure (γ d))

/-- A possible step from `s` to `s'` taking the time `d`: a jump to a state in
    the support of a jump distribution, or a flow along a trajectory of
    duration `d`. -/
inductive Step : S → ℝ → S → Prop
  | jump {s s' : S} {μ : PMF S} : μ ∈ h.jump s → s' ∈ μ.support → Step s 0 s'
  | flow {γ : ℝ → S} {d : ℝ} : h.IsTrajectory γ d → Step (γ 0) d (γ d)

/-- A move goes to any state in the support of its distribution. -/
theorem Move.step {h : Hybrid I S} {s s' : S} {d : ℝ} {μ : PMF S} (hm : h.Move s d μ)
    (hs' : s' ∈ μ.support) : h.Step s d s' := by
  cases hm with
  | jump hμ => exact .jump hμ hs'
  | flow hγ =>
    rw [PMF.support_pure, Set.mem_singleton_iff] at hs'
    exact hs' ▸ .flow hγ

/-- The states reachable, with positive probability, from an initial state. -/
inductive Reachable : S → Prop
  | init {μ : PMF S} {s : S} : μ ∈ h.init → s ∈ μ.support → Reachable s
  | step {s s' : S} {d : ℝ} : Reachable s → h.Step s d s' → Reachable s'

/-- A possible run: the states `ρ n`, each step `n` taking the time `δ n`. -/
structure IsRun (ρ : ℕ → S) (δ : ℕ → ℝ) : Prop where
  /-- The run starts in an initial state. -/
  init : ∃ μ ∈ h.init, ρ 0 ∈ μ.support
  /-- Every step is a possible step. -/
  step : ∀ n, h.Step (ρ n) (δ n) (ρ (n + 1))

/-- The time of a run diverges: it is not Zeno. -/
def Divergent (δ : ℕ → ℝ) : Prop :=
  Tendsto (fun n => ∑ i ∈ Finset.range n, δ i) atTop atTop

/-- Every state of a run is reachable. -/
theorem IsRun.reachable {h : Hybrid I S} {ρ : ℕ → S} {δ : ℕ → ℝ} (hρ : h.IsRun ρ δ) :
    ∀ n, h.Reachable (ρ n)
  | 0 => let ⟨_, hμ, hs⟩ := hρ.init; .init hμ hs
  | n + 1 => .step (hρ.reachable n) (hρ.step n)

end Hybrid

end Zrth
