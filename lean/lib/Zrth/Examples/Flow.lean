import Zrth.Atom

/-!
# The flow `dx = x dt`

The flow of a single real variable `x` over real time `t`, typed as
`ℝ × T ℝ → T ℝ` with `T` the tangent bundle.
-/

namespace Zrth.Examples.Flow

open Manifold Bundle

/-- `T ℝ`: the tangent bundle of `ℝ` viewed as a manifold modelled on itself. -/
abbrev Tℝ := TangentBundle 𝓘(ℝ, ℝ) ℝ

/-- The flow `dx = x dt`: from the state `x` and a time tangent `(t, dt)`,
    produce the tangent vector `dx = x * dt` at base point `x`. -/
noncomputable def flow : ℝ × Tℝ → Tℝ
  | (x, ⟨_t, dt⟩) => ⟨x, x • dt⟩

/-- The output lies over the state: `flow` is a section-like map over `x`. -/
theorem flow_proj (x : ℝ) (τ : Tℝ) : (flow (x, τ)).proj = x := rfl

example (x t dt : ℝ) : (flow (x, ⟨t, dt⟩)).2 = x • dt := rfl

/-- The variables of the atom: the state `x` and the time `t`. -/
inductive Var
  | x
  | t
  deriving DecidableEq

/-- Both variables range over `ℝ`, modelled on itself. -/
noncomputable abbrev I (_ : Var) := 𝓘(ℝ, ℝ)
abbrev M (_ : Var) := ℝ

/-- The differential atom of `dx = x dt`: it controls `x`, reads `x` and `t`
    and awaits the tangent `dt` of the time `t`; `x` starts at `1` and the update
    skips (reading `x` latched). The flow is `flow` at the current values of `x`
    and `t`. -/
noncomputable def atom : Atom I M where
  ctrl := {.x}
  wait := {.t}
  read := {.x, .t}
  disjoint_ctrl_wait := by simp
  init _ := {.pure fun _ => 1}
  update | (r, _) => {.pure fun _ => r ⟨.x, by simp⟩}
  flow c p :=
    let x := c ⟨.x, Finset.mem_singleton_self _⟩
    let t := p.1 ⟨.t, by simp⟩
    let dt := p.2 ⟨.t, Finset.mem_singleton_self _⟩
    {fun _ => (flow (x, ⟨t, dt⟩)).2}

end Zrth.Examples.Flow
