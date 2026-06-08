import Lean
import Mathlib.Tactic
import Smt

open Lean Elab Tactic

-- Default stub macros; certificate files redefine these for their specific module.
macro "simp_mat"     : tactic => `(tactic| simp)
macro "simp_defs"    : tactic => `(tactic| simp only [])
macro "mat_collapse" : tactic => `(tactic| simp only [])

syntax "zeroth_hammer" : tactic

/-- Zeroth hammer: cascading automated prover for reactive module goals.
    Phase 0: simp alone (closes trivially-True invariants)
    Phase 1: fast arithmetic — omega, norm_cast+omega, simp+omega, simp+linarith
    Phase 2: push_neg + simp + omega (negated arithmetic)
    Phase 3: simp + deep case-split + omega/linarith/norm_cast (branching goals)
    Phase 4: simp_defs + case-split (unfold defs before splitting)
    Phase 5: full reduction + mat_collapse + split_ifs + omega/linarith (hrank)
    Phase 6: aesop (general-purpose proof search)
    Phase 7: smt fallback (cvc5)
    Phase 8: sorry (explicit give-up) -/
elab_rules : tactic
  | `(tactic| zeroth_hammer) => do
      -- Phase 0: simp_mat alone (closes trivial True goals without needing omega)
      -- Note: simp never throws when it makes partial progress, so we must
      -- check goals explicitly rather than relying on try/return/catch.
      try evalTactic (← `(tactic| simp_mat)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      -- Phase 1: fast arithmetic passes
      -- 1a: omega alone (goal already in linear arithmetic fragment after intros)
      try evalTactic (← `(tactic| omega)); return catch _ => pure ()
      -- 1b: norm_cast + omega (normalises Nat/Int coercions, e.g. Int.toNat in rankings)
      try evalTactic (← `(tactic| norm_cast <;> omega)); return catch _ => pure ()
      -- 1c: simp_mat + omega (main fast path)
      try
        evalTactic (← `(tactic| simp_mat <;> omega))
        return
      catch _ => pure ()
      -- 1d: simp_mat + linarith (ordered-ring arithmetic; fallback when omega is too weak)
      try
        evalTactic (← `(tactic| simp_mat <;> linarith))
        return
      catch _ => pure ()
      -- Phase 2: push_neg normalises negated arithmetic before omega
      -- (e.g. ¬(x > 0) → x ≤ 0; useful when inv or P contains negated comparisons)
      try
        evalTactic (← `(tactic| push_neg; simp_mat <;> omega))
        return
      catch _ => pure ()
      -- Phase 3: simp_mat + deep case-split cascade (branching state machines)
      -- Four levels of if-branch splitting; tries omega/linarith/norm_cast at each leaf
      try
        evalTactic (← `(tactic|
          simp_mat
          <;> first
            | omega
            | linarith
            | (norm_cast; omega)
            | (push_neg; omega)
            | (simp_all; omega)
            | (split <;> simp_all <;> omega)
            | (split <;> split <;> simp_all <;> omega)
            | (split <;> split <;> split <;> simp_all <;> omega)
            | (split <;> split <;> split <;> split <;> simp_all <;> omega)))
        return
      catch _ => pure ()
      -- Phase 4: unfold definitions everywhere first, then case-split
      -- Useful when inv/P/ranking are not yet visible to the split heuristic
      try
        evalTactic (← `(tactic|
          simp_defs
          <;> first
            | omega
            | linarith
            | (norm_cast; omega)
            | (simp_all; omega)
            | (split <;> simp_all <;> omega)
            | (split <;> split <;> simp_all <;> omega)
            | (split <;> split <;> split <;> simp_all <;> omega)))
        return
      catch _ => pure ()
      -- Phase 5: full pipeline for ranking proofs
      -- Unfold defs in hypotheses → reduce matrices → collapse Mat 1 1 to scalar
      -- → split all ite → omega / linarith / norm_cast
      try evalTactic (← `(tactic| simp_defs)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      -- Check for contradictory hypotheses (e.g. ¬True from vacuous hrank)
      try evalTactic (← `(tactic| contradiction)); return catch _ => pure ()
      -- Try decide/native_decide after full reduction (works for finite Bool state)
      try evalTactic (← `(tactic| decide)); return catch _ => pure ()
      try evalTactic (← `(tactic| native_decide)); return catch _ => pure ()
      -- Reduce matrices and collapse Mat 1 1 to bare scalar arithmetic
      try evalTactic (← `(tactic| simp_mat)) catch _ => pure ()
      try evalTactic (← `(tactic| mat_collapse)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      try
        evalTactic (← `(tactic|
          split_ifs at *
          <;> first
            | omega
            | linarith
            | (norm_cast; omega)
            | (norm_cast; linarith)
            | simp_all
            | (simp_all; omega)
            | (simp_all; linarith)
            | positivity))
        return
      catch _ => pure ()
      -- Phase 6: aesop (general-purpose proof search before SMT)
      try
        evalTactic (← `(tactic| aesop))
        return
      catch _ => pure ()
      -- Phase 7: smt after full reduction
      try
        evalTactic (← `(tactic| smt))
        return
      catch _ => pure ()
      -- Phase 8: sorry (explicit give-up)
      evalTactic (← `(tactic| sorry))
