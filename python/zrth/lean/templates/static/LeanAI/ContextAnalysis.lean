/-
Copyright (c) 2024. All rights reserved.
Released under Apache 2.0 license.
Authors: Your Name

Context analysis for extracting relevant information from definitions.
-/
import Lean

namespace LeanAI.ContextAnalysis

open Lean Meta Elab

/-- Extract function names mentioned in an expression -/
partial def extractFunctionNames (e : Expr) : MetaM (Array Name) := do
  let mut names : Array Name := #[]

  -- Recursively traverse the expression
  let rec visit (expr : Expr) : StateT (Array Name) MetaM Unit := do
    match expr with
    | Expr.const name _ =>
        -- Found a constant (function name)
        modify fun ns => ns.push name
    | Expr.app f arg =>
        visit f
        visit arg
    | Expr.lam _ ty body _ =>
        visit ty
        visit body
    | Expr.forallE _ ty body _ =>
        visit ty
        visit body
    | Expr.letE _ ty val body _ =>
        visit ty
        visit val
        visit body
    | Expr.proj _ _ struct =>
        visit struct
    | Expr.mdata _ e =>
        visit e
    | _ => pure ()

  let (_, finalNames) ← StateT.run (visit e) names
  return finalNames

/-- Check if a string contains a substring -/
def String.containsSubstr (s : String) (sub : String) : Bool :=
  let parts := s.splitOn sub
  parts.length > 1

/-- Extract if-condition from an if-expression recursively -/
partial def extractIfCondition (e : Expr) : MetaM (List Expr) := do
  let rec visit (expr : Expr) : StateT (List Expr) MetaM Unit := do
    match expr with
    | Expr.app (Expr.app (Expr.app (Expr.app (Expr.app (Expr.const `ite _) _) cond) _) thenBranch) elseBranch =>
        -- Found if-then-else: ite type condition decidableInst thenBranch elseBranch
        -- The condition is the 2nd argument (after type)
        modify (cond :: ·)
        -- Recursively search in branches
        visit thenBranch
        visit elseBranch
    | Expr.app f arg =>
        visit f
        visit arg
    | Expr.lam _ _ body _ =>
        visit body
    | Expr.letE _ _ val body _ =>
        visit val
        visit body
    | _ => pure ()

  let (_, conditions) ← StateT.run (visit e) []
  return conditions.reverse

/-- Clean up an expression for pretty printing (remove typeclass instances) -/
partial def cleanupExpr (e : Expr) : Expr :=
  -- Just clean up annotations and traverse the structure
  match e.cleanupAnnotations with
  | Expr.mdata _ inner => cleanupExpr inner
  -- If it's Decidable applied to something, unwrap it
  | Expr.app (Expr.const `Decidable _) arg => cleanupExpr arg
  | expr => expr

/-- Try to find if-conditions in a function definition -/
def extractIfConditions (defName : Name) : MetaM (Array String) := do
  let env ← getEnv

  -- Try to get the definition
  match env.find? defName with
  | none => return #[]
  | some info =>
      match info with
      | ConstantInfo.defnInfo val =>
          -- Get the value (body) and type of the definition
          let body := val.value
          let type := val.type

          -- Use lambdaTelescope to bring function parameters into scope
          Meta.lambdaTelescope body fun params bodyWithParams => do
            -- Look for if-expressions in the body
            let conditions ← extractIfCondition bodyWithParams
            if conditions.isEmpty then
              return #[]
            else
              -- Extract and format each condition
              let mut results : Array String := #[]
              for cond in conditions do
                -- Clean up the condition expression
                let cleanCond := cleanupExpr cond
                -- Instantiate any metavariables
                let instantiatedCond ← instantiateMVars cleanCond
                -- Pretty print it with parameters in scope
                let condStr ← Meta.ppExpr instantiatedCond
                let condText := toString condStr

                -- Return both the hint and the exact condition
                results := results.push s!"Function '{defName}' has if-condition: {condText}"
                results := results.push s!"Proof pattern: (1) unfold definitions, (2) obtain to destructure ∧, (3) by_cases hcond : {condText} <;> (simp [hcond] <;> omega)"
              return results
      | _ => return #[]

/-- Extract relevant context hints from the goal -/
def extractContextHints (goal : MVarId) : MetaM (Array String) := do
  -- Get the goal type
  let goalType ← goal.getType

  -- Extract function names from the goal
  let funcNames ← extractFunctionNames goalType

  -- For each function, try to extract if-conditions
  let mut hints : Array String := #[]

  for name in funcNames do
    let conditions ← extractIfConditions name
    hints := hints ++ conditions

  -- Check local hypotheses for conjunctions that should be destructured
  let lctx ← getLCtx
  for localDecl in lctx do
    if localDecl.isImplementationDetail then continue
    let type ← instantiateMVars localDecl.type
    -- Check if it's a conjunction (And)
    if type.isAppOf `And then
      -- Count the number of conjuncts
      let rec countConjuncts (e : Expr) : Nat :=
        match e with
        | Expr.app (Expr.app (Expr.const `And _) left) right =>
            countConjuncts left + countConjuncts right
        | _ => 1
      let numParts := countConjuncts type
      hints := hints.push s!"CRITICAL: Hypothesis '{localDecl.userName}' is a conjunction with {numParts} parts - MUST use obtain ⟨var1, var2, ...⟩ := {localDecl.userName} with {numParts} variables"

  return hints

end LeanAI.ContextAnalysis
