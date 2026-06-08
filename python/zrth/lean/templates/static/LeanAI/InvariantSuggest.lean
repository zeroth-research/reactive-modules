/-
Copyright (c) 2024. All rights reserved.
Released under Apache 2.0 license.

Invariant suggestion using LLM analysis.
Automatically discovers and suggests stronger invariants for state machines.
-/
import Lean
import LeanAI.Backend
import LeanAI.Backend.Ollama
import LeanAI.Backend.OpenAI
import LeanAI.Backend.Claude

open Lean Elab Command Meta

namespace LeanAI.InvariantSuggest

/-- Build a prompt for invariant suggestion -/
def buildInvariantPrompt (stateDef : String) (initDef : String) (updateDef : String) (weakInvariant : String) : String :=
  s!"You are a Lean 4 formal verification expert analyzing a state machine.

STATE STRUCTURE:
{stateDef}

INITIALIZATION FUNCTION:
{initDef}

UPDATE FUNCTION:
{updateDef}

CURRENT WEAK INVARIANT:
{weakInvariant}

ANALYSIS STEPS:
1. Look at the State structure and identify ALL field names (e.g., x0, x1, x2)
2. Look at init: What values are fields initialized to?
   - If a field is set to 0 or Int.natAbs, it's non-negative
   - Note which fields never change
3. Look at update: How are fields modified?
   - If a field resets to 0, you need to prove 0 satisfies the original invariant
   - This often requires non-negativity constraints on other fields
4. The weak invariant fails because when update resets a field to 0, we can't prove the original condition holds

TASK: Suggest a STRONGER invariant by:
1. Taking the original invariant condition exactly as written
2. Adding conjunction (∧) with additional constraints:
   - Non-negativity for fields that are initialized with Int.natAbs or 0
   - Any bounds implied by initialization
   - Relationships that are preserved

CRITICAL: This is NOT a tactic proof! DO NOT output tactics like 'intro', 'constructor', 'exact'.
You MUST output a DEFINITION, not tactics!

WRONG (tactics - DON'T DO THIS):
intro s
constructor
exact s.x0 ≥ 0

CORRECT (definition - DO THIS):
def strong_invariant : State → Prop :=
  fun ⟨x0, x1, x2⟩ ↦
    (x0 ≤ x1 ∨ x0 ≤ x2) ∧ x0 ≥ 0 ∧ x1 ≥ 0 ∧ x2 ≥ 0

OUTPUT REQUIREMENTS:
- Start with: def strong_invariant : State → Prop :=
- Use fun ⟨...⟩ ↦ pattern matching
- Use the EXACT field names from the State structure above
- Include the original invariant condition with parentheses
- Add ∧ (and) for additional constraints
- Output ONLY the definition - no text before or after

Now analyze the state machine above and write ONLY the definition (starting with 'def'):"

/-- Extract definition as string -/
def getDefAsString (name : Name) : CommandElabM String := do
  let env ← getEnv
  match env.find? name with
  | none => throwError s!"Definition {name} not found"
  | some constInfo =>
      let fmt ← liftTermElabM do
        -- For structures, just get the type
        if constInfo.isInductive then
          let typeFmt ← PrettyPrinter.ppExpr constInfo.type
          return s!"structure {name} : {typeFmt}"
        -- For definitions with values
        else if let some value := constInfo.value? then
          let valueFmt ← PrettyPrinter.ppExpr value
          let typeFmt ← PrettyPrinter.ppExpr constInfo.type
          return s!"def {name} : {typeFmt} := {valueFmt}"
        else
          let typeFmt ← PrettyPrinter.ppExpr constInfo.type
          return s!"{name} : {typeFmt}"
      return fmt

/-- Suggest a stronger invariant using LLM -/
def suggestInvariant (stateName initName updateName weakInvName : Name) : CommandElabM String := do
  -- Get the definitions
  let stateDef ← getDefAsString stateName
  let initDef ← getDefAsString initName
  let updateDef ← getDefAsString updateName
  let weakInvDef ← getDefAsString weakInvName

  -- Build the prompt
  let prompt := buildInvariantPrompt stateDef initDef updateDef weakInvDef

  -- Get config and create backend
  let config ← Backend.getConfigFromEnv

  -- Call the LLM
  logInfo "[InvariantSuggest] Analyzing state machine..."
  logInfo s!"[InvariantSuggest] Calling LLM with model {config.model}..."

  let result ← match config.backend.toLower with
    | "ollama" => do
        let backend ← Backend.OllamaBackend.create
        Backend.generateProof backend
          "Suggest a stronger invariant for this state machine"
          prompt
          config
    | "openai" => do
        let backend ← Backend.OpenAIBackend.create
        Backend.generateProof backend
          "Suggest a stronger invariant for this state machine"
          prompt
          config
    | "claude" => do
        let backendOpt ← Backend.ClaudeBackend.create
        match backendOpt with
        | some backend =>
            Backend.generateProof backend
              "Suggest a stronger invariant for this state machine"
              prompt
              config
        | none => pure (Result.failure "Claude backend requires ANTHROPIC_API_KEY environment variable")
    | _ => pure (Result.failure s!"Unknown backend: {config.backend}. Valid options: ollama, openai, claude")

  match result with
  | Result.success suggestion =>
      return suggestion.trim
  | Result.failure err =>
      throwError s!"Failed to generate invariant suggestion: {err}"

/-- Clean up the LLM response to extract just the definition -/
def cleanInvariantSuggestion (suggestion : String) : String :=
  suggestion.trim
    |>.replace "```lean" ""
    |>.replace "```" ""
    |>.trim

/-- Syntax for the #suggest_invariant command -/
syntax "#suggest_invariant" ident ident ident ident : command

/-- Implementation of #suggest_invariant command -/
elab_rules : command
  | `(command| #suggest_invariant $state $init $update $weakInv) => do
      let stateName := state.getId
      let initName := init.getId
      let updateName := update.getId
      let weakInvName := weakInv.getId

      try
        -- Call the LLM to suggest an invariant
        let suggestion ← suggestInvariant stateName initName updateName weakInvName
        let cleaned := cleanInvariantSuggestion suggestion

        -- Display the suggestion
        logInfo "\n╔══════════════════════════════════════════════════════════════╗"
        logInfo "║           🤖 SUGGESTED STRONGER INVARIANT 🤖                 ║"
        logInfo "╚══════════════════════════════════════════════════════════════╝\n"
        logInfo s!"{cleaned}\n"
        logInfo "💡 NEXT STEPS:"
        logInfo "1. Review the suggested invariant"
        logInfo "2. Copy it to your file"
        logInfo "3. Prove: theorem init_strong : strong_invariant (init ep) := by ai_solve"
        logInfo "4. Prove: theorem update_strong : strong_invariant s → strong_invariant (update s)"
        logInfo "\n✨ The stronger invariant should make your proofs easier!"

      catch e =>
        let msg ← e.toMessageData.toString
        logError s!"Error suggesting invariant: {msg}"

end LeanAI.InvariantSuggest
