/-
Copyright (c) 2024. All rights reserved.
Released under Apache 2.0 license.
Authors: Your Name

Main tactic implementation for AI-powered proof generation.
-/
import Lean
import Lean.Meta.Tactic.TryThis
import LeanAI.Types
import LeanAI.Backend
import LeanAI.Backend.Ollama
import LeanAI.Backend.Claude
import LeanAI.Backend.OpenAI
import LeanAI.ContextAnalysis

open Lean Elab Tactic Meta Lean.Meta.Tactic.TryThis

namespace LeanAI.Tactic

/-- Source location information for auto-replacement -/
structure SourceLocation where
  fileName : System.FilePath
  startPos : String.Pos.Raw
  endPos : String.Pos.Raw
  fileMap : FileMap

/-- Register trace option for ai_solve tactic -/
register_option trace.ai_solve : Bool := {
  defValue := false
  descr := "Enable tracing for ai_solve tactic"
}

/-- Extract the current goal as a string -/
def getCurrentGoalString : TacticM String := do
  let goal ← getMainGoal
  let goalType ← goal.getType
  let goalStr ← Meta.ppExpr goalType
  return toString goalStr

/-- Extract local context as a string -/
def getContextString : TacticM String := do
  let goal ← getMainGoal
  let lctx ← goal.getDecl
  let mut contextStr := ""
  for localDecl in lctx.lctx do
    if !localDecl.isImplementationDetail then
      let name := localDecl.userName
      let type ← Meta.ppExpr localDecl.type
      contextStr := contextStr ++ s!"{name} : {type}\n"
  return contextStr

/-- Clean up tactics string from LLM response -/
def cleanTactics (tactics : String) : String :=
  let cleaned := tactics.trim
    |>.replace "```lean" ""
    |>.replace "```" ""
    |>.replace "<tactics>" ""
    |>.replace "</tactics>" ""
    |>.trim
  -- Split by newlines and filter out empty lines and comments
  let lines := cleaned.splitOn "\n"
    |>.map String.trim
    |>.filter (fun s => !s.isEmpty && !s.startsWith "--" && !s.startsWith "//")  -- Remove comment lines
  -- Join with newlines (Lean tactics can be on separate lines)
  String.intercalate "\n" lines

/-- Calculate indentation of the ai_solve line -/
def getIndentation (loc : SourceLocation) (content : String) : Nat :=
  let pos := loc.startPos
  let linePos := loc.fileMap.toPosition pos
  let lineStart := loc.fileMap.lineStart linePos.line
  let lineContent := String.Pos.Raw.extract content lineStart pos
  (lineContent.takeWhile (· == ' ')).length

/-- Format tactics with proper indentation and comment -/
def formatTacticsWithIndent (tactics : String) (indent : Nat) : String :=
  let indentStr := String.ofList (List.replicate indent ' ')
  let lines := tactics.splitOn "\n" |>.map String.trim |>.filter (!·.isEmpty)
  let formattedLines := lines.map (indentStr ++ ·)
  String.intercalate "\n" formattedLines ++ s!"\n{indentStr}-- proof generated with ai_solve"

/-- Check if running in interactive IDE mode -/
def isInteractiveMode : IO Bool := do
  -- Check for VS Code or other IDE environment
  let lspMode ← IO.getEnv "LEAN_LSP_MODE"
  let vscode ← IO.getEnv "VSCODE_PID"
  return lspMode.isSome || vscode.isSome

/-- Replace ai_solve with generated tactics in source file -/
def tryReplaceInFile (loc : SourceLocation) (tactics : String) (config : Config) : TacticM Unit := do
  if !config.autoReplace then
    return

  -- Check if we should skip IDE mode
  let forceReplace ← IO.getEnv "LEAN_AI_FORCE_REPLACE"
  let interactive ← isInteractiveMode
  if interactive && forceReplace.isNone then
    logInfo "[AI] Note: Auto-replace disabled in IDE mode. Set LEAN_AI_FORCE_REPLACE=true to enable (requires manual file reload)."
    return

  try
    -- Read entire file
    let content ← IO.FS.readFile loc.fileName

    -- Calculate indentation and format replacement
    let indent := getIndentation loc content
    let replacement := formatTacticsWithIndent tactics indent

    -- Find the start of the line containing ai_solve (to avoid double indentation)
    let linePos := loc.fileMap.toPosition loc.startPos
    let lineStart := loc.fileMap.lineStart linePos.line

    -- Build new content (before line + replacement + after)
    let before := String.Pos.Raw.extract content 0 lineStart
    let after := String.Pos.Raw.extract content loc.endPos content.rawEndPos
    let newContent := before ++ replacement ++ after

    -- Write back atomically
    IO.FS.writeFile loc.fileName newContent
    logInfo s!"[AI] ✓ Automatically replaced ai_solve in source file"

  catch e =>
    let msg ← e.toMessageData.toString
    logWarning s!"[AI] Warning: Could not auto-replace (file unchanged): {msg}"

/-- Parse and apply tactics from LLM response -/
def applyTactics (tacticsStr : String) (loc : SourceLocation) (config : Config) (tacticStx : Syntax) : TacticM (Option String) := do
  let cleanedTactics := cleanTactics tacticsStr

  logInfo s!"[AI] Applying tactics:\n{cleanedTactics}"

  -- Split into individual tactic lines
  let tacticLines := cleanedTactics.splitOn "\n" |>.map String.trim |>.filter (fun s => !s.isEmpty)

  -- Track successful tactics
  let mut successfulTactics : Array String := #[]

  -- Apply each tactic line sequentially
  for tacticLine in tacticLines do
    -- Check for errors before applying each tactic
    let msgsBefore ← Core.getMessageLog

    try
      match Parser.runParserCategory (← getEnv) `tactic tacticLine with
      | Except.ok stx =>
          try
            evalTactic stx

            -- Check if any errors were added to the message log
            let msgsAfter ← Core.getMessageLog
            let newMsgs := msgsAfter.toList.drop msgsBefore.toList.length
            let errors := newMsgs.filter (·.severity == MessageSeverity.error)
            if !errors.isEmpty then
              let errorMsg ← errors.head!.data.toString
              -- Show partial progress before returning error
              if !successfulTactics.isEmpty then
                let copyableTactics := successfulTactics.toList.map (fun t => t.replace " at *" "")
                let skeleton := String.intercalate "\n  " copyableTactics ++ "\n  sorry"
                logInfo s!"[AI] ✓ Partial progress - Copy this to continue manually:\n  {skeleton}"
              return some s!"Tactic '{tacticLine}' produced error: {errorMsg}"

            -- Tactic succeeded!
            successfulTactics := successfulTactics.push tacticLine

            -- Check if proof is complete after this tactic
            let goals ← getGoals
            if goals.isEmpty then
              let copyableTactics := successfulTactics.toList.map (fun t => t.replace " at *" "")
              let tacticsStr := String.intercalate "\n" copyableTactics

              -- Show copy-ready tactics in InfoView
              logInfo s!"[AI] ✓ Proof successful! Click the suggestion below to replace ai_solve:\n{tacticsStr}\n-- proof generated with ai_solve"

              -- Add clickable TryThis suggestion for VS Code
              let proofCode := tacticsStr ++ "\n-- proof generated with ai_solve"
              let suggestion : Suggestion := { suggestion := proofCode }
              addSuggestion tacticStx suggestion (origSpan? := tacticStx)

              -- Auto-replace in file (CLI only)
              tryReplaceInFile loc tacticsStr config

              return none  -- Success! Don't apply remaining tactics

          catch e =>
            let msg ← e.toMessageData.toString
            -- Show partial progress before returning error
            if !successfulTactics.isEmpty then
              let copyableTactics := successfulTactics.toList.map (fun t => t.replace " at *" "")
              let skeleton := String.intercalate "\n  " copyableTactics ++ "\n  sorry"
              logInfo s!"[AI] ✓ Partial progress - Copy this to continue manually:\n  {skeleton}"
            return some s!"Tactic '{tacticLine}' failed: {msg}"
      | Except.error err =>
          -- Show partial progress before returning error
          if !successfulTactics.isEmpty then
            let copyableTactics := successfulTactics.toList.map (fun t => t.replace " at *" "")
            let skeleton := String.intercalate "\n  " copyableTactics ++ "\n  sorry"
            logInfo s!"[AI] ✓ Partial progress - Copy this to continue manually:\n  {skeleton}"
          return some s!"Parse error in '{tacticLine}': {err}"
    catch e =>
      let msg ← e.toMessageData.toString
      -- Show partial progress before returning error
      if !successfulTactics.isEmpty then
        let copyableTactics := successfulTactics.toList.map (fun t => t.replace " at *" "")
        let skeleton := String.intercalate "\n  " copyableTactics ++ "\n  sorry"
        logInfo s!"[AI] ✓ Partial progress - Copy this to continue manually:\n  {skeleton}"
      return some s!"Error applying '{tacticLine}': {msg}"

  -- Check if all goals are solved
  let goals ← getGoals
  if goals.isEmpty then
    -- Show the successful tactics in InfoView (copyable version)
    let copyableTactics := successfulTactics.toList.map (fun t => t.replace " at *" "")
    let tacticsStr := String.intercalate "\n" copyableTactics

    -- Show copy-ready tactics in InfoView
    logInfo s!"[AI] ✓ Proof successful! Click the suggestion below to replace ai_solve:\n{tacticsStr}\n-- proof generated with ai_solve"

    -- Add clickable TryThis suggestion for VS Code
    let proofCode := tacticsStr ++ "\n-- proof generated with ai_solve"
    let suggestion : Suggestion := { suggestion := proofCode }
    addSuggestion tacticStx suggestion (origSpan? := tacticStx)

    -- Auto-replace in file (CLI only)
    tryReplaceInFile loc tacticsStr config

    return none  -- Success
  else
    -- Show partial progress with skeleton proof
    if !successfulTactics.isEmpty then
      let copyableTactics := successfulTactics.toList.map (fun t => t.replace " at *" "")
      let skeleton := String.intercalate "\n  " copyableTactics ++ "\n  sorry"
      logInfo s!"[AI] ✓ Partial progress - Copy this to continue manually:\n  {skeleton}"
    return some s!"Tactics applied but {goals.length} goal(s) remain unsolved"

/-- Try to simplify the goal automatically before calling AI -/
def tryAutoSimplify : TacticM Bool := do
  try
    -- Try unfold + simp
    evalTactic (← `(tactic| try simp))
    return true
  catch _ =>
    return false

/-- AI solve with retry loop -/
def aiSolveTacticWithRetry (config : Config) (loc : SourceLocation) (tacticStx : Syntax) (maxRetries : Nat := 3) : TacticM Unit := do
  logInfo "[AI] Starting AI proof generation..."

  -- Extract goal and context
  let goal ← getCurrentGoalString
  let context ← getContextString

  logInfo s!"[AI] Goal: {goal}"
  if config.verbose then
    logInfo s!"[AI] Context:\n{context}"

  -- Extract context hints (Phase 6: Context Optimization)
  let mainGoal ← getMainGoal
  let hints ← ContextAnalysis.extractContextHints mainGoal
  let hintsStr := if hints.isEmpty then ""
    else s!"\n\nHelpful context from definitions:\n{String.intercalate "\n" hints.toList}"

  if config.verbose && !hints.isEmpty then
    logInfo s!"[AI] Extracted hints:{hintsStr}"

  -- Retry loop
  let mut attempt := 0
  let mut previousError : Option String := none
  let mut lastTactics := ""

  while attempt < maxRetries do
    attempt := attempt + 1
    logInfo s!"[AI] Attempt {attempt}/{maxRetries}"

    -- Save the tactic state before attempting tactics
    let savedState ← saveState

    -- Build error context for retry
    let errorContext := match previousError with
      | none => ""
      | some err => s!"\nPrevious attempt failed with error:\n{err}\nPrevious tactics:\n{lastTactics}\n\nPlease try a different approach."

    -- Call the backend to generate proof with context hints
    let fullContext := context ++ hintsStr ++ errorContext
    let result ← match config.backend.toLower with
      | "ollama" => do
          let backend ← Backend.OllamaBackend.create
          Backend.generateProof backend goal fullContext config
      | "openai" => do
          let backend ← Backend.OpenAIBackend.create
          Backend.generateProof backend goal fullContext config
      | "claude" => do
          let backendOpt ← Backend.ClaudeBackend.create
          match backendOpt with
          | some backend => Backend.generateProof backend goal fullContext config
          | none => pure (Result.failure "Claude backend requires ANTHROPIC_API_KEY environment variable")
      | _ => pure (Result.failure s!"Unknown backend: {config.backend}. Valid options: ollama, openai, claude")

    match result with
    | Result.success tactics =>
        lastTactics := tactics
        -- Try to apply the generated tactics
        let maybeError ← applyTactics tactics loc config tacticStx
        match maybeError with
        | none =>
            -- Success!
            logInfo "[AI] Proof completed successfully!"
            return
        | some err =>
            -- Tactic failed, restore state before retrying
            savedState.restore
            previousError := some err
            logInfo s!"[AI] Attempt {attempt} failed: {err}"
            if attempt < maxRetries then
              logInfo "[AI] Retrying with error feedback..."
    | Result.failure err =>
        throwError s!"AI backend error: {err}"

  -- All retries exhausted - show what was attempted
  let errorMsg := if !lastTactics.isEmpty then
    let copyableTactics := (cleanTactics lastTactics).splitOn "\n"
      |>.map String.trim
      |>.filter (fun s => !s.isEmpty)
      |>.map (fun t => t.replace " at *" "")
    let skeleton := String.intercalate "\n  " copyableTactics ++ "\n  sorry"
    s!"AI solve failed after {maxRetries} attempts.\n\nLast attempt (copy to continue manually):\n  {skeleton}\n\nLast error: {previousError.getD "unknown"}"
  else
    s!"AI solve failed after {maxRetries} attempts. Last error: {previousError.getD "unknown"}"

  throwError errorMsg

/-- Main AI solve tactic implementation -/
def aiSolveTactic (config : Config) (loc : SourceLocation) (tacticStx : Syntax) : TacticM Unit := do
  -- First, try auto-simplification
  let simplified ← tryAutoSimplify
  if simplified then
    logInfo "[AI] Applied automatic simplification"

  -- Check if goal is already solved
  let goals ← getGoals
  if goals.isEmpty then
    logInfo "[AI] Goal solved by simplification!"
    return

  -- Try AI with retry loop
  aiSolveTacticWithRetry config loc tacticStx 3

/-- Syntax for the ai_solve tactic -/
syntax "ai_solve" : tactic

/-- Implementation of ai_solve tactic -/
elab stx:"ai_solve" : tactic => do
  let config ← Backend.getConfigFromEnv
  let fileName ← getFileName
  let fileMap ← getFileMap
  let location : SourceLocation := {
    fileName := fileName
    startPos := stx.getPos?.getD 0
    endPos := stx.getTailPos?.getD (stx.getPos?.getD 0)
    fileMap := fileMap
  }
  aiSolveTactic config location stx

end LeanAI.Tactic
