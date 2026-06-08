/-
Copyright (c) 2024. All rights reserved.
Released under Apache 2.0 license.
Authors: Your Name

Ollama backend implementation for local LLM inference.
-/
import LeanAI.Backend
import LeanAI.HTTP
import LeanAI.JSON

namespace LeanAI.Backend

/-- Ollama backend for local LLM -/
structure OllamaBackend where
  url : String := "http://localhost:11434"

namespace OllamaBackend

/-- Default Ollama API endpoint -/
def defaultUrl : String := "http://localhost:11434"

/-- Create an Ollama backend instance -/
def create : IO OllamaBackend := do
  let url ← IO.getEnv Backend.envVarOllamaUrl
  return { url := url.getD defaultUrl }

/-- Build Ollama API request body -/
def buildRequest (model : String) (prompt : String) : String :=
  JSON.mkObjectRaw [
    ("model", model),
    ("prompt", prompt),
    ("stream", "false")
  ]

/-- Extract response text from Ollama JSON response -/
def parseResponse (jsonStr : String) : IO (Result String) := do
  match JSON.parse jsonStr with
  | Except.error err =>
      return Result.failure s!"Failed to parse JSON: {err}"
  | Except.ok json =>
      match JSON.getString json "response" with
      | some response => return Result.success response
      | none => return Result.failure "No 'response' field in Ollama output"

/-- Health check for Ollama service -/
def healthCheck (backend : OllamaBackend) : IO Bool := do
  try
    -- Try to connect to Ollama API
    let _ ← HTTP.get s!"{backend.url}/api/tags"
    return true
  catch _ =>
    return false

/-- Build prompt for Ollama -/
def buildPrompt (goal : String) (context : String) : String :=
  let contextStr := if context.isEmpty then "" else s!"Context:\n{context}\n\n"
  s!"You are a Lean 4 proof assistant. Generate ONLY the tactics needed to prove the following goal.

{contextStr}Goal: {goal}

CRITICAL RULES:
- Output ONLY valid Lean 4 tactic code - NO comments, NO explanations
- Do NOT include ANY lines starting with '--' or '//'
- Do NOT include 'theorem' or 'by' keywords
- Do NOT use markdown code blocks
- Do NOT use 'sorry'
- Do NOT reference lemma names (like natAbs_nonneg, le_refl, etc) - they might not exist
- Do NOT use 'apply' or 'exact' with specific lemma names
- For 'cases' tactic: use 'cases h' not 'cases s' or 'cases update s with us'

Available tactics:
- unfold, rfl, simp, omega, intro, cases, constructor, assumption, rw
- obtain: destructure hypotheses (e.g., obtain ⟨a, b⟩ := h)
- by_cases: case analysis on conditions (e.g., by_cases h : condition)
- split: split if-then-else in goal
- <;>: apply tactic to all subgoals - MUST be on ONE line
  - CORRECT: by_cases h : cond <;> (simp <;> omega)
  - WRONG: multi-line <;> expressions don't parse

Strategy:
1. Unfold definitions in hypotheses first: unfold my_def at *
2. IMPORTANT: After unfolding, if hypothesis has ∧ (and), you MUST use 'obtain' to destructure
   - Example: if h : A ∧ B ∧ C, use: obtain ⟨h1, h2, h3⟩ := h
   - ALWAYS count the ∧ symbols correctly - if there are 3 ∧, you need 4 variables
3. Unfold definitions in goal: unfold other_def
4. If goal has 'if' with conditions: use 'by_cases' with exact condition
5. Use <;> combinator to handle all branches: by_cases h : cond <;> (simp [h] <;> omega)
6. For by_cases with <;>, put continuation tactics in parentheses

Example 1 (simple goal):
unfold strong_invariant at *
unfold init
simp

Example 2 (with conjunctive hypothesis):
unfold my_def at *
obtain ⟨h1, h2, h3⟩ := h
omega

Example 3 (with if-then-else - COMPLETE SEQUENCE):
unfold strong_invariant at *
obtain ⟨hinv, hpos0, hpos1, hpos2⟩ := h
unfold update
by_cases hcond : s.x0 < s.x1 ∨ s.x0 < s.x2 <;> (simp [hcond] <;> omega)

Example 4 (WRONG - multi-line tactic):
unfold foo
by_cases h : cond <;> (
  simp <;>
  omega
)

Example 5 (WRONG - has comments):
unfold foo
-- this comment is WRONG
simp

Example 6 (WRONG - references non-existent lemma):
exact natAbs_nonneg _

Your tactics (each tactic on ONE line, NO comments!):"

/-- Generate proof using Ollama -/
def generateProof (backend : OllamaBackend) (goal : String) (context : String) (config : Config) : IO (Result String) := do
  try
    -- Build the prompt
    let prompt := buildPrompt goal context

    if config.verbose then
      IO.println s!"[Ollama] Prompt:\n{prompt}\n"

    -- Build request body
    let requestBody := buildRequest config.model prompt

    -- Make the HTTP request
    IO.println s!"[Ollama] Calling {backend.url}/api/generate with model {config.model}..."
    let responseStr ← HTTP.post s!"{backend.url}/api/generate" requestBody

    if config.verbose then
      IO.println s!"[Ollama] Raw response:\n{responseStr}\n"

    -- Parse response
    let result ← parseResponse responseStr
    match result with
    | Result.success tactics =>
        IO.println s!"[Ollama] Generated tactics: {tactics}"
        return Result.success tactics.trim
    | Result.failure err =>
        return Result.failure err

  catch e =>
    return Result.failure s!"Ollama error: {e}"

instance : Backend OllamaBackend where
  name := "Ollama"
  generateProof := generateProof
  healthCheck := healthCheck

end OllamaBackend

end LeanAI.Backend
