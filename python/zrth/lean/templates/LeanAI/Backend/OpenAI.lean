/-
Copyright (c) 2024. All rights reserved.
Released under Apache 2.0 license.
Authors: Your Name

OpenAI-compatible API backend implementation.
Works with OpenAI API, LMStudio, and other OpenAI-compatible services.
-/
import LeanAI.Backend
import LeanAI.HTTP
import LeanAI.JSON

namespace LeanAI.Backend

/-- OpenAI-compatible backend -/
structure OpenAIBackend where
  apiKey : String
  url : String := "https://api.openai.com/v1/chat/completions"

namespace OpenAIBackend

/-- Default OpenAI API endpoint -/
def defaultUrl : String := "https://api.openai.com/v1/chat/completions"

/-- Create an OpenAI backend instance -/
def create : IO OpenAIBackend := do
  let apiKey ← IO.getEnv Backend.envVarOpenAIKey
  let url ← IO.getEnv Backend.envVarOpenAIUrl
  return {
    apiKey := apiKey.getD ""
    url := url.getD defaultUrl
  }

/-- Build OpenAI API request body -/
def buildRequest (model : String) (prompt : String) : String :=
  let messageObj := JSON.mkObjectRaw [("role", "user"), ("content", prompt)]
  JSON.mkObjectRaw [
    ("model", model),
    ("messages", s!"[{messageObj}]"),
    ("temperature", "0.7"),
    ("max_tokens", "1000")
  ]

/-- Extract response text from OpenAI JSON response -/
def parseResponse (jsonStr : String) : IO (Result String) := do
  match JSON.parse jsonStr with
  | Except.error err =>
      return Result.failure s!"Failed to parse JSON: {err}"
  | Except.ok json =>
      -- Navigate: json.choices[0].message.content
      -- First get the choices array
      match json.getObjVal? "choices" with
      | Except.ok choicesJson =>
          match choicesJson.getArr? with
          | Except.ok choicesArr =>
              if choicesArr.size > 0 then
                let choice := choicesArr[0]!
                -- Get message object from the choice
                match choice.getObjVal? "message" with
                | Except.ok messageJson =>
                    -- Get content from message
                    match messageJson.getObjVal? "content" with
                    | Except.ok contentJson =>
                        match contentJson.getStr? with
                        | Except.ok content => return Result.success content
                        | Except.error _ => return Result.failure "Content is not a string"
                    | Except.error _ => return Result.failure "No 'content' field in message"
                | Except.error _ => return Result.failure "No 'message' field in choice"
              else
                return Result.failure "Empty choices array"
          | Except.error _ => return Result.failure "Choices is not an array"
      | Except.error _ => return Result.failure "No 'choices' field in OpenAI response"

/-- Health check for OpenAI service -/
def healthCheck (backend : OpenAIBackend) : IO Bool := do
  -- Check if API key is set
  if backend.apiKey.isEmpty then
    IO.println "[OpenAI] Warning: API key not set (OPENAI_API_KEY environment variable)"
    return false
  return true

/-- Build prompt for OpenAI -/
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

/-- Generate proof using OpenAI API -/
def generateProof (backend : OpenAIBackend) (goal : String) (context : String) (config : Config) : IO (Result String) := do
  try
    -- Build the prompt
    let prompt := buildPrompt goal context

    if config.verbose then
      IO.println s!"[OpenAI] Prompt:\n{prompt}\n"

    -- Build request body
    let requestBody := buildRequest config.model prompt

    -- Prepare headers for authentication
    let headers := [
      ("Authorization", s!"Bearer {backend.apiKey}"),
      ("Content-Type", "application/json")
    ]

    -- Make the HTTP request
    IO.println s!"[OpenAI] Calling {backend.url} with model {config.model}..."
    let responseStr ← HTTP.postWithHeaders backend.url requestBody headers

    if config.verbose then
      IO.println s!"[OpenAI] Raw response:\n{responseStr}\n"

    -- Parse response
    let result ← parseResponse responseStr
    match result with
    | Result.success tactics =>
        IO.println s!"[OpenAI] Generated tactics: {tactics}"
        return Result.success tactics.trim
    | Result.failure err =>
        return Result.failure err

  catch e =>
    return Result.failure s!"OpenAI error: {e}"

instance : Backend OpenAIBackend where
  name := "OpenAI"
  generateProof := generateProof
  healthCheck := healthCheck

end OpenAIBackend

end LeanAI.Backend
