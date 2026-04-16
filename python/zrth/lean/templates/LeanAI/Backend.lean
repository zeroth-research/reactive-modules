/-
Copyright (c) 2024. All rights reserved.
Released under Apache 2.0 license.
Authors: Your Name

Abstract backend interface for LLM providers.
-/
import LeanAI.Types

namespace LeanAI

/-- Abstract interface for LLM backends -/
class Backend (β : Type) where
  /-- Name of this backend (for logging) -/
  name : String

  /-- Generate proof tactics for a goal -/
  generateProof (backend : β) (goal : String) (context : String) (config : Config) : IO (Result String)

  /-- Health check - verify backend is available -/
  healthCheck (backend : β) : IO Bool

namespace Backend

/-- Environment variable names -/
def envVarBackend : String := "LEAN_AI_BACKEND"
def envVarModel : String := "LEAN_AI_MODEL"
def envVarOllamaUrl : String := "OLLAMA_URL"
def envVarAnthropicKey : String := "ANTHROPIC_API_KEY"
def envVarOpenAIKey : String := "OPENAI_API_KEY"
def envVarOpenAIUrl : String := "OPENAI_API_URL"

/-- Get configuration from environment variables -/
def getConfigFromEnv : IO Config := do
  let backendOpt ← IO.getEnv envVarBackend
  let backendName := backendOpt.getD "ollama"

  -- Get model from unified LEAN_AI_MODEL variable
  let modelOpt ← IO.getEnv envVarModel

  let model := match backendName.toLower with
    | "openai" => modelOpt.getD "gpt-4"
    | "claude" => modelOpt.getD "claude-sonnet-4-5-20251101"
    | _ => modelOpt.getD "qwen2.5:32b"

  -- Auto-replace configuration
  let autoReplace ← IO.getEnv "LEAN_AI_AUTO_REPLACE"

  return {
    backend := backendName
    model := model
    autoReplace := match autoReplace with
      | some "false" => false  -- Explicitly disabled
      | _ => true              -- Default to true (enabled)
  }

end Backend

end LeanAI
