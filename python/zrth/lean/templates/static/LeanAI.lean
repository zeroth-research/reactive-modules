/-
Copyright (c) 2024. All rights reserved.
Released under Apache 2.0 license.

Main entry point for the LeanAI tactic library.
Import this file to use the ai_solve tactic in your proofs.
-/
import LeanAI.Types
import LeanAI.Backend
import LeanAI.Backend.Ollama
import LeanAI.Backend.Claude
import LeanAI.Backend.OpenAI
import LeanAI.ContextAnalysis
import LeanAI.Tactic
import LeanAI.InvariantSuggest

-- The ai_solve tactic and #suggest_invariant command are automatically available when importing this module
