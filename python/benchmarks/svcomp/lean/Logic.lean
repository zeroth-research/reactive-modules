/-
  Denotational semantics for existentially-quantified linear formulas.

  A `Form` is a list of predicates over integer-valued variables. Its denotation is the proposition
  that *there exists* an assignment of integers to variable names making every predicate hold — i.e.
  all variables are implicitly existentially quantified.

  Everything used here (Int, List, String, ∃, ∧, ≤, <) is in Lean core, so no imports are required.
-/

/-- Arithmetic expressions over integer-valued string variables. -/
inductive Expr : Type where
  | var   : String → Expr
  | lit   : Int → Expr
  | add   : Expr → Expr → Expr
  | scale : Int → Expr → Expr

open Expr

inductive Pred : Type where
  | lessThanOrEquals : Expr → Expr → Pred
  | lessThan : Expr → Expr → Pred
  | equals : Expr → Expr → Pred

open Pred

abbrev Form : Type := List Pred

def example₁ : Form :=
  [lessThan (var "x") (var "y"), lessThan (lit 10) (var "x")]

abbrev Env : Type := String → Int

def denoteExpr (env : Env) : Expr → Int
  | var x     => env x
  | lit n     => n
  | add a b   => denoteExpr env a + denoteExpr env b
  | scale k e => k * denoteExpr env e

def denotePred (env : Env) : Pred → Prop
  | lessThanOrEquals a b => denoteExpr env a ≤ denoteExpr env b
  | lessThan a b         => denoteExpr env a < denoteExpr env b
  | equals a b           => denoteExpr env a = denoteExpr env b

def denoteForm (env : Env) : Form → Prop
  | []      => True
  | p :: ps => denotePred env p ∧ denoteForm env ps

def denote (f : Form) : Prop :=
  ∃ env : Env, denoteForm env f

example : denote example₁ := by
  refine ⟨fun s => if s = "x" then 11 else 12, ?_⟩
  -- unfold the denotation to concrete integer arithmetic, then decide.
  -- (`decide` alone can't synthesise `Decidable` through the `denote*` defs.)
  simp only [example₁, denoteForm, denotePred, denoteExpr]; decide