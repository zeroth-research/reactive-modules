import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Basic

/-! A simple version of wiring diagrams. The main object
    is `Box` which is a node in the wiring diagram.
    There are no explicit wires, the diagrams are represented
    as terms. That is, a `Box` is essentially just a function
    annotated with its domain and codomain so that we can
    define and reason about sequential and parallel composition
    on Boxes.
-/

/-- Data type of values a Box can work with.
    TODO: should we just use `Type` directly?
-/
inductive Ty where
  | int
  | bool
  | mat (m n : Nat)
  deriving DecidableEq, Repr

abbrev Ty.denote : Ty → Type
  | .int  => Int
  | .bool => Bool
  | .mat m n => Fin m → Fin n → Int

instance (t : Ty) : DecidableEq t.denote := by
  cases t <;> simp [Ty.denote] <;> infer_instance

/-- tuple of values -/
@[simp] def ValTuple : List Ty → Type
  | []      => Unit
  | t :: ts => t.denote × ValTuple ts

instance : DecidableEq (ValTuple []) := fun () () => .isTrue rfl

instance [DecidableEq (t.denote)] [DecidableEq (ValTuple ts)] :
    DecidableEq (ValTuple (t :: ts)) :=
  inferInstanceAs (DecidableEq (t.denote × ValTuple ts))

namespace ValTuple

@[simp] def append : (xs : List Ty) → ValTuple xs → ValTuple ys → ValTuple (xs ++ ys)
  | [],     (),       bs => bs
  | _::xs, (a, rest), bs => (a, append xs rest bs)

@[simp] def split : (xs : List Ty) → ValTuple (xs ++ ys) → ValTuple xs × ValTuple ys
  | [],     h         => ((), h)
  | _::xs, (a, rest)  =>
    let (l, r) := split xs rest
    ((a, l), r)

@[simp] theorem split_nil (s : ValTuple ts) :
    (ValTuple.split [] s).1 = () := by
  simp [split]

@[simp] theorem split_nil_snd (s : ValTuple ts) :
    (ValTuple.split [] s).2 = s := by
  simp [split]

@[simp] theorem append_split :
    (xs : List Ty) → (s : ValTuple (xs ++ ys)) →
    ValTuple.append xs (ValTuple.split xs s).1 (ValTuple.split xs s).2 = s
  | [],     s        => rfl
  | _::xs, (a, rest) => by simp [split, append, append_split xs rest]

@[simp] theorem append_ite (p : Prop) [Decidable p]
    (xs : List Ty) (a b : ValTuple xs) (c : ValTuple ys) :
    ValTuple.append xs (if p then a else b) c =
    if p then ValTuple.append xs a c else ValTuple.append xs b c := by
  split <;> rfl

syntax "val!" "(" term,* ")" : term

macro_rules
  | `(val!())              => `(())
  | `(val!($x))            => `(($x, ()))
  | `(val!($x, $xs,*))     => `(($x, val!($xs,*)))
end ValTuple

/-  Matrix operations as plain functions -/

def MatAdd (a b : Fin m → Fin n → Int) : Fin m → Fin n → Int :=
  fun i j => a i j + b i j

def MatMul (a : Fin m → Fin k → Int) (b : Fin k → Fin n → Int) : Fin m → Fin n → Int :=
  fun i j => Finset.sum Finset.univ (fun l => a i l * b l j)

def MatZero : Fin m → Fin n → Int := fun _ _ => 0

/-  Box structure -/

structure Box (dom cod : List Ty) where
  fn : ValTuple dom → ValTuple cod

namespace Box

@[simp] def seq (a : Box α β) (b : Box β γ) : Box α γ :=
  ⟨b.fn ∘ a.fn⟩

@[simp] def par (a : Box α β) (b : Box γ δ)
  : Box (α ++ γ) (β ++ δ) :=
  ⟨fun inp =>
    let (l, r) := ValTuple.split α inp
    ValTuple.append β (a.fn l) (b.fn r)⟩

infixl:70 " ≫ " => seq
infixr:75 " ⊗ " => par

@[simp] def dup {t: Ty}: Box [t] [t, t] :=
  ⟨fun val!(x) => val!(x, x)⟩

/-- destroy a value -/
@[simp] def destr {t: Ty}: Box [t] [] :=
  ⟨fun val!(_) => val!()⟩

@[simp] def id {t: Ty}: Box [t] [t] :=
  ⟨fun x => x⟩

@[simp] def const (t: Ty) (c: t.denote): Box [] [t] :=
  ⟨fun _ => val!(c)⟩

@[simp] def swap {t₁ t₂: Ty}: Box [t₁, t₂] [t₂, t₁] :=
  ⟨fun val!(x, y) => val!(y, x)⟩

@[simp] def add : Box [.int, .int] [.int] :=
  ⟨fun val!(a, b) => val!(a + b)⟩

@[simp] def mul : Box [.int, .int] [.int] :=
  ⟨fun val!(a, b) => val!(a * b)⟩

@[simp] def lt : Box [.int, .int] [.bool] :=
  ⟨fun val!(a, b) => val!(a < b)⟩

@[simp] def or : Box [.bool, .bool] [.bool] :=
  ⟨fun val!(a, b) => val!(a ∨ b)⟩

@[simp] def min : Box [.int, .int] [.int] :=
  ⟨fun val!(a, b) => val!(Min.min a b)⟩

@[simp] def max : Box [.int, .int] [.int] :=
  ⟨fun val!(a, b) => val!(Max.max a b)⟩

@[simp] def ite {t: Ty}: Box [.bool, t, t] [t] :=
  ⟨fun val!(c, a, b) => if c then val!(a) else val!(b)⟩

@[simp] def matAdd {m n : Nat} : Box [.mat m n, .mat m n] [.mat m n] :=
  ⟨fun val!(a, b) => val!(MatAdd a b)⟩

@[simp] def matMul {m n p : Nat} : Box [.mat m n, .mat n p] [.mat m p] :=
  ⟨fun val!(a, b) => val!(MatMul a b)⟩

@[simp] def matGet {m n : Nat} (i : Fin m) (j : Fin n) : Box [.mat m n] [.int] :=
  ⟨fun val!(a) => val!(a i j)⟩

@[simp] def sub : Box [.int, .int] [.int] :=
  ⟨fun val!(a, b) => val!(a - b)⟩

@[simp] def neg : Box [.int] [.int] :=
  ⟨fun val!(a) => val!(-a)⟩

@[simp] def and : Box [.bool, .bool] [.bool] :=
  ⟨fun val!(a, b) => val!(a ∧ b)⟩

@[simp] def not : Box [.bool] [.bool] :=
  ⟨fun val!(a) => val!(¬a)⟩

@[simp] def eq {t : Ty} [DecidableEq t.denote] : Box [t, t] [.bool] :=
  ⟨fun val!(a, b) => val!(a == b)⟩

@[simp] def neq {t : Ty} [DecidableEq t.denote] : Box [t, t] [.bool] :=
  ⟨fun val!(a, b) => val!(a != b)⟩

@[simp] def le : Box [.int, .int] [.bool] :=
  ⟨fun val!(a, b) => val!(a ≤ b)⟩

@[simp] def gt : Box [.int, .int] [.bool] :=
  ⟨fun val!(a, b) => val!(b < a)⟩

@[simp] def ge : Box [.int, .int] [.bool] :=
  ⟨fun val!(a, b) => val!(b ≤ a)⟩

end Box

/-! Helper lemmas for simp -/
@[simp] theorem MatAdd_apply {m n : Nat} (a b : Fin m → Fin n → Int) (i : Fin m) (j : Fin n) :
    MatAdd a b i j = a i j + b i j := rfl

@[simp] theorem MatMul_apply {m k n : Nat} (a : Fin m → Fin k → Int) (b : Fin k → Fin n → Int) (i : Fin m) (j : Fin n) :
    MatMul a b i j = Finset.sum Finset.univ (fun l => a i l * b l j) := rfl

@[simp] theorem MatZero_apply {m n : Nat} (i : Fin m) (j : Fin n) :
    MatZero i j = 0 := rfl

@[simp] theorem ite_fst (p : Prop) [Decidable p] (a b : α × β) :
    (if p then a else b).1 = if p then a.1 else b.1 := by
  split <;> rfl

@[simp] theorem ite_snd (p : Prop) [Decidable p] (a b : α × β) :
    (if p then a else b).2 = if p then a.2 else b.2 := by
  split <;> rfl

@[simp] theorem ite_prod_fst (p : Prop) [Decidable p] (a b : α × β) :
    (if p then a else b).1 = if p then a.1 else b.1 := by
  split <;> rfl

@[simp] theorem ite_prod_snd (p : Prop) [Decidable p] (a b : α × β) :
    (if p then a else b).2 = if p then a.2 else b.2 := by
  split <;> rfl

@[simp] theorem ite_fun_apply (p : Prop) [Decidable p] (f g : α → β) (x : α) :
    (if p then f else g) x = if p then f x else g x := by
  split <;> rfl

@[simp] theorem ite_vt_head (p : Prop) [Decidable p] (a b : ValTuple (t :: ts)) :
    (if p then a else b).1 = if p then a.1 else b.1 := by
  split <;> rfl

@[simp] theorem ite_vt_tail (p : Prop) [Decidable p] (a b : ValTuple (t :: ts)) :
    (if p then a else b).2 = if p then a.2 else b.2 := by
  split <;> rfl


theorem parseq (A: Box α β) (B: Box α' β'):
    (A ⊗ B).fn = (fun (x: ValTuple (α ++ α')) =>
                    let a := A.fn (x.split α).1
                    let b := B.fn (x.split α).2
                    ValTuple.append β a b)
  := by simp [Box.par]


