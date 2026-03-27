import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Real.Basic

/-! A simple version of wiring diagrams. The main object
    is `Box` which is a node in the wiring diagram.
    There are no explicit wires, the diagrams are represented
    as terms. That is, a `Box` is essentially just a function
    annotated with its domain and codomain so that we can
    define and reason about sequential and parallel composition
    on `Box`es.
-/

abbrev Mat (t: Type) (m n: Nat)  := Fin m → Fin n → t

/-  Matrix operations as plain functions -/

def MatAdd [HAdd t t t] (a b : Fin m → Fin n → t) : Fin m → Fin n → t :=
  fun i j => a i j + b i j

def MatMul [HMul t t t] [AddCommMonoid t](a : Mat t m k) (b : Mat t k n) : Mat t m n :=
  fun i j => Finset.sum Finset.univ (fun l => a i l * b l j)

instance [HAdd t t t]: HAdd (Mat t m n) (Mat t m n) (Mat t m n) where
  hAdd := MatAdd

instance [HMul t t t][AddCommMonoid t]: HMul (Mat t m k) (Mat t k n) (Mat t m n) where
  hMul := MatMul

def MatZero [OfNat t 0]: Mat t m n := fun _ _ => (0 : t)

def MatTranspose (A : Mat t m n) : Mat t n m := fun i j => A j i


/-- Affine layer: `nnLinear x A b = x * A + b` (NOTE: `A` is not transposed
    as in `torch.nn.Linear`!)
-/
def affineLinear [HMul t t t] [AddCommMonoid t] [HAdd t t t]
    (x : Mat t m k) (A : Mat t k n) (b : Mat t m n) : Mat t m n :=
  MatMul x A + b

/-- Element-wise ReLU: `relu x i j = max 0 (x i j)` -/
def ReLu [Max t] [OfNat t 0] (x : Mat t m n) : Mat t m n :=
  fun i j => Max.max 0 (x i j)


/-! Helper lemmas for simp -/
@[simp] theorem affineLinear_apply [HMul t t t] [AddCommMonoid t] [HAdd t t t]
    (x : Mat t m k) (A : Mat t k n) (b : Mat t m n) (i : Fin m) (j : Fin n) :
    affineLinear x A b i j = Finset.sum Finset.univ (fun l => x i l * A l j) + b i j := by
  simp [affineLinear]
  unfold MatMul
  exact (congrArg (((fun i j => ∑ l, x i l * A l j) + b) i) ∘ fun a => a) rfl


@[simp] theorem relu_apply [Max t] [OfNat t 0]
    (x : Mat t m n) (i : Fin m) (j : Fin n) :
    ReLu x i j = Max.max 0 (x i j) := rfl

@[simp] theorem MatAdd_apply {m n : Nat} (a b : Fin m → Fin n → Int) (i : Fin m) (j : Fin n) :
    MatAdd a b i j = a i j + b i j := rfl

@[simp] theorem MatMul_apply {m k n : Nat} (a : Fin m → Fin k → Int) (b : Fin k → Fin n → Int) (i : Fin m) (j : Fin n) :
    MatMul a b i j = Finset.sum Finset.univ (fun l => a i l * b l j) := rfl

@[simp] theorem MatZero_apply {m n : Nat} (i : Fin m) (j : Fin n) :
    MatZero i j = 0 := rfl

@[simp] theorem MatAdd_eq_add {m n : Nat} (a b : Fin m → Fin n → Int) :
    MatAdd a b = a + b := rfl

@[simp] theorem MatMul_eq_mul {m n : Nat} [HMul t t t] [AddCommMonoid t]
  (a : Mat t m k) (b : Mat t k n) :
    MatMul a b = a * b := rfl

/- coercion between single-element matrix and the element -/
instance {t: Type} : Coe t (Mat t 1 1) where
  coe x := fun _ _ => x

/-
instance {t: Type} : Coe (Mat t 1 1) t where
  coe m := m 0 0
-/


/-- Flattened tuple of values. We could use regular tuples, but then
    the composition of boxes would give us nested tuples, e.g.
    `((Int × Int) × (Float × Int))`. However, we always want the flattened
    tuple `Int × Int × Float × Int`.
-/
@[simp] def ValTuple : List Type → Type
  | []      => Unit
  | t :: ts => t × ValTuple ts

instance : DecidableEq (ValTuple []) := fun () () => .isTrue rfl

instance [DecidableEq t] [DecidableEq (ValTuple ts)] :
    DecidableEq (ValTuple (t :: ts)) :=
  inferInstanceAs (DecidableEq (t × ValTuple ts))

namespace ValTuple

@[simp] def append : (xs : List Type) → ValTuple xs → ValTuple ys → ValTuple (xs ++ ys)
  | [],     (),       bs => bs
  | _::xs, (a, rest), bs => (a, append xs rest bs)

@[simp] def split : (xs : List Type) → ValTuple (xs ++ ys) → ValTuple xs × ValTuple ys
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
    (xs : List Type) → (s : ValTuple (xs ++ ys)) →
    ValTuple.append xs (ValTuple.split xs s).1 (ValTuple.split xs s).2 = s
  | [],     s        => rfl
  | _::xs, (a, rest) => by simp [split, append, append_split xs rest]

@[simp] theorem append_ite (p : Prop) [Decidable p]
    (xs : List Type) (a b : ValTuple xs) (c : ValTuple ys) :
    ValTuple.append xs (if p then a else b) c =
    if p then ValTuple.append xs a c else ValTuple.append xs b c := by
  split <;> rfl


@[simp] theorem split_cons_fst_fst (a : t) (rest : ValTuple (ts ++ ys)) :
    (ValTuple.split (t :: ts) (a, rest)).1.1 = a := rfl

@[simp] theorem split_cons_fst_snd (a : t) (rest : ValTuple (ts ++ ys)) :
    (ValTuple.split (t :: ts) (a, rest)).1.2 = (ValTuple.split ts rest).1 := rfl


-- @[simp] theorem split_singleton_snd (v : ValTuple (t :: ts)) :
--     (ValTuple.split [t] v).2 = v.2 := rfl

-- @[simp] theorem split_singleton_fst (v : ValTuple (t :: ts)) :
--     (ValTuple.split [t] v).1 = (v.1, ()) := rfl

-- @[simp] theorem split_cons_snd (a : t.denote) (rest : ValTuple (ts ++ ys)) :
--     (ValTuple.split (t :: ts) (a, rest)).2 = (ValTuple.split ts rest).2 := rfl

-- @[simp] theorem split_pair_fst (a : t₁.denote) (b : t₂.denote) (rest : ValTuple ys) :
--     (ValTuple.split [t₁, t₂] (a, b, rest)).1 = ((a, b, ()): ValTuple [t₁, t₂]) := rfl

-- @[simp] theorem split_pair_snd (a : t₁.denote) (b : t₂.denote) (rest : ValTuple ys) :
--     (ValTuple.split [t₁, t₂] (a, b, rest)).2 = rest := rfl

  @[simp] theorem split_singleton_fst (v : ValTuple ([t] ++ ys)) :
      (ValTuple.split [t] v).1 = (v.1, ()) := rfl

  @[simp] theorem split_singleton_snd (v : ValTuple ([t] ++ ys)) :
      (ValTuple.split [t] v).2 = v.2 := rfl

  @[simp] theorem split_2_fst (v : ValTuple ([t₁, t₂] ++ ys)) :
      (ValTuple.split [t₁, t₂] v).1 = (v.1, v.2.1, ()) := rfl

  @[simp] theorem split_2_snd (v : ValTuple ([t₁, t₂] ++ ys)) :
      (ValTuple.split [t₁, t₂] v).2 = v.2.2 := rfl

  @[simp] theorem split_3_fst (v : ValTuple ([t₁, t₂, t₃] ++ ys)) :
      (ValTuple.split [t₁, t₂, t₃] v).1 = (v.1, v.2.1, v.2.2.1, ()) := rfl

  @[simp] theorem split_3_snd (v : ValTuple ([t₁, t₂, t₃] ++ ys)) :
      (ValTuple.split [t₁, t₂, t₃] v).2 = v.2.2.2 := rfl


syntax "val!" "(" term,* ")" : term

macro_rules
  | `(val!())              => `(())
  | `(val!($x))            => `(($x, ()))
  | `(val!($x, $xs,*))     => `(($x, val!($xs,*)))
end ValTuple

/--  Box structure -/
structure Box (dom cod : List Type) where
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

@[simp] def dup {t: Type}: Box [t] [t, t] :=
  ⟨fun val!(x) => val!(x, x)⟩

/-- destroy a value -/
@[simp] def destr {t: Type}: Box [t] [] :=
  ⟨fun val!(_) => val!()⟩

@[simp] def id {t: Type}: Box [t] [t] :=
  ⟨fun x => x⟩

@[simp] def const {t: Type} (c: t): Box [] [t] :=
  ⟨fun _ => val!(c)⟩

@[simp] def swap {t₁ t₂: Type}: Box [t₁, t₂] [t₂, t₁] :=
  ⟨fun val!(x, y) => val!(y, x)⟩

@[simp] def add {t: Type} [HAdd t t t] : Box [t, t] [t] :=
  ⟨fun val!(a, b) => val!(a + b)⟩

@[simp] def mul [HMul ta tb tc] : Box [ta, tb] [tc] :=
  ⟨fun val!(a, b) => val!(a * b)⟩

@[simp] def sub {t: Type} [HSub t t t] : Box [t, t] [t] :=
  ⟨fun val!(a, b) => val!(a - b)⟩

@[simp] def neg {t: Type} [Neg t]: Box [t] [t] :=
  ⟨fun val!(a) => val!(-a)⟩

@[simp] def eq {t : Type} [DecidableEq t] : Box [t, t] [Bool] :=
  ⟨fun val!(a, b) => val!(a == b)⟩

@[simp] def neq {t : Type} [DecidableEq t] : Box [t, t] [Bool] :=
  ⟨fun val!(a, b) => val!(a != b)⟩

@[simp] def le {t: Type} [LE t] [DecidableRel (· ≤ ·: t → t → Prop)]: Box [t, t] [Bool] :=
  ⟨fun val!(a, b) => val!(a ≤ b)⟩

@[simp] def gt {t: Type} [LE t] [DecidableRel (· ≤ ·: t → t → Prop)]: Box [t, t] [Bool] :=
  ⟨fun val!(a, b) => val!(¬(b ≤ a))⟩

@[simp] def ge {t: Type} [LT t] [DecidableRel (· < · : t → t → Prop)] : Box [t, t] [Bool] :=
  ⟨fun val!(a, b) => val!(¬(b < a))⟩

@[simp] def lt {t: Type} [LT t] [DecidableRel (· < · : t → t → Prop)] : Box [t, t] [Bool] :=
  ⟨fun val!(a, b) => val!(a < b)⟩

@[simp] def and : Box [Bool, Bool] [Bool] :=
  ⟨fun val!(a, b) => val!(a ∧ b)⟩

@[simp] def not : Box [Bool] [Bool] :=
  ⟨fun val!(a) => val!(¬a)⟩

@[simp] def or : Box [Bool, Bool] [Bool] :=
  ⟨fun val!(a, b) => val!(a ∨ b)⟩

@[simp] def min {t: Type} [Min t]: Box [t, t] [t] :=
  ⟨fun val!(a, b) => val!(Min.min a b)⟩

@[simp] def max {t: Type} [Max t]: Box [t, t] [t] :=
  ⟨fun val!(a, b) => val!(Max.max a b)⟩

@[simp] def ite {t: Type}: Box [Bool, t, t] [t] :=
  ⟨fun val!(c, a, b) => if c then val!(a) else val!(b)⟩

-- @[simp] def matAdd {m n : Nat} : Box [.mat m n, .mat m n] [.mat m n] :=
--   ⟨fun val!(a, b) => val!(MatAdd a b)⟩

-- @[simp] def matMul {m n p : Nat} : Box [.mat m n, .mat n p] [.mat m p] :=
--   ⟨fun val!(a, b) => val!(MatMul a b)⟩

-- @[simp] def matGet {m n : Nat} (i : Fin m) (j : Fin n) : Box [.mat m n] [.int] :=
--   ⟨fun val!(a) => val!(a i j)⟩



@[simp] def nnLinear [HMul t t t] [AddCommMonoid t] [HAdd t t t] {m k n : Nat}
    : Box [Mat t m k, Mat t k n, Mat t m n] [Mat t m n] :=
  ⟨fun val!(x, A, b) => val!(affineLinear x A b)⟩

@[simp] def relu [Max t] [OfNat t 0] {m n : Nat}
    : Box [Mat t m n] [Mat t m n] :=
  ⟨fun val!(x) => val!(ReLu x)⟩

end Box

/- ite simplifications -/
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

@[simp] theorem ite_pair (p : Prop) [Decidable p] (a b : α) (c d : β) :
    (if p then (a, c) else (b, d)) = (if p then a else b, if p then c else d) := by
  split <;> rfl


/- theorem saying that we can simulate parallel composition
   by executing A and B independently on their parts of input
   and then composing the results -/
theorem parseq (A: Box α β) (B: Box α' β'):
    (A ⊗ B).fn = (fun (x: ValTuple (α ++ α')) =>
                    let a := A.fn (x.split α).1
                    let b := B.fn (x.split α).2
                    ValTuple.append β a b)
  := by simp [Box.par]
