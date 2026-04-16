/- Code generated for reactive module `Counter` -/
import Core.Box

/- Concrete constants -/

@[simp] def c0 : (Mat Int 3 2) := fun i j =>
  match i, j with
  | 0, 0 => 0 | 0, 1 => 0
  | 1, 0 => 1 | 1, 1 => 0
  | 2, 0 => 0 | 2, 1 => 1

@[simp] def c1 : (Mat Int 1 3) := fun i j =>
  match i, j with
  | 0, 0 => 1 | 0, 1 => 0 | 0, 2 => 0

@[simp] def c2 : (Mat Int 1 3) := fun i j =>
  match i, j with
  | 0, 0 => 0 | 0, 1 => 1 | 0, 2 => 0

@[simp] def c3 : (Mat Int 1 3) := fun i j =>
  match i, j with
  | 0, 0 => 0 | 0, 1 => 0 | 0, 2 => 1

@[simp] def c4 : (Mat Int 3 1) := fun i j =>
  match i, j with
  | 0, 0 => 1
  | 1, 0 => 0
  | 2, 0 => 0

@[simp] def c5 : (Mat Int 3 3) := fun i j =>
  match i, j with
  | 0, 0 => 0 | 0, 1 => 0 | 0, 2 => 0
  | 1, 0 => 0 | 1, 1 => 1 | 1, 2 => 0
  | 2, 0 => 0 | 2, 1 => 0 | 2, 2 => 1


namespace Circ
@[simp] def init_l0 : Box [(Mat Int 2 1)] [(Mat Int 2 1)] :=
  @Box.id (Mat Int 2 1)

@[simp] def init_l1 : Box [(Mat Int 2 1)] [(Mat Int 3 2), (Mat Int 2 1)] :=
  @Box.const (Mat Int 3 2) c0 ⊗ @Box.id (Mat Int 2 1)

@[simp] def init_l2 : Box [(Mat Int 3 2), (Mat Int 2 1)] [(Mat Int 3 1)] :=
  Box.mul

@[simp] def init : Box [(Mat Int 2 1)] [(Mat Int 3 1)] :=
  init_l0 ≫ init_l1 ≫ init_l2

@[simp] def update_l0 : Box [(Mat Int 3 1), (Mat Int 2 1)] [(Mat Int 3 1)] :=
  @Box.id (Mat Int 3 1) ⊗ @Box.destr (Mat Int 2 1)

@[simp] def update_l1 : Box [(Mat Int 3 1)] [(Mat Int 3 1), (Mat Int 3 1)] :=
  @Box.dup (Mat Int 3 1)

@[simp] def update_l2 : Box [(Mat Int 3 1), (Mat Int 3 1)] [(Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1)] :=
  @Box.dup (Mat Int 3 1) ⊗ @Box.id (Mat Int 3 1)

@[simp] def update_l3 : Box [(Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1)] [(Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1)] :=
  @Box.dup (Mat Int 3 1) ⊗ @Box.dup (Mat Int 3 1) ⊗ @Box.dup (Mat Int 3 1)

@[simp] def update_l4 : Box [(Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1)] [(Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1)] :=
  @Box.id (Mat Int 3 1) ⊗ @Box.id (Mat Int 3 1) ⊗ @Box.id (Mat Int 3 1) ⊗ @Box.id (Mat Int 3 1) ⊗ @Box.id (Mat Int 3 1) ⊗ @Box.id (Mat Int 3 1)

@[simp] def update_l5 : Box [(Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1)] [(Mat Int 1 3), (Mat Int 3 1), (Mat Int 1 3), (Mat Int 3 1), (Mat Int 1 3), (Mat Int 3 1), (Mat Int 1 3), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1)] :=
  @Box.const (Mat Int 1 3) c1 ⊗ @Box.id (Mat Int 3 1) ⊗ @Box.const (Mat Int 1 3) c2 ⊗ @Box.id (Mat Int 3 1) ⊗ @Box.const (Mat Int 1 3) c1 ⊗ @Box.id (Mat Int 3 1) ⊗ @Box.const (Mat Int 1 3) c3 ⊗ @Box.id (Mat Int 3 1) ⊗ @Box.id (Mat Int 3 1) ⊗ @Box.id (Mat Int 3 1)

@[simp] def update_l6 : Box [(Mat Int 1 3), (Mat Int 3 1), (Mat Int 1 3), (Mat Int 3 1), (Mat Int 1 3), (Mat Int 3 1), (Mat Int 1 3), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 1)] [(Mat Int 1 1), (Mat Int 1 1), (Mat Int 1 1), (Mat Int 1 1), (Mat Int 3 1), (Mat Int 3 1)] :=
  Box.mul ⊗ Box.mul ⊗ Box.mul ⊗ Box.mul ⊗ @Box.id (Mat Int 3 1) ⊗ @Box.id (Mat Int 3 1)

@[simp] def update_l7 : Box [(Mat Int 1 1), (Mat Int 1 1), (Mat Int 1 1), (Mat Int 1 1), (Mat Int 3 1), (Mat Int 3 1)] [(Mat Bool 1 1), (Mat Bool 1 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 3), (Mat Int 3 1)] :=
  Box.lt ⊗ Box.lt ⊗ @Box.id (Mat Int 3 1) ⊗ @Box.const (Mat Int 3 1) c4 ⊗ @Box.const (Mat Int 3 3) c5 ⊗ @Box.id (Mat Int 3 1)

@[simp] def update_l8 : Box [(Mat Bool 1 1), (Mat Bool 1 1), (Mat Int 3 1), (Mat Int 3 1), (Mat Int 3 3), (Mat Int 3 1)] [(Mat Bool 1 1), (Mat Int 3 1), (Mat Int 3 1)] :=
  Box.or ⊗ Box.add ⊗ Box.mul

@[simp] def update_l9 : Box [(Mat Bool 1 1), (Mat Int 3 1), (Mat Int 3 1)] [(Mat Int 3 1)] :=
  Box.ite

@[simp] def update : Box [(Mat Int 3 1) , (Mat Int 2 1)] [(Mat Int 3 1)] :=
  update_l0 ≫ update_l1 ≫ update_l2 ≫ update_l3 ≫ update_l4 ≫ update_l5 ≫ update_l6 ≫ update_l7 ≫ update_l8 ≫ update_l9

end Circ

@[simp] def init (s : (Mat Int 2 1)) : (Mat Int 3 1) :=
  let x0 := c0
  let x1 := MatMul x0 s
  x1

@[simp] def update (s : (Mat Int 3 1) × (Mat Int 2 1)) : (Mat Int 3 1) :=
  let x0 := c1
  let x1 := MatMul x0 s.1
  let x2 := c2
  let x3 := MatMul x2 s.1
  let x4 := c3
  let x5 := MatMul x4 s.1
  let x6 := decide (x1 0 0 < x3 0 0)
  let x7 := decide (x1 0 0 < x5 0 0)
  let x8 := (x6 || x7)
  let x9 := c4
  let x10 := (s.1 + x9)
  let x11 := c5
  let x12 := MatMul x11 s.1
  let x13 := if x8 then x10 else x12
  x13


/-- Reduce one circuit layer: unfolds the given lemma,
    then simplifies all Box/ValTuple plumbing. -/
macro "simp_circ" "[" ls:Lean.Parser.Tactic.simpLemma,* "]" : tactic =>
  `(tactic| simp only [$ls,*,
    Box.par,
    ValTuple.split,
    ValTuple.append,
    ValTuple.append_split,
    ValTuple.append_ite,
    ValTuple.split_singleton_fst,
    ValTuple.split_singleton_snd,
    ValTuple.split_cons_fst_fst,
    ValTuple.split_cons_fst_snd,
    ValTuple.split_2_fst,
    ValTuple.split_2_snd,
    ValTuple.split_3_fst,
    ValTuple.split_3_snd,
    ValTuple.split_nil,
    ValTuple.split_nil_snd,
    Box.id,
    Box.dup,
    Box.swap,
    Box.destr,
    Box.const,
    Box.not,
    Box.and,
    Box.or,
    Box.ite,
    Box.add,
    Box.sub,
    Box.mul,
    Box.neg,
    Box.lt,
    Box.le,
    Box.gt,
    Box.ge,
    Box.eq,
    Box.neq,
    Box.min,
    Box.max,
    Box.nnLinear,
    Box.relu,
    ite_pair])

theorem init_circ_eq : ∀ (s : (Mat Int 2 1)),
    Circ.init.fn (s, ()) =
    let r := init s
    (r, ()) := by
  intro s
  simp_circ [Circ.init, Box.seq]
  simp_circ [Circ.init_l0]
  simp_circ [Circ.init_l1]
  simp_circ [Circ.init_l2]
  simp [init, ite_pair, c0, c1, c2, c3, c4, c5]
  try exact List.ofFn_inj.mp rfl
  try grind
  try simp only [Mat_1_1_eq]
  try simp
  try grind
  try simp [init, ite_pair, c0, c1, c2, c3, c4, c5]
  try exact List.ofFn_inj.mp rfl

theorem update_circ_eq : ∀ (s : (Mat Int 3 1) × (Mat Int 2 1)),
    Circ.update.fn (s.1, (s.2, ())) =
    let r := update s
    (r, ()) := by
  intro s
  simp_circ [Circ.update, Box.seq]
  simp_circ [Circ.update_l0]
  simp_circ [Circ.update_l1]
  simp_circ [Circ.update_l2]
  simp_circ [Circ.update_l3]
  simp_circ [Circ.update_l4]
  simp_circ [Circ.update_l5]
  simp_circ [Circ.update_l6]
  simp_circ [Circ.update_l7]
  simp_circ [Circ.update_l8]
  simp_circ [Circ.update_l9]
  simp [update, ite_pair, c0, c1, c2, c3, c4, c5]
  try exact List.ofFn_inj.mp rfl
  try grind
  try simp only [Mat_1_1_eq]
  try simp
  try grind
  try simp [update, ite_pair, c0, c1, c2, c3, c4, c5]
  try exact List.ofFn_inj.mp rfl

