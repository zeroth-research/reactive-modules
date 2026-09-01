# What gets generated into Lean

`zrth/lean` takes a Python reactive module and writes out a Lean 4 package.

The package contains three kinds of things:

1. **The module itself**, encoded five different ways.
2. **Theorems saying the encodings agree** with each other.
3. **A certificate**: an invariant, a property, a ranking function, and a proof
   that the property holds infinitely often.

This document describes *what* comes out and what shape it has. It does not
describe how the translation works — see [`README.md`](README.md) for that.

---

## The running example: `twobit`

### In plain Python

Before any of the API, here is what the example *means*. Two state bits and one
input bit, as ordinary Python:

```python
def init():
    return False, False


def update(b0, b1, enable):
    new_b0 = (not b0) if enable else b0
    new_b1 = (not b1) if (b0 and enable) else b1
    return new_b0, new_b1
```

`init` gives the starting values. `update` takes the current values plus the
input, and returns the next values. That is the whole system.

It is a 2-bit counter: `b0` is the low bit and flips on every enabled step,
`b1` is the high bit and flips whenever `b0` rolls over. Driving it with
`enable = True` gives

```
b1 b0 = 00   (value 0)
b1 b0 = 01   (value 1)
b1 b0 = 10   (value 2)
b1 b0 = 11   (value 3)
b1 b0 = 00   (value 0)   <- back to the start
```

The property we want to prove is that the counter visits `00` infinitely
often assuming `enable` is `true`.

### As a reactive module

The actual fixture, `python/tests/fixtures/twobit.py`, is built by hand out of
wires and terms instead. Each line of the plain version becomes one operation
on explicitly named wires:

```python
def module() -> Module:
    b0 = (Wire(dt.BitVec(1, [1, 1])), Wire(dt.BitVec(1, [1, 1])))
    b1 = (Wire(dt.BitVec(1, [1, 1])), Wire(dt.BitVec(1, [1, 1])))
    enable = (Wire(dt.BitVec(1, [1, 1])), Wire(dt.BitVec(1, [1, 1])))

    not_b0 = Wire(dt.BitVec(1, [1, 1]))
    not_b1 = Wire(dt.BitVec(1, [1, 1]))
    b0_and_enable = Wire(dt.BitVec(1, [1, 1]))

    init = [
        Term(BV.Const(torch.tensor([[0]])), [b0[1]]),
        Term(BV.Const(torch.tensor([[0]])), [b1[1]]),
    ]
    update = [
        Term(BV.Not(), [not_b0], [b0[0]]),
        Term(BV.Ite(), [b0[1]], [enable[1], not_b0, b0[0]]),
        Term(BV.And(), [b0_and_enable], [b0[0], enable[1]]),
        Term(BV.Not(), [not_b1], [b1[0]]),
        Term(BV.Ite(), [b1[1]], [b0_and_enable, not_b1, b1[0]]),
    ]
    return Module.sequential(init, update, obs=[b0, b1, enable])
```

Three things to notice, because they explain the Lean output later on.

- Every variable is a **pair of wires**: `b0[0]` is its value at the start of
  the step ("latched"), `b0[1]` is the value being written ("next"). The plain
  version's `b0` parameter is `b0[0]`; its `new_b0` return value is `b0[1]`.
- Intermediate results get their own wires. `not_b0`, `not_b1` and
  `b0_and_enable` are the subexpressions of the plain version, named.
- `enable[1]` is read, not `enable[0]`. The module awaits this step's input
  rather than using the previous one. That matches `update(..., enable)` taking
  the current input.

Types differ slightly from the plain version: the fixture uses 1-bit
bitvectors, `dt.BitVec(1, [1, 1])`, where plain Python used `bool`.
`tests/fixtures/twobit_lia.py` is the same counter over `dt.Bool`.


### Generating everything

The whole package comes from one command:

```bash
uv run verith tests/fixtures/twobit.py \
    -P "(and (= s0 (_ bv0 1)) (= s1 (_ bv0 1)))" \
    -o out/ -p TwoBit
```

`-P` is the property, written in SMT-LIB. State variables are called `s0`,
`s1`, ... in the order they were listed when creating the module (omitted here).

---

## The generated tree

```
out/TwoBit/
  lakefile.toml            generated  Lean package: libs Core, System, Certificate, ZerothHammer, LeanAI
  lean-toolchain           generated  pinned Lean version (currently v4.28.0)

  Core/                    copied     hand-written library, same for every module
    Mat.lean                            matrices, ReLu, argmax, matVecAffine
    Basic.lean                          ReactiveModule, transition systems, invariant & Büchi proof rules
    Box.lean                            wiring-diagram (circuit) algebra
    LTL.lean                            LTL syntax and semantics
  LeanAI/                  copied     optional LLM-in-Lean helpers (not needed for proofs)
  ZerothHammer.lean        generated  the `zeroth_hammer` proof tactic (identical for every module)

  System.lean              generated  root file, just imports the six System/* files
  System/
    System.lean            generated  ENCODING 1 — functional
    Circ.lean              generated  ENCODING 2 — circuit      + equivalence to 1
    Scalar.lean            generated  ENCODING 3 — scalar       + equivalence to 1
    Rel.lean               generated  ENCODING 4 — relational, matrix domain + equivalence to 1
    ScalarRel.lean         generated  ENCODING 5 — relational, scalar domain + equivalence to 3 and 1
    FBK.lean               generated  ENCODING 6 — relational, Bool-valued    + equivalence to 5
    Data.lean              generated  the certificate data: pre, inv, P, ranking

  Certificate.lean         generated  one-line shim: `import Certificate.Certificate`
  Certificate/
    Certificate.lean       generated  the reactive module as `RM`, tactic macros, the proofs

  dbg/
    system.txt             generated  the module printed in text form (debug aid)
    system.py              generated  a copy of the input Python file (debug aid)
```

Two extra files appear on demand:

- `Main.lean` — with `--executable`. It runs `init`/`update` in a
  stdin/stdout loop, so the encoded module can be executed.
- Nothing else is optional; all six encodings are always emitted (at this moment).

---

## How a variable looks in Lean

Every wire becomes a **matrix**. `Mat T m n` is just `Fin m → Fin n → T`, i.e.
a function from two indices to a value. A scalar is a 1×1 matrix, and it is
read as `x 0 0`.

| Python wire | Lean type |
|---|---|
| `dt.Bool([1, 1])` | `Mat Bool 1 1` |
| `dt.Int([1, 1])` | `Mat Int 1 1` |
| `dt.Real([1, 1])` | `Mat Real 1 1` |
| `dt.BitVec(1, [1, 1])` | `Mat (BitVec 1) 1 1` |
| `dt.Int([3, 2])` | `Mat Int 3 2` |

Three parameter names show up everywhere:

- `ctrl` — the state, as a tuple: `ctrl.1` is `b0`, `ctrl.2` is `b1`.
- `extl_n` — the external inputs of *this* step ("next").
- `extl_l` — the external inputs of the *previous* step ("latched").

`twobit` never reads `extl_l`, but the parameter is still there, because the
shape of `init` and `update` is fixed.

---

## Encoding 1 — functional (`System/System.lean`)

This is the plain, readable version. Two Lean functions: `init` produces the
first state, `update` produces the next state. The body is a chain of `let`
bindings, one per operation in the Python term list, and it ends with the
output tuple.

```lean
@[simp] def init (extl_n: (Mat (BitVec 1) 1 1)) : (Mat (BitVec 1) 1 1) × (Mat (BitVec 1) 1 1) :=
  let x0 : (Mat (BitVec 1) 1 1) := (fun _ _ => (BitVec.ofNat 1 0))
  let x1 : (Mat (BitVec 1) 1 1) := (fun _ _ => (BitVec.ofNat 1 0))
  (x0, x1)

@[simp] def update (ctrl: (Mat (BitVec 1) 1 1) × (Mat (BitVec 1) 1 1)) (extl_l: (Mat (BitVec 1) 1 1)) (extl_n: (Mat (BitVec 1) 1 1)) : (Mat (BitVec 1) 1 1) × (Mat (BitVec 1) 1 1) :=
  let x0 : (Mat (BitVec 1) 1 1) := (fun _ _ => !(ctrl.1 0 0))
  let x1 : (Mat (BitVec 1) 1 1) := (if extl_n 0 0 then x0 else ctrl.1)
  let x2 : (Mat (BitVec 1) 1 1) := (fun _ _ => (ctrl.1 0 0 && extl_n 0 0))
  let x3 : (Mat (BitVec 1) 1 1) := (fun _ _ => !(ctrl.2 0 0))
  let x4 : (Mat (BitVec 1) 1 1) := (if x2 0 0 then x3 else ctrl.2)
  (x1, x4)
```

Everything else in the package is compared against these two functions.

Note the two shapes that repeat throughout. `fun _ _ => e` wraps a plain value
back into a 1×1 matrix. `m 0 0` reads a plain value out of one.

---

## Encoding 2 — circuit (`System/Circ.lean`)

The same computation, drawn as a wiring diagram. A `Box dom cod` is a gate (or
a whole block) with a typed list of inputs and a typed list of outputs. `⊗`
puts two boxes side by side, `≫` puts one after another.

The generated file is a stack of *layers*. Each layer is one `⊗`-row: either
plumbing (`Box.id` passes a wire through, `Box.dup` copies it, `Box.swap`
crosses two, `Box.destr` drops one, `Box.const` injects a constant) or real
gates (`Box.not`, `Box.and`, `Box.ite`, `Box.add`, …).

```lean
namespace Circ
@[simp] def init_l0 : Box [(Mat (BitVec 1) 1 1)] [] :=
  @Box.destr (Mat (BitVec 1) 1 1)

@[simp] def init_l1 : Box [] [(Mat (BitVec 1) 1 1), (Mat (BitVec 1) 1 1)] :=
  @Box.const (Mat (BitVec 1) 1 1) (fun _ _ => (BitVec.ofNat 1 0)) ⊗ @Box.const (Mat (BitVec 1) 1 1) (fun _ _ => (BitVec.ofNat 1 0))

@[simp] def init : Box [(Mat (BitVec 1) 1 1)] [(Mat (BitVec 1) 1 1) , (Mat (BitVec 1) 1 1)] :=
  init_l0 ≫ init_l1
```

`twobit`'s `update` needs eleven layers. Nine of them are pure routing; the
real work is in the last two:

```lean
@[simp] def update_l9 : Box [...] [...] :=
  @Box.id (Mat (BitVec 1) 1 1) ⊗ Box.not ⊗ @Box.id (Mat (BitVec 1) 1 1) ⊗ Box.and ⊗ Box.not ⊗ @Box.id (Mat (BitVec 1) 1 1)

@[simp] def update_l10 : Box [...] [...] :=
  Box.ite ⊗ Box.ite

@[simp] def update : Box [...] [...] :=
  update_l0 ≫ update_l1 ≫ update_l2 ≫ update_l3 ≫ update_l4 ≫ update_l5 ≫ update_l6 ≫ update_l7 ≫ update_l8 ≫ update_l9 ≫ update_l10
end Circ
```

Then come two theorems. They say: running the circuit gives the same answer as
calling the function. `Box.fn` is the function a box denotes, and its arguments
and results are nested pairs instead of tuples, so the statement re-packs them.

```lean
theorem init_circ_eq : ∀ (extl_n : (Mat (BitVec 1) 1 1)),
    Circ.init.fn (extl_n, ()) =
    let r := init extl_n
    (r.1, (r.2, ())) := by ...

theorem update_circ_eq : ∀ (ctrl : ...) (extl_l : ...) (extl_n : ...),
    Circ.update.fn (ctrl.1, (ctrl.2, (extl_l, (extl_n, ())))) =
    let r := update ctrl extl_l extl_n
    (r.1, (r.2, ())) := by ...
```

The file also defines a helper tactic macro `simp_circ`, used inside those
proofs to unfold one layer at a time.

---

## Encoding 3 — scalar (`System/Scalar.lean`)

The same computation again, but with the matrix wrapper stripped off. Types are
bare `BitVec 1` / `Bool` / `Int` / `Real` instead of `Mat T 1 1`. This version
is much easier for `omega` and other arithmetic tactics to chew on.

Three ingredients: converters between the two worlds, the functions themselves,
and the theorems tying them to Encoding 1.

```lean
namespace Scalar

@[simp] def unpack_ctrl (ctrl : (Mat (BitVec 1) 1 1) × (Mat (BitVec 1) 1 1)) : (BitVec 1) × (BitVec 1) :=
  (ctrl.1 0 0, ctrl.2 0 0)

@[simp] def unpack_extl_n (extl_n : (Mat (BitVec 1) 1 1)) : (BitVec 1) :=
  extl_n 0 0

@[simp] def pack (r : (BitVec 1) × (BitVec 1)) : (Mat (BitVec 1) 1 1) × (Mat (BitVec 1) 1 1) :=
  ((fun _ _ => r.1), (fun _ _ => r.2))

@[simp] def update (ctrl: (BitVec 1) × (BitVec 1)) (extl_l: (BitVec 1)) (extl_n: (BitVec 1)) : (BitVec 1) × (BitVec 1) :=
  let x0 : (BitVec 1) := (!ctrl.1)
  let x1 : (BitVec 1) := (if extl_n then x0 else ctrl.1)
  let x2 : (BitVec 1) := (ctrl.1 && extl_n)
  let x3 : (BitVec 1) := (!ctrl.2)
  let x4 : (BitVec 1) := (if x2 then x3 else ctrl.2)
  (x1, x4)

end Scalar
```

Compare with Encoding 1: the `fun _ _ =>` wrappers and the `0 0` reads are all
gone. `unpack_extl_l` and `Scalar.init` are emitted too, in the same style.

The two theorems say the scalar version is the real one sandwiched between
`unpack` and `pack`:

```lean
theorem init_scalar_eq : ∀ (extl_n : (Mat (BitVec 1) 1 1)),
    init extl_n = Scalar.pack (Scalar.init (Scalar.unpack_extl_n extl_n)) := by ...

theorem update_scalar_eq : ∀ (ctrl : ...) (extl_l : ...) (extl_n : ...),
    update ctrl extl_l extl_n =
      Scalar.pack (Scalar.update (Scalar.unpack_ctrl ctrl)
                                 (Scalar.unpack_extl_l extl_l)
                                 (Scalar.unpack_extl_n extl_n)) := by ...
```

---

## Encoding 4 — relational, matrix domain (`System/Rel.lean`)

So far every encoding was a *function*: give it a state, get the next state
(non-determinism can be modelled as a new input).
This one is a *relation*: a `Prop` that says whether a pair of states is a
legal step. That is the form model checkers and SMT solvers want.

It is built up in three steps. First, one function per state variable, holding
just the code that variable needs — the rest is pruned away:

```lean
namespace Rel

@[simp] def effect_0 (ctrl : ...) (extl_l : ...) (extl_n : ...) : (Mat (BitVec 1) 1 1) :=
  let x0 : (Mat (BitVec 1) 1 1) := (fun _ _ => !(ctrl.1 0 0))
  let x1 : (Mat (BitVec 1) 1 1) := (if extl_n 0 0 then x0 else ctrl.1)
  x1

@[simp] def effect_1 (ctrl : ...) (extl_l : ...) (extl_n : ...) : (Mat (BitVec 1) 1 1) :=
  let x0 : (Mat (BitVec 1) 1 1) := (fun _ _ => (ctrl.1 0 0 && extl_n 0 0))
  let x1 : (Mat (BitVec 1) 1 1) := (fun _ _ => !(ctrl.2 0 0))
  let x2 : (Mat (BitVec 1) 1 1) := (if x0 0 0 then x1 else ctrl.2)
  x2
```

Second, each effect becomes a one-variable relation, and their conjunction is
the transition relation:

```lean
def R_0 (old new : ...) (extl_l : ...) (extl_n : ...) : Prop :=
  new.1 = effect_0 old extl_l extl_n

def R_1 (old new : ...) (extl_l : ...) (extl_n : ...) : Prop :=
  new.2 = effect_1 old extl_l extl_n

def TransRel (old new : ...) (extl_l : ...) (extl_n : ...) : Prop :=
  R_0 old new extl_l extl_n ∧
  R_1 old new extl_l extl_n
```

Third, the same for the initial state: `init_0`, `init_1`, then `Init_0`,
`Init_1`, then `InitCond`.

Two theorems close the loop back to Encoding 1, plus one small theorem per
effect and per init:

```lean
theorem effect_0_eq : ∀ ctrl extl_l extl_n,
    effect_0 ctrl extl_l extl_n = (update ctrl extl_l extl_n).1 := by ...

theorem TransRel_func_eq : ∀ (old new : ...) (extl_l : ...) (extl_n : ...),
    TransRel old new extl_l extl_n ↔ new = update old extl_l extl_n := by ...

theorem InitCond_func_eq : ∀ (s : ...) (extl_n : ...),
    InitCond s extl_n ↔ s = init extl_n := by ...
```

Read `TransRel_func_eq` as: "the relation holds exactly when `new` is what
`update` computes". So nothing is lost by switching to the relational view.

---

## Encoding 5 — relational, scalar domain (`System/ScalarRel.lean`)

Encoding 4 again, but over bare scalars instead of matrices. Same names
(`effect_i`, `R_i`, `TransRel`, `init_i`, `Init_i`, `InitCond`), inside
`namespace ScalarRel`.

```lean
namespace ScalarRel

@[simp] def effect_0 (ctrl : (BitVec 1) × (BitVec 1)) (extl_l : (BitVec 1)) (extl_n : (BitVec 1)) : (BitVec 1) :=
  let x0 : (BitVec 1) := (!ctrl.1)
  let x1 : (BitVec 1) := (if extl_n then x0 else ctrl.1)
  x1

def TransRel (old new : (BitVec 1) × (BitVec 1)) (extl_l : (BitVec 1)) (extl_n : (BitVec 1)) : Prop :=
  R_0 old new extl_l extl_n ∧
  R_1 old new extl_l extl_n
```

This one gets *two* levels of equivalence theorem. One down to `Scalar.update`,
and one all the way to the matrix `update`:

```lean
theorem TransRel_scalar_eq : ∀ (old new : ...) (extl_l : ...) (extl_n : ...),
    TransRel old new extl_l extl_n ↔ new = Scalar.update old extl_l extl_n := by ...

theorem TransRel_func_eq : ∀ (ctrl ctrl' : (Mat (BitVec 1) 1 1) × (Mat (BitVec 1) 1 1)) (extl_l : ...) (extl_n : ...),
    TransRel (Scalar.unpack_ctrl ctrl) (Scalar.unpack_ctrl ctrl')
             (Scalar.unpack_extl_l extl_l) (Scalar.unpack_extl_n extl_n)
    ↔ ctrl' = update ctrl extl_l extl_n := by ...
```

`InitCond_scalar_eq` and `InitCond_func_eq` are the matching pair for `init`.

---

## Encoding 6 — relational, `Bool`-valued (`System/FBK.lean`)

The last one drops two more things. The state is no longer a tuple but a
*function from indices*: `state 0` is the first variable, `state 1` the second.
And the relations return `Bool` (a computable yes/no) rather than `Prop`.

```lean
namespace FBK

abbrev TypeMap : Nat → Type
  | _ => (BitVec 1)

abbrev StateType := (n : Nat) → TypeMap n

variable (state newstate s : StateType) (extl_l : (BitVec 1)) (extl_n : (BitVec 1))

abbrev var_0 := state 0
abbrev var_1 := state 1

abbrev effect_0 : (BitVec 1) :=
  let x0 : (BitVec 1) := (!(state 0))
  let x1 : (BitVec 1) := (if extl_n then x0 else (state 0))
  x1

abbrev R_0 : Bool :=
  (newstate 0) == effect_0 state extl_n

abbrev TransRel : Bool :=
  R_0 state newstate extl_n &&
  R_1 state newstate extl_n
```

Note `∧` became `&&` and `=` became `==`. Everything is `abbrev`, so Lean
unfolds it automatically. The theorems link each `Bool` back to the
corresponding `Prop` in Encoding 5:

```lean
theorem TransRel_iff : (TransRel state newstate extl_n = true) ↔
    ScalarRel.TransRel ((state 0), (state 1)) ((newstate 0), (newstate 1)) extl_l extl_n := by ...

theorem InitCond_iff : (InitCond s = true) ↔
    ScalarRel.InitCond ((s 0), (s 1)) extl_n := by ...
```

---

## The certificate data (`System/Data.lean`)

Five definitions, and they are the only place where *your* input about the
problem lands. Everything above was derived mechanically from the module.

```lean
import Core.Basic

def init_pre (e : ((Mat (BitVec 1) 1 1)) × ((Mat (BitVec 1) 1 1))) : Prop := True

def update_pre (e : ((Mat (BitVec 1) 1 1)) × ((Mat (BitVec 1) 1 1))) : Prop := True

def inv (s : (Mat (BitVec 1) 1 1) × (Mat (BitVec 1) 1 1)) : Prop := True

def P : (Mat (BitVec 1) 1 1) × (Mat (BitVec 1) 1 1) → Prop :=
  fun s => (((s.1 0 0) = (BitVec.ofNat 1 0)) ∧ ((s.2 0 0) = (BitVec.ofNat 1 0)))

instance : DecidablePred P := fun s => by unfold P; first | infer_instance | dsimp; infer_instance

def ranking (s : (Mat (BitVec 1) 1 1) × (Mat (BitVec 1) 1 1)) : Nat := sorry
```

- `init_pre` / `update_pre` — assumptions on the external inputs. `True` unless
  you pass `--pre`.
- `inv` — the invariant. `True` unless you pass `--invariant` or `--infer`.
- `P` — the property. This is where the `-P "(and (= s0 ...) (= s1 ...))"`
  argument ended up. The `DecidablePred` instance next to it is needed by the
  proof rule.
- `ranking` — the ranking function, a `Nat` that must strictly decrease while
  `P` is false. `sorry` unless you pass `--ranking` or `--infer`.

`sorry` is Lean's "trust me" hole. A file with `sorry` still compiles, but
nothing is really proved. So a bare run gives you a well-typed skeleton, and
filling in `inv` and `ranking` is the actual work — by hand, or by an LLM with
`--infer`.

With `--ranking "(ite (= s1 (_ bv0 1)) 1 2)"`, for instance, the last line
becomes:

```lean
def ranking : (Mat (BitVec 1) 1 1) × (Mat (BitVec 1) 1 1) → Nat :=
  fun s => (((if ((s.2 0 0) = (BitVec.ofNat 1 0)) then 1 else 2) : Int)).toNat
```

---

## The certificate (`Certificate/Certificate.lean`)

This is the certificate entry file. Its structure is fixed; only the types and the list of
definition names change per module.

**First**, the module is packaged as a `ReactiveModule` record — a four-field
bundle of `init`, `update` and the two preconditions:

```lean
def RM : ReactiveModule (((Mat (BitVec 1) 1 1)) × ((Mat (BitVec 1) 1 1))) ((Mat (BitVec 1) 1 1) × (Mat (BitVec 1) 1 1)) := {
    init := fun e => init e.2
    update := fun x e => update x e.1 e.2
    init_pre := init_pre
    update_pre := update_pre
}
```

The first type argument is the external input type — a pair, `(latched, next)`.
The second is the state type.

**Second**, three tactic macros, each carrying this module's definition names:

```lean
macro "simp_mat"     : tactic => `(tactic| simp [RM, init, update, inv, init_pre, update_pre, P, ranking, MatAdd_apply, MatMul_apply, ...])
macro "simp_defs"    : tactic => `(tactic| (simp only [RM, init, update, inv, init_pre, update_pre, P, ranking] at *; try dsimp at *))
macro "mat_collapse" : tactic => `(tactic| simp only [Mat_1_1_lt_iff, Mat_1_1_le_iff, Mat_1_1_eq_iff, ...] at *)
```

Roughly: `simp_mat` reduces matrix arithmetic in the goal, `simp_defs` unfolds
the module's definitions everywhere, and `mat_collapse` turns 1×1 matrix
comparisons into plain scalar ones.

**Third**, the proof chain. Six declarations, always the same six:

```lean
theorem init_inv : ∀ s, RM.init_pre s → inv (RM.init s)
theorem step_inv : ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e)

def lts := RM.toTS                                          -- module viewed as a transition system

theorem hinv' : lts.StateSet_isInductiveInitial inv         -- inv holds initially and is preserved
theorem hinv  : lts.StateSet_isInvariant inv                -- therefore inv holds on all reachable states

theorem hrank : ∀ s s', (inv s ∧ ¬(P s) ∧ (∃ l, lts.Tr s l s')) →
    ranking s' < ranking s                                  -- ranking drops on every step where P is false

def buchi := rule_buchi lts P inv hinv ranking hrank
```

Each theorem body is a canned tactic script: `simp_mat` / `simp_defs`, then a
cascade of `split_ifs`, `omega`, `linarith`, `norm_num`, `tauto`.

`buchi` is the final artefact. `rule_buchi` lives in `Core/Basic.lean` and its
conclusion is:

```lean
∀ ss μs, lts.ωTrace ss μs → ss ⊧ G (F (AP P))
```

That reads: on every infinite run of the system, `P` holds infinitely often.
For `twobit`: the counter always comes back to `00`.

The argument is the standard one. `hinv` says `inv` over-approximates the
reachable states. `hrank` says that from any `inv` state where `P` is false, a
natural number strictly decreases. A natural number cannot decrease forever, so
`P` must keep recurring.

---

## The tactic (`ZerothHammer.lean`)

One file, identical for every module. It defines a tactic `zeroth_hammer` that
tries nine strategies in order and stops at the first that closes the goal:

| Phase | What it tries |
|---|---|
| 0 | `simp_mat` alone |
| 1 | `omega`, `norm_cast; omega`, `simp_mat; omega`, `simp_mat; linarith` |
| 2 | `push_neg; simp_mat; omega` |
| 3 | `simp_mat`, then up to four nested `split`s, then `omega`/`linarith` |
| 4 | `simp_defs`, then nested `split`s, then `omega` |
| 5 | `simp_defs` → `decide` → `simp_mat` → `mat_collapse` → `split_ifs` → `omega` |
| 6 | `aesop` |
| 7 | `smt` (hands the goal to cvc5) |
| 8 | `sorry` |

It expects `simp_mat`, `simp_defs` and `mat_collapse` to exist. The file ships
trivial stubs for them; `Certificate/Certificate.lean` overrides those with the
module-specific versions shown above.

The certificate's canned proofs currently spell their tactics out rather than
calling `zeroth_hammer`; the tactic is there for hand-written proofs and for
regenerating stuck goals. You can also produce this one file on its own with
`--hammer-file`.

---

## Standalone mode

With `--cert-file out/TwoBitCert.lean` you get four files instead of a project.
They are meant to be dropped into an existing lake project, such as
`python/tests/lean/`:

```
out/TwoBitCert.lean            init/update inlined + Data + the whole certificate, in one file
out/TwoBitCertRel.lean         Encoding 4
out/TwoBitCertScalar.lean      Encoding 3
out/TwoBitCertScalarRel.lean   Encoding 5
```

Differences from project mode: `init`/`update` are inlined into the certificate
rather than imported, the `Data` definitions sit in the same file, and there is
no circuit encoding and no `FBK`.

---

## Summary of what is claimed

| Theorem | Says |
|---|---|
| `init_circ_eq`, `update_circ_eq` | circuit = functional |
| `init_scalar_eq`, `update_scalar_eq` | functional = pack ∘ scalar ∘ unpack |
| `Rel.effect_i_eq`, `Rel.init_i_eq` | each per-variable function = the matching projection of `update`/`init` |
| `Rel.TransRel_func_eq`, `Rel.InitCond_func_eq` | matrix relation ↔ functional |
| `ScalarRel.TransRel_scalar_eq`, `ScalarRel.InitCond_scalar_eq` | scalar relation ↔ scalar functional |
| `ScalarRel.TransRel_func_eq`, `ScalarRel.InitCond_func_eq` | scalar relation ↔ matrix functional |
| `FBK.TransRel_iff`, `FBK.InitCond_iff`, `FBK.R_i_iff`, `FBK.Init_i_iff` | `Bool` relation ↔ scalar relation |
| `init_inv`, `step_inv`, `hinv'`, `hinv` | `inv` is an invariant of the system |
| `hrank` | `ranking` decreases whenever `P` is false |
| `buchi` | every run satisfies `G F P` |

The first seven rows are proved outright by the generator. The last three
depend on `inv` and `ranking`; they are only real once those stop being `True`
and `sorry`.

---

## What changes for other modules

`twobit` is the simple case: every wire is a single bit. Other modules bring in
a few more shapes.

- **Matrix wires.** A wire of shape `[3, 2]` is `Mat Int 3 2`, and it stays a
  matrix in Encoding 1. The scalar encodings flatten it: each entry becomes its
  own component of the tuple, and a `let _m0 : Mat Int 3 2 := ...` line rebuilds
  the matrix where an operation needs the whole thing.
- **Constant matrices.** Non-scalar constants are lifted to top-level
  definitions, `@[simp] def c0 : Mat Int 3 2 := ...`, in a `/- Concrete
  constants -/` block at the top of `System/System.lean`. Scalar constants are
  written inline.
- **Affine maps.** A `Linear(A, B)` op is emitted in reflected form, as
  `matVecAffine m ([[1, 0, 0], ...] : List (List Int)) ([1, 0, 0] : List Int) x`
  — the matrix as plain list literals. `Core/Mat.lean` proves this equals the
  intended affine map.
- **Real-typed modules.** `Real` has no decidable equality in Lean, so every
  generated definition is marked `noncomputable`.
- **`argmax`.** Modules using it get extra unrolled helper definitions plus a
  `@[simp] theorem ..._eq` relating each one to `argmax_1d`, above the `Scalar`
  namespace.
- **Rank-3 wires and higher.** Cannot be flattened. Encodings 3, 5 and 6 are
  replaced by a one-line comment (`-- scalar encoding not available: ...`), and
  the root `System.lean` stops importing them.
