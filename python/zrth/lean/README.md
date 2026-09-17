# `zrth/lean` — Lean 4 Certificate Generation

This package translates Python reactive modules (an SSA-style IR) into Lean 4
source files that encode the module and carry machine-checked proofs of
safety/liveness properties.

Known defects in this pipeline, fixed and open, are catalogued in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).  The one link the `--fbk-proveit` route
is missing -- a proof that the NA model *is* the module -- is designed, with
measurements, in [FBK_EQUIVALENCE.md](FBK_EQUIVALENCE.md).  What cvc5 is used for -- the obligation
pre-check, sharing in the printer, and the solver-informed tactics -- is in
[SMT_ASSIST.md](SMT_ASSIST.md), together with what the synthesis routes
(`--infer sygus`, `--infer smt-linear`) were measured against and the
cold-start plan for abduction that is still one.  The 77-case limit matrix those two are measured on, with its
runners and baseline, is in [`tests/limits/`](../../tests/limits/README.md).

---

## Reactive Module IR

A **`Module`** is the top-level object.  It has:

- **`ctrl`** — the module's mutable state, as a sequence of `Var`s.  A `Var`
  stands for its *latched* wire (the value from the previous step), and
  `X(v)` is its *next* wire, which `update` writes.
- **`extl`** — external/environment inputs, also a sequence of `Var`s.
- **atoms** — currently always a single atom (the combinational logic block).
  Each atom has an **`init`** term list and an **`update`** term list.

A **`Wire`** carries a `Sort` (`Bool`, `Int`, `Real`, `BitVec`) and a shape,
always 2-D here: `[1, 1]` = scalar, `[1, n]` = row vector, `[m, n]` = matrix.

A **`Term`** is one SSA node: a single operation drawn from a theory
namespace (`LIA`, `LRA`, `BV`), reading from some wires (`term.read`) and
writing one output wire (`term.write[0]`).  The term lists form a
topologically-sorted dataflow graph.

---

## Module Files

| File | Role |
|------|------|
| `common.py` | Shared utilities: `LeanContext`, type helpers, `ConstantRegistry`, wire-binding helpers, `flat_layout` (the row-major element order every scalar encoding shares) |
| `ops.py` | The op table: one row per theory variant, one column per backend |
| `native.py` | Translates term lists → Lean functional `let`-binding bodies |
| `circ.py` | Translates term lists → `Box` circuit layers |
| `translate.py` | Top-level `ModuleToLean4` class; assembles functional + circuit encodings |
| `cert.py` | Certificate generation: `CertificateData`, `generate_certificate_lean`, ZerothHammer |

---

## Type Mapping (`common.py`)

`dtype_to_lean_type(wire, simple_types=False)` maps wire types to Lean:

| Python DType | Lean type (default) | Lean type (`simple_types=True`) |
|---|---|---|
| `Bool` (scalar) | `(Mat Bool 1 1)` | `Bool` |
| `Int` (scalar) | `(Mat Int 1 1)` | `Int` |
| `Float` (scalar) | `(Mat Real 1 1)` | `Real` |
| `Int` shape `[n]` | `(Mat Int 1 n)` | — |
| `Int` shape `[m,n]` | `(Mat Int m n)` | — |

All intermediate values are `Mat T m n` (Lean: `Fin m → Fin n → T`) even when
scalar.  This keeps the algebra uniform.  Float-typed modules require
`noncomputable` on all definitions because `Real` lacks decidable equality.

**`_accessor(pos, total)`** builds right-nested product accessors:
- `total=1` → `""` (value itself)
- `total=2` → `.1`, `.2`
- `total=3` → `.1`, `.2.1`, `.2.2`

---

## `LeanContext` (`common.py`)

Created once per module; shared across all code generators.  It extracts:

- `extl_latched`, `extl_next`, `ctrl_latched`, `ctrl_next` — the four wire
  groups.
- `constants` — a `ConstantRegistry` that assigns top-level `@[simp] def c0`
  names to non-scalar `Tensor` constants (scalar tensors are inlined at use).
- `init_wire_names`, `update_wire_names` — `dict[wire_id, lean_accessor]`
  mapping each input wire to its Lean expression (e.g. `"ctrl.1"`).
- `uses_real` — True if any wire is Float-typed (triggers `noncomputable`).

`_bind_wires(params)` builds the wire-id → accessor dict from a list of
`(param_name, [wire, ...])` pairs.

---

## Encoding 1 — Functional (`native.py`, `translate.py`)

`_translate_terms(terms, input_bindings, block_outputs, constants)` produces a
Lean function body as a string of `let` bindings followed by the output tuple:

```lean
  let x0 : (Mat Bool 1 1) := (fun _ _ => decide (ctrl.1 0 0 < ctrl.2 0 0))
  let x1 : (Mat Int 1 1) := (fun _ _ => (1 : Int))
  ...
  (x6, x5)
```

Each term is looked up in the op table's `mat` column (`ops.mat_emitter`,
keyed by theory and variant name).  Operations produce `Mat T 1 1` values
even for scalar results — conditions extract `0 0`, boolean ops wrap in
`fun _ _ => ...`.  Non-scalar ops (`MatMul`, `Linear`, `ReLU`) use their
native matrix forms.

`ModuleToLean4.atom_to_lean_functional()` wraps the body into:

```lean
@[simp] def init (extl_n: Unit) : (Mat Int 1 1) × (Mat Int 1 1) := ...
@[simp] def update (ctrl: ...) (extl_l: ...) (extl_n: ...) : ... := ...
```

---

## Encoding 2 — Circuit (`circ.py`, `translate.py`)

The same computation is re-expressed as a `Box` (categorical wiring diagram)
inside `namespace Circ`.  The translation:

1. Builds a reversed dependency graph layer by layer.
2. Emits swap/dup/delete layers to route wires into the right positions.
3. Each operation becomes `Box.add`, `Box.lt`, etc. composed with `⊗` (par) and
   `≫` (seq).

`ModuleToLean4.atom_to_lean_circuit()` emits one `@[simp] def init_l0 ...`
per layer and a composed `def init := init_l0 ≫ init_l1 ≫ ...`.

**Equivalence theorems** (`to_lean_equiv_theorems`) prove that the circuit and
functional encodings compute the same result:

```lean
theorem init_circ_eq : ∀ (extl_n : Unit),
    Circ.init.fn ⟨extl_n, ()⟩ =
    let r := init extl_n
    (r.1, (r.2, ())) := by
  intro extl_n
  simp_circ [Circ.init, Box.seq]
  simp_circ [Circ.init_l0]
  ...
  simp [init]
```

The helper macro `simp_circ` unfolds one layer name plus all Box/ValTuple
plumbing lemmas.

---

## Encoding 3 — Scalar (`native.py`, `translate.py`)

`_translate_terms_scalar` is like `_translate_terms` but uses bare scalar types
(`Bool`, `Int`, `Real`) instead of `Mat T 1 1` for all 1×1 wires:

```lean
namespace Scalar
@[simp] def update (ctrl: (Mat Int 1 1) × (Mat Int 1 1)) ... : Int × Int :=
  let x0 : Bool := (decide (ctrl.1 0 0 < ctrl.2 0 0))
  let x1 : Int  := (1 : Int)
  let x2 : Int  := (ctrl.1 0 0 + x1)
  ...
  (x7, x8)
end Scalar
```

Key differences from functional:
- Input parameter types are **unchanged** — they still accept `Mat T m n`.
- Input scalars are pre-extracted at the call site via `ctrl.1 0 0`.
- All intermediate `let` bindings use `Bool`/`Int`/`Real` directly.
- Output type is a product of bare scalars (e.g. `Int × Int`).
- Non-scalar wires (actual matrices, `MatMul` results with shape `[m,n]`) fall
  back to matrix types transparently.

`_bind_wires_scalar` builds the input bindings, appending `" 0 0"` for scalar
wires so downstream ops receive bare values.

**Equivalence theorems** (`to_lean_scalar_equiv`) connect the scalar and
functional encodings:

```lean
theorem update_scalar_eq : ∀ ctrl extl_l extl_n,
    update ctrl extl_l extl_n =
    let r := Scalar.update ctrl extl_l extl_n
    (fun _ _ => r.1, fun _ _ => r.2) := by
  intro ctrl extl_l extl_n
  simp only [Scalar.update, update]
  try rfl
  try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])
  try (funext i j; simp [Fin.fin_one_eq_zero])
```

The reconstruction `fun _ _ => r.i` wraps each scalar output back into a
`Mat T 1 1`.  The proof reduces via `Fin.fin_one_eq_zero` (every `i : Fin 1`
equals `0`), making the two sides propositionally equal.

**Scalar encoding lives in its own file** (`<Name>Scalar.lean`), separate from
the certificate.  It contains: functional `init`/`update`, `namespace Scalar`,
and the equivalence theorems.

---

## `ModuleToLean4` (`translate.py`)

The main entry point.  Accepts a `Module` or a pre-built `LeanContext`.

| Method | Returns |
|--------|---------|
| `to_lean_functional()` | Constants block + `init`/`update` defs |
| `to_lean_circ()` | Circuit namespace + equivalence theorems |
| `to_lean_scalar()` | `namespace Scalar` block + equivalence theorems |
| `to_lean(circuit=True, scalar=True)` | All three encodings combined |
| `atom_to_lean_functional()` | Just `init`/`update` bodies |
| `atom_to_lean_circuit()` | Just `namespace Circ` block |
| `atom_to_lean_scalar()` | Just `namespace Scalar` block |
| `to_lean_equiv_theorems()` | `init_circ_eq` / `update_circ_eq` theorems |
| `to_lean_scalar_equiv()` | `init_scalar_eq` / `update_scalar_eq` theorems |

---

## Certificate Generation (`cert.py`)

A **certificate** adds five predicates and a proof skeleton on top of the
module encoding.  `CertificateData` holds them as either:

- `None` → a `True` / `sorry` placeholder is emitted.
- `str` → a raw Lean expression string is emitted verbatim.
- `list[Term]` → compiled from the Python term IR into a Lean body (via
  `_translate_terms` with `e`/`s` as the param names).

`generate_certificate_lean(project_name, module_name, ctx, cert_data, ...)`:

1. Emits imports (`Mathlib`, `Core.Basic`, module import or inline, hammer).
2. Emits `init_pre`, `update_pre`, `inv`, `P`, `DecidablePred P`, `ranking`.
3. Emits `def RM : ReactiveModule Extl State` wiring `init`/`update` in.
4. Emits module-specific macros `simp_mat`, `simp_defs`, `mat_collapse` that
   include all local definition names and matrix reduction lemmas.
5. Emits the ZerothHammer tactic (inlined or via import).
6. Emits proof skeletons: `init_inv`, `step_inv`, `hinv`, `hrank`, `buchi`.

**Wire bindings in certificates:**  State is accessed as `s`, externals as
`e.1` (latched) and `e.2` (next).

### `smt_predicates_to_lean`

Converts SMT-LIB string predicates (from a CEGIS loop) into Lean expression
strings by parsing them with cvc5 and translating the AST.

---

## Tactic plans (`tactics.py`)

The three obligations used to be closed by one hardcoded chain, textually
identical in every certificate — wasteful in both directions: an integer
module with a linear ranking paid for `decide` and `bv_decide` on every goal,
and a shape nobody had anticipated had no way to ask for the step it needed.

`plan_for` reads the module and its predicates and emits two macros,
`cert_prep` (canonicalise) and `cert_close` (decide), which the three proofs
share; each certificate carries a comment saying what was detected.

| detected | consequence |
|---|---|
| Int / Bool in the state | `omega` is available |
| Real | `omega` is dropped; `linarith`, `norm_num` and `norm_cast; linarith` take over |
| Real *and* equalities in the predicates | `simp_all` moves into **prep**, so the closers see numerals, not an opaque `⌊4 - x⌋` |
| a product of two state-dependent terms | `nlinarith`, `positivity` |
| finite state (Bool/BitVec only) | `decide`, last, because it is the most expensive step |
| an `Ite`/`ReLU`/`Min`/`Max` anywhere | `split_ifs` in prep |
| `∧` / `∨` in the predicates | `casesm*` and `simp only [not_and_or]` in prep |
| an `=` in the predicates | the `ne_iff_lt_or_gt` split, *after* `norm_num at *` |
| term count | `maxRecDepth` scales with it |
| branch points in the predicates | scales the heartbeat budget — this, not module size, is what a net costs |
| nothing known to be slow | heartbeat budget stays at 400 000 for economy (it does *not* bound failures — see "Nets over ℝ") |

Two decisions are worth recording because measurement contradicted the
obvious guess. **`linarith` belongs in every plan** — gating it on `Real`
looked right and cost two integer certificates that were closing on it; an
integer goal `omega` cannot phrase is still ordinary linear arithmetic.
**`bv_decide` belongs in none** — it only ever reaches goals every cheaper
prover has already failed on, and there it bit-blasts: including it took a
BitVec certificate from 10 s to 977 s while closing nothing `decide` had not
already closed.

Because the plan depends on the predicates, `Certificate.lean` is not stable
across a change of invariant: `--infer` rewrites it alongside `Data.lean`.
`tests/test_lean_tactics.py` pins these decisions in the fast suite, so a
plan regression does not wait for a Lean build.

## ZerothHammer (`cert.py`)

A Lean 4 elaborator tactic that cascades proof strategies:

| Phase | Tactic |
|-------|--------|
| 0 | `simp_mat` alone (trivial / True goals) |
| 1 | `omega`, `norm_cast; omega`, `simp_mat; omega`, `simp_mat; linarith` |
| 2 | `push_neg; simp_mat; omega` |
| 3 | `simp_mat` + nested `split` (up to 4 levels) + `omega`/`linarith` |
| 4 | `simp_defs` + nested `split` + `omega` |
| 5 | `simp_defs` → `simp_mat` → `mat_collapse` → `split_ifs` → `omega` |
| 6 | `aesop` |
| 7 | `smt` (cvc5) |
| 8 | `sorry` (give-up) |

The three macro names (`simp_mat`, `simp_defs`, `mat_collapse`) are expected to
be defined in the importing file; `ZerothHammer.lean` ships stub definitions
that certificate files override.

---

## Key Lemmas (in `Core.Basic`)

| Lemma | Effect |
|-------|--------|
| `MatAdd_apply`, `MatMul_apply` | Reduce matrix `+`, `*` to pointwise scalar ops |
| `Mat_1_1_lt_iff`, `_eq_iff`, `_le_iff`, `_ne_iff` | Collapse `(m : Mat T 1 1) 0 0 op n` to bare scalar comparison |
| `ite_fun_apply` | Push `if` through function application |
| `Fin.sum_univ_one/two/three` | Reduce finite sums |

---

## `uv run verith` CLI (`cli.py`, `main.py`, `project.py`)

`verith` is the command-line tool that drives the full pipeline: Python module
→ Lean project or standalone certificate files.

### Module file format

```python
# mymodule.py
from zrth import Module, Int, Var, X
from zrth.analyzer import convert_method

def init():
    return 0

def update(old_x):
    x = old_x + 1
    if x == 10:
        return 0
    return x

def module() -> Module:
    state = Var(Int([1, 1]))
    init_terms   = convert_method(init,   {},               [X(state)])
    update_terms = convert_method(update, {"old_x": state}, [X(state)])
    return Module.sequential([state], init_terms, update_terms)
```

### Common invocations

```bash
# Bare Lean project (all certificate fields left as sorry)
uv run verith mymodule.py -o out/ -p MyProject

# Specify the property to prove (SMT-LIB 2 Bool over s0..sN-1). Which flag it
# goes under is the choice of proof rule -- see "Safety or Büchi" below.
uv run verith mymodule.py --buchi "(= s0 0)" -o out/ -p MyProject
uv run verith mymodule.py --safety "(<= s0 10)" -o out/ -p MyProject

# Also supply invariant and ranking manually
uv run verith mymodule.py --buchi "(= s0 0)" \
    --invariant "(and (>= s0 0) (<= s0 10))" \
    --ranking   "(ite (= s0 0) 0 s0)" \
    -o out/ -p MyProject

# A safety certificate is an invariant alone -- no ranking function exists
uv run verith mymodule.py --safety "(<= s0 10)" \
    --invariant "(and (>= s0 0) (<= s0 10))" \
    -o out/ -p MyProject

# AI inference with Claude (requires ANTHROPIC_API_KEY + pip install zrth[ai])
uv run verith mymodule.py --buchi "(= s0 0)" --infer -o out/ -p MyProject

# ... and for a safety property, through the cvc5-checked loop
uv run verith mymodule.py --safety "(<= s0 10)" --infer ai-cegis -o out/ -p MyProject

# AI inference with a local LLM via Ollama (requires pip install zrth[ai-local])
uv run verith mymodule.py --buchi "(= s0 0)" --infer \
    --model qwen3-coder --base-url http://localhost:11434/v1 \
    -o out/ -p MyProject

# No LLM: learn a ranking function and certify it before it is offered
uv run verith mymodule.py --buchi "(= s0 0)" --infer nuterm -o out/ -p MyProject

# Standalone self-contained certificate (no project scaffold)
uv run verith mymodule.py --buchi "(= s0 0)" --cert-file out/MyCert.lean
# → writes out/MyCert.lean  (certificate)
# → writes out/MyCertScalar.lean  (scalar encoding + equivalence theorems)
```

### What gets generated

**Full project** (`-o`, `-p`):

```
<output_dir>/<ProjectName>/
  lakefile.toml
  lean-toolchain
  ZerothHammer.lean          # standalone zeroth_hammer tactic
  <ProjectName>.lean          # functional + circuit encodings of init/update
  Certificate/
    Certificate.lean          # certificate predicates + proofs
    Scalar.lean               # scalar encoding + equivalence theorems
  Certificate.lean            # re-export shim
  Core/                       # library files (Mat, Box, ReactiveModule, …)
```

**Standalone certificate** (`--cert-file path/Name.lean`):

```
path/Name.lean               # self-contained: init/update + certificate
path/NameScalar.lean         # scalar encoding + equivalence theorems
```

### Safety or Büchi

The property comes in under one of two flags, and the choice is not a label:
it decides which proof rule the certificate is built on, and so what the
certificate consists of.

| | `--safety P` | `--buchi P` |
|---|---|---|
| proves | `G P` — `P` in every reachable state | `G (F P)` — `P` infinitely often |
| proof rule | `rule_globally` | `rule_buchi` |
| certificate | an invariant that **implies** `P` | an invariant **and** a ranking function that decreases wherever `P` is false |
| obligations | `init_inv`, `step_inv`, `inv_imp_P` | `init_inv`, `step_inv`, `hrank` |
| `--ranking` | rejected — there is nowhere to put one | the other half of the certificate |
| routes | `--fbk-proveit` (ic3ia finds the invariant), `--infer ai-cegis`, `--infer nuterm`, `--infer sygus`, `--infer smt-linear`, or `--infer vampire` | `--infer ai`, `--infer ai-cegis`, `--infer nuterm`, `--infer smt-linear`, or `--infer vampire` |

Neither flag *requires* a route: with neither `--infer` nor `--fbk-proveit`,
the project is generated from whatever predicates were supplied, and the two
flags simply say which certificate to emit. `--safety` and `--buchi` are
mutually exclusive, and `--fbk-proveit` takes only `--safety` — ic3ia decides
reachability of `¬P`, which is not a question about recurrence.

The distinction is not academic. Countdown starts at 100 and counts down, so
`(= s0 0)` is reached over and over — true under `--buchi`, and false under
`--safety` at step 0.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `-o` / `--output-dir` | `.` | Where to create the project |
| `-p` / `--project-name` | `Rea` | Lean package name |
| `-d` / `--module-def` | `module` | Name of the factory function in the Python file |
| `--safety` | — | SMT-LIB 2 Bool over `s0..sN-1`, to hold in every reachable state (`G P`) |
| `--buchi` | — | SMT-LIB 2 Bool over `s0..sN-1`, to hold infinitely often (`G (F P)`) |
| `--invariant` | — | SMT-LIB 2 Bool invariant (skips invariant inference) |
| `--ranking` | — | SMT-LIB 2 Int ranking (skips ranking inference) |
| `--infer` | — | Which route finds the certificate: `ai`, `ai-cegis` (default when the flag is given without a value), `nuterm`, `sygus`, `smt-linear`, `vampire`, or `fbk-proveit` (see below) |
| `--model` | `claude-sonnet-4-6` | LLM model for inference; rejected by a route that calls none |
| `--base-url` | — | OpenAI-compatible endpoint for local LLMs |
| `--cert-file` | — | Write standalone `.lean` file instead of full project |
| `--hammer-file` | — | Regenerate `ZerothHammer.lean` only |
| `--artifacts` | `use` | What to do with the project's `artifacts/`: `use` lets a route resume from what an earlier run left, `ignore` searches afresh, `reset` empties it first |
| `--sygus-grammar` | `congruence` | `--infer sygus`: what an atom of the synthesised invariant may be; `linear` drops the `(= (mod … k) 0)` atoms |
| `--sygus-conjuncts` | `3` | `--infer sygus`: how many atoms the invariant may be a conjunction of — the bound is what makes the space finite, and so decidably empty |
| `--linear-rows` | `2` | `--infer smt-linear --safety`: how many linear inequalities the invariant may be a conjunction of |
| `--vampire` | `$VAMPIRE`, then `vampire` on PATH | `--infer vampire`: the Vampire binary, or a directory holding one |
| `--vampire-timeout` | `120` | `--infer vampire`: seconds for all Vampire calls together; each call's limit climbs 2 s, 10 s, 60 s, then the rest |
| `--vampire-cores` | `4` | `--infer vampire`: processes each call's portfolio spreads over |
| `--proveit-dir` | — | `--infer fbk-proveit`: path to a `lean-ltl-certifying` checkout |
| `--ic3ia` | — | `--infer fbk-proveit`: path to the `ic3ia` binary, forwarded to `proveit.py` |
| `--fbk-simplify` | `cvc5` | `--infer fbk-proveit`: `none` leaves the NA model's transition the shape the module's own terms give it |
| `--build-cert` | off | `lake update` + `lake build Certificate` in the generated project |

Each route is one row of `infer_route.ROUTES`, and a flag in the bottom half
of the table belongs to a route rather than to verith: passed without its
route selected it is an error, not a no-op.  `--fbk-proveit DIR` is the older
spelling of `--infer fbk-proveit --proveit-dir DIR` and still works.

### State variable naming in SMT-LIB predicates

`ctrl` wires are named `s0`, `s1`, … in left-to-right order (matching the
`vars` list passed to `Module.sequential`).  For tuple/matrix wires, use
SMT-LIB tuple selectors: `((_ tuple.select 0) s0)`.  External inputs are
`e0..eM-1` (next) and `el0..elM-1` (latched).

### Inferring without an LLM (`--infer nuterm`)

`--infer nuterm` searches for the certificate the way a termination prover
does, and proves it before offering it:

1. **The invariant** — Houdini over a candidate lattice of sign and pairwise
   facts (`benchmarks/svcomp/_invariants.py`): seed the candidates, drop the
   ones that do not hold at entry or are not preserved by a step, and certify
   the survivors as one inductive `Safety` claim.  With `--safety`, the
   property itself is seeded as a candidate, so it is an invariant exactly
   when it survives with the rest.
2. **The ranking function** (`--buchi` only) — a small ReLU network is trained
   on rollouts of the module to drop on the rounds where the property does not
   hold, its weights are rounded to integers, and the candidate is composed
   into the module as two ordinary atoms, `V(s)` and `V(s')`.  The obligation
   `V(s) - V(s') >= 1` on that domain goes to a decision procedure over the
   composed module's wires: Farkas-certified linear regions, with CEGAR
   finding the regions (`benchmarks/svcomp/_farkas.py`).  Only a candidate it
   certifies is returned.

So unlike the `ai` routes, what reaches the certificate has already been
proved — `--pre-check cvc5` and `lake build Certificate` confirm it rather
than discovering it.  The trade is reach:

| | `--infer nuterm` | `--infer ai-cegis` | `--infer fbk-proveit` |
|---|---|---|---|
| needs | nothing but the repo | an API key or a local LLM | an `ic3ia` build and a `lean-ltl-certifying` checkout |
| state it reads | scalar integers | whatever cvc5 encodes | whatever the NA encoding expresses |
| property | `--safety` or `--buchi` | `--safety` or `--buchi` | `--safety` |
| invariants it can find | Houdini's lattice | whatever the LLM proposes and cvc5 confirms | ic3ia's interpolants |
| deterministic | yes (a fixed seed) | no | yes |

The procedure refuses by name what it has no rule for — a matrix-shaped or
real-valued component, a next value that reads an awaited input, an operation
with no cell rule — rather than degrading quietly.  `--invariant`,
`--ranking` and `--pre` are rejected alongside it: the route computes the
whole certificate, and its invariant holds at entry for *every* input, which
is stronger than any precondition would make it.

### Searching a shape instead of proposing one (`--infer smt-linear`, `--infer sygus`)

Two more routes need no LLM, and they differ from `nuterm` in what they do
when they fail.  Both fix the *space* the certificate may live in and hand
the whole space to cvc5 at once, so "not found" is an answer with content:

* **`--infer smt-linear`** fixes the shape and leaves the coefficients open,
  which makes the search one query — `exists c. forall s. obligations(c . s)`.
  Every scalar component is a column whatever its sort: an `Int` is itself, a
  `Bool` is `0`/`1` (`(ite s0 1 0)`), a bitvector is its unsigned value
  (`(ubv_to_int s0)`) — so a Bool-state module, which `--infer nuterm`
  refuses, is in reach, and so is a BitVec one.
  `--buchi` asks it for a ranking function `c0 + c1*s0 + …` over a fixed
  invariant (supplied with `--invariant`, resumed from `artifacts/`, or
  `true`); `--safety` asks it for the invariant itself, as a conjunction of
  `a0 + a1*s0 + … >= 0` rows, one width at a time up to `--linear-rows`.
* **`--infer sygus`** (`--safety` only) fixes a *grammar* instead and hands
  cvc5's SyGuS invariant track the three formulas `G P` is made of:
  `addSygusInvConstraint(inv, pre, trans, post)`.  Its grammar carries
  congruences — `(= (mod a0 + a1*s0 + … k) 0)` — which is the one fact
  Houdini's lattice cannot state, and the reason the route exists beside
  `nuterm`.  The conjunction is bounded (`--sygus-conjuncts`, default 3),
  which bounds the certificate and, more to the point, makes the space
  *finite*: cvc5 can then report it empty rather than merely unsearched.

**A refuted search is a proof, and it is kept.**  When the template query
comes back `unsat`, nothing of that shape satisfies the obligations — a fact
about the module, not a failure to look.  It is written to `artifacts/` as a
`no_solution` note, and `--infer ai-cegis` reads those notes into its prompt
on the next run, so an attempt is not spent proposing what a decision
procedure has already ruled out.  A search that runs out of budget writes an
`unknown` note instead and is *not* carried into any prompt: the difference
between a proof and a timeout is the whole value of the note.

```bash
# 1. the cheap question first: is there a linear ranking function at all?
uv run verith m_toward5.py --buchi "(= s0 5)" \
    --invariant "(and (>= s0 0) (<= s0 10))" --infer smt-linear -o out/ -p Rea
# error: --infer: ... No ranking function linear in the state ... cvc5 refuted
# the whole shape at once ... so this is a proof that the space is empty.

# 2. the expensive one, now knowing that
uv run verith m_toward5.py --buchi "(= s0 5)" \
    --invariant "(and (>= s0 0) (<= s0 10))" --infer ai-cegis -o out/ -p Rea
# .. resuming from note-0003-smt-linear.md: a space an earlier run ruled out
```

The same workspace carries what was *found*.  `--infer sygus` writes its
invariant as a resumable `inv`, so the next run takes it as given:

```bash
uv run verith m_step2.py --safety "(not (= s0 1))" --infer sygus -o out/ -p Rea
# [sygus] inv: (= (mod (+ (- 2) (* (- 1) s0)) 2) 0)        -- `x` is even

uv run verith m_step2.py --safety "(not (= s0 1))" --infer ai-cegis \
    --pre-check cvc5 -o out/ -p Rea
# .. resuming from inv-0002-sygus.smt2 (proved): taking it as the certificate
# [CEGAR] all obligations UNSAT — accepted        -- and no LLM call was made
```

What each is for, measured on the `tests/limits` fixtures:

| | `--infer smt-linear` | `--infer sygus` |
|---|---|---|
| property | `--safety` or `--buchi` | `--safety` |
| finds | coefficients of a fixed shape | any term its grammar generates |
| answers `no` | yes, as a proof — 2–30 ms for a ranking function | yes, as a proof — the bounded grammar is finite (10–166 ms) |
| cost | milliseconds per width; a second invariant row can cost more than 30 s | 11 ms for `m_step2`'s congruence |
| engine | one quantified query, or — when cvc5 will not state it, which a bitvector column always does — a counterexample loop over a bounded coefficient box | cvc5's SyGuS invariant track |
| state it reads | scalar `Int`, `Bool` and `BitVec` | scalar integers, transition in `LIA` |
| seeds | `--invariant` (strengthened, not replaced), `--pre` | `--pre` |

Neither is a better `nuterm`: within the scalar-integer class they overlap
with it and lose on ranking functions, where a learned rank handles shapes a
linear template has no room for.  What they add is the *no*, and one
invariant shape — a congruence — that nothing else here can state.

### Proposing candidates and proving them (`--infer vampire`)

[Vampire](https://github.com/vprover/vampire) is a first-order theorem
prover that reads SMT-LIB. Handed the negation of an obligation it either
refutes it — the obligation holds — or runs out of time: it never answers
*false* and never returns a model. So this route does not ask it to find
anything. It proposes candidates, drops the ones a cheap test refutes, and
keeps a candidate only once Vampire proves it:

* **invariant** — Houdini over facts that hold on simulated runs of the
  module: bounds stated with the program's own constants, congruences,
  sums and differences of two components against zero, bounds on each side
  of a Bool flag, and under `--safety` the property's conjuncts.
* **ranking function** — affine forms of one or two components, `K*x + y`,
  and piecewise `(ite c f g)` over the conditions the property and
  transition branch on and the order of each pair of components. Each is
  shifted to be positive on the rounds where the property fails and offered
  zeroed where it holds.
* **smaller certificate** — Vampire's unsat cores name the facts a proof
  used; the invariant handed on is their closure, then minimised one fact
  at a time.

Because a false candidate costs a whole time limit, nothing is put to
Vampire that a sampled round refutes: states near the reached ones — and
from a range far wider than the program's constants, which is what exposes
a shift fitted to small numbers — where the invariant holds, with inputs as
`--pre` allows. On `ChenFlurMukhopadhyay-SAS2012-Ex3.01` the wide sample
took the ranking search from 46 Vampire calls and 84 s to 3 calls and
0.1 s. A `--safety` property a run of the module violates is reported as
not holding, before Vampire is started.

Vampire runs as its `smtcomp` portfolio. A whole search runs at one time
limit per call — 2 s, then 10 s, then 60 s, then what is left of
`--vampire-timeout` — because a short limit is a different schedule rather
than a truncated long one. Measured on the 399 obligations of certificates
the benchmark matrix had verified, the portfolio refuted 369 within 2 s,
nearly all in 10–20 ms; the rest were encodings it cannot read or
obligations that were false.

```bash
uv run verith mymodule.py --buchi "(= s0 0)" --infer vampire \
    --vampire ~/vampire/vampire -o out/ -p Rea
# [vampire] Houdini kept 2 of 2: (<= 0 s0) (<= s0 100)
# [vampire] ranking function proved: s0
# [vampire] invariant minimised to 1 of 2 facts
```

It reads scalar `Int` and `Bool` state and inputs; Vampire has no bitvector
theory, and a matrix-shaped or `Real` component is refused as it is by
`smt-linear`. A tuple the transition builds internally is folded away by
cvc5's rewriter before Vampire sees it. When nothing is found, the note in
`artifacts/` says `unknown`, not `no_solution`: a prover that cannot refute
proves no space empty.

---

## Building the certificate (`--build-cert`)

`--build-cert` is route-independent: it builds whatever certificate the run
produced, with `lake update` then `lake build Certificate` in the generated
project. All three routes that fill a certificate qualify —

```bash
# predicates supplied
uv run verith mymodule.py --buchi "(= s0 0)" --invariant "(<= s0 100)" \
    --ranking "s0" --build-cert -o out/ -p Counter

# predicates inferred
uv run verith mymodule.py --buchi "(= s0 0)" --infer --build-cert -o out/ -p Counter

# predicates from ic3ia
uv run verith mymodule.py --safety "(not (= s0 15))" --build-cert -o out/ -p Counter \
    --fbk-proveit ~/zeroth/proof-prototyping/lean-ltl-certifying
```

— and the two that do not are refused rather than silently doing nothing:

| refused | because |
|---|---|
| no `--invariant`/`--ranking`, no `--infer`, no `--fbk-proveit` | every obligation in a bare project is `sorry`. Lake compiles that and exits 0, so "built" would mean nothing |
| `--cert-file`, `--hammer-file` | they write a file and return; there is no project to build |

`--invariant` without `--ranking` (or either without a property) is the
first row: a missing predicate is a `sorry`, not a weaker proof.

Two checks sit past the parse-time gate, because a flag cannot promise what
inference returns and lake cannot be trusted to fail loudly:

* **After the route, before the build.** `--infer` can come back without an
  invariant or a ranking — an LLM that answers nothing usable — and the
  error names which one is missing instead of building a proof of `sorry`.
  (`--fbk-proveit` is exempt: its certificate is ic3ia's, and no
  `CertificateData` describes it.)
* **After the build.** `zeroth_hammer` leaves a `sorry` where it cannot
  close an obligation, and lake reports that as a *warning* and exits 0. So
  `build_certificate` also scans the log for `sorry` under `Certificate/` —
  a dependency's own `sorry` (lean-smt ships one) is not ours — and raises
  if it finds any.

A build that fails is a `LakeBuildError`, and the run exits non-zero with
the Lean diagnostics restated: lake's log ends with thousands of replayed
Mathlib jobs, so a bare "exit 1" as the last line would bury them.

Off by default because the build resolves and compiles `cslib`, Mathlib,
`lean-smt` and cvc5 — minutes against a warm `.lake/packages`, far longer
against a cold one. Because the require set is pinned to the checkout's, one
already-built `.lake/packages` serves both: copying (or cloning, on APFS)
the checkout's into the generated project makes the first `lake update` a
no-op fetch.

## Certifying through `lean-ltl-certifying` (`--infer fbk-proveit`)

`--fbk-proveit=<dir>` swaps verith's own invariant machinery for the
`proveit.py` driver of the [`lean-ltl-certifying`][ltl] repository, which
model-checks the system with ic3ia and renders the inductive invariant it
finds back as a Lean certificate:

```bash
uv run verith mymodule.py --safety "(not (= s0 15))" -o out/ -p Counter \
    --fbk-proveit ~/zeroth/proof-prototyping/lean-ltl-certifying \
    --ic3ia ~/ic3ia/build/ic3ia
```

Add `--build-cert` to have the certificate compiled rather than merely
written (see [Building the certificate](#building-the-certificate---build-cert)).

```
mymodule.py ──verith──▶ out/Counter/                       (bare, as usual)
                        out/Counter/ProveIt/CounterNA.lean (the NA encoding)
                                  │
                                  ▼   proveit.py
                        lean2vmt ─▶ ic3ia ─▶ vmt2lean
                                  │
                                  ▼
                        out/Counter/ProveIt/CounterCert.lean
                                  │
                                  ▼   "processing" — a copy, for now
                        out/Counter/Certificate/Certificate.lean
```

The installed file is the project's *own* certificate — the same
`Certificate/Certificate.lean` that `Certificate.lean` root-imports and that
`lake build Certificate` reaches, overwriting the `sorry` stub the bare
project would have carried. For that reach to end in a build rather than an
unknown identifier, `create_project` is given the checkout and the lakefile
gains two entries:

| lakefile entry | resolves |
|---|---|
| `[[require]] name = "LTL_Certifying", path = <checkout>` | `import LTLCertifying.Safety.Lemmas` and its two siblings. A path require because the route already holds the checkout, and because that checkout pins `cslib` and `smt` to the revisions the generated project already requires — the two package sets resolve as one |
| `[[lean_lib]] name = "<Proj>NA", srcDir = "ProveIt"` | `import <Proj>NA`. `proveit.py` compiles an out-of-tree model into an olean and imports it under the file *stem*, so the stem is the module name the certificate carries; the lib maps it back onto `ProveIt/<Proj>NA.lean`. `project.na_module_name` is the single owner of that name, used to write the model and to declare the lib |

`import Smt` needs nothing new — verith's own route already requires
`smt` — and neither does the cvc5 shared library the `smt` tactic dlopens:
lean-cvc5 declares `precompileModules := true`, so `lake build` loads it
without the `--plugin=` that `proveit.py` has to pass to a bare `lean`.

Of the *invariant* the project is still **bare**: it comes from ic3ia, so
`--infer`, `--invariant`, `--ranking` and `--pre` are rejected rather than
silently ignored, and `--safety` is required.

### Encoding 6 — NA, for the FBK toolchain (`translate/fbk.py`)

`proveit.py`'s first step, `lake exe lean2vmt`, pattern-matches on a very
specific Lean shape. This encoding is emitted only for that route, and is
not one of the five above — though the project does build it, as the
`<Proj>NA` lean_lib, because the certificate imports it:

| `lean2vmt` requires | why | what it would otherwise get |
|---|---|---|
| binders `state` / `statenext` | `emitDefs` looks them up by name to tell current from next | `state`, `newstate`, `s` |
| state read as `var_i state` | `exprToSMT`'s `.fvar` case returns the bare binder name and **drops the index**, collapsing every slot onto one variable | `(state i)` |
| next state as `var_i statenext` | this is how `collectLatchesIndices` finds the latches at all | `(newstate i)` |
| `INIT` / `TRANS` / `PROPERTY` | they become `:init`, `:trans`, `:invar-property` | `InitCond`, `TransRel`, nothing |
| top-level `StateType`, `abbrev M : … NA …` | the certificate template imports the model and names both | everything inside a namespace |
| self-contained imports | it is elaborated inside the `lean-ltl-certifying` package | imports `Core.Basic`, `System.Scalar`, … |

The right-hand column is what the project's own Bool-valued relational
encoding emitted. That encoding — `System/FBK.lean`, one of six until this
one replaced it — missed every row, which is why the two could never be the
same file; nothing else read it, so it is gone and its name is here.

The property is translated to **`Bool`**-valued Lean (`&&`, `!`,
`decide (… ≤ …)`), not the `Prop` form `smt_to_lean` emits: `lean2vmt` reads
`decide`'s *instance* argument, so `decide` applied to a compound proposition
hands it `instDecidableAnd`, which it prints as an unapplied leaf.  `>` / `≥`
are normalised to `<` / `≤` with swapped operands for the same reason.

### What the route refuses

Everything `lean2vmt`/`vmt2lean.py` cannot represent aborts with a non-zero
exit instead of producing a transition system that parses but does not
describe the module:

| rejected | because |
|---|---|
| external input wires | `lean2vmt` models only `state`/`statenext` |
| state elements other than `Int`/`Bool` | `vmt2lean.py`'s `tp()` maps back only those two |
| — | a state *mixing* `Int` and `Bool` is fine: `TypeMap` gets an arm per slot, and the `TypeMap.match_1` the equation compiler generates never reaches the VMT, because `emitDefs` keeps only declarations whose return type whnfs to `Prop`/`Int`/`Bool` |
| — | a ctrl wire *wider* than 1×1 is fine too, and used to be the largest refusal here: the state is flattened to one VMT variable per **element**, so `R_k` compares two scalars where it once would have compared two tuples |
| an op `smt_encode` has no term for (`Transpose`, `Uninterpreted`) | the transition is `smt_encode`'s term for each element, printed by `smt_to_lean_bool` — no term, no model |
| an op the Bool printer cannot render | same boundary from the other side: the printer raises rather than emit something `exprToSMT` would misread. `Linear`, `Argmax`, `Max`/`Min`, `Xor` and `Ne` all pass it — as an affine sum, a nested `ite` chain, `max`/`min`, `!(a == b)` and a negated equality — which is why they are no longer refused |
| — | `Ne`, `ReLU`, `Max` and `Min` need a `lean2vmt` carrying the commit *"translate mod, max/min and a negated decidable instance"*; against an older one they go back to being printed as `instDecidableNot` / `max` / `min` |
| a property outside that same fragment | ditto — `smt_to_lean_bool` raises rather than guess. `mod` is refused here even though `lean2vmt` translates it and ic3ia proves such a property: MathSAT eliminates the mod from the *witness*, as `x + (-2) * to_int ((1/2) * to_real x) = 0`, and `vmt2lean.py` renders neither those operators nor a Real inside a `Bool` `INVAR` |
| `lake` missing, `mathsat` not importable, `proveit.py` failing or writing nothing | checked before and after the subprocess |

The transition itself is not written by `translate/fbk.py`. Each state slot's
next value is `smt_encode`'s term for that element — the encoder `--pre-check`
and `--infer ai-cegis` run on — simplified by cvc5 and printed by
`smt_to_lean_bool`. So the model `lean2vmt` reads and the obligations cvc5
answers about the same module are one encoding rather than two readings, and
the refusal list above is exactly what those two components cannot express.

**`--fbk-simplify none`** turns the rewriter off. It is a spelling, not a
semantics: a module whose update reads `if x + 1 = 10 then 0 else x + 1`
emits

```lean
-- default
abbrev effect_0 (state : StateType) : Int :=
  (if ((var_0 state) == (9 : Int)) then (0 : Int) else ((1 : Int) + (var_0 state)))
-- --fbk-simplify none
abbrev effect_0 (state : StateType) : Int :=
  (if (((var_0 state) + (1 : Int)) == (10 : Int)) then (0 : Int) else ((var_0 state) + (1 : Int)))
```

— the same transition, and only the second one can be read against the
source. Bool slots are never simplified either way, for a reason worth
knowing: the rewriter turns `ite c b (¬b)` into an equality, which moves the
slot from a branch into a *condition*, and `vmt2lean`'s `generalizeNatVar`
then retypes it to the unreduced `TypeMap` match where the `Decidable`
instance cannot be synthesised. What the rewriter buys on the arithmetic is
size: a 32-wide affine layer is 97 KB of term unfolded and 1.9 KB folded, so
`none` is for reading small models, not for running the sweep.

Two limits are by design rather than by defect. First, the model's imports
are `lake build`-ed before `proveit.py` runs: `lean2vmt` elaborates the model
with `processHeader`, so they must already exist as oleans, and `lake exe
lean2vmt` builds only the executable (whose own imports are just `Lean`).
The model imports `Cslib.Computability.Automata.NA.Basic` and nothing else —
`M`'s type is all it needs — so that pre-build is one target;
`NA_IMPORTS` and `fbk_proveit._LAKE_TARGETS` are the same list, pinned by a
test, because drift between them leaves `lean2vmt` unable to elaborate.
Second, the installed certificate is by default **written but not
compiled** — not because the project cannot compile it (it can: the lakefile
requires the checkout and declares the NA lib) but because doing so resolves
and builds Mathlib. That is `--build-cert`, which is not this route's flag:
see [Building the certificate](#building-the-certificate---build-cert).

### The `lean-ltl-certifying` side

The route needs that checkout on the same toolchain as the generated
projects, **v4.28.0**; it was pinned to v4.27.0 with `mathlib a3a10db` and
`cslib d69fa7d`. Ported by matching `project.py`'s package set exactly
(`cslib v4.28.0`, `smt f58d19d…`, Mathlib inherited through cslib), which
lets one already-built `.lake/packages` — e.g. `python/tests/lean`'s — serve
both. One source fix was needed: `bv_decide_light` dropped `bvSimprocs`
along with the `seval` simp sets, and without it the `Bool` structure of a
hypothesis reaches the bitblaster unnormalised, which abstracts whole
`||`/`&&`/`==` compounds as opaque variables and reports a "potentially
spurious counterexample". That is the SAT path (`vmt2lean -m sat`), not the
`smt` one this route uses, but it is what `NACounterBoolTS_cert` and
`lmcs06mutex0` are built on.

[ltl]: https://github.com/zeroth/proof-prototyping

---

## Adding a New Op (`ops.py`)

Every backend dispatches through one table, so an op is added in one place:
a row in `ops.OPS` giving its name, the theories that expose it, and a cell
for each of `mat`, `scalar`, `box` and `smt`.  A cell is an emitter, or
`ViaMat`/`Inline`/`Unsupported` saying why it is not one — `Unsupported`'s
reason is what the caller is told when it hits the gap.

`python -m zrth.lean.ops` prints the whole matrix and every gap in it.
`tests/test_lean_ops.py` checks the table against `LIA`/`LRA`/`BV` in both
directions: a variant a theory gained but the table has not fails there, and
so does a row no theory backs.

---

## Adding an Inference Route (`infer_route.py`)

Every way verith can be handed a certificate is one row of
`infer_route.ROUTES`, and `--infer <name>` selects a row.  A route is added
there and nowhere else: `cli.py` reads the rows, so the `--infer` choices,
the help text, every cross-flag rule and the dispatch all follow from the
row rather than from an edit apiece.

A row says:

* **the property** — `kinds` (`"safety"`, `"buchi"`, or both) and
  `kinds_refusal`, the sentence printed when the property is the other one.
  The check is generic and the reason is not: why the `ai` route cannot
  certify `G P` is prose about `rule_globally`, and why `nuterm` discards a
  precondition is prose about Houdini.
* **what it seeds from** — `seeds`, the `CertificateData` fields it will
  *use* rather than discard, and `seeds_refusal`.  A predicate outside the
  set is refused where it is passed, not dropped four steps later.
* **the project** — `reads` (encoding suffixes; `""` is the functional one)
  and `owns` (project-relative paths it may write).  The route is handed a
  `ProjectHandle`, so these two are the whole of its access, and a route
  that needs more of the project than the predicates says which more.
* **`artifacts/`** — `reads_artifacts`, the roles it may resume from.  It is
  declared access, like `reads`: `ProjectHandle.resume(role, …)` refuses an
  undeclared role, and the filtering (stale module, stale property, stale
  proof rule, plus the languages and statuses asked for) is the store's, so
  there is one filter and one place that reports what it dropped.
* **the hooks** — `resolve` at parse time (a route whose lakefile names a
  path has to reject a bad one before any of the project exists), `precheck`
  once the module is loaded and before it is generated (a module shape the
  route cannot express is met there rather than as a traceback out of an
  encoder it never wanted), and `run`.
* **what comes back** — `returns`: `"smt"` (SMT-LIB predicates, which
  `--pre-check` can restate the obligations from), `"lean"` (nothing parses
  it back, so they cannot be restated), or `"installed"` (the route wrote the
  certificate itself).  The row *declares* it and the `InferResult` carries
  it, and `main` checks the two agree once rather than each reader trusting
  whichever is nearer to hand.  Plus `uses_llm`, and `errors_self_named` for
  a route whose refusals already name their own flag.
* **its own flags** — `options`, a tuple of `Opt`.  Passed without the route
  selected, each is an error rather than a no-op.  `aliases` carries any
  older spelling that selects the route and fills one of those options
  (`--fbk-proveit DIR`), so a deprecated flag goes away with its row.

A route that renders its own Lean hands **both** spellings back in its
`InferResult`.  Rendering means encoding the module into cvc5 again, and
`magic_cegar` and `magic_learn` both already have the Lean — so the pipeline
renders only for a route that returns SMT alone.

`tests/test_infer_routes.py` checks the table the way `test_lean_ops.py`
checks the op matrix: a row declaring an encoding no row emits, an artifact
role that is not one, a refusal with no reason, or two routes claiming one
flag fails there rather than in generated code.

### `artifacts/` — the project as a workspace

The project is generated *before* inference runs, and what a run leaves in
`<project>/artifacts/` outlives it: an invariant that was inductive but too
weak, a ranking function that used an op the certificate cannot express, a
note for the next prompt.  A second `uv run verith` into the same `-o`
continues rather than restarts — take that invariant and strengthen it, or
take it as given and infer only the ranking function.

Artifacts are arbitrary files (`.smt2` and `.md`, mostly, plus whatever a
subprocess drops there); `artifacts/index.json` types the ones it knows, and
a file with no entry is still listed, as `role="other"`.  Each entry carries
`status` — what a consumer filters on — and `why`, the prose it puts in a
prompt.  Three more fields exist only so a stale artifact is not trusted:
`module_digest`, `prp` and `kind`.  An invariant found for another property,
another proof rule or a since-edited module is not about this run, and
`usable()` drops it and says so, because two identical command lines that
resume differently is the one thing a workspace must not do quietly.

`--artifacts ignore` searches afresh; `--artifacts reset` empties the
directory first, which is what to pass when a bad artifact is being
inherited by every run.  `artifacts/` belongs to no route, so a regenerating
run must never treat it as its own output to clean up.

#### The questions, not only the answers

The same directory holds what the run *asked*, which is the other half of
being able to check it.  A run encodes a good deal on its way to a
certificate — the module's transition, the property, the invariant, one query
per obligation — and until it was written down, all of it lived and died
inside the process.  What reached the user was a project and a line saying
`holds` or `REFUTED`: the conclusion without the question.

| file | what it is |
|---|---|
| `system.smt2` | the whole problem as cvc5 encoded it: `init<i>` / `next<i>` per state component, and `P`, `inv`, `ranking` over the same state |
| `obligation-<name>.smt2` | one obligation, **negated**, exactly as it was asked |
| `property.smt`, `inv.smt`, `ranking.smt` | the predicate source, as a flag was given it or a route found it |

`.smt2` is a script — declarations, definitions, `(check-sat)` — so
`cvc5 artifacts/obligation-step_inv.smt2` answers without `verith` in the
loop, and `unsat` there is the `holds` the pre-check printed.  That is what
makes it worth having when the answer surprises: a refuted obligation becomes
a file to bisect rather than a message to trust.  `.smt` is one predicate and
not a script; the encoded form of the same thing is in `system.smt2`.

These are written by `smt_query` as it builds each query, so they appear when
something was encoded — `--pre-check cvc5` is what asks for the obligations.
`--artifacts ignore` still records them: it is about where a run *starts*, not
about what it leaves.

#### `README.md` — the index components write

`artifacts/README.md` is the human-readable half of `index.json`, and nothing
assembles it centrally: no one place knows what a run will produce, which
routes will run, or what a component added later will want to leave behind.
So each component passes `what=` to `store.put` / `store.encoded` — one
sentence saying what its file *is* — and the README is rendered from the index
after every write.  It is therefore always exactly what was written, in the
order it was written, and a file a later run rewrites replaces its own section
rather than gaining a second one.

`what` is the counterpart of `why`: `why` says what was wrong with a candidate,
`what` says what the file is.  A question the run asked has a `what` and no
`why`, which is why its status (`encoded`) is not flagged in the README.

---

## Adding a New Encoding

1. Add a translation function in `native.py` (or a new file) analogous to
   `_translate_terms`.  It receives `input_bindings: dict[wire_id, str]` and
   returns a Lean body string.
2. Add a method to `ModuleToLean4` in `translate.py` that calls it and wraps
   the result in the appropriate namespace / function signature.
3. Optionally add an equivalence theorem connecting the new encoding to the
   existing functional one.
4. Wire it into `to_lean()`.

If the new encoding is *relational* — a per-slot body def, a per-slot
relation, and a conjunction over them — do not spell that shape a fourth
time: build a `RelSyntax` and a `RelBlock` and call `emit_rel_block` from
`translate/_skeleton.py`, as `Rel`, `ScalarRel` (both in
`translate/relational.py`) and the NA model (`translate/fbk.py`) do.  It is
parameterised on the type builder, the projection, and the spelling
(`def`/`abbrev`, `Prop`/`Bool`, `=`/`==`, `∧`/`&&`); the body producer stays
yours, and the per-slot `*_eq` theorems are emitted only if you name a
functional counterpart to prove them against.
