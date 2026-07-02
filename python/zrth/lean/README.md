# `zrth/lean` — Lean 4 Certificate Generation

This package translates Python reactive modules (an SSA-style IR) into Lean 4
source files that encode the module and carry machine-checked proofs of
safety/liveness properties.

---

## Reactive Module IR

A **`Module`** is the top-level object.  It has:

- **`ctrl`** wires — the module's mutable state: list of `(latched, next)` wire
  pairs.  `latched` holds the value from the previous step; `next` is what
  `update` writes.
- **`extl`** wires — external/environment inputs, also `(latched, next)` pairs.
- **atoms** — currently always a single atom (the combinational logic block).
  Each atom has an **`init`** term list and an **`update`** term list.

A **`Wire`** carries a `DType` (`Bool`, `Int`, `Float`) and a shape
(`[]`/`[1]` = scalar, `[n]` = row vector, `[m, n]` = matrix).

A **`Term`** is one SSA node: a single `IType` operation, reading from some
wires (`term.read`) and writing one output wire (`term.write[0]`).  The term
lists form a topologically-sorted dataflow graph.

---

## Module Files

| File | Role |
|------|------|
| `common.py` | Shared utilities: `LeanContext`, type helpers, `ConstantRegistry`, wire-binding helpers |
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

Each term is looked up in `_LEAN_OP` (keyed by `IType` variant name).
Operations produce `Mat T 1 1` values even for scalar results — conditions
extract `0 0`, boolean ops wrap in `fun _ _ => ...`.  Non-scalar ops
(`MatMul`, `Linear`, `ReLU`) use their native matrix forms.

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

## `uv run verith` CLI (`main.py`, `project.py`)

`verith` is the command-line tool that drives the full pipeline: Python module
→ Lean project or standalone certificate files.

### Module file format

```python
# mymodule.py
from zrth import Wire, Module, DType as dt
from zrth.analyzer import convert_method

def init():
    return 0

def update(old_x):
    x = old_x + 1
    if x == 10:
        return 0
    return x

def module() -> Module:
    state = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    init_terms   = convert_method(init,   {},               [state[1]])
    update_terms = convert_method(update, {"old_x": state}, [state[1]])
    return Module.sequential(init_terms, update_terms, obs=[state])
```

### Common invocations

```bash
# Bare Lean project (all certificate fields left as sorry)
uv run verith mymodule.py -o out/ -p MyProject

# Specify the property to prove (SMT-LIB 2 Bool over s0..sN-1)
uv run verith mymodule.py -P "(= s0 0)" -o out/ -p MyProject

# Also supply invariant and ranking manually
uv run verith mymodule.py -P "(= s0 0)" \
    --invariant "(and (>= s0 0) (<= s0 10))" \
    --ranking   "(ite (= s0 0) 0 s0)" \
    -o out/ -p MyProject

# AI inference with Claude (requires ANTHROPIC_API_KEY + pip install zrth[ai])
uv run verith mymodule.py -P "(= s0 0)" --infer -o out/ -p MyProject

# AI inference with a local LLM via Ollama (requires pip install zrth[ai-local])
uv run verith mymodule.py -P "(= s0 0)" --infer \
    --model qwen3-coder --base-url http://localhost:11434/v1 \
    -o out/ -p MyProject

# Standalone self-contained certificate (no project scaffold)
uv run verith mymodule.py -P "(= s0 0)" --cert-file out/MyCert.lean
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

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `-o` / `--output-dir` | `.` | Where to create the project |
| `-p` / `--project-name` | `Rea` | Lean package name |
| `-d` / `--module-def` | `module` | Name of the factory function in the Python file |
| `-P` / `--property` | — | SMT-LIB 2 Bool over `s0..sN-1` |
| `--invariant` | — | SMT-LIB 2 Bool invariant (skips invariant inference) |
| `--ranking` | — | SMT-LIB 2 Int ranking (skips ranking inference) |
| `--infer` | — | `ai` or `ai-cegar` (default when flag given without value) |
| `--model` | `claude-sonnet-4-6` | LLM model for inference |
| `--base-url` | — | OpenAI-compatible endpoint for local LLMs |
| `--cert-file` | — | Write standalone `.lean` file instead of full project |
| `--hammer-file` | — | Regenerate `ZerothHammer.lean` only |

### State variable naming in SMT-LIB predicates

`ctrl` wires are named `s0`, `s1`, … in left-to-right order (matching the
`obs=` list passed to `Module.sequential`).  For tuple/matrix wires, use
SMT-LIB tuple selectors: `((_ tuple.select 0) s0)`.  External inputs are
`e0..eM-1` (next) and `el0..elM-1` (latched).

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
