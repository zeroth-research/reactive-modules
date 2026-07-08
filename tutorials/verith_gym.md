# Generating a Lean certificate for a gymnasium module with `verith`

`verith` turns a Python reactive module into a Lean 4 project that encodes the
module's transition system and carries a machine-checkable *certificate* — an
invariant, a property, and a ranking function, together with the proof
obligations that connect them. This tutorial walks through the pipeline for a
module extracted from a plain [gymnasium](https://gymnasium.farama.org/)
environment:

```
gym.Env  ──zrth.gym.Env──▶  symbolic Module (SSA term IR)  ──verith──▶  Lean 4 project + certificate
```

## Prerequisites

Run everything from the repository checkout (`uv run verith` works from the
repo root or from `python/`; it is not installed globally):

```bash
uv sync          # builds zrth via maturin and installs dependencies
uv run verith --help
```

To actually *check* the generated certificate you also need a Lean 4
toolchain with `lake` (the generated project pins its own `lean-toolchain`
and depends on Mathlib, so the first `lake build` downloads a large cache).

## Step 1 — Wrap a gymnasium environment as a module

A *module file* is any Python file exposing a callable named `module`
(override the name with `-d`) that returns a `zrth.Module`. For a gymnasium
environment you don't hand-write the module — `zrth.gym.Env` extracts it by
symbolically analyzing the environment's `reset` and `step` methods.

Save this as `chain_env.py`:

```python
"""A simple gymnasium chain environment wrapped as a reactive module."""
import gymnasium as gym
from gymnasium import spaces

from zrth import Module
from zrth.gym import Env


class ChainEnv(gym.Env):
    """State moves along a chain 0..2; action 1 = right, 0 = left."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)

    def _get_observation(self):
        return 1 if self.state == 2 else 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self._get_observation(), {}

    def step(self, q_values):
        action = q_values.argmax().item()
        if action == 1:
            self.state = min(self.state + 1, 2)
        else:
            self.state = max(self.state - 1, 0)
        reward = 1.0 if self.state == 2 else 0.0
        terminated = self._get_observation() == 1
        return self._get_observation(), reward, terminated, False, {}


def module() -> Module:
    return Env(ChainEnv())
```

What the extraction does:

- `reset` becomes the module's **init** block, `step` its **update** block.
- Instance attributes written in `reset`/`step` (here `self.state`) become
  **private state wires**; observation, reward, terminated, and truncated
  become **interface wires**; the action becomes an **external input**.
- With the default theory (LRA) everything numeric is real-valued. A
  `Discrete(n)` action space is encoded as a one-hot real vector of width
  `n` — which is why `step` reads its action with `argmax`. (An integer
  theory can be selected with `Env(..., theory=LIA)`, but discrete *actions*
  are still one-hot real vectors — see [An integer-only environment
  (LIA)](#step-6--an-integer-only-environment-lia) for how to design an env that
  stays in integer arithmetic.)
- The analyzer handles straight-line arithmetic, comparisons, `if`/`else`,
  `min`/`max`, `argmax`, etc. It analyzes the *code* of `reset`/`step`, so
  keep them free of I/O, randomness, and unbounded loops.

You can sanity-check that the extracted symbolic module matches the real
environment before generating any Lean — `interpret=True` runs the extracted
IR instead of delegating to the backing env:

```python
import torch
from chain_env import ChainEnv
from zrth.gym import Env

real = Env(ChainEnv())
sim = Env(ChainEnv(), interpret=True)   # runs the symbolic IR only
real.reset(seed=42); sim.reset()
right = torch.tensor([0., 1.])          # one-hot action "go right"
for _ in range(3):
    real.step(right); sim.step(right)
    assert real.state == sim.state
```

## Step 2 — Generate a bare Lean project

```bash
uv run verith chain_env.py -o out/ -p ChainCert
```

This prints the extracted module (wires and SSA terms) and creates
`out/ChainCert/`:

```
out/ChainCert/
├── lakefile.toml, lean-toolchain    # Lean package pinned to a toolchain
├── Core/                            # library: Mat, Box, LTL, ReactiveModule
├── ZerothHammer.lean                # the zeroth_hammer proof-search tactic
├── System.lean                      # root import
├── System/
│   ├── System.lean                  # functional encoding of init/update
│   ├── Circ.lean                    # circuit (Box) encoding + equivalence
│   ├── Scalar.lean, ScalarRel.lean  # scalar encoding + relational form
│   ├── Rel.lean, FBK.lean           # matrix-domain relational encodings
│   └── Data.lean                    # certificate data: init_pre, update_pre, inv, P, ranking
├── Certificate/Certificate.lean     # proof obligations wired to the data
└── dbg/system.txt                   # human-readable wire/term dump
```

Since we passed no property, `System/Data.lean` is a skeleton: `inv` is
`True` and `P`/`ranking` are `sorry`, waiting to be filled in.

## Step 3 — Find your state variable names

Properties are written in SMT-LIB 2 over the module's **controlled wires**,
named `s0, s1, …` in declaration order; external inputs are `e0, …` (next
value) and `el0, …` (latched value). For an extracted gym environment the
order is: interface wires first (observation, reward, terminated,
truncated), then private wires. For `ChainEnv`:

| SMT var | Wire | Lean type |
|---------|------|-----------|
| `s0` | observation | `Mat Real 1 1` |
| `s1` | reward | `Mat Real 1 1` |
| `s2` | terminated | `Mat Bool 1 1` |
| `s3` | truncated | `Mat Bool 1 1` |
| `s4` | `self.state` (private) | `Mat Real 1 1` |
| `e0` / `el0` | action (one-hot, width 2) | `Mat Real 1 2` |

When in doubt, read `dbg/system.txt`: the `interface` and `private` sections
list the wire pairs in exactly this order, and the state tuple in
`System/System.lean`'s `update` signature matches it component by component.
Elements of vector wires like `e0` are addressed with SMT-LIB tuple
selectors: `((_ tuple.select 1) e0)` is the second component of the action.

## Step 4 — Generate the certificate

The certificate proves a **recurrence property**: `P` holds infinitely often
along every run. It is built from three ingredients you pass on the command
line (all SMT-LIB 2 expressions over `s0..sN-1`):

- `-P` — the property,
- `--invariant` — an inductive invariant,
- `--ranking` — a non-negative integer ranking function that strictly
  decreases on every step where `P` does not hold.

If you only know the property, Step 7 shows how to have an LLM infer the
other two.

For a pure safety property ("the chain position always stays in `[0, 2]`")
set `P` equal to the invariant and use the constant ranking `0` — if `P` is
invariant it trivially holds infinitely often:

```bash
uv run verith chain_env.py \
    -P          "(and (>= s4 0.0) (<= s4 2.0))" \
    --invariant "(and (>= s4 0.0) (<= s4 2.0))" \
    --ranking   "0" \
    -o out/ -p ChainCert
```

The SMT-LIB predicates are compiled into Lean in `System/Data.lean` (state
components appear as nested product accessors — `s4` becomes `s.2.2.2.2`):

```lean
def inv : (Mat Real 1 1) × (Mat Real 1 1) × (Mat Bool 1 1) × (Mat Bool 1 1) × (Mat Real 1 1) → Prop :=
  fun s => (((s.2.2.2.2 0 0) ≥ 0.0) ∧ ((s.2.2.2.2 0 0) ≤ 2.0))

def P : ... → Prop := fun s => (((s.2.2.2.2 0 0) ≥ 0.0) ∧ ((s.2.2.2.2 0 0) ≤ 2.0))

def ranking : ... → Nat := fun s => ((0 : Int)).toNat
```

`Certificate/Certificate.lean` states the obligations and tries to discharge
them with generated tactic scripts:

- `init_inv` — every initial state satisfies `inv`,
- `step_inv` — `inv` is preserved by `update` (inductiveness),
- `hinv` — hence `inv` holds on all reachable states,
- `hrank` — `ranking` strictly decreases while `¬P`,
- `buchi` — the final certificate, via the `rule_buchi` principle.

## Step 5 — Check the certificate with Lean

```bash
cd out/ChainCert
lake build Certificate
```

The first build fetches and compiles Mathlib and takes a long while
(subsequent builds reuse the cache). To skip compiling Mathlib from source,
run `lake exe cache get` inside the project first — the standard Mathlib
binary cache download. For this example the generated tactic
cascades close all five obligations — the certificate checks with no `sorry`.
If a goal survives the cascade for a harder module, the build reports it as
an unsolved goal — open the file in a Lean editor and finish the proof by
hand, or refine the invariant and re-run `verith`.

`lake build` (no target) additionally builds the auxiliary encodings
(`System/Circ.lean`, `System/FBK.lean`, …). These are not needed for the
certificate, and currently do not compile for gym modules that use `argmax`
over real-valued wires: the circuit encoding types `argmax_1d` as
integer-valued while the functional encoding treats it as real, and the FBK
relational encoding applies its `effect_*` abbreviations with a mismatched
argument list. Until those translator bugs are fixed, build the
`Certificate` target.

## Step 6 — An integer-only environment (LIA)

The chain example above lives in real arithmetic because of two encoding
rules: a `Discrete` *action* space always becomes a one-hot real vector
(read back with `argmax`), and rewards default to real-valued. Real-typed
modules work, but they cost you: every Lean definition becomes
`noncomputable`, the auxiliary circuit encoding currently rejects `argmax`,
and the proofs lean on `linarith` instead of the more robust `omega`.

If the dynamics are genuinely discrete, design the environment so the
extraction stays in the integer theory:

- pass `theory=LIA` to `Env`;
- use an **integer `Box` action space** instead of `Discrete`, and branch on
  the action value directly — no one-hot, no `argmax`;
- return **integer rewards** (`1`, not `1.0`) — under LIA the reward wire is
  integer-typed.

Save this as `cycle_env.py`:

```python
"""An integer-only gymnasium environment wrapped as a reactive module (LIA)."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from zrth import Module, LIA
from zrth.gym import Env


class CycleEnv(gym.Env):
    """4-state cyclic counter: a positive action advances the state, 3 wraps to 0."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.int64)
        self.observation_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        bumped = self.state + 1
        bumped = 0 if bumped == 4 else bumped
        self.state = bumped if action > 0 else self.state
        reward = 1 if self.state == 0 else 0
        terminated = self.state == 0
        return self.state, reward, terminated, False, {}


def module() -> Module:
    return Env(CycleEnv(), theory=LIA)
```

The wire layout is the same as before — `s0` observation, `s1` reward,
`s2` terminated, `s3` truncated, `s4` the private `self.state` — except that
every numeric wire is now `Mat Int 1 1`, and the action is a plain integer
external (`e0`/`el0`, no tuple selectors).

This time we prove a genuine **liveness** property: *the state returns to 0
infinitely often*. Unlike the safety-style certificate of Step 4, all three
ingredients now pull real weight — and the property is simply false without
a precondition, since a zero action freezes the state forever:

- `--pre "(>= el0 1)"` restricts runs to those where the action is always
  positive. Note it constrains `el0`, the *latched* input: `update` computes
  the step from the input latched at the previous step, so that is the copy
  the precondition must speak about (`e0`, the next value, is what will be
  latched for the following step).
- `--invariant "(and (>= s4 0) (<= s4 3))"` — inductive as before, but now
  load-bearing: from an unreachable state above 3 the counter would never
  wrap, and the ranking argument below would collapse.
- `--ranking "(ite (= s4 0) 0 (- 4 s4))"` counts the steps left until the
  wrap: it strictly decreases on every step from a state with `s4 ≠ 0`
  (3, 2, 1 as the state advances 1 → 2 → 3, then dropping to 0 on the wrap
  to `s4 = 0`, where `P` holds and no further decrease is required).

```bash
uv run verith cycle_env.py \
    -P          "(= s4 0)" \
    --pre       "(>= el0 1)" \
    --invariant "(and (>= s4 0) (<= s4 3))" \
    --ranking   "(ite (= s4 0) 0 (- 4 s4))" \
    -o out/ -p CycleCert

cd out/CycleCert
lake build Certificate
```

The generated project contains no `Real` and no `noncomputable` anywhere,
and the certificate obligations reduce to pure integer arithmetic, which the
proof cascade closes with `omega` — including the case split on the action
in `hrank`, where the precondition rules out the frozen branch.

### A matrix-form variant: one-hot state and a permutation matrix

Everything so far lives on 1×1 wires. The same 4-state cycle can be encoded
with genuine matrix state: keep the position as a **one-hot integer column
vector** and advance it by multiplying with a constant **permutation
matrix**. Save this as `matcycle_env.py`:

```python
"""Matrix-form LIA example: one-hot state advanced by a permutation matrix."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from zrth import Module, LIA
from zrth.gym import Env


class MatCycleEnv(gym.Env):
    """4-state cycle in one-hot form: a positive action rotates the state vector."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.int64)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 1), dtype=np.int64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([[1], [0], [0], [0]])
        return self.state, {}

    def step(self, action):
        # cyclic rotation: position k moves to position (k+1) mod 4
        rotated = np.array([[0, 0, 0, 1],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]]) @ self.state
        self.state = rotated if action > 0 else self.state
        return self.state, 0, False, False, {}


def module() -> Module:
    return Env(MatCycleEnv(), theory=LIA)
```

Two encoding constraints shape the Python:

- The state is a **column** vector (`shape (4, 1)`) and the matrix is the
  **left** operand of `@`: the integer theory lowers a matrix product to a
  `Linear` op with the convention `Y = A·X`, where `A` must be a constant.
- The rotation matrix is written **inline** so the analyzer bakes it as a
  constant term (an `np.array` literal); in Lean it appears as a constant
  `Mat Int 4 4` inside an `affineLinear` application.

The state tuple now contains real matrices: `s0` (observation) and `s4`
(`self.state`) are `Mat Int 4 1`. In SMT-LIB, their components are
addressed with tuple selectors — `((_ tuple.select k) s4)` is row `k` (in
Lean: `s.2.2.2.2 k 0`). The liveness certificate is the same "returns to
position 0" statement, but the invariant now says *the state vector is
one-hot*, and the ranking is a linear functional over the components —
weights 3, 2, 1 count the steps remaining until the wrap:

```bash
uv run verith matcycle_env.py \
    -P   '(= ((_ tuple.select 0) s4) 1)' \
    --pre '(>= el0 1)' \
    --invariant '(and (>= ((_ tuple.select 0) s4) 0) (<= ((_ tuple.select 0) s4) 1)
                      (>= ((_ tuple.select 1) s4) 0) (<= ((_ tuple.select 1) s4) 1)
                      (>= ((_ tuple.select 2) s4) 0) (<= ((_ tuple.select 2) s4) 1)
                      (>= ((_ tuple.select 3) s4) 0) (<= ((_ tuple.select 3) s4) 1)
                      (= (+ ((_ tuple.select 0) s4) ((_ tuple.select 1) s4)
                            ((_ tuple.select 2) s4) ((_ tuple.select 3) s4)) 1))' \
    --ranking '(+ (* 3 ((_ tuple.select 1) s4)) (* 2 ((_ tuple.select 2) s4)) ((_ tuple.select 3) s4))' \
    -o out/ -p MatCycleCert

cd out/MatCycleCert
lake build Certificate
```

In the proofs, `simp_mat` reduces the `affineLinear` application — the
`Fin 4` sums unfold and the permutation shuffles the components — after
which every obligation is again linear integer arithmetic over the vector
components and `omega` closes it. The one-hot invariant is load-bearing
twice over: it bounds each component (so the ranking is non-negative) and
pins their sum to 1 (so after a rotation exactly one component is hot).

## Step 7 (optional) — Infer the invariant and ranking with an LLM (`--infer`)

In Steps 4 and 6 we wrote the invariant and ranking function by hand. For
anything beyond toy modules that is the hard part of the certificate — and
it is the part `verith` can search for automatically. With `--infer`, you
supply only the property (and precondition) and an LLM proposes the rest.
Dropping `--invariant` and `--ranking` from the Step 6 liveness example:

```bash
uv run verith cycle_env.py \
    -P    "(= s4 0)" \
    --pre "(>= el0 1)" \
    --infer -o out/ -p CycleCert
```

The default `ai-cegar` mode prints each proposal and cvc5's verdict:

```
[CEGAR] attempt 0
  inv: (and (<= 0 s4) (<= s4 3) (= s3 false))
  ranking: (ite (= s4 0) 0 (- 4 s4))
[CEGAR] all obligations UNSAT — accepted
```

Here the model found the same ranking we wrote by hand in Step 6 (plus a
slightly stronger invariant — it also pinned the truncated flag to false),
and cvc5 discharged all obligations on the first attempt. When a proposal is
wrong, the loop reports which obligation failed, e.g. `[CEGAR] failures:
['rank_decrease']`, and re-prompts the model with the concrete
counterexample until a proposal passes (or the attempt budget runs out).

Requirements: `--infer` needs `-P` and, for the default Claude backend
(model `claude-sonnet-4-6`), an `ANTHROPIC_API_KEY` in the environment plus
the `ai` extra (`uv sync --extra ai`, or `pip install zrth[ai]`). Any other
backend — a local model or a hosted OpenAI-compatible provider — goes
through `--base-url`; see [Using a local LLM or another
provider](#using-a-local-llm-or-another-provider-ollama-vllm-openrouter-)
below.

What happens:

1. The project is generated as in Step 2, with the property compiled in.
2. The inference loop runs. `--infer` without a value means `ai-cegar`: each
   LLM proposal is checked with cvc5, and counterexamples are fed back to
   the model until the invariant is inductive and the ranking decreases
   (`--infer ai` skips the cvc5 loop and relies on LLM self-checking).
   Both integer (LIA) and real-valued (LRA) modules are validated — the
   Step 1 chain works here just as well as the cycle example.
3. Only `System/Data.lean` is rewritten with the inferred `inv` and
   `ranking` — `Certificate/Certificate.lean` and the module encodings are
   stable, so re-running inference never touches the proof skeleton.

Then check the result exactly as in Step 5 (`lake build Certificate`). The
inferred predicates are also printed as SMT-LIB, so you can freeze a good
result by passing it back explicitly with `--invariant`/`--ranking` on
future runs. The two modes mix: fixing `--invariant` by hand while adding
`--infer` searches only for a ranking function, and vice versa — and fixing
*both* makes no LLM call at all: the loop simply verifies your hand-written
certificate with cvc5, a fast sanity check before the full Lean build.

### Using a local LLM or another provider (Ollama, vLLM, OpenRouter, …)

Passing `--base-url` switches the backend from the Anthropic API to any
OpenAI-compatible endpoint; `--model` is the model name as that endpoint
knows it. The only extra dependency is the `openai` client package:

```bash
uv sync --extra ai-local        # or: pip install zrth[ai-local]
```

For **local servers** no API key is needed (the client is created with a
placeholder). For **hosted providers** the key is read from the
`OPENROUTER_API_KEY` or `OPENAI_API_KEY` environment variable. For example,
via [OpenRouter](https://openrouter.ai) — which fronts models from many
vendors under one key:

```bash
export OPENROUTER_API_KEY=sk-or-...

uv run verith cycle_env.py -P "(= s4 0)" --pre "(>= el0 1)" --infer \
    --model anthropic/claude-haiku-4.5 \
    --base-url https://openrouter.ai/api/v1 -o out/ -p CycleCert
```

With [Ollama](https://ollama.com) (serves on `http://localhost:11434/v1`
once the daemon is running):

```bash
ollama pull qwen3-coder         # once

uv run verith cycle_env.py -P "(= s4 0)" --pre "(>= el0 1)" --infer \
    --model qwen3-coder --base-url http://localhost:11434/v1 -o out/ -p CycleCert
```

With [vLLM](https://docs.vllm.ai) (serves on `http://localhost:8000/v1` by
default):

```bash
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct   # in another terminal

uv run verith cycle_env.py -P "(= s4 0)" --pre "(>= el0 1)" --infer \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --base-url http://localhost:8000/v1 -o out/ -p CycleCert
```

A local model is typically weaker at proposing invariants than a frontier
one, which is exactly what the default `ai-cegar` mode is for: cvc5 rejects
unsound proposals and returns concrete counterexamples to the model, so a
smaller model usually still converges — it just needs more refinement
rounds. Prefer code- or reasoning-tuned models; tiny chat models tend to
produce SMT-LIB that does not parse.

## Going further

### Constrain the environment's inputs (`--pre`)

The action is an unconstrained external input, so liveness properties like
"the goal is reached infinitely often" are false in general — the agent
could always move left. `--pre` adds a precondition over the inputs to both
`init_pre` and `update_pre` (the [integer example
above](#step-6--an-integer-only-environment-lia) uses this to prove a full liveness
certificate). For the chain env, restrict runs to those where the one-hot
action always prefers "right":

```bash
uv run verith chain_env.py \
    -P "(and (>= s4 0.0) (<= s4 2.0))" \
    --pre "(> ((_ tuple.select 1) e0) ((_ tuple.select 0) e0))" \
    --invariant "(and (>= s4 0.0) (<= s4 2.0))" \
    --ranking "0" \
    -o out/ -p ChainCert
```

### Standalone certificate files (`--cert-file`)

Instead of a full project scaffold, emit self-contained `.lean` files
(useful when you already have a Lean project with the `Core` library):

```bash
uv run verith chain_env.py \
    -P          "(and (>= s4 0.0) (<= s4 2.0))" \
    --invariant "(and (>= s4 0.0) (<= s4 2.0))" \
    --ranking   "0" \
    --cert-file out/ChainCert.lean
```

writes `ChainCert.lean` (init/update + certificate) plus `ChainCertRel.lean`,
`ChainCertScalar.lean`, and `ChainCertScalarRel.lean` (alternative encodings
with equivalence theorems). Pass the invariant and ranking too — without
them those certificate fields are emitted as `sorry` placeholders and the
proof obligations cannot close. To check the files, add them as `lean_lib`
entries in a lake project that also contains the `Core` library and
`ZerothHammer.lean` (copy both from any `verith`-generated project).

### Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-o` / `--output-dir` | `.` | Where to create the project |
| `-p` / `--project-name` | `Rea` | Lean package name |
| `-d` / `--module-def` | `module` | Factory function name in the module file |
| `-P` / `--property` | — | SMT-LIB 2 Bool over `s0..sN-1` |
| `--pre` | — | SMT-LIB 2 Bool over `e0..`/`el0..` input vars |
| `--invariant` | — | SMT-LIB 2 Bool invariant |
| `--ranking` | — | SMT-LIB 2 Int ranking function |
| `--infer [ai\|ai-cegar]` | `ai-cegar` | Infer missing invariant/ranking with an LLM |
| `--model`, `--base-url` | `claude-sonnet-4-6`, — | LLM selection for `--infer` |
| `--cert-file` | — | Standalone certificate files instead of a project |
| `-x` / `--executable` | off | Also generate a runnable `Main.lean` |
