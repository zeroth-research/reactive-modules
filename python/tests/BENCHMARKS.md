# `uv run verith` benchmark commands

Every `verith` invocation the two measurement harnesses make, written out so
a single case can be copy-pasted into a terminal. Generated from
[`limits/cases.py`](limits/cases.py) and [`lean/fbk/probes.py`](lean/fbk/probes.py);
those two files remain the source of truth, and the operational notes live in
[`limits/README.md`](limits/README.md) and [`lean/fbk/README.md`](lean/fbk/README.md).

Two independent suites:

| suite | what it measures | route |
|---|---|---|
| [Limit matrix](#a-limit-matrix) (77 cases) | which certificates `verith` can generate *and* Lean can discharge | verith's own `inv` + `ranking` machinery |
| [`--fbk-proveit` sweep](#b---fbk-proveit-sweep) (29 probes) | which safety properties survive `lean2vmt` → ic3ia → `vmt2lean` | the `lean-ltl-certifying` driver, no invariant supplied |

The properties are **not** interchangeable between the two. The limit matrix's
`-P` is a reachability target (`inv` + `ranking` ⇒ `P` is reached); the sweep
proves `□ PROPERTY`. Countdown starts at 100, so feeding it `-P '(= s0 0)'`
as a safety property is false at step 0 and reports UNSAFE for essentially
every case.

---

## Before you start

All commands are run from `python/`:

```bash
cd ~/zeroth/reactive-modules/python
```

Generation alone needs nothing else. To **build** what a command emits you
also need `tests/lean/.lake` populated (Mathlib, cslib, lean-smt), and the
generated project's `.lake` pointed at it — that plumbing is what
`run_limits.py` does, and doing it by hand is:

```bash
(cd tests/lean && lake build)                  # once, ~an hour cold

P=/tmp/verith.noindex/Countdown/Rea            # the project the command below emits
mkdir -p "$P/.lake"
ln -s "$PWD/tests/lean/.lake/packages" "$P/.lake/packages"
cp tests/limits/lake-manifest.json "$P/"
(cd "$P" && lake build)
```

Three things worth knowing before you time anything:

- **Keep the `.noindex` suffix** on the output directory. A pass writes tens
  of thousands of `.olean` files; Spotlight indexing them turned a 9 s case
  into a 928 s one. Directories ending in `.noindex` are skipped.
- **One build directory takes one writer.** Module names are identical across
  generated projects, so two concurrent builds overwrite each other's oleans
  and the symptom is `unknown constant 'hrank'`, which reads like a codegen bug.
- **Ask cvc5 before calling anything a tool limit.** Adding `--pre-check cvc5`
  answers all three obligations in milliseconds with a counterexample, where a
  failing `lake build` takes 9–79 s to not tell you whether the certificate is
  wrong or the tactics are weak:

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' --invariant '(and (>= s0 0) (<= s0 100))' --ranking '(- 100 s0)' \
    --pre-check cvc5 -o /tmp/out.noindex -p Rea
```

---

## A. Limit matrix

77 cases, `tests/limits/`. Each is one full Lean project. `expect` is
what the case is *meant* to show (`ok` / `fail` / `?` — the probe's open
questions); `baseline` is the verdict recorded in
[`limits/baseline.json`](limits/baseline.json), so a difference is a
regression or a fix.

Whole suite, or a few cases by name:

```bash
uv run python tests/limits/run_limits.py                    # all, 20–35 min
uv run python tests/limits/run_limits.py Countdown NN2Deep5
uv run python tests/limits/run_regress.py                   # baseline.json only, diffs verdicts
```

`P`, `inv`, `rank` and `pre` below are SMT-LIB 2 over `s0..sN-1` (state) and
`e0…` / `el0…` (inputs); a handful of the vector cases use the Python
predicate DSL instead.

### Baseline and bare projects

Known-good controls, plus what a project contains when the certificate flags are left off.

**`Countdown`** — LIA 1x1, conjunctive bound, Ite ranking — the known-good control  
<sub>`m_countdown` · expect `ok` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(ite (= s0 0) 0 s0)' \
    -o /tmp/verith.noindex/Countdown -p Rea
```

**`TwoVars`** — two 1x1 wires, relational invariant, difference ranking  
<sub>`m_twovars` · expect `ok` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_twovars.py \
    -P '(= s0 s1)' \
    --invariant '(and (>= s0 0) (<= s0 s1) (= s1 10))' \
    --ranking '(ite (= s0 s1) 0 (- s1 s0))' \
    -o /tmp/verith.noindex/TwoVars -p Rea
```

**`NoCert`** — no -P at all: what does a bare `verith` project contain?  
<sub>`m_countdown` · expect `fail` · baseline `PROOF-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -o /tmp/verith.noindex/NoCert -p Rea
```

**`PropOnly`** — -P but no --invariant/--ranking: inv defaults to True, ranking to sorry  
<sub>`m_countdown` · expect `fail` · baseline `PROOF-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    -o /tmp/verith.noindex/PropOnly -p Rea
```

### Invariant shapes

One module, one connective at a time: what the SMT-LIB invariant may contain before the certificate stops discharging.

**`InvTrue`** — trivially inductive invariant; hrank has no bound to work with  
<sub>`m_countdown` · expect `fail` · baseline `PROOF-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant true \
    --ranking '(ite (= s0 0) 0 s0)' \
    -o /tmp/verith.noindex/InvTrue -p Rea
```

**`InvDisj`** — 6-way disjunctive invariant — needs the casesm* _ v _ branch  
<sub>`m_step2` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_step2.py \
    -P '(= s0 0)' \
    --invariant '(or (= s0 0) (= s0 2) (= s0 4) (= s0 6) (= s0 8) (= s0 10))' \
    --ranking '(ite (= s0 0) 0 (- 12 s0))' \
    -o /tmp/verith.noindex/InvDisj -p Rea
```

**`InvMod`** — parity invariant: INTS_MODULUS through smt_to_lean, then omega  
<sub>`m_step2` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_step2.py \
    -P '(= s0 0)' \
    --invariant '(and (= (mod s0 2) 0) (>= s0 0) (<= s0 10))' \
    --ranking '(ite (= s0 0) 0 (- 12 s0))' \
    -o /tmp/verith.noindex/InvMod -p Rea
```

**`InvNe`** — DISTINCT in an invariant  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100) (distinct s0 101))' \
    --ranking '(ite (= s0 0) 0 s0)' \
    -o /tmp/verith.noindex/InvNe -p Rea
```

**`InvIte`** — Ite in Prop position over a Bool state component  
<sub>`m_boolint` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_boolint.py \
    -P '(= s1 0)' \
    --invariant '(and (>= s1 0) (<= s1 5) (ite s0 (<= s1 5) (>= s1 0)))' \
    --ranking '(ite (and (not s0) (= s1 0)) 0 (+ (ite s0 (- 12 s1) s1) 1))' \
    -o /tmp/verith.noindex/InvIte -p Rea
```

**`InvImplies`** — Implies in a Prop position — same module and ranking as InvIte, so the only difference from that case is the connective  
<sub>`m_boolint` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_boolint.py \
    -P '(= s1 0)' \
    --invariant 'And(s1[0][0] >= 0, s1[0][0] <= 5, Implies(s0[0][0], s1[0][0] <= 5))' \
    --ranking '(ite (and (not s0) (= s1 0)) 0 (+ (ite s0 (- 12 s1) s1) 1))' \
    -o /tmp/verith.noindex/InvImplies -p Rea
```

**`InvMixed`** — 1x1 + 3x1 state: wire index and flat slot disagree; x is unbounded  
<sub>`m_mixed` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_mixed.py \
    -P 's1[0][0] == 0' \
    --invariant 'And(s1[0][0] >= 0, s1[0][0] <= 1, s1[1][0] == 2, s1[2][0] == 3)' \
    --ranking 'Ite(s1[0][0] == 0, 0, s1[0][0])' \
    -o /tmp/verith.noindex/InvMixed -p Rea
```

### Ranking shapes

The same question for the ranking function, which `hrank` has to show strictly decreases.

**`RankConst`** — constant ranking — hrank needs 0 < 0  
<sub>`m_countdown` · expect `fail` · baseline `PROOF-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking 0 \
    -o /tmp/verith.noindex/RankConst -p Rea
```

**`RankLinear`** — bare linear ranking, no Ite guard  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking s0 \
    -o /tmp/verith.noindex/RankLinear -p Rea
```

**`RankLex`** — nested loops; lexicographic (y,x) folded into one Nat  
<sub>`m_lex` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_lex.py \
    -P '(and (= s0 0) (= s1 0))' \
    --invariant '(and (>= s0 0) (<= s0 3) (>= s1 0) (<= s1 3))' \
    --ranking '(+ (* 4 s1) s0)' \
    -o /tmp/verith.noindex/RankLex -p Rea
```

**`RankQuadratic`** — nonlinear but correct ranking — omega/linarith territory  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(* s0 s0)' \
    -o /tmp/verith.noindex/RankQuadratic -p Rea
```

**`RankToInt`** — non-integral Real state (steps of 0.5): to_int has to scale  
<sub>`m_lra_half` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_lra_half.py \
    -P '(= s0 0.0)' \
    --invariant '(or (= s0 0.0) (= s0 0.5) (= s0 1.0) (= s0 1.5) (= s0 2.0) (= s0 2.5) (= s0 3.0))' \
    --ranking '(to_int (* 2.0 s0))' \
    -o /tmp/verith.noindex/RankToInt -p Rea
```

### ReLU in the module

A ReLU in the *transition* rather than in the certificate: `x' = relu(x-1)` reaches Lean as `Max.max 0 (x-1)`.

**`Relu`** — ReLU in the transition: x' = relu(x-1) becomes Max.max 0 (x-1)  
<sub>`m_relu` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_relu.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 5))' \
    --ranking s0 \
    -o /tmp/verith.noindex/Relu -p Rea
```

**`ReluRankRelu`** — ReLU-shaped ranking (no max kind in smt_to_lean, so Ite)  
<sub>`m_relu` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_relu.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 5))' \
    --ranking '(ite (> s0 0) s0 0)' \
    -o /tmp/verith.noindex/ReluRankRelu -p Rea
```

**`ReluInvRelu`** — ReLU-shaped invariant: `x = relu(x)` written as an Ite  
<sub>`m_relu` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_relu.py \
    -P '(= s0 0)' \
    --invariant '(and (= s0 (ite (>= s0 0) s0 0)) (<= s0 5))' \
    --ranking s0 \
    -o /tmp/verith.noindex/ReluInvRelu -p Rea
```

**`ReluVec`** — element-wise ReLU on a 3-vector state  
<sub>`m_relu_vec` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_relu_vec.py \
    -P 's[0][0] == 0' \
    --invariant 'And(s[0][0] >= 0, s[0][0] <= 3, s[1][0] >= 0, s[2][0] >= 0)' \
    --ranking 's[0][0]' \
    -o /tmp/verith.noindex/ReluVec -p Rea
```

**`ReluLRA`** — ReLU over Real — noncomputable RM, linarith instead of omega  
<sub>`m_relu_lra` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_relu_lra.py \
    -P '(= s0 0.0)' \
    --invariant '(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))' \
    --ranking '(to_int s0)' \
    -o /tmp/verith.noindex/ReluLRA -p Rea
```

**`ReluNet`** — Linear -> ReLU -> Linear, the shape a small Q-network compiles to  
<sub>`m_relu_net` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_relu_net.py \
    -P 's[0][0] == 0' \
    --invariant 'And(s[0][0] >= 0, s[0][0] <= 4, s[1][0] >= 0)' \
    --ranking 's[0][0]' \
    -o /tmp/verith.noindex/ReluNet -p Rea
```

**`ReluInput`** — ReLU + external input, sound under the precondition  
<sub>`m_relu_input` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_relu_input.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 5))' \
    --ranking s0 \
    --pre '(>= e0 1)' \
    -o /tmp/verith.noindex/ReluInput -p Rea
```

**`ReluInputNoPre`** — same module without --pre: e is unconstrained, ranking cannot decrease  
<sub>`m_relu_input` · expect `fail` · baseline `PROOF-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_relu_input.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 5))' \
    --ranking s0 \
    -o /tmp/verith.noindex/ReluInputNoPre -p Rea
```

### Theories other than LIA

Bool, BitVec and Real state. `omega` does not model BitVec and does not apply over ℝ, so each needs a different closing tactic.

**`BoolState`** — Bool state (2-bit counter), Bool->Int ranking via Ite  
<sub>`TESTS/twobit_lia` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/fixtures/twobit_lia.py \
    -P '(and (not s0) (not s1))' \
    --invariant true \
    --ranking '(ite (and (not s0) (not s1)) 0 (- 4 (+ (ite s0 1 0) (* 2 (ite s1 1 0)))))' \
    --pre e0 \
    -o /tmp/verith.noindex/BoolState -p Rea
```

**`BVState`** — BitVec state: omega does not model BitVec, and a branch that is contradictory only because a 1-bit vector has two values needs the state enumerated  
<sub>`TESTS/twobit` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/fixtures/twobit.py \
    -P '(and (= s0 (_ bv0 1)) (= s1 (_ bv0 1)))' \
    --invariant true \
    --ranking '(ite (and (= s0 (_ bv0 1)) (= s1 (_ bv0 1))) 0 (- 4 (+ (ite (= s0 (_ bv1 1)) 1 0) (* 2 (ite (= s1 (_ bv1 1)) 1 0)))))' \
    --pre '(= e0 (_ bv1 1))' \
    -o /tmp/verith.noindex/BVState -p Rea
```

**`LRALinear`** — Real state, no ReLU — an interval invariant is not inductive over the reals, so the invariant pins exact values  
<sub>`m_lra_lin` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_lra_lin.py \
    -P '(= s0 0.0)' \
    --invariant '(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))' \
    --ranking '(to_int s0)' \
    -o /tmp/verith.noindex/LRALinear -p Rea
```

### Matrix and reduction operators

Operators that are unary reductions over a matrix. `matMin`, `matMax` and `argmax_1d` are the three standing limits — `--pre-check cvc5` says the certificates are true, so the defect is Lean-side reduction, not the certificate.

**`OpMax`** — Max as a unary reduction: x' = max(x-1, 0), i.e. ReLU spelled Max  
<sub>`m_max` · expect `?` · baseline `PROOF-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_max.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 5))' \
    --ranking s0 \
    -o /tmp/verith.noindex/OpMax -p Rea
```

**`OpMin`** — Min as a unary reduction: x' = min(x+1, 5)  
<sub>`m_min` · expect `?` · baseline `PROOF-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_min.py \
    -P '(= s0 5)' \
    --invariant '(and (>= s0 0) (<= s0 5))' \
    --ranking '(- 5 s0)' \
    -o /tmp/verith.noindex/OpMin -p Rea
```

**`OpArgmax`** — Argmax in the transition — does argmax_1d reduce under the hammer?  
<sub>`m_argmax` · expect `?` · baseline `PROOF-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_argmax.py \
    -P '(= s0 0)' \
    --invariant '(= s0 0)' \
    --ranking s0 \
    -o /tmp/verith.noindex/OpArgmax -p Rea
```

**`OpTranspose`** — Transpose in the transition (MatTranspose / Box.transpose)  
<sub>`m_transpose` · expect `fail` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_transpose.py \
    -P 's[0][0] == 1' \
    --invariant 's[0][0] == 1' \
    --ranking 0 \
    -o /tmp/verith.noindex/OpTranspose -p Rea
```

**`OpUninterp`** — an uninterpreted symbol — no Lean counterpart exists  
<sub>`m_uninterp` · expect `fail` · baseline `GEN-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_uninterp.py \
    -P '(= s0 0)' \
    --invariant true \
    --ranking 0 \
    -o /tmp/verith.noindex/OpUninterp -p Rea
```

### Scale

Width and depth of the module itself, independent of the certificate. Both are slow; see the timeouts in `cases.py`.

**`Vec32`** — 32-wide state, all six encodings — the scaling ceiling  
<sub>`m_vec32` · expect `?` · baseline `VERIFIED` · timeout 2400s</sub>

```bash
uv run verith tests/limits/mods/m_vec32.py \
    -P 's[0][0] == 0' \
    --invariant 'And(s[0][0] >= 0, s[0][0] <= 100)' \
    --ranking 'Ite(s[0][0] == 0, 0, s[0][0])' \
    -o /tmp/verith.noindex/Vec32 -p Rea
```

**`Deep64`** — 64-deep straight-line transition body  
<sub>`m_deep` · expect `?` · baseline `VERIFIED` · timeout 1200s</sub>

```bash
uv run verith tests/limits/mods/m_deep.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(ite (= s0 0) 0 s0)' \
    -o /tmp/verith.noindex/Deep64 -p Rea
```

### Neural certificates, first pass

The certificate predicate *is* a ReLU net, built from explicit weight matrices rather than reverse-engineered from what the prover closes.

**`NNRank`** — ranking is a 2-unit ReLU net: 2·relu(x-1) + relu(x)  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 2 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0)) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)))' \
    -o /tmp/verith.noindex/NNRank -p Rea
```

**`NNRankWide`** — ranking is a 3-unit ReLU net with distinct thresholds  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 3 (ite (>= (+ (* 1 s0) (- 2)) 0) (+ (* 1 s0) (- 2)) 0)) (* 2 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0)) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)))' \
    -o /tmp/verith.noindex/NNRankWide -p Rea
```

**`NNRankDeep`** — ranking is a two-hidden-layer ReLU net  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)))' \
    -o /tmp/verith.noindex/NNRankDeep -p Rea
```

**`NNInv`** — invariant is a ReLU net: relu(x) + relu(9-x) = 9, i.e. 0 ≤ x ≤ 9  
<sub>`TESTS/counter` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/fixtures/counter.py \
    -P '(= s0 0)' \
    --invariant '(= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 9) 0) (+ (* (- 1) s0) 9) 0))) 9)' \
    --ranking '(ite (= s0 0) 0 (- 10 s0))' \
    -o /tmp/verith.noindex/NNInv -p Rea
```

**`NNInvIneq`** — net invariant in inequality form — the shape a learned barrier/Lyapunov function takes  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(<= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 100) 0) (+ (* (- 1) s0) 100) 0))) 100)' \
    --ranking '(ite (= s0 0) 0 s0)' \
    -o /tmp/verith.noindex/NNInvIneq -p Rea
```

**`NNBoth`** — invariant and ranking both nets, over the same state  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 100) 0) (+ (* (- 1) s0) 100) 0))) 100)' \
    --ranking '(+ (* 2 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0)) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)))' \
    -o /tmp/verith.noindex/NNBoth -p Rea
```

**`NNTwoInput`** — two-input net invariant: relu(y-x) + relu(x) = y  
<sub>`m_twovars` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_twovars.py \
    -P '(= s0 s1)' \
    --invariant '(and (= (+ (* 1 (ite (>= (+ (* (- 1) s0) (* 1 s1)) 0) (+ (* (- 1) s0) (* 1 s1)) 0)) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0))) s1) (= s1 10))' \
    --ranking '(ite (= s0 s1) 0 (- s1 s0))' \
    -o /tmp/verith.noindex/NNTwoInput -p Rea
```

**`NNNetModule`** — the module is Linear->ReLU->Linear *and* both certificate predicates are nets — the fully neural case  
<sub>`m_relu_net` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_relu_net.py \
    -P 's[0][0] == 0' \
    --invariant 'And(((1 * Ite(((1 * s[0][0])) >= 0, (1 * s[0][0]), 0)) + (1 * Ite(((-1 * s[0][0]) + 4) >= 0, (-1 * s[0][0]) + 4, 0))) == 4, ((1 * Ite(((1 * s[1][0])) >= 0, (1 * s[1][0]), 0))) == s[1][0])' \
    --ranking '(2 * Ite(((1 * s[0][0]) + -1) >= 0, (1 * s[0][0]) + -1, 0)) + (1 * Ite(((1 * s[0][0])) >= 0, (1 * s[0][0]), 0))' \
    -o /tmp/verith.noindex/NNNetModule -p Rea
```

### Hard Real cases

Real arithmetic where `omega` does not apply and the invariant has to pin exact values or case on disjunctions.

**`RealConjDisj`** — Real, invariant is a conjunction of two 4-way disjunctions: omega does not apply and `constructor <;> linarith` cannot prove a disjunct, so the plan has to case on both  
<sub>`m_lra_conv` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_lra_conv.py \
    -P '(and (= s0 0.0) (= s1 2.0))' \
    --invariant '(and (or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0)) (or (= s1 2.0) (= s1 3.0) (= s1 4.0) (= s1 5.0)))' \
    --ranking '(+ (to_int s0) (to_int (- s1 2.0)))' \
    -o /tmp/verith.noindex/RealConjDisj -p Rea
```

**`RealNonlin`** — Real *and* nonlinear: needs nlinarith over an ordered field, not omega  
<sub>`m_lra_lin` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_lra_lin.py \
    -P '(= s0 0.0)' \
    --invariant '(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))' \
    --ranking '(to_int (* s0 s0))' \
    -o /tmp/verith.noindex/RealNonlin -p Rea
```

**`RealNet`** — a ReLU net over Real as the ranking: ite branches, real literals and a floor, all at once  
<sub>`m_lra_lin` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_lra_lin.py \
    -P '(= s0 0.0)' \
    --invariant '(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))' \
    --ranking '(to_int (+ (* 2.0 (ite (>= (+ (* 1.0 s0) (- 1.0)) 0.0) (+ (* 1.0 s0) (- 1.0)) 0.0)) (* 1.0 (ite (>= (* 1.0 s0) 0.0) (* 1.0 s0) 0.0))))' \
    -o /tmp/verith.noindex/RealNet -p Rea
```

### Negative controls

Cases that must fail. A harness that passes these is not measuring anything.

**`RealConjDisjUnsat`** — two Real components that both *cycle*: the product of their ranges admits (1,2)->(2,5)->(3,2)->(0,5)->(1,2), which never reaches P, so hrank is false for every ranking. Must be rejected  
<sub>`m_lra_two` · expect `fail` · baseline `PROOF-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_lra_two.py \
    -P '(and (= s0 0.0) (= s1 2.0))' \
    --invariant '(and (or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0)) (or (= s1 2.0) (= s1 5.0)))' \
    --ranking '(ite (and (= s0 0.0) (= s1 2.0)) 0 (+ (to_int (- 4.0 s0)) (ite (= s1 2.0) 0 4)))' \
    -o /tmp/verith.noindex/RealConjDisjUnsat -p Rea
```

**`BadInv`** — not inductive (init is 100) — init_inv must fail  
<sub>`m_countdown` · expect `fail` · baseline `PROOF-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 50))' \
    --ranking '(ite (= s0 0) 0 s0)' \
    -o /tmp/verith.noindex/BadInv -p Rea
```

**`BadRankDir`** — ranking increases along the transition — hrank must fail  
<sub>`m_countdown` · expect `fail` · baseline `PROOF-FAIL`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(- 100 s0)' \
    -o /tmp/verith.noindex/BadRankDir -p Rea
```

### Neural certificates, width and depth sweep

The scaling study: net width, net depth, weight size, Real vs Int, and nets over vector state. Folding a ReLU back to `max` took `NN2Width6` from 491 s to 9.6 s and turned 11 of these green.

**`NN2Width4`** — 4-unit ranking net, distinct thresholds — 8 ite in hrank  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 4 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 3 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0)) (* 2 (ite (>= (+ (* 1 s0) (- 2)) 0) (+ (* 1 s0) (- 2)) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 3)) 0) (+ (* 1 s0) (- 3)) 0)))' \
    -o /tmp/verith.noindex/NN2Width4 -p Rea
```

**`NN2Width6`** — 6-unit ranking net — 12 ite in hrank  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 6 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 5 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0)) (* 4 (ite (>= (+ (* 1 s0) (- 2)) 0) (+ (* 1 s0) (- 2)) 0)) (* 3 (ite (>= (+ (* 1 s0) (- 3)) 0) (+ (* 1 s0) (- 3)) 0)) (* 2 (ite (>= (+ (* 1 s0) (- 4)) 0) (+ (* 1 s0) (- 4)) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 5)) 0) (+ (* 1 s0) (- 5)) 0)))' \
    -o /tmp/verith.noindex/NN2Width6 -p Rea
```

**`NN2Width8`** — 8-unit ranking net — 16 ite in hrank  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 8 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 7 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0)) (* 6 (ite (>= (+ (* 1 s0) (- 2)) 0) (+ (* 1 s0) (- 2)) 0)) (* 5 (ite (>= (+ (* 1 s0) (- 3)) 0) (+ (* 1 s0) (- 3)) 0)) (* 4 (ite (>= (+ (* 1 s0) (- 4)) 0) (+ (* 1 s0) (- 4)) 0)) (* 3 (ite (>= (+ (* 1 s0) (- 5)) 0) (+ (* 1 s0) (- 5)) 0)) (* 2 (ite (>= (+ (* 1 s0) (- 6)) 0) (+ (* 1 s0) (- 6)) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 7)) 0) (+ (* 1 s0) (- 7)) 0)))' \
    -o /tmp/verith.noindex/NN2Width8 -p Rea
```

**`NN2Width10`** — 10-unit ranking net — 20 ite in hrank  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED` · timeout 600s</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 10 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 9 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0)) (* 8 (ite (>= (+ (* 1 s0) (- 2)) 0) (+ (* 1 s0) (- 2)) 0)) (* 7 (ite (>= (+ (* 1 s0) (- 3)) 0) (+ (* 1 s0) (- 3)) 0)) (* 6 (ite (>= (+ (* 1 s0) (- 4)) 0) (+ (* 1 s0) (- 4)) 0)) (* 5 (ite (>= (+ (* 1 s0) (- 5)) 0) (+ (* 1 s0) (- 5)) 0)) (* 4 (ite (>= (+ (* 1 s0) (- 6)) 0) (+ (* 1 s0) (- 6)) 0)) (* 3 (ite (>= (+ (* 1 s0) (- 7)) 0) (+ (* 1 s0) (- 7)) 0)) (* 2 (ite (>= (+ (* 1 s0) (- 8)) 0) (+ (* 1 s0) (- 8)) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 9)) 0) (+ (* 1 s0) (- 9)) 0)))' \
    -o /tmp/verith.noindex/NN2Width10 -p Rea
```

**`NN2Width12`** — 12-unit ranking net — 24 ite in hrank  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED` · timeout 600s</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 12 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 11 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0)) (* 10 (ite (>= (+ (* 1 s0) (- 2)) 0) (+ (* 1 s0) (- 2)) 0)) (* 9 (ite (>= (+ (* 1 s0) (- 3)) 0) (+ (* 1 s0) (- 3)) 0)) (* 8 (ite (>= (+ (* 1 s0) (- 4)) 0) (+ (* 1 s0) (- 4)) 0)) (* 7 (ite (>= (+ (* 1 s0) (- 5)) 0) (+ (* 1 s0) (- 5)) 0)) (* 6 (ite (>= (+ (* 1 s0) (- 6)) 0) (+ (* 1 s0) (- 6)) 0)) (* 5 (ite (>= (+ (* 1 s0) (- 7)) 0) (+ (* 1 s0) (- 7)) 0)) (* 4 (ite (>= (+ (* 1 s0) (- 8)) 0) (+ (* 1 s0) (- 8)) 0)) (* 3 (ite (>= (+ (* 1 s0) (- 9)) 0) (+ (* 1 s0) (- 9)) 0)) (* 2 (ite (>= (+ (* 1 s0) (- 10)) 0) (+ (* 1 s0) (- 10)) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 11)) 0) (+ (* 1 s0) (- 11)) 0)))' \
    -o /tmp/verith.noindex/NN2Width12 -p Rea
```

**`NN2WidthDup8`** — 8 units but one distinct condition — control separating the cost of net *size* from the cost of net *branching*  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 2 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 3 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 4 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 5 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 6 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 7 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 8 (ite (>= (* 1 s0) 0) (* 1 s0) 0)))' \
    -o /tmp/verith.noindex/NN2WidthDup8 -p Rea
```

**`NN2Deep3`** — 3 dense hidden layers — 6 ite per copy, text doubles per layer  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)))' \
    -o /tmp/verith.noindex/NN2Deep3 -p Rea
```

**`NN2Deep4`** — 4 dense hidden layers — 8 ite per copy, 16 in hrank  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED` · timeout 600s</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0)))' \
    -o /tmp/verith.noindex/NN2Deep4 -p Rea
```

**`NN2Deep5`** — 5 dense hidden layers — 10 ite per copy, 20 in hrank  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED` · timeout 600s</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0) (+ (* 1 (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0)) (* (- 1) (ite (>= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0) (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* (- 1) (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0))) 0))) 0))) 0))) 0)))' \
    -o /tmp/verith.noindex/NN2Deep5 -p Rea
```

**`NN2Narrow8`** — 8 layers of one unit each: same 16 ite as NN2Width8 but linear text — does depth or does size cost?  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED` · timeout 600s</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0) (* 1 (ite (>= (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0) (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 0)) 0)) 0)) 0)) 0)) 0)) 0))' \
    -o /tmp/verith.noindex/NN2Narrow8 -p Rea
```

**`NN2InvWide4`** — invariant is a 4-unit net equality that *is* the box 0..100  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 100) 0) (+ (* (- 1) s0) 100) 0)) (* 1 (ite (>= (+ (* 1 s0) 1) 0) (+ (* 1 s0) 1) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 101) 0) (+ (* (- 1) s0) 101) 0))) 202)' \
    --ranking '(ite (= s0 0) 0 s0)' \
    -o /tmp/verith.noindex/NN2InvWide4 -p Rea
```

**`NN2InvWide8`** — the same, 8 units wide — step_inv now carries 16 ite  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED` · timeout 600s</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 100) 0) (+ (* (- 1) s0) 100) 0)) (* 1 (ite (>= (+ (* 1 s0) 1) 0) (+ (* 1 s0) 1) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 101) 0) (+ (* (- 1) s0) 101) 0)) (* 1 (ite (>= (+ (* 1 s0) 2) 0) (+ (* 1 s0) 2) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 102) 0) (+ (* (- 1) s0) 102) 0)) (* 1 (ite (>= (+ (* 1 s0) 3) 0) (+ (* 1 s0) 3) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 103) 0) (+ (* (- 1) s0) 103) 0))) 412)' \
    --ranking '(ite (= s0 0) 0 s0)' \
    -o /tmp/verith.noindex/NN2InvWide8 -p Rea
```

**`NN2RealAllPos4`** — 4-unit net over Real whose every ReLU is non-negative under the invariant — the shape --smt-tactics is built for  
<sub>`m_lra_lin` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_lra_lin.py \
    -P '(= s0 0.0)' \
    --invariant '(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))' \
    --ranking '(to_int (+ (* 4.0 (ite (>= (* 1.0 s0) 0.0) (* 1.0 s0) 0.0)) (* 3.0 (ite (>= (+ (* 1.0 s0) 1.0) 0.0) (+ (* 1.0 s0) 1.0) 0.0)) (* 2.0 (ite (>= (+ (* 1.0 s0) 2.0) 0.0) (+ (* 1.0 s0) 2.0) 0.0)) (* 1.0 (ite (>= (+ (* 1.0 s0) 3.0) 0.0) (+ (* 1.0 s0) 3.0) 0.0))))' \
    -o /tmp/verith.noindex/NN2RealAllPos4 -p Rea
```

**`NN2RealWide4`** — 4-unit net over Real — no omega, linarith on 2^8 branches  
<sub>`m_lra_lin` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_lra_lin.py \
    -P '(= s0 0.0)' \
    --invariant '(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))' \
    --ranking '(to_int (+ (* 4.0 (ite (>= (* 1.0 s0) 0.0) (* 1.0 s0) 0.0)) (* 3.0 (ite (>= (+ (* 1.0 s0) (- 1.0)) 0.0) (+ (* 1.0 s0) (- 1.0)) 0.0)) (* 2.0 (ite (>= (+ (* 1.0 s0) (- 2.0)) 0.0) (+ (* 1.0 s0) (- 2.0)) 0.0)) (* 1.0 (ite (>= (+ (* 1.0 s0) (- 3.0)) 0.0) (+ (* 1.0 s0) (- 3.0)) 0.0))))' \
    -o /tmp/verith.noindex/NN2RealWide4 -p Rea
```

**`NN2RealFrac`** — weights 1.5 / 0.25 / -0.5 and a negative output weight — the weights a trained net actually has  
<sub>`m_lra_lin` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_lra_lin.py \
    -P '(= s0 0.0)' \
    --invariant '(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))' \
    --ranking '(to_int (+ (* 1.0 (ite (>= (+ (* 1.5 s0) (- 0.5)) 0.0) (+ (* 1.5 s0) (- 0.5)) 0.0)) (* 2.0 (ite (>= (* 0.25 s0) 0.0) (* 0.25 s0) 0.0)) (* (- 1.0) (ite (>= (+ (* (- 0.5) s0) 2.0) 0.0) (+ (* (- 0.5) s0) 2.0) 0.0)) 1.0))' \
    -o /tmp/verith.noindex/NN2RealFrac -p Rea
```

**`NN2RealNetInv`** — a net *invariant* over Real: the exact-value disjunction keeps it inductive, the net box is the learned part  
<sub>`m_lra_lin` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_lra_lin.py \
    -P '(= s0 0.0)' \
    --invariant '(and (or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0)) (= (+ (* 1.0 (ite (>= (* 1.0 s0) 0.0) (* 1.0 s0) 0.0)) (* 1.0 (ite (>= (+ (* (- 1.0) s0) 5.0) 0.0) (+ (* (- 1.0) s0) 5.0) 0.0))) 5.0))' \
    --ranking '(to_int s0)' \
    -o /tmp/verith.noindex/NN2RealNetInv -p Rea
```

**`NN2VecNet`** — 3-input net invariant over a 3-vector state: Σ relu(vᵢ) = Σ vᵢ is componentwise non-negativity, and Σ relu(vᵢ) ≤ 6 bounds it  
<sub>`m_relu_vec` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_relu_vec.py \
    -P 's[0][0] == 0' \
    --invariant 'And(((1 * Ite(((1 * s[0][0])) >= 0, (1 * s[0][0]), 0)) + (1 * Ite(((1 * s[1][0])) >= 0, (1 * s[1][0]), 0)) + (1 * Ite(((1 * s[2][0])) >= 0, (1 * s[2][0]), 0))) == s[0][0] + s[1][0] + s[2][0], ((1 * Ite(((1 * s[0][0])) >= 0, (1 * s[0][0]), 0)) + (1 * Ite(((1 * s[1][0])) >= 0, (1 * s[1][0]), 0)) + (1 * Ite(((1 * s[2][0])) >= 0, (1 * s[2][0]), 0))) <= 6)' \
    --ranking '(1 * Ite(((1 * s[0][0])) >= 0, (1 * s[0][0]), 0))' \
    -o /tmp/verith.noindex/NN2VecNet -p Rea
```

**`NN2NetMod8`** — the module's own net is 8 wide and both predicates are nets — module width against certificate width  
<sub>`m_relu_net8` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_relu_net8.py \
    -P 's[0][0] == 0' \
    --invariant 'And(((1 * Ite(((1 * s[0][0])) >= 0, (1 * s[0][0]), 0)) + (1 * Ite(((-1 * s[0][0]) + 6) >= 0, (-1 * s[0][0]) + 6, 0))) == 6, ((1 * Ite(((1 * s[1][0])) >= 0, (1 * s[1][0]), 0))) == s[1][0])' \
    --ranking '(3 * Ite(((1 * s[0][0])) >= 0, (1 * s[0][0]), 0)) + (2 * Ite(((1 * s[0][0]) + -1) >= 0, (1 * s[0][0]) + -1, 0)) + (1 * Ite(((1 * s[0][0]) + -2) >= 0, (1 * s[0][0]) + -2, 0))' \
    -o /tmp/verith.noindex/NN2NetMod8 -p Rea
```

**`NN2NetMod16`** — the same with a 16-wide module net  
<sub>`m_relu_net16` · expect `?` · baseline `VERIFIED` · timeout 900s</sub>

```bash
uv run verith tests/limits/mods/m_relu_net16.py \
    -P 's[0][0] == 0' \
    --invariant 'And(((1 * Ite(((1 * s[0][0])) >= 0, (1 * s[0][0]), 0)) + (1 * Ite(((-1 * s[0][0]) + 6) >= 0, (-1 * s[0][0]) + 6, 0))) == 6, ((1 * Ite(((1 * s[1][0])) >= 0, (1 * s[1][0]), 0))) == s[1][0])' \
    --ranking '(3 * Ite(((1 * s[0][0])) >= 0, (1 * s[0][0]), 0)) + (2 * Ite(((1 * s[0][0]) + -1) >= 0, (1 * s[0][0]) + -1, 0)) + (1 * Ite(((1 * s[0][0]) + -2) >= 0, (1 * s[0][0]) + -2, 0))' \
    -o /tmp/verith.noindex/NN2NetMod16 -p Rea
```

**`NN2WideInput`** — a net that reads 8 state slots but has only 2 units — input width without branch width, over the 32-wide state  
<sub>`m_vec32` · expect `?` · baseline `VERIFIED` · timeout 900s</sub>

```bash
uv run verith tests/limits/mods/m_vec32.py \
    -P 's[0][0] == 0' \
    --invariant 'And(((1 * Ite(((1 * s[0][0]) + (1 * s[1][0]) + (1 * s[2][0]) + (1 * s[3][0]) + (1 * s[4][0]) + (1 * s[5][0]) + (1 * s[6][0]) + (1 * s[7][0])) >= 0, (1 * s[0][0]) + (1 * s[1][0]) + (1 * s[2][0]) + (1 * s[3][0]) + (1 * s[4][0]) + (1 * s[5][0]) + (1 * s[6][0]) + (1 * s[7][0]), 0)) + (1 * Ite(((-1 * s[0][0]) + (-1 * s[1][0]) + (-1 * s[2][0]) + (-1 * s[3][0]) + (-1 * s[4][0]) + (-1 * s[5][0]) + (-1 * s[6][0]) + (-1 * s[7][0]) + 100) >= 0, (-1 * s[0][0]) + (-1 * s[1][0]) + (-1 * s[2][0]) + (-1 * s[3][0]) + (-1 * s[4][0]) + (-1 * s[5][0]) + (-1 * s[6][0]) + (-1 * s[7][0]) + 100, 0))) == 100, s[1][0] == 0, s[2][0] == 0, s[3][0] == 0, s[4][0] == 0, s[5][0] == 0, s[6][0] == 0, s[7][0] == 0)' \
    --ranking '(3 * Ite(((1 * s[0][0])) >= 0, (1 * s[0][0]), 0)) + (2 * Ite(((1 * s[0][0]) + -1) >= 0, (1 * s[0][0]) + -1, 0)) + (1 * Ite(((1 * s[0][0]) + -2) >= 0, (1 * s[0][0]) + -2, 0))' \
    -o /tmp/verith.noindex/NN2WideInput -p Rea
```

**`NN2MixedSign`** — mixed-sign weights, thresholds at 0 and 10 so two branches are live at once, output bias keeping the value non-negative  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 2 (ite (>= (+ (* 1 s0) (- 10)) 0) (+ (* 1 s0) (- 10)) 0)) (* (- 1) (ite (>= (+ (* (- 1) s0) 10) 0) (+ (* (- 1) s0) 10) 0)) (* 3 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) 10)' \
    -o /tmp/verith.noindex/NN2MixedSign -p Rea
```

**`NN2BigWeights`** — weights 10007 / 3001 / 499 — coefficient size, not branch count  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 10007 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 3001 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0)) (* 499 (ite (>= (+ (* 1 s0) (- 2)) 0) (+ (* 1 s0) (- 2)) 0)))' \
    -o /tmp/verith.noindex/NN2BigWeights -p Rea
```

**`NN2ToNatClamp`** — net(x) = 3x - 2 is negative at x = 0, so Int.toNat clamps; the obligation is still true because the clamp only bites under P  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 1 (ite (>= (* 3 s0) 0) (* 3 s0) 0)) (- 2))' \
    -o /tmp/verith.noindex/NN2ToNatClamp -p Rea
```

**`NN2CmpBoth`** — a net on each side of ≤ — relu(x)+relu(y-x) ≤ relu(y) is 0≤x≤y  
<sub>`m_twovars` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_twovars.py \
    -P '(= s0 s1)' \
    --invariant '(and (<= (+ (* 1 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 1 (ite (>= (+ (* (- 1) s0) (* 1 s1)) 0) (+ (* (- 1) s0) (* 1 s1)) 0))) (* 1 (ite (>= (* 1 s1) 0) (* 1 s1) 0))) (= s1 10))' \
    --ranking '(ite (= s0 s1) 0 (- s1 s0))' \
    -o /tmp/verith.noindex/NN2CmpBoth -p Rea
```

**`NN2Lyapunov`** — |x-5| as a 2-unit net, used as *both* invariant and ranking over a plant that converges from either side: a genuine piecewise-linear Lyapunov function whose decrease needs the ReLU split  
<sub>`m_toward5` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_toward5.py \
    -P '(= s0 5)' \
    --invariant '(<= (+ (* 1 (ite (>= (+ (* 1 s0) (- 5)) 0) (+ (* 1 s0) (- 5)) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 5) 0) (+ (* (- 1) s0) 5) 0))) 5)' \
    --ranking '(+ (* 1 (ite (>= (+ (* 1 s0) (- 5)) 0) (+ (* 1 s0) (- 5)) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 5) 0) (+ (* (- 1) s0) 5) 0)))' \
    -o /tmp/verith.noindex/NN2Lyapunov -p Rea
```

**`NN2Lyap2D`** — the same in two dimensions: a 4-unit Lyapunov net over a plant with two independent piecewise-linear legs  
<sub>`m_toward2d` · expect `?` · baseline `VERIFIED` · timeout 600s</sub>

```bash
uv run verith tests/limits/mods/m_toward2d.py \
    -P '(and (= s0 5) (= s1 5))' \
    --invariant '(<= (+ (* 1 (ite (>= (+ (* 1 s0) (- 5)) 0) (+ (* 1 s0) (- 5)) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 5) 0) (+ (* (- 1) s0) 5) 0)) (* 1 (ite (>= (+ (* 1 s1) (- 5)) 0) (+ (* 1 s1) (- 5)) 0)) (* 1 (ite (>= (+ (* (- 1) s1) 5) 0) (+ (* (- 1) s1) 5) 0))) 10)' \
    --ranking '(+ (* 1 (ite (>= (+ (* 1 s0) (- 5)) 0) (+ (* 1 s0) (- 5)) 0)) (* 1 (ite (>= (+ (* (- 1) s0) 5) 0) (+ (* (- 1) s0) 5) 0)) (* 1 (ite (>= (+ (* 1 s1) (- 5)) 0) (+ (* 1 s1) (- 5)) 0)) (* 1 (ite (>= (+ (* (- 1) s1) 5) 0) (+ (* (- 1) s1) 5) 0)))' \
    -o /tmp/verith.noindex/NN2Lyap2D -p Rea
```

**`NN2Width3`** — 3-unit ranking net — the last width that closes  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 3 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 2 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 2)) 0) (+ (* 1 s0) (- 2)) 0)))' \
    -o /tmp/verith.noindex/NN2Width3 -p Rea
```

**`NN2Width5`** — 5-unit ranking net — bisects the knee  
<sub>`m_countdown` · expect `?` · baseline `VERIFIED`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= s0 0)' \
    --invariant '(and (>= s0 0) (<= s0 100))' \
    --ranking '(+ (* 5 (ite (>= (* 1 s0) 0) (* 1 s0) 0)) (* 4 (ite (>= (+ (* 1 s0) (- 1)) 0) (+ (* 1 s0) (- 1)) 0)) (* 3 (ite (>= (+ (* 1 s0) (- 2)) 0) (+ (* 1 s0) (- 2)) 0)) (* 2 (ite (>= (+ (* 1 s0) (- 3)) 0) (+ (* 1 s0) (- 3)) 0)) (* 1 (ite (>= (+ (* 1 s0) (- 4)) 0) (+ (* 1 s0) (- 4)) 0)))' \
    -o /tmp/verith.noindex/NN2Width5 -p Rea
```

**`NN2Width6Big`** — the 6-unit net of NN2Width6 over the 32-slot module: identical certificate and identical transition shape, but the plan calls the module slow and raises the heartbeat budget 5x  
<sub>`m_vec32` · expect `?` · baseline `VERIFIED` · timeout 900s</sub>

```bash
uv run verith tests/limits/mods/m_vec32.py \
    -P 's[0][0] == 0' \
    --invariant 'And(s[0][0] >= 0, s[0][0] <= 100)' \
    --ranking '(6 * Ite(((1 * s[0][0])) >= 0, (1 * s[0][0]), 0)) + (5 * Ite(((1 * s[0][0]) + -1) >= 0, (1 * s[0][0]) + -1, 0)) + (4 * Ite(((1 * s[0][0]) + -2) >= 0, (1 * s[0][0]) + -2, 0)) + (3 * Ite(((1 * s[0][0]) + -3) >= 0, (1 * s[0][0]) + -3, 0)) + (2 * Ite(((1 * s[0][0]) + -4) >= 0, (1 * s[0][0]) + -4, 0)) + (1 * Ite(((1 * s[0][0]) + -5) >= 0, (1 * s[0][0]) + -5, 0))' \
    -o /tmp/verith.noindex/NN2Width6Big -p Rea
```

---

## B. `--fbk-proveit` sweep

29 probes, `tests/lean/fbk/`. No invariant is supplied: `verith`
emits the NA encoding, `proveit.py` runs `lean2vmt` → ic3ia → `vmt2lean`, and
the invariant comes back from the model checker. `-P` here is a safety
property, proved as `□ P`.

This route needs three things that are **not** vendored — MathSAT with its
Python bindings, ic3ia, and a `lean-ltl-certifying` checkout on Lean v4.28.0.
[`lean/fbk/README.md`](lean/fbk/README.md) has the cold start for all three.
(That README predates the fixtures being checked in and still says they live
in a scratch directory; `--mods tests/limits/mods` is the in-repo path, and it
is what every command below uses.) Set them up once:

```bash
export PYTHONPATH=$MSAT/python          # MathSAT bindings, built by the same interpreter
export IC3IA=~/ic3ia/build/ic3ia        # --ic3ia falls back to $IC3IA, then to PATH
export LTL=~/proof-prototyping/lean-ltl-certifying
```

Whole sweep, or one group:

```bash
uv run python tests/lean/fbk/run_fbk.py --mods tests/limits/mods --screen        # no Lean/ic3ia needed
uv run python tests/lean/fbk/run_fbk.py --mods tests/limits/mods --ltl "$LTL" --json /tmp/fbk.json
uv run python tests/lean/fbk/run_fbk.py --mods tests/limits/mods --ltl "$LTL" --only '^Ni' --no-check
```

**Do not run probes in parallel** — they share one `lean-ltl-certifying` build
directory. Roughly 18 s per certified probe: ~5 s route, ~12 s for Lean to
check the certificate.

The route installs its certificate **without checking it** (it imports
`LTLCertifying.*` and `Smt`, which the generated project does not provide), so
`run_fbk.py` asks Lean separately. To reproduce that second half by hand:

```bash
uv run python "$LTL/proveit.py" /tmp/fbk.noindex/InvBase/InvBase/ProveIt/InvBaseNA.lean \
    -o /tmp/fbk.noindex/InvBase/checked.lean -c
```

`expect` is what the route *should* report:

| verdict | meaning |
|---|---|
| `certified` | route succeeds and Lean checks the certificate |
| `unsafe` | ic3ia finds a counterexample; the route aborts |
| `unknown` | ic3ia cannot decide; the route aborts |
| `abort` | rejected before ic3ia (unsupported shape or property) |
| `lean-fail` | certificate produced, Lean rejects it — an upstream `lean-smt` bug |

### Invariants lifted from the limit matrix

The `--invariant` of the case matrix, reused as the safety property. These are inductive by construction, so they measure the plumbing rather than ic3ia.

**`InvBase`** — the Countdown bound from the case matrix — the known-good control  
<sub>`m_countdown` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(and (>= s0 0) (<= s0 100))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/InvBase -p InvBase
```

**`InvNe`** — the same bound with a `distinct` conjoined  
<sub>`m_countdown` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(and (>= s0 0) (<= s0 100) (distinct s0 101))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/InvNe -p InvNe
```

**`InvDisj`** — 6-way disjunction over the parity walker  
<sub>`m_step2` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_step2.py \
    -P '(or (= s0 0) (= s0 2) (= s0 4) (= s0 6) (= s0 8) (= s0 10))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/InvDisj -p InvDisj
```

**`InvTwoVars`** — relational, over two 1x1 wires  
<sub>`m_twovars` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_twovars.py \
    -P '(and (>= s0 0) (<= s0 s1) (= s1 10))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/InvTwoVars -p InvTwoVars
```

**`InvLex`** — bounds on both variables of the nested-loop module  
<sub>`m_lex` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_lex.py \
    -P '(and (>= s0 0) (<= s0 3) (>= s1 0) (<= s1 3))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/InvLex -p InvLex
```

**`InvHalf`** — negative control: x reaches 100, so a bound at 50 is false  
<sub>`m_countdown` · expect `unsafe`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(and (>= s0 0) (<= s0 50))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/InvHalf -p InvHalf
```

**`InvMod`** — parity via `mod`. `lean2vmt` translates `%` and ic3ia proves it safe — but MathSAT eliminates the mod from the *witness*, returning `x + (-2) * to_int ((1/2) * to_real x) = 0`, and `vmt2lean` renders neither `to_real`/`to_int` nor a Real inside a Bool `INVAR`. So verith refuses it up front, where the message can say why  
<sub>`m_step2` · expect `abort`</sub>

```bash
uv run verith tests/limits/mods/m_step2.py \
    -P '(and (= (mod s0 2) 0) (>= s0 0) (<= s0 10))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/InvMod -p InvMod
```

**`InvTrue`** — trivially safe, so ic3ia's invariant is literally `true`. Two `vmt2lean` bugs used to fire here at once: `MSAT_TAG_TRUE` rendered as the Lean *Prop* `True` inside `abbrev INVAR : Bool`, and `INVAR` binding nothing when the invariant mentions no state  
<sub>`m_countdown` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P true \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/InvTrue -p InvTrue
```

### Net-shaped invariants

A ReLU net *is* the safety property — the shape a learned barrier takes.

**`NetBox`** — relu(x) + relu(100-x) = 100, i.e. 0 ≤ x ≤ 100  
<sub>`m_countdown` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(= (+ (ite (>= (* 1 s0) 0) (* 1 s0) 0) (ite (>= (+ (* (- 1) s0) 100) 0) (+ (* (- 1) s0) 100) 0)) 100)' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/NetBox -p NetBox
```

**`NetBoxIneq`** — the same as an inequality — the shape a learned barrier takes  
<sub>`m_countdown` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(<= (+ (ite (>= (* 1 s0) 0) (* 1 s0) 0) (ite (>= (+ (* (- 1) s0) 100) 0) (+ (* (- 1) s0) 100) 0)) 100)' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/NetBoxIneq -p NetBoxIneq
```

**`NetTwoInput`** — two-input net over (x, y): relu(y-x) + relu(x) = y  
<sub>`m_twovars` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_twovars.py \
    -P '(= (+ (ite (>= (+ (* (- 1) s0) (* 1 s1)) 0) (+ (* (- 1) s0) (* 1 s1)) 0) (ite (>= (* 1 s0) 0) (* 1 s0) 0)) s1)' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/NetTwoInput -p NetTwoInput
```

**`NetLyapunov`** — |x-5| as a two-unit net: a Lyapunov function for a converging plant  
<sub>`m_toward5` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_toward5.py \
    -P '(<= (+ (ite (>= (+ (* 1 s0) (- 5)) 0) (+ (* 1 s0) (- 5)) 0) (ite (>= (+ (* (- 1) s0) 5) 0) (+ (* (- 1) s0) 5) 0)) 5)' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/NetLyapunov -p NetLyapunov
```

### Written for this route

Properties that make ic3ia work: true-but-not-inductive, relational over two variables, nonlinear, `ite` in the property, and false-but-only-refutable-after-100-steps.

**`NiCdNe101`** — true but **not** inductive: x ≠ 101 holds (the reachable set is 0..100) but has a predecessor outside itself, namely 102, so ic3ia has to synthesise the bound  
<sub>`m_countdown` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(distinct s0 101)' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/NiCdNe101 -p NiCdNe101
```

**`NiT5Ne4`** — the same shape on the converging plant  
<sub>`m_toward5` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_toward5.py \
    -P '(not (= s0 4))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/NiT5Ne4 -p NiT5Ne4
```

**`NiLexNe4`** — the same on the nested-loop module  
<sub>`m_lex` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_lex.py \
    -P '(not (= s0 4))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/NiLexNe4 -p NiLexNe4
```

**`NiTvNe11`** — the same over two wires  
<sub>`m_twovars` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_twovars.py \
    -P '(not (= s0 11))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/NiTvNe11 -p NiTvNe11
```

**`NiS2Odd3`** — needs *parity*, not just a bound: 3 is inside 0..10 and unreachable only because x steps by 2. ic3ia finds the invariant; the `smt` tactic then miscompiles the proof — a `sum_ub` step returns an equality where a `≤` is wanted. See [`lean-smt-bug.md`](lean/fbk/lean-smt-bug.md)  
<sub>`m_step2` · expect `lean-fail`</sub>

```bash
uv run verith tests/limits/mods/m_step2.py \
    -P '(not (= s0 3))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/NiS2Odd3 -p NiS2Odd3
```

**`RelT2dSum`** — relational: `m_toward2d` walks x down from 10 and y up from 0 in lockstep, so the reachable set is the diagonal x + y = 10  
<sub>`m_toward2d` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_toward2d.py \
    -P '(= (+ s0 s1) 10)' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/RelT2dSum -p RelT2dSum
```

**`RelT2dPoint`** — a single excluded point on the same plant  
<sub>`m_toward2d` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_toward2d.py \
    -P '(not (and (= s0 6) (= s1 0)))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/RelT2dPoint -p RelT2dPoint
```

**`RelTvImplies`** — an implication between the two state variables  
<sub>`m_twovars` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_twovars.py \
    -P '(=> (= s0 5) (= s1 10))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/RelTvImplies -p RelTvImplies
```

**`NonlinLexMul`** — nonlinear, and out of scope by construction: ic3ia is IC3 with implicit predicate abstraction over *linear* arithmetic. The route aborts cleanly  
<sub>`m_lex` · expect `unknown`</sub>

```bash
uv run verith tests/limits/mods/m_lex.py \
    -P '(<= (* s0 s1) 9)' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/NonlinLexMul -p NonlinLexMul
```

**`BmcCdReach0`** — false, and only refutable after a long unrolling — x = 0 is first reached at step 100. Exercises the counterexample path rather than the invariant path  
<sub>`m_countdown` · expect `unsafe`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(not (= s0 0))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/BmcCdReach0 -p BmcCdReach0
```

**`BmcT5Reach5`** — the same, on the converging plant  
<sub>`m_toward5` · expect `unsafe`</sub>

```bash
uv run verith tests/limits/mods/m_toward5.py \
    -P '(not (= s0 5))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/BmcT5Reach5 -p BmcT5Reach5
```

**`IteCd`** — `ite` in the property itself, so the Bool printer has to nest one  
<sub>`m_countdown` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_countdown.py \
    -P '(ite (= s0 0) (>= s0 0) (and (> s0 0) (<= s0 100)))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/IteCd -p IteCd
```

**`DeepBody64`** — transition-body size: a 64-deep straight-line body, with a non-inductive property so ic3ia does real work at each depth  
<sub>`m_deep` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_deep.py \
    -P '(distinct s0 101)' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/DeepBody64 -p DeepBody64
```

**`DeepBody48`** — the same, 48 deep  
<sub>`m_depth48` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_depth48.py \
    -P '(distinct s0 101)' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/DeepBody48 -p DeepBody48
```

**`DeepBody8`** — the same, 8 deep — the small control  
<sub>`m_depth8` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_depth8.py \
    -P '(distinct s0 101)' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/DeepBody8 -p DeepBody8
```

**`MixedBoolInt`** — mixed Bool+Int state through a per-index `TypeMap`, so `var_0 : Bool` and `var_1 : Int` have to reach the VMT as different sorts  
<sub>`m_boolint` · expect `certified`</sub>

```bash
uv run verith tests/limits/mods/m_boolint.py \
    -P '(and (>= s1 0) (<= s1 5))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/MixedBoolInt -p MixedBoolInt
```

**`ReluTrans`** — ReLU in the *transition*. The scalar encoding spells it `Max.max 0 (x - 1)`, which `lean2vmt` expands to an `ite`; the VMT is right and ic3ia proves it, but `smt` then dies on the `Max` still in the model — "incorrect number of universe levels Max". Second `lean-smt` symptom  
<sub>`m_relu` · expect `lean-fail`</sub>

```bash
uv run verith tests/limits/mods/m_relu.py \
    -P '(and (>= s0 0) (<= s0 5))' \
    --fbk-proveit "$LTL" \
    -o /tmp/fbk.noindex/ReluTrans -p ReluTrans
```

### Modules the encoding refuses

These never reach ic3ia. `run_fbk.py --screen` checks that each still fails
for the reason recorded here, so a lifted restriction shows up as a surprise
rather than silently. `verith` = a restriction in `translate/na.py`;
`lean2vmt` / `vmt2lean` = an upstream limit that has to be fixed there first.

```bash
uv run python tests/lean/fbk/run_fbk.py --mods tests/limits/mods --screen
```

| module | where | why |
|---|---|---|
| `m_mixed` | verith | ctrl wire holds more than one element |
| `m_relu_net` | verith | ctrl wire holds more than one element |
| `m_relu_net8` | verith | ctrl wire holds more than one element |
| `m_relu_net16` | verith | ctrl wire holds more than one element |
| `m_relu_vec` | verith | ctrl wire holds more than one element |
| `m_transpose` | verith | ctrl wire holds more than one element |
| `m_vec32` | verith | ctrl wire holds more than one element |
| `m_lra_conv` | vmt2lean | Real state: tp() maps only Int and Bool |
| `m_lra_half` | vmt2lean | Real state: tp() maps only Int and Bool |
| `m_lra_lin` | vmt2lean | Real state: tp() maps only Int and Bool |
| `m_lra_two` | vmt2lean | Real state: tp() maps only Int and Bool |
| `m_relu_lra` | vmt2lean | Real state: tp() maps only Int and Bool |
| `m_argmax` | lean2vmt | Argmax/Linear reach exprToSMT as a leaf |
| `m_max` | lean2vmt | Linear reaches exprToSMT as a leaf |
| `m_min` | lean2vmt | Linear reaches exprToSMT as a leaf |
| `m_uninterp` | lean2vmt | no Lean counterpart for an uninterpreted op |
| `m_relu_input` | lean2vmt | models only state/statenext, no inputs |

