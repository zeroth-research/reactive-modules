# Tying the FBK model back to the module

The `--fbk-proveit` route ends in a Lean certificate that says

```lean
∀ ss μs, M.ωTrace ss μs → ss ⊧ G (AP PROPERTY)
```

— a statement about `M`, the NA model `translate/fbk.py` writes. Nothing in
the tree says `M` is the module. Every other encoding has that link and is
useless without it (`init_circ_eq`, `update_scalar_eq`, `Rel.effect_i_eq`,
`ScalarRel.TransRel_func_eq`); this one has never had it, because the route
was designed around trust leaving Lean through ic3ia and coming back as
`vmt2lean.py`'s certificate. So a mistranslation anywhere in `smt_encode`,
`_scalar_element`, cvc5's rewriter or `smt_to_lean_bool` produces a model
that parses, model-checks, carries a machine-checked certificate — and
describes a different system.

This file designs the missing link. Everything below marked **measured** was
run; the Lean shown compiled.

---

## What the statement has to be

The route rejects modules with external inputs, so the functional encoding is
`init () : CtrlNative` and `update ctrl () () : CtrlNative`, and the model is
`INIT : StateType → Bool`, `TRANS : StateType → StateType → Bool` over
`StateType := (n : Nat) → TypeMap n`, one slot per state *element*.

The link that matters is one-directional:

> every run of the module is a run of the model

which is what makes `G P` on the model's traces transfer to the module. The
converse is only needed to transfer a *counterexample* — see
[Limitations](#limitations). As it happens both directions come out of the
same proof, because the obligations are equalities rather than implications.

---

## The decomposition

Not one theorem but a chain, and the split is the point: each link is
discharged by something that can actually do it.

```
  TRANS s s' = true
     ↕  (1) structural: `rfl` + `simp`
  slots s' = (effect_0_fn (slots s), …, effect_{N-1}_fn (slots s))
     ↕  (2) semantic: one obligation per slot, `simp` + `omega`
  slots s' = Scalar.update (slots s) () ()
     ↕  (3) already proven: `update_scalar_eq`, emitted by `translate/scalar.py`
  unpack s' = update (unpack s) () ()          -- the functional encoding
```

Link (3) is free: `Scalar.update` is the flat-slot transition function and
`update_scalar_eq : update ctrl () () = Scalar.pack (Scalar.update
(Scalar.unpack_ctrl ctrl) () ())` is generated and proved today, together
with the `pack`/`unpack` round-trip lemmas (`_recon_lemma_lines`). Targeting
`Scalar` rather than `update` directly is what keeps the SMT-shaped
obligation free of matrices.

### (1) is structural

The model's `effect_k` takes the whole `state`; the bridge wants it as a
function of the slot values, so that no goal carries a variable of type
`TypeMap k` — a stuck `match` that `omega` will not accept as an `Int` and
`smt` will not accept at all. **Measured:** `rfl` crosses that boundary,
because `TypeMap k` reduces to `Int`/`Bool`:

```lean
abbrev effect_1 (state : StateType) : Int := (if (var_0 state) then … else …)
abbrev effect_1_fn (x0 : Bool) (x1 : Int) : Int := (if x0 then … else …)

theorem link_1 (state : StateType) : effect_1 state = effect_1_fn (var_0 state) (var_1 state) := rfl
```

and the relation level follows by `simp`:

```lean
theorem TRANS_iff (s s' : StateType) :
    TRANS s s' = true ↔ slots s' = (effect_0_fn (slots s), effect_1_fn (slots s)) := by
  simp [TRANS, R_0, R_1, link_0, link_1, slots, Prod.ext_iff]
```

Both compiled on `m_boolint`, whose state has one `Bool` slot and one `Int`
slot — the case where `TypeMap` is a real per-index match rather than a
constant function.

### (2) is the semantic step, and it is the whole risk

```lean
theorem bridge_k (x0 : …) … (xN : …) :
    effect_k_fn x0 … xN = (Scalar.update (x0, …, xN) () ()).k
```

Two independent translators on the two sides — `smt_encode` + cvc5 +
`smt_to_lean_bool` on the left, `_translate_terms_scalar` on the right — which
is exactly why the theorem is worth proving and why `rfl` cannot prove it.
cvc5 rewrites `x - 1` to `-1 + x`, folds `x + 1 = 10` into `x = 9`, expands
affine layers, and reshapes `ite c b (¬b)` into an equality.

**Measured**, on every module the FBK encoding accepts, with the cascade
below:

| module | slots | shape | time |
|---|---|---|---|
| `m_countdown` | 1 | `ite`, `==` vs `decide (· = ·)` | 3 s |
| `m_boolint` | 2 | `Bool` + `Int` slots | 2 s |
| `m_relu_vec` | 3 | ReLU: `ite` on one side, `Max.max` on the other | 3 s |
| `m_mixed` | 4 | `Linear` + `Ite` | 3 s |
| `m_relu_net` | 2 | `Linear → ReLU → Linear` | 4 s |
| `m_relu_net8` | 8 | 8-wide net | 5 s |
| `m_relu_net16` | 16 | 16-wide net | 18 s |
| `m_vec32` | 32 | 32-wide affine | 73 s |

The cascade:

```lean
first
  | rfl
  | (simp [effect_k_fn, Scalar.update]; done)
  | (simp [effect_k_fn, Scalar.update]; omega)
  | (simp [effect_k_fn, Scalar.update, <matrix set>]; done)
  | (simp [effect_k_fn, Scalar.update, <matrix set>]; omega)
  | (simp [effect_k_fn, Scalar.update, <matrix set>]; split_ifs <;> omega)
  | (simp [effect_k_fn, Scalar.update, <matrix set>]; simp_all)
```

`<matrix set>` is the certificate's own `simp_mat` list — `matVecAffine`,
`dotL`, `List.ofFn_succ`, `Fin.sum_univ_*`, `MatAdd_apply`, … Without it the
`Linear` cases fail: `_translate_terms_scalar` does **not** scalarise a
`Linear`, it rebuilds the `Mat` and calls `matVecAffine`, so the right-hand
side arrives as `matVecAffine 2 [[1,0],[0,1]] [0,1] (fun i j => match i, j
with …)` and nothing arithmetic can touch it until that is computed away.
That was the one failure mode the experiment turned up, and it is fixed by
the simp set, not by a better prover.

Three things the measurements settle:

* **`smt` cannot be in the cascade at all.** Every case closed without it,
  which was measured first and is why the arm was only ever insurance — a
  module with a `Mul` of two state elements would have needed it. But the
  arm was measured in a *single file*, where the model and the bridge share
  an environment; in the generated project they are two, and `smt` is only
  in scope after `import Smt`. That import pulls in `auto` and with it
  `Auto.instBEqInt_auto`, a `BEq Int` instance outranking the
  `instBEqOfDecidableEq` the NA model elaborated its `==` against — so
  `effect_k_fn` is built over a different instance than `effect_k` and
  `link_k := rfl` stops typechecking. The insurance breaks the mechanism,
  so the bridge carries no `smt` arm and does not depend on `lean-smt`,
  whose two reconstruction bugs are catalogued in
  [`../../tests/lean/fbk/lean-smt-bug.md`](../../tests/lean/fbk/lean-smt-bug.md).
* **`decide` is useless here** and should not be in the cascade: the goal has
  free variables, and it fails with "Expected type must not contain free
  variables".
* **`split_ifs` before `simp` is a trap.** The two sides spell the same test
  `(x == 0)` and `decide (x = 0)`; `split_ifs` treats them as independent
  conditions and produces four branches, two of them unprovable. Plain `simp`
  normalises `==` first, and then there is one branch.

### The two-step design is not needed — measured

The obvious worry is that cvc5's rewriter puts the optimized body out of
reach, and that the way through is to emit the *unsimplified* model as a
middle term (`--fbk-simplify none`), prove optimized ≡ unsimplified with the
solver, and unsimplified ≡ functional by `rfl`.

Neither half of that holds up:

* the middle term would not be `rfl`-provable against `Scalar` anyway.
  `--fbk-simplify none` only disables `solver.simplify`; `smt_encode` has
  already flattened, scalarised and reassociated during term construction, so
  the unsimplified body is *still* a different expression from
  `_translate_terms_scalar`'s (`DESIGN_REVIEW.md` lines 17-52 makes this
  point, and it is why the body producers cannot be swapped);
* and the direct bridge closes. **Measured** on `m_countdown`, `m_boolint`,
  `m_relu_vec` and `m_relu_net8`, the unsimplified bodies close with the same
  cascade in the same time (3 s / 3 s / 3 s / 7 s) as the optimized ones.

So: one step, and `--fbk-simplify` stays what it is — a readability switch,
not a proof obligation.

---

## Above the transition: transferring the trace property

Links (1)-(3) give `INIT`/`TRANS` agreement. Turning that into a statement
about the module is one module-independent lemma, and it belongs in
`Core/Basic.lean` beside `rule_globally`:

```lean
theorem TS.transfer {S T L} (A : TS S L) (B : TS T L) (f : S → T)
    (hinit : ∀ s, A.start s → B.start (f s))
    (hstep : ∀ s l s', A.Tr s l s' → B.Tr (f s) l (f s'))
    (Q : StateSet T) :
    (∀ ts μs, B.ωTrace ts μs → ts ⊧ G (AP Q)) →
    ∀ ss μs, A.ωTrace ss μs → ss ⊧ G (AP (fun s => Q (f s)))
```

with `A := RM.toTS`, `B := M`, and `f := toSlots : CtrlNative → StateType`
the generated per-element projection (the same layout `_slot_layout` already
computes). `hinit`/`hstep` are `INIT_iff`/`TRANS_iff` read left to right.

The final artefact is then

```lean
theorem module_safety : ∀ ss μs, RM.toTS.ωTrace ss μs → ss ⊧ G (AP P)
```

which is the same shape the Büchi route's `def safety := rule_globally …`
produces, and says something about the *module*.

### The property has to be bridged too

`P` above cannot be `fun s => PROPERTY (toSlots s) = true`: `PROPERTY` is
`smt_to_lean_bool`'s reading of the user's `--safety` formula, so that `P`
would be the model's opinion of the property, not an independent one. The
missing piece is the same predicate through the *other* translator —
`smt_predicates_to_lean`, which the Büchi route already uses to write
`Data.lean`'s `P` — and one more obligation in the same cascade:

```lean
theorem PROPERTY_iff (s : CtrlNative) : PROPERTY (toSlots s) = true ↔ P s
```

Then the `--safety` formula is translated twice, independently, and the two
readings are proved equal. Without this the chain is sound about the
transition and silent about the property.

---

## Built

Implemented as designed, with the numbers below measured on the emitted
files rather than on prototypes. `translate/fbk_bridge.py` emits
`<Proj>/Certificate/Equivalence.lean`, the `Certificate` lean_lib globs its
submodules so `--build-cert` builds it, `Core/Basic.lean` carries
`TS.transfer`, and `--fbk-equiv none` turns it off.

The glob rather than an import from the root `Certificate.lean` is forced:
the bridge is stated in `Core.LTL`'s vocabulary and the certificate in
`LTLCertifying`'s, and both declare a top-level `LTLFormula`, so a module
importing the two is rejected before it is elaborated. Nothing is lost
today — `module_safety` takes the model-side statement as a hypothesis
rather than reading it out of the certificate — but discharging that
hypothesis means putting the two in one environment, so the name clash is
part of the work `KNOWN_ISSUES.md` #31 still records as outstanding.

**14 of the 17 module shapes the route accepts get a complete bridge.** The
three that do not are `m_max`, `m_min` and `m_argmax`, and they fail for the
reason already recorded as `KNOWN_ISSUES.md` #26 and #30: `matMin`, `matMax`
and `argmax_1d` are folds that never reduce, so the *right-hand* side of
`bridge_k` cannot be computed. The same defect blocks those modules' own
Büchi certificates. It fails loudly -- an unclosed goal in
`Equivalence.lean`, not a silent gap -- and `--fbk-equiv none` is the escape
hatch until the fold is fixed.

Two things the implementation added that the design did not have:

* **the budgets.** `set_option maxRecDepth 100000` / `maxHeartbeats 2000000`,
  the same the certificate and the scalar equivalence carry. Without them a
  64-deep straight-line transition (`m_deep`) exhausts the default recursion
  depth *inside `simp`* before any prover sees the goal.
* **the fallback arm of `toSlots`.** `TypeMap` is stuck at a variable index,
  so `| _ => …` does not typecheck -- the arm's expected type is
  `TypeMap x✝`, which reduces to nothing. `| _ + N => …` does, because the
  successor pattern lets the match reach its own fallback.

The plan it was built from:

1. **`Core/Basic.lean`** — add `TS.transfer` (above). Module-independent,
   written once, ~15 lines.
2. **`translate/fbk.py`** — `_slot_bodies` already returns the slot texts;
   emit the `_fn` forms and `toSlots` from the same data. The substitution is
   textual and total: `(var_k state)` ↦ `xk`.
3. **New `translate/fbk_bridge.py`** (or a section of `fbk.py`) — emit
   `<Proj>/Certificate/Equivalence.lean`:
   `import System.System`, `System.Scalar`, `Certificate.Data`, `<Proj>NA`; the
   `_fn` abbrevs; `link_k` (`rfl`); `bridge_k` (cascade); `INIT_iff`,
   `TRANS_iff`, `PROPERTY_iff`; `module_safety` via `TS.transfer`.
4. **`fbk_proveit.py`** — write it after the certificate is installed, and
   include it in the `--build-cert` build. The lakefile already declares the
   NA lib, and `Certificate/` is already a lake target.
5. **`main.py`** — `--fbk-equiv {lean,none}`, default `lean`. `none` for the
   cases where the bridge is the expensive part (`m_vec32`: 73 s against the
   ~5 s the rest of the route takes) and the user only wants the model.
   Refused without `--fbk-proveit`, like `--ic3ia` and `--fbk-simplify`.
6. **Tests** — `tests/test_lean_fbk.py` gets the emitter's shape (one `_fn`
   per slot, the substitution is total, the cascade is present); one
   slow-marked case per shape class (`m_countdown`, `m_boolint`,
   `m_relu_net`) builds the file, as `Certs/` does today. The existing
   `test_the_emitted_transition_agrees_with_the_module` stays as the fast
   check — it is the same claim, evaluated rather than proved.

### Cost

2-5 s for the shapes in the matrix, 18 s at 16 slots, 73 s at 32. That is
per generated project, only under `--fbk-proveit`, and it buys the one
statement the route does not otherwise have. The 32-slot case is also the
one whose *certificate* already exhausts Lean's heartbeats
(`tests/lean/fbk/README.md`), so both limits bite in the same place.

---

## Limitations

* **One direction.** `hinit`/`hstep` as implications make every module run a
  model run, which is what `G P` needs. An ic3ia **UNSAFE** verdict is not
  transferred by this: to conclude the *module* is unsafe you need the model
  run to be a module run, i.e. the converse. The obligations as measured are
  equalities, so the converse is available for free — it just has to be
  stated, and the counterexample itself is not currently brought back into
  Lean at all.
* **The NA model is not re-read.** The bridge proves things about the
  `effect_k` the generator emitted. If `proveit.py` were pointed at a
  hand-edited model, the bridge would still be about the generated one. The
  file is written and consumed in one run, so this is theoretical — but it is
  the reason the bridge is emitted from `_slot_bodies` rather than parsed
  back out of the `.lean` file.
* **`Real` and `BitVec` states are out of scope** because the route already
  refuses them (`vmt2lean.py` maps only `Int` and `Bool`). If that changes,
  `omega` stops applying and the cascade needs an arm that is not `omega` —
  `linarith`, since `smt` is ruled out by the instance clash above.
