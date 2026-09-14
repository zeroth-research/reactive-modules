# Design review: the encoding paths in `zrth/lean`

A read of the whole package — `common.py`, `native.py`, `circ.py`,
`translate/*`, `smt_encode.py`, `smt_module.py`, `smt_to_lean.py`,
`smt_query.py`, `cert.py`, `project.py`, `main.py`, `fbk_proveit.py` — asking
one question: where is the structure duplicated, and how much of the SMT path
`translate/fbk.py` opened up can the other encodings reuse.

Defects already catalogued in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) are not
repeated here; this is about shape, not breakage. Where the two overlap, the
issue number is cited.

`MEASURED` was reproduced by running the code. `READ` comes from reading it.

---

## The question that prompted this: can `Rel`/`ScalarRel` reuse the SMT path?

**Not as a body producer, and the reason is structural rather than
incidental.**

`translate/fbk.py` can build its transition from `smt_encode` because the NA
model carries no Lean-side proof obligation. Nothing has to show `effect_i`
equals anything: trust leaves Lean through ic3ia and comes back as
`vmt2lean.py`'s own certificate. That frees the body to be whatever cvc5
hands back.

`Rel` and `ScalarRel` exist *for* their equivalence theorems — rows 3-6 of
`GENERATED.md`'s "Summary of what is claimed". Those proofs are:

```python
# translate/rel.py:75-79
f"  simp only [{scalar_func}, {func_name}, Fin.cons_zero, Fin.cons_succ]",
f"  try rfl",
```

They close only because both sides are emitted by the *same*
`_translate_terms_scalar` over the same `_reachable_terms`, so `effect_i` and
the matching slice of `Scalar.update` are syntactically equal. A cvc5-printed
body is normalised — constant folding, `ite` reshaping, affine expansion;
`fbk.py`'s own docstring measures 97 KB to 1.9 KB on `m_vec32` — and `rfl`
will not bridge that. `--fbk-simplify none` does not help either: it only
disables `solver.simplify`, and `smt_encode` has already flattened and
reassociated during term construction.

Swapping the body producer would trade a `rfl`-provable equivalence for one
nothing can discharge.

**What the fbk work does make reusable** is the skeleton (A), the slot layout
(B), and the SMT path *as a checker* rather than a generator (G).

---

## Findings

### A. Three relational emitters spell one skeleton three times · MEASURED

| emitter | namespace | domain | body from |
|---|---|---|---|
| `translate/mat_rel.py:16` `atom_to_lean_mat_rel` | `Rel` | `Mat`, Prop | `_translate_terms` |
| `translate/rel.py:19` `atom_to_lean_rel` | `ScalarRel` | flat scalars, Prop | `_translate_terms_scalar` |
| `translate/fbk.py:315` `atom_to_lean_na` | `Definition` | flat scalars, Bool | `smt_encode` + `smt_to_lean_bool` |

All three emit the same shape: a per-slot body def (`effect_i` / `init_i`), a
per-slot relation (`R_i` / `Init_i`), and a conjunction over them
(`TransRel` / `INIT`).

94 of `mat_rel.py`'s 162 body lines are byte-identical to lines in `rel.py`
(`diff -u`, counting context lines).

They vary on exactly three axes:

1. **type builder** — `_product_type` vs `_product_type_scalar` vs the
   per-slot type list;
2. **body producer** — the three in the table above;
3. **projection** — `_accessor(i, n)` vs `_flat_slice(offset, size, total)`
   vs `var_k state`.

One emitter parameterised on those three, with the `*_eq` theorem block
optional, collapses `rel.py` and `mat_rel.py` into one and lets `fbk.py`
supply its own body producer without owning a copy of the skeleton. It
touches no proof: the theorem block stays exactly where it is, emitted only
by the two instances that can prove it.

### B. The element flattening has three owners · READ

Three views of one row-major flattening, derived independently:

| function | returns |
|---|---|
| `translate/_shared.py:14` `_flat_layout` | `[(offset, size)]` per wire, plus total |
| `translate/fbk.py:192` `_slot_layout` | `(wire index, row, col)` per slot |
| `translate/fbk.py:199` `_slot_accessors` | `s{i}` to its `(var_k state)` reads |

`KNOWN_ISSUES` #20 and #22 are both exactly these drifting apart — a per-wire
layout meeting a per-element one. One owner returning all three views makes
that class of bug unrepresentable rather than merely fixed.

### E. The NA route encodes the module into cvc5 two to three times per run · MEASURED

`atom_to_lean_na` calls `check_na_supported`, which runs `_slot_bodies` to
test-print the transition, and then calls `_slot_bodies` again for the real
emission. Instrumented on a one-wire counter: **2** calls, and **2** cvc5
`TermManager` + `Solver` pairs appended to `translate/fbk.py:218 _LIVE` and
retained for the process's lifetime. `main.py`'s `check_module` adds a third
before generation.

Having `check_na_supported` return the bodies it already computed removes two
thirds of it. The `_LIVE` retention is a deliberate workaround for the cvc5
bindings' shutdown ordering and should stay — there is just less to retain.

### F. `ModuleToLean4` is silently order-dependent · MEASURED

`to_lean_scalar_equiv()` reads `self._argmax_scalar_variants`, which only
`atom_to_lean_scalar()` populates. Both are public (README documents them as
a table of methods). Called alone on a module using `Argmax`:

```
with atom_to_lean_scalar first, argmax lemma present: True
called alone,                   argmax lemma present: False
```

The missing `argmax1d_scalar_int_3_eq` drops out of the `simp only` list and
the emitted proof fails at `lake build` rather than at generation. Same
coupling for `atom_to_lean_circuit()` to `to_lean_equiv_theorems()` through
`_init_layer_names` / `_update_layer_names`.

Current callers happen to be in order. Returning the variants from the
producer, or caching on first use, removes the trap.

### G. The SMT encoding is never used to *check* the Lean encoding · READ

`pre_check` asks cvc5 about the **certificate** — `init_inv`, `step_inv`,
`hrank`. Nothing asks whether `native.py`'s Lean and `smt_encode`'s cvc5
terms agree about the **module**. They are asserted equal by construction,
and finding C shows they already diverge in coverage.

This is the reuse with the best payoff, and the one `KNOWN_ISSUES`' own
lesson points at:

> the equivalence theorems can stay provable while both sides are wrong

Re-parsing the emitted Lean would be circular. The non-circular version is to
give `native.py`'s op table a cvc5 *evaluator* sibling in the same registry
(finding C) and check the two agree on random concrete inputs — milliseconds
per module. That is what catches the `argmax` seeding and tie-breaking class
of bug, which `argmax1d_scalar_n_eq` could not.

### H. Minor

* Emission glue is hand-rolled five times in `project.py:551-610` and three
  times in `main.py:644-676` — header comment, imports, `m2l.to_lean_*()`. A
  `(filename, header, imports, producer)` table would also make the
  `--cert-file` and project paths provably consistent instead of
  consistent-by-inspection.
* `common.py:_constant_expr` and `native.py:_constant_expr_scalar` take a
  `const_name` argument neither uses.
* Two independent cvc5 leak-lists (`smt_query.py:111`,
  `translate/fbk.py:218`) for the same shutdown-ordering workaround.

---

## Deliberately not changed

* **`smt_module.ModuleSMT` as the single SMT front-end.** Prompts
  (`smt_prompt`), queries (`smt_query`) and the NA encoding all route through
  it. That consolidation is already done and is what made the fbk route
  possible.
* **`LeanContext`** — built once, read-only, shared by every consumer.
* **The `translate/` split by encoding.** The right seam. What is missing is
  the shared skeleton behind it (A), not a different split.
* **`smt_to_lean`'s two printers** (`_walk` for Prop, `_walk_bool` for Bool).
  They look like duplication and are not: they target different Lean
  fragments for a documented reason — `lean2vmt` reads `decide`'s *instance*
  argument, so a compound proposition under `decide` becomes an unapplied
  leaf. Merging them would cost more than it saves.
* **The sharing/`let` logic, the `min`/`max` folding, the rule that
  `_simplify` never touches a Bool slot.** Each is a measured fix carrying its
  measurement in the comment.

---

## Order of work, if any

1. **C** — one op registry with an explicit coverage matrix and a
   completeness test. Highest value, independent of everything else.
2. **A + B** — one relational skeleton, one slot-layout owner. Removes the
   duplication that #20 and #22 came out of.
3. **G** — the SMT path as a cross-check on the Lean path. Builds on C.
4. **D, E, F** — small, independent, each a few lines.

Overall the package is in good shape; the discipline of recording *why* in
the comment next to the code is unusually strong and should survive any of
the above.
