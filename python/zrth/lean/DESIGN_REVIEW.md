Defects already catalogued in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) are not
repeated here; this is about shape, not breakage. Where the two overlap, the
issue number is cited.

---


## Findings


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
