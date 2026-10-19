# Phase 6 plan: codegen cleanup

Done 2026-10-19 as `c4f5b00e..26c41773`; `PIPELINE_REVIEW.md` names each
item's commits. Kept for the answers and the reasoning.

Phase 6 of the `verith` refactor ([`PIPELINE_REVIEW.md`](PIPELINE_REVIEW.md),
"Order of work"): CG-1, CG-2, CG-4..CG-6, CG-8, CG-9, CG-12, CG-13, RT-8.
Phase 7, a README for the pipeline with a picture the user supplies, comes
after it. Checking and working rules are phase 5's
([`PHASE5_HANDOFF.md`](PHASE5_HANDOFF.md), "How to check a commit" and
"Working rules").

## The user's answers (2026-10-19)

- **RT-8**: nuterm stays where it is. The user has it moved on another
  branch, which will be rebased onto this one, so this phase does not
  touch `magic/learn.py` or `benchmarks/` (not even import migrations).
  cvc5 becomes a runtime dependency.
- **CG-8 literals**: unify them into one printer. The generated Lean
  changes in text, and `just test-lean` checks that the proofs still
  build.
- **CG-4 shared Lean**: Python constants. One Python source renders `RM`
  and the `simp_mat` list wherever they appear, so `RM` and `Bridge.RM`
  stay two Lean constants. The project layout does not change.
- **fbk-proveit patches** (`_REVERT`, `_SMT_CLOSER`): keep them as they
  are; a TODO item says to revisit them.

Decided without asking, as technical choices:

- Tactic planning for a Lean-only (`ai`) answer keeps a text scan, but
  only for the verbatim fields, and merged with the facts. Today the
  facts *replace* the scan's `nonlinear`, which drops `nlinarith` for a
  nonlinear Lean invariant.
- Error text keeps naming CLI flags. Codegen raises role-tagged
  refusals, and `main` maps each role to its flag.
- CG-6 is closed by policy rather than by rewriting the encoders: Jinja
  renders file skeletons and Python renders expressions. The
  `--cert-file` definitions stop being a hand copy of `Data.lean.j2`.

## Commits

1. deps: cvc5 is a runtime dependency (`pyproject.toml`, `uv.lock`; then
   recover with `uv sync --inexact --no-build-isolation-package zrth`).
2. CG-13: drop `flatten_outputs` (always flattened); fix the README's
   `ModuleToLean4` table.
3. CG-5: `ModuleToLean4` keeps no state between methods. The argmax
   variants are derived where they are used, and the circuit names come
   from a cached circuit result. Add tests that call each `*_equiv` alone.
4. CG-12: add `scalar_eq_name` and one `has_init`/`has_update` guard,
   shared by scalar, relational and the bridge. The bridge refuses a
   stateless module itself. Add a test.
5. CG-4a: add `lean_names.py`, a leaf module holding the layout and the
   names of the Lean modules, files and lemmas. The bridge's imports, the
   fbk route's `owns` and files, and `build_certificate` read it. Add a
   test that every module the bridge imports is a file the project
   writes.
6. CG-4b: one source each for `MAT_SIMP` and `RM`, used by the template
   and by the bridge.
7. CG-8a: delete the dead helpers; one element-type function.
8. CG-9: split `common.py` into `errors`, `sorts`, `literals`, `layout`
   and `context`, keeping `common.py` as a re-export shim. Importers
   migrate except `magic/learn.py` and `benchmarks/`.
9. CG-8b: one shape rule. `MatShape`/`wire_shape` move out of
   `smt_encode` (which imports cvc5) into `sorts`, and 1-D `[n]` reads as
   1xn everywhere (latent: no front end builds 1-D).
10. CG-8c: one `lean_literal` (Int ascribed, Real as an exact quotient).
    Goldens change in text; `just test-lean`.
11. CG-8d: the two SMT-to-Lean walkers share their leaf printer, and
    `_walk_bool` refuses an unknown variable with `Refused`.
12. CG-6: one `has_ranking` rule (`not is_safety`); the `--cert-file`
    definitions render from the template's partial.
13. CG-1/CG-2, in steps:
    - a. `cert.prepare` holds the `share` rule once, for `main._render`
      and the golden test. Fix the `nonlinear` override for Lean-only
      answers, and add a golden for a nonlinear `LeanAnswer`.
    - b. `SolverHints` carries terms; the plan prints them.
    - c. `PredicateFacts` gains the floors; `_rebind` and
      `floor_arguments` go.
    - d. `plan_for` plans from the facts. The text scan applies only to
      verbatim fields, merged with the facts.
    - e. `predicate_terms` leaves `cert`. Refusals are role-tagged and
      `main` names the flags.
    - f. Check with a Lean build whether `share=False` is still needed;
      drop the coupling if not.
    - g. fbk: `ModuleEncoding.simplify`. Slot terms are built on the
      route side and `translate/fbk` only prints, so the fallback
      encodings go.
14. The review: mark the items, and correct RT-8's symptom (a
    non-editable install, not "only from `python/`").

Checking: the route transcripts per group of commits that should change
nothing; the goldens per commit; `just test-lean` after every commit that
changes generated Lean (6, 10, 12, 13). Neither the goldens nor the
transcripts cover `System/*.lean` or the fbk bridge, so commits 2-6
also diff the generated encodings over the fixtures, before and after.
