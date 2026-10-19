# `verith` pipeline review: what is still open

What is left of two reviews of `uv run verith` (`zrth.lean.main:main`)
against the shape the [README](README.md#the-pipeline-stage-by-stage)
describes: an input stage, one certificate language, one SMT layer, routes as
modules, and Lean codegen that plans from terms rather than printed text.
Each item has an ID and a status (`open`, `partly fixed`); a fixed item is
removed, and its fix is in the commit message. Paths are relative to
`python/zrth/lean/` unless stated. Defects, as opposed to shape, are in
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Open items

| ID | Status | Finding |
|----|--------|---------|
| PL-10 | open | A Lean-rendering failure after inference raises from `main._regenerate` without the artifact note that `_note_refusal` writes for a route's own refusal. |
| RT-6 | partly fixed | Not every SMT user is on `smt/` yet: nuterm's z3 encoding and `_farkas.Inductive`/`LexDecrease` (RT-8); `sygus._pre`/`_trans` and their existential relations are not on the obligation builder `smt/obligations.py`; smt-linear uses `_bound_vars`/`_fresh_consts` instead of `ModuleEncoding.fresh`; houdini and vampire parse with `env.parse_expr` (their candidates may be Real) where the rest use `ModuleEncoding.parse`. |
| RT-8 | partly fixed | `magic/learn.py` imports `benchmarks.svcomp`, which is not packaged, so `--infer nuterm` needs the editable install (which puts `python/` on the path). Left as is by the user's choice: the engine is moved on another branch, to be rebased onto this one. |
| RT-9 | partly fixed | Private names crossing modules are checked for the SMT layer (`tests/test_layering.py`) but not for Lean codegen. |
| RT-13 | open | `nuterm` records its answer only as `inv.smt` (`smt-src`, encoded), not as a resumable `inv` artifact, so a following `--infer ai-cegis` asks the model again instead of resuming. |
| RT-14 | open | `ai-cegis` builds an LLM client even when every predicate is fixed (`magic/cegar.py`), and its names still say CEGAR (`cegar.py`, `TA2MagicCEGAR`, `CegarPromptEnv`) for what the CLI calls `ai-cegis`. |
| CG-14 | open | The SMT encoding is never used to *check* the Lean encoding. `pre_check` asks cvc5 about the certificate (`init_inv`, `step_inv`, `hrank`); nothing asks whether `native.py`'s Lean and `smt_encode`'s cvc5 terms agree about the module, which they do only by construction, and the equivalence theorems can stay provable while both sides are wrong (`KNOWN_ISSUES.md`, "Traps worth knowing"). Re-parsing the emitted Lean would be circular. The non-circular check gives `native.py`'s op table a cvc5 *evaluator* sibling in the same registry, and checks that the two agree on random concrete inputs, in milliseconds per module. That is what catches the `argmax` seeding and tie-breaking class of bug, which `argmax1d_scalar_n_eq` could not. |

## Deliberately not changed

* **`smt_module.ModuleSMT` as the single SMT front-end.** Prompts
  (`smt_prompt`), queries (`smt_query`) and the NA encoding all route
  through it.
* **`LeanContext`**: built once, read-only, shared by every consumer.
* **The `translate/` split by encoding**, with the shared relational
  skeleton behind it in `translate/_skeleton.py`.
* **`smt_to_lean`'s two printers** (`_walk` for Prop, `_walk_bool` for
  Bool). They share their leaf printer but not their walk, because they
  target different Lean fragments: `lean2vmt` reads `decide`'s *instance*
  argument, so a compound proposition under `decide` becomes an unapplied
  leaf. Merging them would cost more than it saves.
* **The sharing/`let` logic, the `min`/`max` folding, the rule that
  `_simplify` never touches a Bool slot.** Each is a measured fix carrying
  its measurement in the comment.

## How to check a refactor

Record each of these at the commit before the change and after it, then
compare; every difference must be one the change means.

- **Route transcripts**, the main check:
  `uv run python scripts/route_transcripts.py OUTDIR` from `python/`, then
  `diff -r`. 42 cases over all eight routes; LLM replies are scripted, so no
  key is needed, and the docstring names the variables that find the
  external tools. Record on a quiet machine: `vampire_twovars` and
  `vampire_buchi` share a deadline between their templates and can lose
  lines under load, so re-run such a case alone before reading it as a
  change. Keep the directories out of `/private/tmp`, which macOS empties of
  files left untouched for three days.
- **Encoding snapshots**: `scripts/encoding_snapshots.py OUTDIR` writes the
  `System/*.lean` encodings, the bare `--cert-file` certificate and the NA
  model and bridge per fixture, which neither the transcripts nor the
  goldens cover.
- **Goldens**: `tests/test_pipeline_golden.py` pins whole-run output and
  the number of cvc5 encodings per run; `ZRTH_UPDATE_GOLDEN=1` rewrites
  them.
- **Suites**: `VAMPIRE=<binary> uv run pytest -m 'not slow' -q` from
  `python/` (about 1450 tests, 4 minutes), and `just test-lean` for the
  lake builds after any change to the generated Lean.
