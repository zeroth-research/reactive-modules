# `verith` pipeline review: code quality and modularity

Review of `uv run verith` (`zrth.lean.main:main`) against the design it is
meant to have:

```
input (one of several formats)
  -> reactive module (zrth.Module)
  -> Lean project (certificate parts possibly left as `sorry`)
  -> if parts are missing: a decision procedure (--infer) that reads the
     original input, the module, or the Lean project, and infers them
```

The three axes that should be cheap to extend: **a new input adapter**, **a
new decision procedure**, and **a change to the generated Lean** (ignoring
that the procedures reading that Lean would have to follow).

[`DESIGN_REVIEW.md`](DESIGN_REVIEW.md) and [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)
overlap in places; where they do, the item says so. Each item has an ID and
a status (`open`, `fixed <commit>`, `wontfix: why`) so the list can be worked
down. Paths are relative to `python/zrth/lean/` unless stated. A refactor
is checked by diffing `python/scripts/route_transcripts.py` before and after.

## Summary

The CLI and the `--infer` route table are well built: routes are data, the
cross-flag checks are generic, and `tests/test_infer_routes.py` holds the
rows to what they declare. Below them the pipeline does not follow the
design:

- there is **no input stage** (only Python files load);
- the stages talk through one **mutable, stringly-typed `CertificateData`**
  whose text fields are SMT-LIB at one moment and Lean at the next;
- the decision procedures have **no shared SMT layer** (two encoders, six
  obligation builders);
- **Lean codegen is coupled both ways**: routes print Lean, codegen calls
  cvc5, and the tactic planner reads the printed Lean back with regexes.

---

## 1. Input stage: missing

| ID | Status | Finding |
|----|--------|---------|
| IN-1 | fixed 578abb1e, a9c4766e | The only input path is `load_module_from_file`, which execs a Python file, and it lives in the Lean project builder (`project.py:770`). |
| IN-2 | fixed a9c4766e (SMV); wontfix for SV-COMP and Petri nets, which stay shims (answers F, G in `PHASE5_HANDOFF.md`) | Other formats bypass verith: `zrth/smv.parse_smv` already returns `(Module, name_map)` but is unreachable from the CLI; SV-COMP and Petri nets get in through Python shims selected by environment variables (`tests/bench_matrix/svcomp_mod.py`, `petri_mod.py`). |
| IN-3 | fixed 23dc5fbc, 333f8d66 (claims in the module's own terms; the bench harness keeps internal rows and converts at argv) | `--safety`/`--buchi`/`--pre` are positional SMT-LIB over `s0..sN` / `e0..eM`, so a property cannot be written in the input's own names. The bench harness translates (`tests/bench_matrix/suites.py` `smt()`, `entry_pre`). |
| IN-4 | fixed d344db30 | `module_file` is threaded into `create_project` only to copy it to `dbg/system.py` -- assumes Python input. |
| IN-5 | fixed 1cbd0688 (the fence), d12906c8 (`reads_source`; no route reads the source yet, answer E) | The original input never reaches a decision procedure: `InferInput` has no field for it. `main.py`'s docstring says `init()`/`update()` are there "so that the AI inference can read the high-level logic", but `ai` reads the generated Lean -- and the prompt fences it as ```` ```python ```` (`magic/ai.py:307,326`, `smt_prompt.py:573`). |
| IN-6 | fixed 7c6149fa, fcd83a25, b816ec10, 07012f9d | `zrth/smv` changes a model's meaning silently: a non-assignment `TRANS`/`INIT`/`INVAR` is dropped (`_promote_constraints`, `smv/parser.py:189-211`), and a VAR with no `init` starts at 0 and with no `next` holds its value (`:700,707`) where NuSMV leaves both unconstrained. Specs (`INVARSPEC`/`LTLSPEC`) do not parse; `JUSTICE`/`FAIRNESS` parse and are ignored. Every type lowers to BitVec (`integer` is BV32), and none of `tests/smv_fixtures/*.smv` has a Lean form. Must be fixed before SMV becomes an input adapter. |

**Direction:** an `InputAdapter` registry shaped like `ROUTES`, each adapter
returning a `Problem(module, names, source, prp?, pre?)`. Move the Python
loader to `inputs/python.py`, add SMV, and let properties use names.

## 2. `CertificateData`: the weakest point

| ID | Status | Finding |
|----|--------|---------|
| CD-1 | fixed 7b293e24..ea995fba | `prp`, `inv`, `ranking`, `init_pre`, `update_pre` are `str \| Expr \| None`, and a `str` is SMT-LIB, the Python DSL or Lean depending on the pipeline stage (`cert.py:24-38`). `main` keeps `cert_data` (SMT) and `project_cert_data` (Lean) side by side, and `_infer` rewrites the first into Lean in place (`main.py:408-429`). |
| CD-2 | fixed a4666f23 | Readers disagree: `smt_query.py:185-188` reads a `str` as SMT (`inv_smt or inv`), `cert._cert_def_context` reads it as Lean. `pre_check` detects Lean-where-SMT-was-expected by swallowing every exception (`ModuleQueries.build`, `smt_query.py:206`) and encoding the module a second time to tell the two failures apart. |
| CD-3 | fixed e1f45825..ed5980b6 | Routes pick their own convention: sygus/linear/houdini/vampire write SMT into both `inv` and `inv_smt`; cegar/learn write Lean in `inv`, SMT in `inv_smt`; ai writes Lean only. |
| CD-4 | fixed 8790cc61 | Tactic-planning data (`facts`, `hints`, typed `object`) rides on the certificate. `smt_predicates_to_lean` drops `hints` and `main` re-attaches them. |
| CD-5 | fixed 88dce29f | Routes mutate the pipeline's own object: `_take_inv` (`infer_route.py:438`) and every `infer(cd)`, although `InferInput` is frozen. |
| CD-6 | fixed 8790cc61 | `Expr` is in the annotation but never handled: `cert._field` (`cert.py:147-154`) silently falls through to the default `True` / `sorry`. The `list` (compiled IR) case is produced only by `tests/test_lean_twocounter.py:101` and `tutorials/counter_cvc5_lean.*` (the latter already broken by `from zrth import IType`). |
| CD-7 | fixed ea995fba | Bug: on `--infer ai` runs `_record_predicates` falls back from `inv_smt` to `inv` and writes the Lean answer to `artifacts/inv.smt` as `language="smt-src"`. Also `tests/test_lean_counter.py:95` hands Python-DSL strings straight to `create_project`, so its `Data.lean` holds `def P ... := s[0][0] == 0` -- CD-1 inside a test. |

**Direction:** three types -- an immutable `Problem` (property, kind,
preconditions), a `Certificate` in one canonical language (SMT-LIB source),
and a Lean rendering produced in exactly one place. Routes return SMT and
never mutate their input; `returns="lean"` (the `ai` route) becomes an
explicit exception instead of a convention.

## 3. `main.py`: orchestration

| ID | Status | Finding |
|----|--------|---------|
| PL-1 | fixed f42fc88f | `main()` is ~200 lines running a dozen concerns in sequence; inference alone is spread over `precheck`, `_infer`, `_replan`, a re-render and `write_predicates`. No function per stage. |
| PL-2 | fixed 309ba849..8996c985, c3cecc54 (fbk-proveit). Every golden run encodes once, `bare` never; `tests/test_pipeline_golden.py` and `test_lean_fbk.py` pin the count | The module is encoded into cvc5 four to six times per run (`main.py:286,313,418,529,538`, each `pre_check`, then inside each route), while the `InferResult` docstring argues against re-encoding. A per-run encoding object would fix it. Related: `DESIGN_REVIEW.md` E. |
| PL-3 | fixed e2dd5aaf | With a route, the pre-inference hints and Lean rendering are computed and then discarded by `_replan`. |
| PL-4 | fixed ce8c794b (the route hands its files back in `InferResult.files` and the pipeline writes them through the handle; `returns="files"` only skips regenerating from predicates there are none of) | `returns="installed"` (fbk-proveit) is a second control path that bypasses codegen. |
| PL-5 | fixed a4e6b159 | Bug: with no `--infer`, a module with no Lean form reports `error: --infer: ...` -- `_refused` falls back to that prefix when there is no route (`main.py:156,493`). |
| PL-6 | fixed d5b31785 | Bug: `_record_predicates(..., inferred=True)` also labels the *property* "Inferred by this run's route" (`main.py:230`). |
| PL-7 | fixed 16e37bf1 | Dead condition: `settings.hammer_file` at `main.py:504` (already returned at `:437`). |
| PL-8 | fixed 16e37bf1 | Leftovers: `_build_cert(args, ...)` receives `Settings`; `".. Doing TA2Magic"` prints with no route. |
| PL-9 | fixed 16e37bf1 | `lean_gaps(LeanContext(module))` (`main.py:491`) runs a discovery pass and drops it, though `lean_context_for` exists so that pass runs once. |
| PL-10 | open | Since phase 2 a Lean-rendering failure after inference raises from `main._regenerate` without the artifact note `_note_refusal` used to write when the route rendered its own Lean. |

## 4. Decision procedures

Good: rows with declarative `reads`/`owns`/`seeds`/`kinds`, an artifact store
that tracks staleness, generic CLI checks, good tests.

| ID | Status | Finding |
|----|--------|---------|
| RT-1 | fixed 33e34328 (`infer_route/<name>.py` exports `ROUTE`; resume helpers are `InferInput.resumable_inv`/`taking_inv`/`ruled_out`) | Adding a route means editing the 1,235-line `infer_route.py`: adapter, row, every option's help, and the resume helpers (`_resume_inv`, `_take_inv`, `_ruled_out`, `FIXABLE`) that other routes cannot reuse. |
| RT-2 | fixed 631c7026, e1f45825..ed5980b6, 862c904f (the contract is the row's `run(InferInput) -> InferResult`; `TA2Magic` only names `infer(claim, given)` and holds nothing; `ai` answers in Lean, which its row declares) | `TA2Magic` (`magic/__init__.py:25`) is vestigial: no two constructors agree, `learn` ignores `source` (the adapter passes `""`), fbk has no class, and dispatch is through `_run_*` adapters. Its docstring says only CEGAR implements safety; five routes do. |
| RT-3 | fixed ce8c794b (fbk writes `ProveIt/` through `handle.path`; its certificate and equivalence proof come back as files) | `ProjectHandle` access control is decorative: fbk receives `project_dir` and writes `ProveIt/*`, `Certificate/Certificate.lean`, `Certificate/Equivalence.lean` directly (`fbk_proveit.py:368,497,545`); `handle.path()` has no production caller. |
| RT-4 | fixed d74fed8a (`create_project(remove=...)`, from `infer_route.owned_files()`; `tests/test_layering.py` checks `project.py` imports no route) | `create_project` imports `ROUTES` to delete every route's owned files (`project.py:678`): an import cycle with `infer_route`, and the project builder knows about routes. |
| RT-5 | fixed ae78b487 (the `lakefile` hook returns `project.LakefileParts`; `PROVEIT_DIR`/`na_module_name` live in `fbk_proveit.py`) | fbk-specific code lives in `project.py`: `PROVEIT_DIR`, `na_module_name`, `generate_lakefile(ltl_project=...)`. The `lakefile` hook returns fbk's own kwarg; it should return generic fragments (extra `require`s and `lean_lib`s). |
| RT-6 | partly fixed 405b252e..d7a4936f, 83c13102 (`smt/`: one walker, script printer, Vampire driver, budget, recorder, per-run `ModuleEncoding`, obligation builder `smt/obligations.py`). Left: nuterm's z3 encoding and `_farkas.Inductive/LexDecrease` (RT-8); `sygus._pre/_trans`, existential relations not on the builder; `translate/fbk._slot_bodies` runs its own `Solver` for the rewriter (on the run's encoding since c3cecc54; CG-2); smt-linear's `_bound_vars`/`_fresh_consts` instead of `ModuleEncoding.fresh`; houdini and vampire parse with `env.parse_expr` (their candidates may be Real) where the rest use `ModuleEncoding.parse` | No shared SMT layer. Two module encoders (cvc5 via `smt_encode`/`ModuleSMT`; z3 via `benchmarks.svcomp._farkas` for nuterm). Obligations built six ways: `cegar._check_*`, `ModuleQueries.check_*`, `houdini.Obligations`, `linear._inductive/_safety_at/_buchi_at`, `sygus._pre/_trans`, `_farkas.Inductive/LexDecrease`. Input substitution three times (`cegar._subst_inputs`, `SynthContext.with_inputs`, `ModuleQueries._sub_inputs`), SMT script printing four times, two Vampire subprocess drivers, three near-identical artifact recorders, three ite-condition collectors. |
| RT-7 | fixed fd0ff290 (`tests/test_layering.py`: no route imports another) | `linear` and `vampire` import from `houdini`, so `houdini.py` (~2,000 lines) is a shared library by accident. |
| RT-8 | partly fixed 736e568c (cvc5 is a runtime dependency). Left, by the user's choice: nuterm's engine stays in `benchmarks/svcomp`, moved on another branch to be rebased onto this one. The symptom is a non-editable install, not running from outside `python/`: the editable one puts `python/` on the path | `magic/learn.py:69` imports `benchmarks.svcomp`, which is not packaged: `nuterm` works only when run from `python/`. cvc5 is only a dev dependency (`pyproject.toml:58`) yet most routes need it. |
| RT-9 | fixed 405b252e, 10364d04, 12738ceb for the SMT layer (`tests/test_layering.py`); Lean codegen not checked yet (CG-9) | Private names cross modules: `smt_query._constants` (houdini, modes, houdini_solver), `houdini_solver._kill_group` (vampire), `ai._make_client` / `_describe_preconditions` (cegar). |
| RT-10 | fixed 12738ceb, 73e484bd (`magic/llm.py`; `log=`, `budget=`, `Refused`) | Inconsistent plumbing: ai/cegar `print` while the rest take `log=`; cegar's cvc5 calls have no time limit (`cegar.py:387`); cegar defaults `model="gpt-4"`, ai `"claude-sonnet-4-6"`; cegar raises `RuntimeError`, not `Refused`, when cvc5 is missing. |
| RT-11 | fixed ec35fe5d | **Security:** `smt_prompt.parse_predicate` runs `eval` on LLM replies with `{"__builtins__": {}}` (`smt_prompt.py:273`). That sandbox is escapable, so a reply can run arbitrary Python. |
| RT-12 | fixed ce8c794b for extra Lean (`InferResult.files`, checked against `owns`). Tactic hints from a route: not added, since no route has any; the pipeline asks cvc5 for hints over whatever predicates come back | `InferResult` carries only inv/ranking x smt/lean: no channel for lemmas, tactic hints or extra Lean other than writing files directly. |
| RT-13 | open | `nuterm` records its answer only as `inv.smt` (`smt-src`, encoded), not as a resumable `inv` artifact, so a following `--infer ai-cegis` asks the model again instead of resuming. |
| RT-14 | open | `ai-cegis` builds an LLM client even when every predicate is fixed (`magic/cegar.py`), and its names still say CEGAR (`cegar.py`, `TA2MagicCEGAR`, `CegarPromptEnv`) for what the CLI calls `ai-cegis`. |

**Direction:** one module per route exporting `ROUTE`; resume helpers on
`InferInput`; fbk through `handle.path()`; a generic lakefile hook; an
`smt/` package with one per-run `ModuleEncoding` and one obligation builder
(`smt/obligations.py`) that every SMT route uses.

## 5. Lean code generation

Good: the `ENCODINGS` table with `Layout`; Jinja for the project skeleton and
the certificate's proof structure.

| ID | Status | Finding |
|----|--------|---------|
| CG-1 | fixed daf612c2, 74cdaff7, a98a38cb, 6a68153d (the plan reads facts and printed terms; text is scanned only where no term stands behind it, a Lean answer's fields; the `share=False` coupling was unneeded) | Tactic planning reads printed Lean: `tactics.features_for` tests substrings (`" ∧ " in pred_text`, `"if "`, `"⌊"`), `_VAR` hardcodes `s\|e\|el`, `_rebind` rewrites `\bs\b`. `hints.determined` is Lean text that must match the definitions character for character, which is why `_replan` reprints everything with `share=False` (`main.py:313`). Plan from terms (`facts`); printing is codegen's choice. |
| CG-2 | fixed 13ba8590, a1c4cce5, 26c41773 (solver hints and NA slot terms are built in the SMT layer and printed by codegen; refusals name a predicate by role) | Codegen calls solvers: `smt_predicates_to_lean` builds a cvc5 `ModuleQueries` and writes error text naming CLI flags; `translate/fbk._slot_bodies` runs cvc5's simplifier; `smt_query.py` emits Lean tactic text (`:940-960`, `:1042`). |
| CG-3 | fixed 59da0f6a | Bug: `magic/ai.py:116` finds `init` with `^@\[simp\] def init`, which misses `@[simp] noncomputable def init` (`translate/functional.py:40`), so Real-valued modules lose the state-type line in the prompt. |
| CG-4 | fixed 12042bcc, 6811d572 (`lean_names`: module and file names, `RM`, `MAT_SIMP`; `_REVERT`/`_SMT_CLOSER` kept, TODO.md) | Consumers depend on exact Lean text: `fbk_bridge.py:179-181` hardcodes `System.System`, `Certificate.Data`; `RM` exists twice (`Certificate.lean.j2:25-30`, `fbk_bridge.py:253`); `fbk_bridge._MAT_SIMP` copies `simp_mat`; `fbk_proveit` regex-patches proveit's output (`_REVERT`, `_SMT_CLOSER`). |
| CG-5 | fixed 31b67e4f (the `project.py` comment was rewritten in 631c7026) | Hidden state in `ModuleToLean4`: `_init_layer_names`, `_argmax_scalar_variants` are filled by one method and read by another, so `*_equiv` alone silently drops lemmas (`DESIGN_REVIEW.md` F). The comment at `project.py:224` says row order matters; it does not -- the real coupling is Lean names such as `init_scalar_eq`, captured only by `Encoding.needs`. |
| CG-6 | fixed fc2213e7 (closed by policy, in the README, plus one shared definitions partial) | Four ways of rendering: Jinja; tactics spliced in as Python strings; encodings via f-strings; `_cert_def_lines` (`cert.py:61-117`) duplicating `Data.lean.j2` for `--cert-file`, with a different `has_ranking` rule. |
| CG-7 | fixed 56a5c87d | Dead templates `templates/project/System/{Rel,Scalar,ScalarRel}.lean.j2`, never rendered and already drifted (`KNOWN_ISSUES.md`). |
| CG-8 | fixed eeb0817c, da757ac2, 1ff16cb3, 55784379 (every Sort is 2-D, so the 1-D disagreement was unreachable; literals unified, the Lean changes in text only) | Duplicated helpers that disagree: element type x4 (`dtype_to_lean_type`, `_flat_element_type`, `_sort_elem_ty`, `tactics._sort_name`); literals x4 (`_get_dtype_item`, `_tensor_to_lean_inline`, `_tensor_to_lean_scalar`, `smt_to_lean._walk`) formatting Int/Real differently; shape-to-`(m, n)` at ~10 sites that disagree on 1-D shapes (latent); two SMT-to-Lean walkers (`_walk`, `_walk_bool`). `_bind_wires_scalar` and `_vec_from_scalars` are dead. |
| CG-9 | fixed c2d14eca (`common.py` re-exports, for `magic/learn.py`) | `common.py` is a grab-bag: `Refused` (the CLI-wide error type), type helpers, constants, `LeanContext`, bindings, `FlatLayout`. |
| CG-10 | fixed 8b6caebd | Bug: `native._translate_terms` returns `"sorry /- no terms -/"` (`native.py:190,300`), which is truthy, so the `if init_body:` guards (`translate/functional.py:38,44`, likewise `scalar.py`) never skip and a module with no init terms gets `def init ... := sorry`. |
| CG-11 | fixed 8790cc61 | `write_certificate_lean` builds `LeanContext(module)` without the certificate's terms when no `ctx` is passed (`project.py:583`); `write_data_lean` uses `lean_context_for`. |
| CG-12 | fixed 5ab7c4b7 (`cert._body` went in 8790cc61) | Stateless modules (no controlled variables) after CG-10: the scalar and relational theorem guards still skip theorems for an empty block, but `translate/fbk_bridge.py` cites `init_scalar_eq` unconditionally; `cert._body` raises `IndexError` on an empty term list. |
| CG-13 | fixed 5b5627c1 | `_translate_terms_scalar(flatten_outputs=False)` (the default) has no production caller. The README's `ModuleToLean4` table still describes `to_lean` as "all three encodings" and omits the `rel` flag. |

## 6. Comments

| ID | Status | Finding |
|----|--------|---------|
| DOC-1 | fixed 631c7026, 88dce29f | Comments are long and narrate history and measurements ("used to", "today", "measured, 124 s"), and several are already wrong: the `TA2Magic` docstring (RT-2); `project.py:224` (CG-5); `native.py:373` (describes typing in `rel.py`/`fbk.py` that neither does); `translate/__init__.py:3` and `README.md` refer to `translate.py`; the `InferResult` re-encoding argument (PL-2). Keep invariants in comments; move rationale to commit messages or design docs. |

---

## Target shape

```
inputs/          adapter registry (python, smv, svcomp, petri)  -> Problem
smt/             one ModuleEncoding per run + one obligation builder (RT-6 lists who bypasses it)
infer_route/     <name>.py exports ROUTE; run(InferInput) -> InferResult, no mutation (done)
lean/            Problem + Certificate -> files; tactics planned from terms, not text
pipeline.py      load -> generate -> infer -> regenerate certificate -> check/build
```

## Order of work

1. **Bugs and safety** -- RT-11, PL-5..PL-9, CG-3, CG-10, CG-7, stale
   comments under DOC-1.
2. **One certificate language** -- CD-1..CD-7, PL-1, PL-3, CG-11. Unblocks
   most of the rest.
3. **Shared encoding and obligations** -- PL-2, RT-6, RT-7, RT-9, RT-10.
4. **Routes as modules** -- RT-1..RT-5, RT-12, PL-4.
5. **Input adapters** -- IN-6 first, then IN-1..IN-5. Done:
   0e6b45b3..07012f9d. Plan, answers and how it was checked:
   [`PHASE5_HANDOFF.md`](PHASE5_HANDOFF.md).
6. **Codegen cleanup** -- CG-1, CG-2, CG-4..CG-6, CG-8, CG-9, CG-11, RT-8. Done:
   c4f5b00e..26c41773; the plan and the user's answers are in
   [`PHASE6_PLAN.md`](PHASE6_PLAN.md).
7. **A README for the pipeline** -- last, once the code has its final shape:
   the stages from input to built certificate, with a picture the user
   supplies.
