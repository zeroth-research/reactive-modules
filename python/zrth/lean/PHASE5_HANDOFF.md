# Phase 5 handoff: input adapters

Everything a fresh session needs to start phase 5 of the `verith` pipeline
refactor. [`PIPELINE_REVIEW.md`](PIPELINE_REVIEW.md) is the item list the
phases work down; this file is the plan for its section 1 and the way of
working that phases 2-4 settled on. Paths are relative to `python/zrth/lean/`
unless they start with `python/`.

## Where things stand

- Branch `mch/verith-1`. Phases 1-4 of the review's "Order of work" are done
  and each item's status names its commits. Phase 4 is `33e34328..c3cecc54`
  (code) and `26aeed7a` (docs); this file was added with its review update.
- Phase 5 is IN-1..IN-6. Not part of it: phase 6 (codegen: CG-1, 2, 4-6, 8,
  9, 12, 13 and RT-8) and the open route items RT-6 (remainder), RT-13,
  RT-14 and PL-10.
- The pipeline today: `main.main` loads a Python file
  (`project.load_module_from_file`, via `main._load`), builds the claim from
  positional SMT-LIB flags (`main._claim`), runs the route's precheck,
  generates the project (`main._generate` -> `project.create_project`), runs
  the route (`main._infer`, rows in `infer_route/<name>.py`), rewrites the
  certificate (`main._regenerate`), writes any whole files the route handed
  back (`InferResult.files`, through `ProjectHandle.write_files`) and builds
  it (`main._build_cert`). One cvc5 encoding per run (`smt.run.lazy`), handed
  to routes as `InferInput.smt` and to a row's `precheck`.
- What phase 5 copies from phase 4: `infer_route/` is the shape an
  `inputs/` registry should have (one module per entry exporting a row,
  a tuple in `__init__.py`, types in `api.py`, engines imported lazily so
  `verith --help` stays cheap). `Opt`, the per-route CLI option with its
  "only meaningful with" check, lives in `infer_route/api.py`; an adapter's
  options (`-d` for Python) want the same type, so move it to a leaf module
  both can import rather than importing it from `infer_route`.

## First: the user's answers

Phase 5 cannot start without these; ask them together, before any code.
Each has a recommendation.

**A. SMV variables with no `init` or no `next`.** NuSMV leaves them
unconstrained: no `init` is a free initial value, no `next` a free value at
every step. `zrth.smv` silently starts them at 0 and holds them
(`smv/parser.py` `_build_module`). Options: refuse them, or read them as
inputs (a free initial value is an input read at init, a free next value an
input read each step), which is NuSMV's meaning. *Recommended: inputs.*

**A'. SMV `integer`.** Lowered to a 32-bit bitvector (`_DEFAULT_INT_BW`), so
it wraps where NuSMV's does not. Options: keep BV32, or lower to `Int`
(LIA). Ranges `a..b` and enums are the same question. *Recommended: `Int` for
`integer` and ranges, keep enums as codes.*

**B. SMV specs.** `INVARSPEC`/`LTLSPEC` do not parse today. Should a file's
spec supply the claim (`INVARSPEC p` -> `--safety p`, `LTLSPEC G F p` ->
`--buchi p`, anything else refused)? If the command line also gives a
property: override, or refuse? Several specs: a `--spec N` flag?
*Recommended: the file's spec is used only when no `--safety`/`--buchi` is
given; both is an error; several need `--spec N`. Optional commit, last.*

**C. Names for Python inputs.** Properties are positional (`s0`, `e0`, `el0`)
in every format today. Where do Python modules' names come from: a `Named`
wrapper the file returns, gym's `_wire_names` (`zrth/gym/module.py`), or the
parameter names of a `sugar.Module` update? *Recommended: `Named` only.*

**D. Latched inputs.** `el{i}` is input `i`'s previous value. Give it a named
spelling (e.g. `(prev x)`), or keep `el{i}` only? *Recommended: `el{i}` only
for now.*

**E. The original input in a prompt (IN-5).** No route reads the source
today. Plumb it only (`InferRoute.reads_source`, `InferInput.source`), or
also put it into the ai-cegis prompt and measure on the bench matrix?
*Recommended: plumbing only.*

**F. SV-COMP.** Its modules arrive through Python shims selected by env vars
(`python/tests/bench_matrix/svcomp_mod.py`, `SVCOMP_BENCH`). Make it an
adapter now (names from the C variables, and its precondition conjoined with
`--pre`), or leave the shim? *Recommended: leave it.*

**G. Petri nets (PNML).** Same shape (`petri_mod.py`, `PETRI_NET`). In
scope? *Recommended: no.*

## Design (from the phase-5 planner, before the answers)

- **Registry** `inputs/__init__.py`, shaped like `infer_route/`: one module
  per format exporting `ADAPTER`, plus
  - `SourceText(format, path, text, fence, suffix)` -- the input as read;
  - `Names(ctrl, extl)` -- source names in positional order;
  - `Problem(module, source, names)` -- what an adapter returns;
  - `InputAdapter(name, summary, suffixes, load, options)` with `load` a
    lazy `"pkg.mod:fn"`, so `--help` imports no parser;
  - `ADAPTERS`, `adapter_by_name`, `adapter_for(path)`.
- **CLI**: `--format {python,smv}`, else by suffix. `-d` becomes an option of
  the python adapter (same `Opt`/"only meaningful with" machinery the route
  rows use; move `Opt` to a leaf module both can import).
  `Settings.input_format`, `Settings.input_opts`.
- **Names** (IN-3): `Names.to_positional(src, flag)` is an SMT-LIB lexer that
  rewrites bare source names to `s{i}`/`e{i}`, looks up `|quoted|` names
  always, passes positional names through, and refuses a name that clashes
  with a positional name of another index. Applied in `main._claim`; the
  claim, artifacts and Lean stay positional. Print a note only when the text
  changed. Positional naming is fixed by `smt_module.ModuleSMT` (`s{i}` =
  `X(ctrl[i])`, `e{i}` = `X(extl[i])`, `el{i}` = `extl[i]`) and repeated in
  `smt_prompt.CegarPromptEnv`, `smt_prompt._pyeval_namespace`,
  `smt_to_lean.build_var_map` and `cert.render_certificate`.
- **IN-4**: `create_project(source: SourceText)` writes `dbg/<name><suffix>`
  instead of always `dbg/system.py`. Keep the `Wrote system.py` line for
  Python input: 11 golden transcripts carry it.
- **IN-5**: fence the Lean encoding in the LLM prompts as ```` ```lean ````
  (`magic/ai.py`, `smt_prompt.py`). Add `InferRoute.reads_source` and
  `InferInput.source`/`names`, gated like `reads`.
- **Adapters**: python (the loader moved out of `project.py`, `-d`, the gym
  `Env` wrap, `Named`); smv (`parse_smv(strict=True)`, below). SV-COMP and
  Petri stay shims unless F/G say otherwise.

## Commits

0. **IN-6 first**: `parse_smv(strict=True)` refuses what it would change the
   meaning of -- a non-assignment `TRANS`/`INIT`/`INVAR`
   (`_promote_constraints` drops them), `JUSTICE`/`FAIRNESS` (parsed and
   ignored), and per answer A the unconstrained VARs (or lowers them to
   inputs). Lower `integer` per A'. Add an SMV fixture that has a Lean form
   (`mod10.smv`), since none of `python/tests/smv_fixtures/*.smv` does.
1. Prompt fences (IN-5, cosmetic). The transcripts do not record prompt
   text, so they stay identical, and no test pins the fence today: add one
   with the change.
2. The Python loader becomes the python adapter (transcripts identical).
3. IN-4: `create_project(source=SourceText)`.
4. CLI: `--format`, suffix detection, `-d` scoped to python, docstring.
5. Names, with a byte-identical `Named` fixture test.
6. SMV adapter.
7. (Optional, per E) `reads_source` plumbing.
8. (Optional, per B) SMV specs supply the claim.

Callers of the loader to update in commit 2: `main._load`,
`python/tests/test_pipeline_golden.py`, `python/tests/bench_matrix/test_rows.py`,
`python/tests/bench_matrix/compare_certs.py`, `python/tests/lean/fbk/run_fbk.py`.

## How to check a commit

- **Transcripts** (the main check): `python/scripts/route_transcripts.py`
  (its docstring has the details). Record at the commit before a change and
  after it, then `diff -r`; every difference must be one the commit means.
  42 cases over all eight routes (LLM replies are scripted, no key needed),
  including fbk-proveit and Vampire runs when those tools are present; 6
  minutes on an idle machine, 40 on a busy one. Run it while no agent is
  editing the tree and nothing else loads the machine. `vampire_twovars` is
  timing-sensitive even so: its two templates share one 120 s deadline,
  and when the first one's calls use all of it the second is never asked,
  so its transcript can lose the `differences` lines between two runs of
  the same commit. Re-run that case alone before reading it as a change.
- Keep transcript directories out of the session scratchpad: it lives under
  `/private/tmp`, which macOS empties of files untouched for three days, so a
  baseline recorded there does not survive a pause. Re-record the "before"
  from the parent commit instead of trusting an old one.
- **Golden test**: `python/tests/test_pipeline_golden.py` pins whole-run
  output for several routes and the number of cvc5 encodings per run.
- **Per commit**: the targeted test files, `cd python && uv run pytest -m
  'not slow' -q <files>`; Vampire tests need `VAMPIRE=$HOME/Downloads/vampire`.
- **End of phase**: the fast suite once,
  `cd python && VAMPIRE=$HOME/Downloads/vampire uv run pytest -m 'not slow' -q`
  (about 1300 tests, 3 minutes). `slow` marks the lake/mathlib builds and two
  `m_twovars` smt-linear searches (about 135 s each, see the comment beside
  them); `just test-lean` runs everything marked `slow`.
- **External tools** on this machine: Vampire at `~/Downloads/vampire`; the
  `lean-ltl-certifying` checkout at `~/zeroth/proof-prototyping/`, ic3ia at
  `~/zeroth/fbk/ic3ia/build`, MathSAT's Python bindings at
  `~/zeroth/fbk/mathsat/python` (on `PYTHONPATH` for fbk-proveit runs).

## Working rules for this checkout

- Other Claude sessions share the checkout. Stage explicit paths only; never
  `git stash`, `git reset`, `git add -A`/`.`, `commit -a`, or `git checkout
  --` on a file not verified to hold only your changes. Commit with
  `git add <paths> && git commit --only <paths> -F <msg>`.
- One commit per fix; message `area: what is now true` plus a body.
- `python/tests/bench_matrix/render.py` belongs to another session (it
  imports `zrth.lean.infer_route.ROUTES` and `zrth.lean.cli._SEED_FLAGS`).
  Do not edit or stage it, and keep those names working.
- Do not edit `python/pyproject.toml`: any write makes every `uv run`
  rebuild the extension, and the isolated rebuild picks maturin 1.15, which
  fails (`libshm.dylib`). Recovery: `cd python && uv sync --inexact
  --no-build-isolation-package zrth`. Pinning `maturin==1.9.0` in
  `[build-system]` would fix it for good; that is the user's decision.
- Delegate mechanical commits to Sonnet agents (`model: "sonnet"`) with a
  written spec; keep design for the main session. Agents that touch the same
  files run one at a time, and transcripts are recorded only while none is
  editing.
- Comments state invariants, not history; rationale goes in commit messages.
