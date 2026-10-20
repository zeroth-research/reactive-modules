"""The row a route is declared by (`InferRoute`), and an older flag that
selects one (`Alias`)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.options import Opt


@dataclass(frozen=True)
class Alias:
    """An older spelling that selects a route and fills one of its options.

    `--fbk-proveit DIR` is `--infer fbk-proveit --proveit-dir DIR`. Declared
    by the row rather than known to the CLI, so the deprecated flag goes away
    with the route and a second one is another entry rather than another
    hand-written branch.
    """

    flag: str                   # e.g. "--fbk-proveit"
    fills: str                  # the `dest` of the `Opt` it supplies
    help: str


@dataclass(frozen=True)
class InferRoute:
    """One way of being handed a certificate."""

    name: str                       # the `--infer` value
    summary: str                    # one line, assembled into `--infer`'s help

    # --- the property ---------------------------------------------------
    kinds: frozenset[str]           # "safety", "buchi", or both
    kinds_refusal: str              # why the kind it lacks is not its question

    # --- predicates supplied alongside ----------------------------------
    # The given predicates this route will *use*: "inv", "ranking",
    # "pre".  Anything outside the set would be discarded, so passing it is
    # refused with `seeds_refusal` rather than silently dropped.
    seeds: frozenset[str]
    seeds_refusal: str = ""

    # --- the project ----------------------------------------------------
    reads: tuple[str, ...] = ()     # ENCODINGS suffixes; "" is the functional one
    owns: tuple[str, ...] = ()      # project-relative paths it may write

    # --- the input ------------------------------------------------------
    # Whether the route is handed the input the module was read from and
    # the names claims may use (`InferInput.source` / `.names`). Declared
    # like `reads`: a route that did not ask is handed neither, so what a
    # route reads of the run is what its row says.
    reads_source: bool = False

    # --- `artifacts/`, the workspace across runs ------------------------
    # Roles (`artifacts.ROLES`) this route can start from, so that a second
    # `uv run verith` in the same `-o` continues rather than restarts: take
    # the invariant a previous run found and strengthen it, or take it as
    # given and infer only the ranking function. The pipeline preselects by
    # role and drops what is stale; the route filters by `status`, because
    # whether a `too_weak` invariant is usable depends on what the route
    # would do with it.
    reads_artifacts: frozenset[str] = frozenset()
    # `(config, project_name) -> LakefileParts`: the libraries and requires
    # the lakefile gets for this route.  Declared rather than passed because
    # the lakefile is written while the project is created, before any route
    # runs.
    lakefile: Callable | None = None

    # --- hooks, in the order they fire ----------------------------------
    # `resolve(opts) -> config` at parse time: a route whose lakefile names
    # a path has to reject a bad one before any of the project exists.
    resolve: Callable | None = None
    # `extra_check(settings, opts) -> str | None` at parse time: the rules
    # that are not a cell of this table.  `fbk-proveit` has one -- the
    # project name becomes a Lean module name.
    extra_check: Callable | None = None
    # `precheck(module, opts, config, claim, smt) -> object` once the module
    # is loaded and before it is generated: a module shape the route cannot
    # express is met here rather than as a traceback out of an encoder it
    # never wanted.  What it computed reaches `run` as `prechecked`, so the
    # module is not encoded twice to answer one question.  `smt` is the run's
    # lazy `ModuleEncoding` (`InferInput.smt`), so a precheck that has to
    # encode does it on the encoding the rest of the run uses.
    #
    # It is handed the claim because whether a route can take a `--pre` is
    # not always a question about the flag alone: `fbk-proveit` accepts one
    # over an input its model keeps and refuses one over an input it does
    # not, and only the module's encoding knows which is which.
    precheck: Callable | None = None
    run: Callable | None = None      # `(InferInput) -> InferResult`

    # --- what comes back ------------------------------------------------
    # "smt"       -- SMT-LIB predicates; `--pre-check` can restate the
    #                obligations from them
    # "lean"       -- Lean text only, and nothing parses it back, so the
    #                obligations cannot be restated
    # "files"      -- the certificate is a file the route produced, and there
    #                are no predicates, so `--pre-check` cannot restate the
    #                obligations
    returns: str = "smt"
    uses_llm: bool = False          # `--model` / `--base-url` are meaningful
    # Whether this route's own refusals already name the flag they are about.
    # `--infer nuterm` failing is reported as `error: --infer: ...`; the
    # proveit route's errors say `--fbk-proveit:`, `--ic3ia:` or `--safety:`
    # for themselves, and prefixing those again would name two flags.
    errors_self_named: bool = False

    options: tuple[Opt, ...] = ()
    aliases: tuple[Alias, ...] = ()
