"""Every way verith can be handed a certificate, as one row each.

``--infer <name>`` selects a row.  The row says what the route needs of the
property, what it does with predicates supplied alongside it, what it reads
and writes of the generated project, and what it hands back.  The CLI
(:mod:`zrth.lean.cli`) reads the rows; it knows nothing about any particular
route, which is what makes adding one an edit in this file alone.

Two things are deliberate about the shape.

**The check is generic and the reason is not.**  Whether a route accepts a
safety property is a set membership; *why* the ``ai`` route does not is prose
about ``rule_globally``, and why ``nuterm`` discards a precondition is prose
about Houdini.  So the row carries both: the set the CLI tests, and the
sentence it prints when the test fails.

**The project is generated before inference runs, and that is the point.**
Some routes read the emitted Lean (the LLM ones prompt on the functional
encoding), and generating it twice is a second chance to generate it
differently.  Some routes write more of the project than the predicates --
``fbk-proveit`` writes an NA model, installs a certificate and emits a
bridge.  So a route is handed a :class:`ProjectHandle` rather than a path,
and ``reads``/``owns`` on its row are the whole of its access: what a run
wrote is what the row says it may have.

What a route cannot do after generation is change the *lakefile* -- it is
written while the project is created.  That is why the row declares its
lakefile needs as data (``lakefile``) and resolves what those needs name at
parse time (``resolve``), rather than discovering it later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .artifacts import ARTIFACTS_DIR, Artifact, ArtifactStore
from .cert import CertificateData
from .common import Refused

if TYPE_CHECKING:
    from zrth import Module

    from .common import LeanContext


# ══════════════════════════════════════════════════════════════════════════
# What a route is given
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ProjectHandle:
    """The generated project, as much of it as one route declared.

    A route is given this instead of a directory: ``reads`` and ``owns`` on
    its row are its whole access to the project, so "modifications are
    confined to ``Data.lean`` and ``Certificate.lean``" is enforced for the
    routes that say so, and a route that needs more of the project says
    which more -- by name, in its row, where the project builder can also
    read it.
    """

    dir: Path
    name: str
    module: "Module"
    ctx: "LeanContext"          # built by `create_project`; not built again
    reads: frozenset[str]       # ENCODINGS suffixes this route declared
    owns: tuple[str, ...]       # project-relative paths this route declared
    resumes: frozenset[str]     # artifact roles this route declared
    # `artifacts/` belongs to no route and to every route -- one run's
    # leftovers are the next run's starting point, and the run that wrote one
    # is usually not the run that uses it. So it is not an `owns` entry, it is
    # a field, and `path()` refuses it below so that the only way in is the
    # store's own typed writes.
    artifacts: ArtifactStore = None

    # --- reading --------------------------------------------------------

    def encoding(self, suffix: str = "") -> str:
        """The Lean of one emitted encoding.  ``""`` is the functional one.

        That is the file the LLM routes prompt on, and asking for it by the
        encoding it is removes the layout arithmetic -- ``System/System.lean``
        is the functional row's path only because `SYSTEM_LAYOUT` says so.
        """
        from .project import ENCODINGS, SYSTEM_LAYOUT

        if suffix not in {e.suffix for e in ENCODINGS}:
            raise Refused(f"there is no {suffix!r} encoding to read")
        if suffix not in self.reads:
            raise Refused(
                f"this route did not declare that it reads the {suffix!r} "
                f"encoding (its row says: {sorted(self.reads)})"
            )
        layout = replace(SYSTEM_LAYOUT, directory=self.dir / "System")
        return layout.path(suffix).read_text()

    def resume(
        self,
        role: str,
        *,
        languages: "tuple[str, ...] | None" = None,
        statuses: "tuple[str, ...] | None" = None,
    ) -> tuple[Artifact, ...]:
        """What a previous run left of `role`, newest first.

        Declared access, like `encoding()`: the row's `reads_artifacts` is
        what a route may resume from. The filtering is the store's -- it is
        the half that knows what a stale artifact is, and it reports what it
        dropped, which a filter applied out here would do silently.
        """
        if role not in self.resumes:
            raise Refused(
                f"this route did not declare that it resumes from {role!r} "
                f"artifacts (its row says: {sorted(self.resumes)})"
            )
        return self.artifacts.usable(role, languages=languages, statuses=statuses)

    # --- the two ordinary writes ----------------------------------------
    #
    # Available to every route, and the only writes most of them want: after
    # inference the predicates are new, so the definitions (`Data.lean`) and
    # the tactics generated from their shape (`Certificate.lean`) are both
    # restated.  Going through here rather than through two module-level
    # functions in the caller keeps that pairing in one place.

    def write_data(self, cert_data: CertificateData) -> Path:
        from .project import write_data_lean

        return write_data_lean(
            self.dir, self.name, self.module, cert_data, ctx=self.ctx
        )

    def write_certificate(self, cert_data: CertificateData) -> Path:
        from .project import write_certificate_lean

        return write_certificate_lean(
            self.dir, self.name, self.module, cert_data, ctx=self.ctx
        )

    def write_predicates(self, cert_data: CertificateData) -> None:
        """Both of the above, which is what an inferring route wants."""
        self.write_data(cert_data)
        self.write_certificate(cert_data)

    # --- writing more than that -----------------------------------------

    def path(self, rel: str) -> Path:
        """A file this route declared it owns, with its parent made.

        Refuses anything undeclared.  The point is not to police a route --
        it could import `pathlib` -- but to make the declaration the truth:
        `create_project` can then clean up what no route in *this* run owns,
        instead of hardcoding one route's filenames to unlink (`project.py`
        does that for `Certificate/Equivalence.lean` today).
        """
        if rel == ARTIFACTS_DIR or rel.startswith(ARTIFACTS_DIR + "/"):
            raise Refused(
                f"{ARTIFACTS_DIR}/ is written through `project.artifacts`, not "
                f"`project.path`: an artifact with no role, status or `why` is "
                f"a file the next run can do nothing with but read"
            )
        if not self._owned(rel):
            raise Refused(
                f"{rel} is not among this route's declared paths "
                f"({', '.join(self.owns) or 'none'})"
            )
        out = self.dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def _owned(self, rel: str) -> bool:
        return any(
            rel == o or rel.startswith(o.rstrip("/") + "/") for o in self.owns
        )


@dataclass(frozen=True)
class InferInput:
    """Everything a route is run with.

    One object rather than a constructor signature per route: `TA2Magic`'s
    `__init__(source)` named the Lean text as the input, which two of the
    three routes do not want -- `TA2MagicLearn("", module)` is how it is
    called -- and none of them could ask for anything else.
    """

    project: ProjectHandle
    module: "Module"
    cert_data: CertificateData
    opts: dict                          # this route's own options, by dest
    config: object | None = None        # what `resolve` returned
    prechecked: object | None = None    # what `precheck` returned
    model: str | None = None            # meaningful iff the row uses an LLM
    base_url: str | None = None
    log: Callable = print


# ══════════════════════════════════════════════════════════════════════════
# What a route hands back
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class InferResult:
    """The certificate a route inferred, in whichever languages it holds it.

    A route that renders its own Lean hands *both* spellings over. The SMT is
    what `--pre-check` restates the obligations from -- nothing parses the
    Lean back -- and the Lean is what the project is generated from. Handing
    over the SMT alone would make the pipeline render what the route already
    rendered, and rendering means encoding the module into cvc5 again: one
    more encoding per run, which is the cost the rest of this package spends
    comments avoiding.

    All four empty is the answer of a route that installed the certificate
    itself; its row says `returns="installed"`.
    """

    inv_smt: str | None = None
    ranking_smt: str | None = None
    inv_lean: str | None = None
    ranking_lean: str | None = None

    @property
    def empty(self) -> bool:
        return not any(
            (self.inv_smt, self.ranking_smt, self.inv_lean, self.ranking_lean)
        )


# ══════════════════════════════════════════════════════════════════════════
# The row
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Opt:
    """One command-line option that belongs to a route, not to verith.

    Given without its route selected it is an error rather than a no-op --
    which is what `--ic3ia`, `--fbk-simplify` and `--fbk-equiv` each say by
    hand today, in a block that grows a line per option per route.
    """

    flags: tuple[str, ...]
    kwargs: dict                # forwarded to `add_argument`
    required: bool = False      # the route cannot run without it

    @property
    def dest(self) -> str:
        d = self.kwargs.get("dest")
        return d or self.flags[-1].lstrip("-").replace("-", "_")


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
    # The `CertificateData` fields this route will *use*: "inv", "ranking",
    # "pre".  Anything outside the set would be discarded, so passing it is
    # refused with `seeds_refusal` rather than silently dropped.
    seeds: frozenset[str]
    seeds_refusal: str = ""

    # --- the project ----------------------------------------------------
    reads: tuple[str, ...] = ()     # ENCODINGS suffixes; "" is the functional one
    owns: tuple[str, ...] = ()      # project-relative paths it may write

    # --- `artifacts/`, the workspace across runs ------------------------
    # Roles (`artifacts.ROLES`) this route can start from, so that a second
    # `uv run verith` in the same `-o` continues rather than restarts: take
    # the invariant a previous run found and strengthen it, or take it as
    # given and infer only the ranking function. The pipeline preselects by
    # role and drops what is stale; the route filters by `status`, because
    # whether a `too_weak` invariant is usable depends on what the route
    # would do with it.
    reads_artifacts: frozenset[str] = frozenset()
    # `config -> {template var: value}`, merged into the lakefile render.
    # Declared rather than passed because the lakefile is written while the
    # project is created, before any route runs.
    lakefile: Callable | None = None

    # --- hooks, in the order they fire ----------------------------------
    # `resolve(opts) -> config` at parse time: a route whose lakefile names
    # a path has to reject a bad one before any of the project exists.
    resolve: Callable | None = None
    # `extra_check(settings, opts) -> str | None` at parse time: the rules
    # that are not a cell of this table.  `fbk-proveit` has one -- the
    # project name becomes a Lean module name.
    extra_check: Callable | None = None
    # `precheck(module, opts, config) -> object` once the module is loaded
    # and before it is generated: a module shape the route cannot express is
    # met here rather than as a traceback out of an encoder it never wanted.
    # What it computed reaches `run` as `prechecked`, so the module is not
    # encoded twice to answer one question.
    precheck: Callable | None = None
    run: Callable | None = None      # `(InferInput) -> InferResult`

    # --- what comes back ------------------------------------------------
    # "smt"       -- SMT-LIB predicates; `--pre-check` can restate the
    #                obligations from them
    # "lean"       -- Lean text only, and nothing parses it back, so the
    #                obligations cannot be restated
    # "installed"  -- the route wrote the certificate into the project itself
    returns: str = "smt"
    uses_llm: bool = False          # `--model` / `--base-url` are meaningful
    # Whether this route's own refusals already name the flag they are about.
    # `--infer nuterm` failing is reported as `error: --infer: ...`; the
    # proveit route's errors say `--fbk-proveit:`, `--ic3ia:` or `--safety:`
    # for themselves, and prefixing those again would name two flags.
    errors_self_named: bool = False

    options: tuple[Opt, ...] = ()
    aliases: tuple[Alias, ...] = ()


# ══════════════════════════════════════════════════════════════════════════
# The routes
#
# The `run` adapters below bridge each route's own constructor to
# `InferInput`.  They import lazily, as `main` does: a run that did not ask
# for a route should not need its packages present -- `anthropic`/`openai`
# for the LLM routes, `z3` and the learner for `nuterm`, a
# `lean-ltl-certifying` checkout for `fbk-proveit`.
# ══════════════════════════════════════════════════════════════════════════


def _run_ai(inp: InferInput) -> InferResult:
    from .magic_ai import TA2MagicAI

    magic = TA2MagicAI(
        inp.project.encoding(""), model=inp.model, base_url=inp.base_url
    )
    cd = magic.infer(inp.cert_data)
    # Lean only: it is parsed out of an LLM reply and never had an SMT term
    # behind it, which is what this route's `returns="lean"` declares.
    return InferResult(inv_lean=cd.inv, ranking_lean=cd.ranking)


# Statuses that make a previously-found invariant a *fixed* invariant for
# this route: `magic_cegar` takes `cd.inv` as given and infers only what is
# left, so what it is handed has to be inductive. `too_weak` is inductive
# and merely fails to imply the property, which `rule_buchi` never asked of
# it -- so it is usable for a Buchi certificate and is exactly the wrong
# thing for a safety one.
FIXABLE = ("proved", "too_weak")


def _resume_inv(inp: InferInput) -> "Artifact | None":
    """An invariant from a previous run to take as given, or `None`.

    An explicit `--invariant` wins: inheriting a predicate the user did not
    ask for, over one they did, is the kind of cross-run surprise that makes
    two identical command lines mean different things.
    """
    if inp.cert_data.inv:
        return None
    # An inductive invariant is what this route can take as *given*:
    # `magic_cegar` infers only what is left around a fixed `cd.inv`.
    for a in inp.project.resume("inv", languages=("smt",), statuses=FIXABLE):
        # The one thing the store cannot filter on: `too_weak` means
        # inductive but not implying the property, which is all `rule_buchi`
        # ever wanted of an invariant and exactly what `rule_globally` needs.
        if a.status == "too_weak" and inp.cert_data.is_safety:
            continue
        return a
    return None


def _run_ai_cegar(inp: InferInput) -> InferResult:
    from .magic_cegar import TA2MagicCEGAR

    resumed = _resume_inv(inp)
    if resumed is not None:
        inp.log(
            f".. resuming from {resumed.name} ({resumed.status}): taking its "
            f"invariant as given and inferring the ranking function"
        )
        inp.cert_data.inv = inp.project.artifacts.read(resumed)

    magic = TA2MagicCEGAR(
        inp.project.encoding(""),
        inp.module,
        model=inp.model,
        base_url=inp.base_url,
    )
    cd = magic.infer(inp.cert_data)
    # It renders the Lean from its own cvc5 context, where it is free, so
    # both spellings come back and the pipeline renders nothing.
    return InferResult(
        inv_smt=cd.inv_smt,
        ranking_smt=cd.ranking_smt,
        inv_lean=cd.inv,
        ranking_lean=cd.ranking,
    )


def _run_nuterm(inp: InferInput) -> InferResult:
    from .magic_learn import TA2MagicLearn

    magic = TA2MagicLearn("", inp.module, log=inp.log)
    cd = magic.infer(inp.cert_data)
    # `magic_learn._emit` already rendered the Lean, so it comes along rather
    # than being rendered a second time from the same SMT.
    return InferResult(
        inv_smt=cd.inv_smt,
        ranking_smt=cd.ranking_smt,
        inv_lean=cd.inv,
        ranking_lean=cd.ranking,
    )


def _resolve_fbk(opts: dict):
    from .fbk_proveit import resolve_project

    return resolve_project(opts["proveit_dir"])


def _check_fbk_project_name(settings, opts) -> "str | None":
    """`-p` becomes a Lean module name on this route, so it has to be one.

    Not a cell of the table: it is a rule about another flag's *value*.
    `import 1projNA` is "unexpected token; expected identifier", four steps
    later and in generated code.
    """
    import re

    from .project import na_module_name

    name = settings.project_name
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", name):
        return None
    return (
        f"--infer fbk-proveit needs a -p that is a Lean identifier: "
        f"`{na_module_name(name)}` is the module name the NA model and the "
        f"certificate are written under, and `{name}` does not start one"
    )


def _precheck_fbk(module, opts, config):
    from .fbk_proveit import ProveItError, check_module

    try:
        return check_module(module, opts["fbk_simplify"] == "cvc5")
    except ProveItError as e:
        # One exception type crosses the seam, so `main` catches what every
        # route raises. The message keeps naming `--fbk-proveit` itself,
        # which is what `errors_self_named` is about.
        raise Refused(str(e)) from e


def _lakefile_fbk(config) -> dict:
    """What this route needs of the lakefile: the checkout required, and the
    NA model declared as a lean_lib.  Without the two, the certificate this
    route installs is a file lake reaches and cannot elaborate."""
    return {"ltl_project": str(config)}


def _run_fbk(inp: InferInput) -> InferResult:
    from .fbk_proveit import ProveItError
    from .fbk_proveit import run as run_proveit

    try:
        run_proveit(
            ltl_project=inp.config,
            module=inp.module,
            project_dir=inp.project.dir,
            project_name=inp.project.name,
            property_smt=str(inp.cert_data.prp),
            ic3ia=inp.opts["ic3ia"],
            simplify=inp.opts["fbk_simplify"] == "cvc5",
            equivalence=inp.opts["fbk_equiv"] == "lean",
            bodies=inp.prechecked,
        )
    except ProveItError as e:
        raise Refused(str(e)) from e
    # The installed certificate carries ic3ia's invariant: no predicates come
    # back, which is what this route's `returns="installed"` declares.
    return InferResult()


ROUTES: tuple[InferRoute, ...] = (
    InferRoute(
        name="ai",
        summary=(
            "plain LLM self-check: one call proposes the invariant and the "
            "ranking function, a second is asked to find them wrong"
        ),
        kinds=frozenset({"buchi"}),
        kinds_refusal=(
            "--safety needs --infer ai-cegar or --infer nuterm: the `ai` route "
            "infers a ranking function `rule_globally` cannot take, and nothing "
            "checks that the invariant implies the property"
        ),
        seeds=frozenset(),
        seeds_refusal=(
            "the `ai` route prompts for a whole certificate and does not state "
            "the obligations, so it has nothing to hold a supplied predicate to"
        ),
        reads=("",),
        returns="lean",
        uses_llm=True,
        run=_run_ai,
    ),
    InferRoute(
        name="ai-cegar",
        summary=(
            "LLM + cvc5 counterexample-guided refinement: each candidate is "
            "put to the obligations and the model that refutes one is fed "
            "back (the default when --infer is passed without a value)"
        ),
        kinds=frozenset({"safety", "buchi"}),
        kinds_refusal="",
        seeds=frozenset({"inv", "ranking", "pre"}),
        reads=("",),
        # It already takes a fixed invariant from `--invariant` and infers
        # only what is left, so an inductive invariant a previous run left in
        # `artifacts/` is the same seed from the other source.
        reads_artifacts=frozenset({"inv"}),
        returns="smt",
        uses_llm=True,
        run=_run_ai_cegar,
    ),
    InferRoute(
        name="nuterm",
        summary=(
            "no LLM: a ranking function trained on rollouts of the module and "
            "returned only once a Farkas/CEGAR decision procedure has "
            "certified it, with the invariant inferred by Houdini -- scalar "
            "integer state only"
        ),
        kinds=frozenset({"safety", "buchi"}),
        kinds_refusal="",
        seeds=frozenset(),
        seeds_refusal=(
            "the certificate comes from the learner and its invariant holds at "
            "entry for every input, which is stronger than any precondition "
            "would make it"
        ),
        returns="smt",
        run=_run_nuterm,
    ),
    InferRoute(
        name="fbk-proveit",
        summary=(
            "certify through a `lean-ltl-certifying` checkout's proveit.py "
            "(lean2vmt -> ic3ia -> vmt2lean): the project is generated bare, "
            "an NA-encoded model is written to <project>/ProveIt/, and the "
            "certificate ic3ia's invariant makes is installed as "
            "<project>/Certificate/Certificate.lean"
        ),
        kinds=frozenset({"safety"}),
        kinds_refusal=(
            "--infer fbk-proveit is incompatible with --buchi: proveit.py "
            "proves `G PROPERTY` through ic3ia, which is a safety question. "
            "Pass --safety, or infer a Buchi certificate with --infer ai-cegar."
        ),
        seeds=frozenset(),
        seeds_refusal=(
            "the invariant comes from ic3ia and the project is generated bare"
        ),
        owns=(
            "ProveIt/",
            "Certificate/Certificate.lean",
            "Certificate/Equivalence.lean",
        ),
        lakefile=_lakefile_fbk,
        resolve=_resolve_fbk,
        extra_check=_check_fbk_project_name,
        precheck=_precheck_fbk,
        run=_run_fbk,
        aliases=(
            Alias(
                "--fbk-proveit",
                "proveit_dir",
                "Older spelling of `--infer fbk-proveit --proveit-dir DIR`: "
                "it selects the route and names the checkout in one flag.",
            ),
        ),
        returns="installed",
        errors_self_named=True,
        options=(
            Opt(
                ("--proveit-dir",),
                dict(
                    metavar="DIR",
                    help=(
                        "Path to a `lean-ltl-certifying` checkout. The "
                        "lakefile requires it, so the certificate this route "
                        "installs is buildable. `--fbk-proveit DIR` is the "
                        "older spelling and still selects this route."
                    ),
                ),
                required=True,
            ),
            Opt(
                ("--ic3ia",),
                dict(
                    metavar="PATH",
                    help=(
                        "Path to the ic3ia executable, forwarded to "
                        "proveit.py. When omitted proveit.py falls back to "
                        "$IC3IA, then to `ic3ia` on PATH."
                    ),
                ),
            ),
            Opt(
                ("--fbk-simplify",),
                dict(
                    default="cvc5",
                    choices=["cvc5", "none"],
                    help=(
                        "Whether cvc5's rewriter runs over the transition the "
                        "NA model carries (default: cvc5). `none` leaves each "
                        "state slot the shape the module's own terms give it "
                        "-- the same transition, spelled the way the module "
                        "spells it (`x - 1` rather than `-1 + x`), which is "
                        "what to pass when reading the generated Lean against "
                        "the source. The rewriter is what makes a constant "
                        "matrix tractable, though: a 32-wide affine layer is "
                        "97 KB of term unfolded and 1.9 KB folded."
                    ),
                ),
            ),
            Opt(
                ("--fbk-equiv",),
                dict(
                    default="lean",
                    choices=["lean", "none"],
                    help=(
                        "Whether to emit the proof that the NA model is the "
                        "module (default: lean). The certificate proveit.py "
                        "installs is about the *model*; this is what carries "
                        "it back to the module, and without it a "
                        "mistranslation anywhere in the SMT path would be a "
                        "machine-checked certificate about a different "
                        "system. `none` skips it -- it is the expensive part "
                        "on a wide state (73 s at 32 slots against ~5 s for "
                        "the rest of the route)."
                    ),
                ),
            ),
        ),
    ),
)


DEFAULT_ROUTE = "ai-cegar"


def index_routes(routes: "tuple[InferRoute, ...]") -> dict[str, InferRoute]:
    """The rows by name, refusing a duplicate.

    `ops.py` guards its op table the same way: a shadowed row would also
    make `--infer` offer the same name twice.
    """
    by_name: dict[str, InferRoute] = {}
    for r in routes:
        if r.name in by_name:
            raise RuntimeError(
                f"two route-table rows named {r.name!r} in "
                f"infer_route.ROUTES: the second would shadow the first and "
                f"`--infer` would offer the name twice"
            )
        by_name[r.name] = r
    return by_name


_BY_NAME = index_routes(ROUTES)


def route_by_name(name: str) -> InferRoute:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"no inference route named {name!r}; add a row to "
            f"infer_route.ROUTES (have: {', '.join(_BY_NAME)})"
        ) from None


def route_names() -> tuple[str, ...]:
    return tuple(_BY_NAME)


def all_route_options() -> tuple[tuple[InferRoute, Opt], ...]:
    """Every route-scoped option, with the route that owns it."""
    return tuple((r, o) for r in ROUTES for o in r.options)


def all_route_aliases() -> tuple[tuple[InferRoute, Alias], ...]:
    """Every deprecated spelling, with the route it selects."""
    return tuple((r, a) for r in ROUTES for a in r.aliases)
