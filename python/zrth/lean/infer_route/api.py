"""The types a route row is written in: what a route is given, what it hands
back, and the row itself."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from ..artifacts import ARTIFACTS_DIR, Artifact, ArtifactStore
from ..certificate import Certificate, Claim, LeanAnswer
from ..errors import Refused
from ..options import Opt

if TYPE_CHECKING:
    from ..inputs import Names, SourceText
    from zrth import Module

    from ..proof.render import LeanCertificate
    from ..context import LeanContext
    from ..smt.encoding import ModuleEncoding


# ══════════════════════════════════════════════════════════════════════════
# What a route is given
# ══════════════════════════════════════════════════════════════════════════


# Statuses that make a previously-found invariant a *fixed* invariant for
# a route that holds it as given: it has to be inductive. `too_weak` is
# inductive and merely fails to imply the property, which `rule_buchi` never
# asked of it -- so it is usable for a Buchi certificate and is exactly the
# wrong thing for a safety one.
FIXABLE = ("proved", "too_weak")


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
        from ..project import ENCODINGS, SYSTEM_LAYOUT

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

    def write_data(self, lean_cert: "LeanCertificate") -> Path:
        from ..project import write_data_lean

        return write_data_lean(
            self.dir, self.name, self.module, lean_cert, ctx=self.ctx
        )

    def write_certificate(self, lean_cert: "LeanCertificate") -> Path:
        from ..project import write_certificate_lean

        return write_certificate_lean(
            self.dir, self.name, self.module, lean_cert, ctx=self.ctx
        )

    def write_predicates(self, lean_cert: "LeanCertificate") -> None:
        """Both of the above, which is what an inferring route wants."""
        self.write_data(lean_cert)
        self.write_certificate(lean_cert)

    # --- writing more than that -----------------------------------------

    def path(self, rel: str) -> Path:
        """A file this route declared it owns, with its parent made.

        Refuses anything undeclared.  The point is not to police a route --
        it could import `pathlib` -- but to make the declaration the truth:
        the pipeline can then clean up what no route in *this* run owns
        (`create_project` unlinks what every row owns before the project is
        written), instead of hardcoding one route's filenames.
        """
        if rel == ARTIFACTS_DIR or rel.startswith(ARTIFACTS_DIR + "/"):
            raise Refused(
                f"{ARTIFACTS_DIR}/ is written through `project.artifacts`, not "
                f"`project.path`: an artifact with no role, status or `why` is "
                f"a file the next run can do nothing with but read"
            )
        if not self.owns_path(rel):
            raise Refused(
                f"{rel} is not among this route's declared paths "
                f"({', '.join(self.owns) or 'none'})"
            )
        out = self.dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def owns_path(self, rel: str) -> bool:
        """Whether `rel` is one of the declared paths, or under a declared
        directory."""
        return any(
            rel == o or rel.startswith(o.rstrip("/") + "/") for o in self.owns
        )

    def write_files(self, files: Mapping[str, str]) -> None:
        """Write whole files a route handed back in `InferResult.files`, in
        the mapping's order, each through `path()`."""
        for rel, text in files.items():
            out = self.path(rel)
            out.write_text(text)
            print(f"Wrote {out}")


@dataclass(frozen=True)
class InferInput:
    """Everything a route is run with.

    One object rather than a constructor signature per route: each row's
    `run` takes from it what its engine needs -- the Lean text, the module,
    the budget, the artifact store -- and no engine is asked for what it
    does not use.
    """

    project: ProjectHandle
    module: "Module"
    claim: Claim
    # What the route starts from: `--invariant` / `--ranking`, or an
    # invariant `taking_inv` resumed from `artifacts/`.
    given: Certificate
    opts: dict                          # this route's own options, by dest
    config: object | None = None        # what `resolve` returned
    prechecked: object | None = None    # what `precheck` returned
    # This run's `--smt-timeout` / `--smt-budget`, for a route that asks cvc5
    # a question of its own. The obligation queries have always been on it
    # (`smt.limits.SmtBudget`); a route that *searches* has to be, for the
    # same reason and more so -- a synthesis with no limit hangs generation,
    # and `tlimit-per` is the option that actually stops one.
    budget: object | None = None
    model: str | None = None            # meaningful iff the row uses an LLM
    base_url: str | None = None
    log: Callable = print
    # The run's one cvc5 encoding of `module`, built on first call.
    smt: "Callable[[], ModuleEncoding] | None" = None
    # The input verith read and the names a claim may use, for a row that
    # declares `reads_source`; `None` for every other row.
    source: "SourceText | None" = None
    names: "Names | None" = None

    # --- resuming from `artifacts/` -------------------------------------

    def resumable_inv(self, *, strengthen: bool = False) -> "Artifact | None":
        """An invariant from a previous run to start from, or `None`.

        An explicit `--invariant` wins: inheriting a predicate the user did
        not ask for, over one they did, is the kind of cross-run surprise
        that makes two identical command lines mean different things.

        By default the invariant is one to take as *given*: `magic.cegis`
        infers only what is left around a fixed `given.inv`, so it has to be
        inductive and, on a safety claim, imply the property. `too_weak` is
        inductive and merely fails to imply it, which `rule_buchi` never
        asked of an invariant and exactly what `rule_globally` needs -- so it
        is skipped for a safety claim, the one thing the store cannot filter
        on. With `strengthen=True` the invariant is one to *add to*:
        `--infer smt-linear --safety` searches for the conjuncts that are
        missing, so an invariant that is inductive and does not yet imply the
        property is the best thing it can be handed.
        """
        if self.given.inv:
            return None
        for a in self.project.resume(
            "inv", languages=("smt",), statuses=FIXABLE
        ):
            if (a.status == "too_weak" and self.claim.is_safety
                    and not strengthen):
                continue
            return a
        return None

    def taking_inv(self, resumed: "Artifact | None", what: str) -> "InferInput":
        """This input with a resumed invariant as its given one, saying so."""
        if resumed is None:
            return self
        self.log(f".. resuming from {resumed.name} ({resumed.status}): {what}")
        inv = self.project.artifacts.read(resumed)
        return replace(self, given=replace(self.given, inv=inv))

    def ruled_out(self) -> tuple[str, ...]:
        """What earlier runs *proved* is not there, as prose for the prompt.

        Only `no_solution`: a note whose status is `unknown` records a search
        that ran out of budget, and telling a model that a shape is empty when
        all that happened is a timeout would put a falsehood in the prompt.
        The store has already dropped anything found against another module,
        property or proof rule, which is the other half of the same care.
        """
        notes = self.project.resume("note", languages=("md",),
                                    statuses=("no_solution",))
        out = []
        for a in notes:
            self.log(f".. resuming from {a.name}: a space an earlier run "
                     f"ruled out")
            out.append(self.project.artifacts.read(a))
        return tuple(out)


# ══════════════════════════════════════════════════════════════════════════
# What a route hands back
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class InferResult:
    """What a route answered: SMT-LIB in `cert`, or Lean in `lean` for the
    one row that answers in Lean (`returns="lean"`). A route whose answer is
    a certificate it produced whole hands back `files` instead and no
    predicates (`returns="files"`).

    `files` are whole files by project-relative path, which the pipeline
    writes; each has to be a path the row owns.
    """

    cert: Certificate = Certificate()
    lean: LeanAnswer | None = None
    files: Mapping[str, str] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.cert.empty and self.lean is None and not self.files


# ══════════════════════════════════════════════════════════════════════════
# The row
# ══════════════════════════════════════════════════════════════════════════


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

