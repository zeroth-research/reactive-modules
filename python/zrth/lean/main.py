"""CLI entry point for generating a Lean4 project from a reactive module.

Entry point: ``uv run verith``

Quick start
-----------
A *module file* is a Python file that defines a ``module()`` function (or
class) returning a :class:`zrth.Module`.  Optionally it may also define plain
``init()`` / ``update()`` functions so that the AI inference (``--infer``) can
read the high-level logic.

Generate a bare Lean project (all certificate fields left as ``sorry``)::

    uv run verith mymodule.py -o out/ -p MyProject

Pass the property so the certificate knows what to prove.  Which flag it
goes under is the choice of proof rule, and so of what the certificate
contains::

    # `G (F x == 0)` -- holds infinitely often: invariant + ranking function
    uv run verith mymodule.py --buchi "x == 0" -o out/ -p MyProject

    # `G (x <= 100)` -- holds in every reachable state: invariant alone
    uv run verith mymodule.py --safety "x <= 100" -o out/ -p MyProject

Ask the AI to infer the certificate automatically (the ranking function too,
for a Buchi property)::

    uv run verith mymodule.py --buchi "x == 0" --infer -o out/ -p MyProject

Or infer it without an LLM at all: ``--infer nuterm`` trains a ranking function
on rollouts of the module and hands it over only once a Farkas/CEGAR decision
procedure has certified it, with the invariant inferred by Houdini.  It needs
no API key, and it reads modules whose state is scalar integers::

    uv run verith mymodule.py --buchi "(= s0 0)" --infer nuterm -o out/ -p MyProject

Two more routes need no LLM and search a *shape* rather than proposing
candidates, so that failing to find one is an answer with content:
``--infer sygus`` synthesises the invariant of a ``--safety`` property with
cvc5's SyGuS invariant track (congruences included -- ``x`` even -- which no
lattice of signs and pairwise facts can state), and ``--infer smt-linear``
asks for a linear ranking function or a conjunction of linear inequalities as
one quantified query.  A refuted query proves that nothing of that shape
works, and both leave what they found and what they ruled out in the
project's ``artifacts/``, where the next run picks it up::

    uv run verith mymodule.py --safety "(not (= s0 1))" --infer sygus -o out/ -p MyProject
    uv run verith mymodule.py --buchi "(= s0 0)" --infer smt-linear -o out/ -p MyProject

Use a local LLM via Ollama instead of Claude::

    uv run verith mymodule.py --buchi "x == 0" --infer \\
        --model qwen3-coder --base-url http://localhost:11434/v1 \\
        -o out/ -p MyProject

AI inference requirements
-------------------------
* **Claude (default)**: set ``ANTHROPIC_API_KEY`` and install
  ``pip install zrth[ai]``.
* **Local LLM (Ollama, vLLM, …)**: install ``pip install zrth[ai-local]``
  and provide ``--base-url`` pointing to an OpenAI-compatible endpoint.
* **Hosted OpenAI-compatible provider (OpenRouter, …)**: as above, plus set
  ``OPENROUTER_API_KEY`` (or ``OPENAI_API_KEY``), e.g.
  ``--base-url https://openrouter.ai/api/v1 --model anthropic/claude-haiku-4.5``.

Module file format
------------------
The file must expose a callable named ``module`` (override with ``-d``) that
returns a :class:`zrth.Module`::

    # mymodule.py
    from zrth import Module, Int, Var, X
    from zrth.analyzer import convert_method

    def init():
        return 0

    def update(old_x):
        x = old_x + 1
        if x == 10:
            return 0
        return x

    def module() -> Module:
        state = Var(Int([1, 1]))
        init_terms   = convert_method(init,   {},               [X(state)])
        update_terms = convert_method(update, {"old_x": state}, [X(state)])
        return Module.sequential([state], init_terms, update_terms)
"""

from pathlib import Path

from .artifacts import ARTIFACTS_DIR, ArtifactStore, module_digest
from .cert import CertificateData, generate_zeroth_hammer_lean, smt_predicates_to_lean
from .cli import Settings, parse_args
from .common import Refused
from .infer_route import InferInput, ProjectHandle
from .project import (
    ENCODINGS,
    LakeBuildError,
    Layout,
    build_certificate,
    create_project,
    generate_standalone_cert_lean,
    lean_context_for,
    load_module_from_file,
    write_encoding,
)
from .smt_query import ModuleQueries, pre_check, predicate_facts, solver_hints
from .translate import ModuleToLean4


def _build_cert(args, project_dir: Path, cert_data) -> None:
    """Honour `--build-cert`, once the certificate is written.

    `cert_data` is what the project's certificate was generated from, or
    `None` when the certificate came from elsewhere (`--infer fbk-proveit`
    installs ic3ia's). The predicates are re-checked here rather than only
    at parse time because a route can return without one: an LLM that
    answers nothing usable leaves `inv` or `ranking` unset, and building
    that would report a proof of `sorry`.
    """
    if not args.build_cert:
        return
    if cert_data is not None:
        # A safety certificate has no ranking function to be missing.
        required = [("property", cert_data.prp), ("invariant", cert_data.inv)]
        if not cert_data.is_safety:
            required.append(("ranking", cert_data.ranking))
        missing = [name for name, value in required if not value]
        if missing:
            raise SystemExit(
                f"error: --build-cert: the certificate has no "
                f"{', '.join(missing)}, so every obligation it carries is "
                "`sorry`. With --infer this means inference produced none."
            )
    print(f".. Building the certificate in {project_dir}")
    try:
        build_certificate(project_dir)
    except LakeBuildError as e:
        raise SystemExit(f"error: --build-cert: {e}") from e


def _refused(settings: Settings, error: Exception) -> SystemExit:
    """A route's refusal, under the flag it is about.

    Most routes are asked for with `--infer`, so that is what a failure of
    one is reported under. The proveit route's own errors already name
    `--fbk-proveit`, `--ic3ia` or `--safety`, and prefixing those again
    would name two flags for one problem.
    """
    if settings.route is not None and settings.route.errors_self_named:
        return SystemExit(f"error: {error}")
    return SystemExit(f"error: --infer: {error}")


def _hammer_only(settings: Settings) -> None:
    """`--hammer-file`: write ZerothHammer.lean and return. No module needed."""
    out = Path(settings.hammer_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_zeroth_hammer_lean())
    except OSError as e:
        raise SystemExit(f"error: --hammer-file: {e}") from e
    print(f"Wrote {out}")


def _standalone(settings: Settings, module, cert_data) -> None:
    """`--cert-file`: a self-contained certificate and its sibling encodings.

    No project, so no route: the predicates are the ones supplied, which is
    why every route that reads or writes a project is refused alongside it.
    """
    out = Path(settings.cert_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_standalone_cert_lean(module, cert_data))
    except (Refused, OSError) as e:
        raise SystemExit(f"error: --cert-file: {e}") from e
    print(f"Wrote standalone certificate: {out}")
    # The same encoding table the project route walks, so these files and
    # the ones in a generated project carry the same headers and the same
    # imports. The functional encoding is already inlined in the standalone
    # certificate above, which is the base they import.
    m2l = ModuleToLean4(module)
    layout = Layout(base=out.stem, flat=True, directory=out.parent)
    for enc in ENCODINGS:
        if not enc.cert_file:
            continue
        print(f"Wrote {enc.title.lower()}: {write_encoding(enc, m2l, layout, out.stem)}")


# `inv` / `ranking` hold SMT-LIB when they were supplied and Lean once a route
# has rendered them, and `*_smt` is where that route keeps the source. So each
# is looked for in both places, source first.
_PREDICATES = (
    ("property", ("prp",),
     "The property this run was asked to prove, as it was given to "
     "`--safety` / `--buchi`."),
    ("inv", ("inv_smt", "inv"),
     "The invariant the certificate is generated from."),
    ("ranking", ("ranking_smt", "ranking"),
     "The ranking function the certificate is generated from."),
)


def _record_predicates(store, settings: Settings, cert_data, *,
                       inferred: bool = False) -> None:
    """Leave the certificate's predicates in `artifacts/`, as SMT-LIB.

    They are what every obligation beside them is stated against, so a reader
    comparing `system.smt2` with a refuted obligation has the third side of it
    here rather than having to dig it out of `Certificate/Data.lean`, where it is
    Lean and no longer the text any solver was given.
    """
    if store is None or cert_data is None:
        return
    for role, fields, what in _PREDICATES:
        src = next(
            (v for v in (getattr(cert_data, f, None) for f in fields)
             if isinstance(v, str) and v.strip()),
            None,
        )
        if src is None:
            continue
        store.encoded(
            role, role, src.strip(), language="smt-src",
            what=(what + " Inferred by this run's route." if inferred else what),
        )


def _workspace(settings: Settings, module, project_dir: Path) -> ArtifactStore:
    """The project's `artifacts/`, as this run sees it.

    The project is generated before inference runs and `artifacts/` outlives
    the run that wrote it, so a sequence of `uv run verith` in one `-o`
    continues rather than restarts. The three fields that are not the
    directory are what stop a stale artifact being trusted: an invariant
    found for another property, another proof rule or another module is not
    about this run, and `usable()` says so rather than silently omitting it.
    """
    store = ArtifactStore(
        dir=project_dir / ARTIFACTS_DIR,
        module_digest=module_digest(module),
        kind=settings.kind,
        prp=settings.prp or "",
        producer=settings.route.name if settings.route else "",
        enabled=settings.artifacts == "use",
        log=print,
    )
    if settings.artifacts == "reset":
        gone = store.reset()
        print(f".. artifacts: cleared {gone} file(s) from {store.dir}")
    return store


def _replan(settings: Settings, module, inferred, project_cert_data) -> None:
    """Restate the tactic plan's inputs over the predicates that were inferred.

    Both of them are computed before the route runs, which is before there is
    an invariant to compute them *from*. `predicate_facts` then reads the
    shape of the property alone, and `solver_hints` answers nothing at all --
    every question it asks is asked under the invariant, so a `None` one is
    answered by returning immediately. A `--infer` certificate was therefore
    generated from a plan about predicates it does not carry, and in two ways
    that matter:

    * `cert_facts` and `cert_thin` were `skip` on every inferred certificate,
      however branchy the inferred predicates turned out to be -- the run
      printed "nothing cvc5 can add to the plan" before the route that would
      produce the thing to add it about had run;
    * `features_for` takes `nonlinear` from the facts *over* what it read in
      the printed Lean rather than merging the two, so an inferred nonlinear
      predicate lost `nlinarith` and `positivity` from the closer list
      entirely. Measured on a `(* s0 s0)` ranking, that is the whole
      difference between a plan that can close `hrank` and one that cannot.

    Needs the SMT source, since nothing parses the printed Lean back into a
    term: that is what `inv_smt` / `ranking_smt` keep, and what
    `_record_predicates` has just written to `artifacts/` beside the
    obligations stated against it. A route that answers in Lean alone leaves
    nothing to read and the plan stays exactly as it was.
    """
    q = ModuleQueries.build(module, inferred)
    if q is None:
        return

    facts = predicate_facts(q)
    if facts is not None:
        project_cert_data.facts = facts

    if settings.smt_tactics != "cvc5":
        return
    print(
        f".. SMT-informed tactics (cvc5), restated over the inferred "
        f"predicates: <={settings.budget.per_call_ms} ms per query, "
        f"<={settings.budget.phase_ms} ms total"
    )
    hints = solver_hints(q, settings.budget, log=print)
    project_cert_data.hints = hints
    if not hints.determined:
        return

    # The same care the pre-inference path takes, for the same reason: a
    # settled condition is spliced into a tactic in expanded form, so the
    # definition it has to match cannot have its repeated subterms shared
    # behind a `let` the `have` outside cannot name. Every predicate is
    # reprinted, not only the two that were inferred -- a condition can come
    # from the property, which was printed shared when the hints were empty.
    try:
        lean = smt_predicates_to_lean(inferred, module, share=False)
    except Refused:
        # Unshared printing is an optimisation on top of a plan that is
        # already better than the one this function replaced. Losing it is
        # not worth losing the run: the `have`s stay, each one `try`ed, and
        # the ones that cannot match cost nothing.
        return
    for field in ("prp", "inv", "ranking", "init_pre", "update_pre"):
        setattr(project_cert_data, field, getattr(lean, field))


def _infer(
    settings: Settings, module, handle: ProjectHandle, cert_data, prechecked
) -> "CertificateData | None":
    """Run the selected route against the generated project.

    Returns the certificate it inferred, or `None` when the route installed
    one itself -- there are then no predicates for the project to be
    regenerated from.

    Everything route-specific is a field of the row: what of the project it
    may read and write, which artifact roles it resumes from, and which
    languages its answer comes back in.
    """
    route = settings.route
    try:
        result = route.run(
            InferInput(
                project=handle,
                module=module,
                cert_data=cert_data,
                opts=settings.route_opts,
                config=settings.route_config,
                prechecked=prechecked,
                budget=settings.budget,
                model=settings.model,
                base_url=settings.base_url,
                log=print,
            )
        )
    except (Refused, ImportError) as e:
        # A missing package and a missing key are both about how the route
        # was asked for, not about the module.
        raise _refused(settings, e) from e

    # The row *declares* what comes back and the result carries it, so the
    # two are checked against each other once rather than each reader
    # trusting whichever is nearer to hand.
    if (route.returns == "installed") != result.empty:
        raise SystemExit(
            f"error: --infer {route.name}: the route's row declares "
            f"returns={route.returns!r} but it handed back "
            f"{'no predicates' if result.empty else 'predicates'}"
        )
    if route.returns == "installed":
        return None

    # cvc5's own input back, for `--pre-check` to restate the obligations
    # from: nothing parses the Lean below into a term.
    cert_data.inv_smt, cert_data.ranking_smt = result.inv_smt, result.ranking_smt

    if result.inv_lean or result.ranking_lean:
        # The route rendered its own Lean -- `magic.cegar` from the cvc5
        # context it already had, `magic.learn` on its way out. Rendering it
        # again here would encode the module into cvc5 one more time for a
        # string we were handed.
        cert_data.inv, cert_data.ranking = result.inv_lean, result.ranking_lean
    else:
        try:
            lean = smt_predicates_to_lean(
                CertificateData(
                    prp=settings.prp,
                    kind=settings.kind,
                    inv=result.inv_smt,
                    ranking=result.ranking_smt,
                ),
                module,
            )
        except Refused as e:
            raise _refused(settings, e) from e
        cert_data.inv, cert_data.ranking = lean.inv, lean.ranking
    return cert_data


def main():
    settings = parse_args()

    if settings.hammer_file:
        _hammer_only(settings)
        return

    # The loader refuses a path that is not there, a file that is not a
    # module and a `-d` the file does not define. All three are the user's
    # input, so they are reported rather than raised.
    try:
        module = load_module_from_file(
            settings.module_file, module_def=settings.module_def
        )
    except (OSError, AttributeError, RuntimeError) as e:
        raise SystemExit(f"error: {e}") from e
    print(module)

    # The route's own shape check, before a line of Lean is written. Left
    # until the route runs, a module its encoding cannot express is met
    # first by `create_project` -- which fails about the functional
    # encoding, in a traceback, for a route that was never going to use it.
    #
    # Making the check *is* encoding the module, so what it answered with is
    # kept and handed to the route below rather than encoded again.
    prechecked = None
    if settings.route is not None and settings.route.precheck is not None:
        try:
            prechecked = settings.route.precheck(
                module, settings.route_opts, settings.route_config
            )
        except (Refused, ImportError) as e:
            raise _refused(settings, e) from e

    cert_data: CertificateData | None = None
    if settings.prp or settings.pre or settings.invariant or settings.ranking:
        cert_data = CertificateData(prp=settings.prp, kind=settings.kind)
        if settings.pre:
            cert_data.init_pre = settings.pre
            cert_data.update_pre = settings.pre
        if settings.invariant:
            cert_data.inv = settings.invariant
        if settings.ranking:
            cert_data.ranking = settings.ranking

    # The workspace belongs to the project, not to the route: `--artifacts
    # reset` is what to pass when a bad artifact is being inherited by every
    # run, and a run that selects no route is one of those runs. Built before
    # the pre-check rather than after `create_project`, because the pre-check
    # is the run's first encoding and what it encodes is worth keeping.
    store = (
        None if settings.cert_file or settings.hammer_file
        else _workspace(settings, module,
                        settings.output_dir / settings.project_name)
    )
    _record_predicates(store, settings, cert_data)

    # With a route the certificate the project gets is the inferred one, so
    # the check has to wait for it -- run at this point, there is no
    # invariant yet to refute.
    if settings.pre_check == "cvc5" and settings.route is None:
        if cert_data is None:
            print(".. SMT pre-check: nothing to check (no certificate data)")
        else:
            pre_check(module, cert_data, settings.budget, artifacts=store)

    # The hints have to come first: a settled branch condition is spliced
    # into a tactic in expanded form, so the definition it has to match must
    # be printed unshared.
    hints = None
    if settings.smt_tactics == "cvc5" and cert_data is not None:
        print(
            f".. SMT-informed tactics (cvc5): <={settings.budget.per_call_ms} ms per "
            f"query, <={settings.budget.phase_ms} ms total"
        )
        hints = solver_hints(
            ModuleQueries.build(module, cert_data), settings.budget, log=print
        )

    # Translate SMT-LIB predicates to Lean expression strings for codegen.
    # `cert_data` keeps the original SMT source so a route can parse it with
    # its own cvc5 context.
    project_cert_data = cert_data
    if cert_data is not None:
        try:
            project_cert_data = smt_predicates_to_lean(
                cert_data, module, share=not (hints and hints.determined)
            )
        except Refused as e:
            raise SystemExit(f"error: {e}") from e
        project_cert_data.hints = hints

    if settings.cert_file:
        _standalone(settings, module, project_cert_data)
        return

    print(".. Generating lean code")
    # Built once and threaded: the discovery pass interns every init/update
    # term, and the files rewritten after inference need the same one.
    ctx = lean_context_for(module, project_cert_data)
    try:
        project_dir = create_project(
            output_dir=settings.output_dir,
            module=module,
            project_name=settings.project_name,
            executable=settings.executable,
            cert_data=project_cert_data,
            module_file=settings.module_file,
            lakefile_vars=(
                settings.route.lakefile(settings.route_config)
                if settings.route is not None and settings.route.lakefile
                else None
            ),
            ctx=ctx,
        )
    except (Refused, OSError) as e:
        raise SystemExit(f"error: {e}") from e

    print(".. Doing TA2Magic")
    if settings.route is not None:
        handle = ProjectHandle(
            dir=project_dir,
            name=settings.project_name,
            module=module,
            ctx=ctx,
            reads=frozenset(settings.route.reads),
            owns=settings.route.owns,
            resumes=frozenset(settings.route.reads_artifacts),
            artifacts=store,
        )
        inferred = _infer(settings, module, handle, cert_data, prechecked)

        if inferred is None:
            # The route installed its own certificate, so there is nothing
            # of `project_cert_data` to be complete about and nothing to
            # regenerate from it.
            _build_cert(settings, project_dir, cert_data=None)
            print(f"\nProject ready at: {project_dir}")
            return

        if settings.pre_check == "cvc5":
            pre_check(module, inferred, settings.budget, artifacts=store)
        _record_predicates(store, settings, inferred, inferred=True)

        # Merge the inferred predicates into what the project is generated
        # from. The kind comes along: it decides which proof rule the
        # rewritten files state, and `ranking` is `None` for a safety
        # certificate by construction.
        project_cert_data.inv = inferred.inv
        project_cert_data.ranking = inferred.ranking

        # And the tactic plan's two inputs with them: until this point there
        # were no inferred predicates for either to be about.
        _replan(settings, module, inferred, project_cert_data)

        # After inference the predicates are new, so both files are
        # rewritten -- `Data.lean` holds the definitions and
        # `Certificate.lean`'s tactics are generated from their shape. The
        # handle is what does it, so a route that writes more of the project
        # than the predicates and this pipeline go through one writer.
        handle.write_predicates(project_cert_data)

    _build_cert(settings, project_dir, cert_data=project_cert_data)

    print(f"\nProject ready at: {project_dir}")


if __name__ == "__main__":
    main()
