"""CLI entry point for generating a Lean4 project from a reactive module.

Entry point: ``uv run verith``

Quick start
-----------
A *module file* is a Python file that defines a ``module()`` function (or
class) returning a :class:`zrth.Module` (or a ``zrth.lean.inputs.Named``
one, for claims in its own names), or another format ``--format`` lists.
The LLM routes prompt on the module's generated Lean; a route that declares
``reads_source`` is also handed the input file itself.

Generate a bare Lean project (all certificate fields left as ``sorry``)::

    uv run verith mymodule.py -o out/ -p MyProject

Pass the property so the certificate knows what to prove.  Which flag it
goes under is the choice of proof rule, and so of what the certificate
contains::

    # `G (F x == 0)` -- holds infinitely often: invariant + ranking function
    uv run verith mymodule.py --buchi "(= (X x) 0)" -o out/ -p MyProject

    # `G (x <= 100)` -- holds in every reachable state: invariant alone
    uv run verith mymodule.py --safety "(<= (X x) 100)" -o out/ -p MyProject

A claim is written in the module's own terms: ``(X x)`` is the value of ``x``
after the round.  Names like ``x`` come from a module file that returns
``zrth.lean.inputs.Named(module, {"x": var})``, and ``(X @intf.0)`` names
interface variable 0 by position, which works for any module.

Ask the AI to infer the certificate automatically (the ranking function too,
for a Buchi property)::

    uv run verith mymodule.py --buchi "(= (X x) 0)" --infer -o out/ -p MyProject

Or infer it without an LLM at all: ``--infer nuterm`` trains a ranking function
on rollouts of the module and hands it over only once a Farkas/CEGAR decision
procedure has certified it, with the invariant inferred by Houdini.  It needs
no API key, and it reads modules whose state is scalar integers::

    uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer nuterm -o out/ -p MyProject

Two more routes need no LLM and search a *shape* rather than proposing
candidates, so that failing to find one is an answer with content:
``--infer sygus`` synthesises the invariant of a ``--safety`` property with
cvc5's SyGuS invariant track (congruences included -- ``x`` even -- which no
lattice of signs and pairwise facts can state), and ``--infer smt-linear``
asks for a linear ranking function or a conjunction of linear inequalities as
one quantified query.  A refuted query proves that nothing of that shape
works, and both leave what they found and what they ruled out in the
project's ``artifacts/``, where the next run picks it up::

    uv run verith mymodule.py --safety "(not (= (X @intf.0) 1))" --infer sygus -o out/ -p MyProject
    uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer smt-linear -o out/ -p MyProject

Use a local LLM via Ollama instead of Claude::

    uv run verith mymodule.py --buchi "(= (X x) 0)" --infer \\
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

import shlex
import sys
from pathlib import Path

from .artifacts import ARTIFACTS_DIR, ArtifactStore, module_digest
from .proof.render import (
    LeanCertificate,
    generate_zeroth_hammer_lean,
    prepare,
)
from .core.certificate import Certificate, Claim, LeanAnswer
from .cli import Settings, parse_args
from .core.errors import Refused
from .codegen.context import LeanContext
from .codegen.terms import lean_gaps
from . import inputs
from .infer_route import (
    InferInput, InferResult, ProjectHandle, owned_files,
)
from .project import (
    ENCODINGS,
    LakeBuildError,
    Layout,
    build_certificate,
    create_project,
    generate_standalone_cert_lean,
    write_encoding,
)
from .smt.record import note
from .smt.encoding import lazy
from .smt.precheck import pre_check, predicate_terms
from .proof.facts import solver_hints
from .codegen.encodings import ModuleToLean4


def _build_cert(settings: Settings, project_dir: Path,
                lean_cert: "LeanCertificate | None") -> None:
    """Honour `--build-cert`, once the certificate is written.

    `lean_cert` is what the project's certificate was generated from, or
    `None` when the certificate came from elsewhere (`--infer fbk-proveit`
    installs ic3ia's). The predicates are re-checked here rather than only
    at parse time because a route can return without one: an LLM that
    answers nothing usable leaves `inv` or `ranking` unset, and building
    that would report a proof of `sorry`.
    """
    if not settings.build_cert:
        return
    if lean_cert is not None:
        defs = lean_cert.defs
        # A safety certificate has no ranking function to be missing.
        required = [("property", defs.prp), ("invariant", defs.inv)]
        if not defs.is_safety:
            required.append(("ranking", defs.ranking))
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
    """A refusal, under the flag it is about.

    Most routes are asked for with `--infer`, so that is what a failure of
    one is reported under. The proveit route's own errors already name
    `--fbk-proveit`, `--ic3ia` or `--safety`, and prefixing those again
    would name two flags for one problem. With no route there is no flag to
    name: the refusal is about the module.
    """
    if settings.route is None or settings.route.errors_self_named:
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


def _standalone(settings: Settings, module, lean_cert) -> None:
    """`--cert-file`: a self-contained certificate and its sibling encodings.

    No project, so no route: the predicates are the ones supplied, which is
    why every route that reads or writes a project is refused alongside it.
    """
    out = Path(settings.cert_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_standalone_cert_lean(module, lean_cert))
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


# Each predicate recorded: its role, the `Certificate` field it is read from
# (`None` for the claim's property), and what it is.
_PREDICATES = (
    ("property", None,
     "The property this run was asked to prove, as it was given to "
     "`--safety` / `--buchi`."),
    ("inv", "inv",
     "The invariant the certificate is generated from."),
    ("ranking", "ranking",
     "The ranking function the certificate is generated from."),
)


def _record_predicates(store, claim: "Claim | None", cert: Certificate, *,
                       inferred: bool = False) -> None:
    """Leave the certificate's predicates in `artifacts/`, as SMT-LIB.

    They are what every obligation beside them is stated against, so a reader
    comparing `system.smt2` with a refuted obligation has the third side of it
    here rather than having to dig it out of `Certificate/Data.lean`, where it is
    Lean and no longer the text any solver was given. An answer that is Lean
    only (`--infer ai`) has no SMT-LIB to record.
    """
    if store is None or claim is None:
        return
    for role, field, what in _PREDICATES:
        src = claim.prp if field is None else getattr(cert, field)
        if not (src and src.strip()):
            continue
        # A route infers the certificate, never the property it was asked about.
        if inferred and role != "property":
            what += " Inferred by this run's route."
        store.encoded(role, role, src.strip(), language="smt-src", what=what)


def _workspace(settings: Settings, module, project_dir: Path,
               claim: "Claim | None") -> ArtifactStore:
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
        prp=(claim.prp if claim else None) or "",
        producer=settings.route.name if settings.route else "",
        enabled=settings.artifacts == "use",
        log=print,
    )
    if settings.artifacts == "reset":
        gone = store.reset()
        print(f".. artifacts: cleared {gone} file(s) from {store.dir}")
    return store


def _render(settings: Settings, smt, claim: "Claim | None",
            cert: Certificate, *, hints: bool, restated: bool = False,
            verbatim: "LeanAnswer | None" = None) -> "LeanCertificate | None":
    """The certificate as codegen takes it. `None` without a claim: a bare
    run asks cvc5 nothing. `smt` is the run's `smt.encoding.lazy` encoding.

    The predicates are parsed once, and that one parse is printed as Lean,
    read for its shape (`predicate_facts`) and, with `hints` under
    `--smt-tactics cvc5`, put to `solver_hints` (`proof.render.prepare` says how
    the two meet). `verbatim` is an answer that is Lean already.
    """
    if claim is None:
        return None
    q = predicate_terms(smt, claim, cert)
    h = None
    if hints and settings.smt_tactics == "cvc5":
        restating = ", restated over the inferred predicates" if restated else ""
        print(
            f".. SMT-informed tactics (cvc5){restating}: "
            f"<={settings.budget.per_call_ms} ms per query, "
            f"<={settings.budget.phase_ms} ms total"
        )
        h = solver_hints(q, settings.budget)
    lean = prepare(q, h, verbatim)
    if h is not None:
        reasons = [ln.removeprefix("-- cvc5: ")
                   for ln in lean.plan.hints.why().splitlines()]
        for reason in reasons or ["nothing cvc5 can add to the plan"]:
            print(f"   {reason}")
    return lean


def _note_refusal(handle: ProjectHandle, route, e: Exception) -> None:
    """Leave *why* a route gave up in `artifacts/`, if it did not say so itself.

    `NO-CERT` is three measurements wearing one name -- a space proved
    empty, a budget that ran out, a shape the route never looked at -- and
    the prose that distinguishes them is a dozen wordings per route, which
    is why reading it back is guesswork. The status is not: a route that
    searched a bounded shape and refuted it writes `no_solution`, one whose
    budget ran out writes `unknown`, and everything else is a decline.

    Only when the route wrote nothing. `smt-linear`, `sygus`, `houdini` and
    `vampire` each leave a note of their own with more in it than this, and
    a second one saying less would be worse than none.

    An `ImportError` is not about the module at all -- a package or a key is
    missing -- so it is left unrecorded rather than filed against a run that
    never started.
    """
    store = getattr(handle, "artifacts", None)
    if store is None or store.notes or isinstance(e, ImportError):
        return
    searched = getattr(e, "searched", False)
    store.note(
        f"{e}\n",
        status="unknown" if searched else "declined",
        what=(f"A search by `--infer {route.name}` that did not finish."
              if searched else
              f"`--infer {route.name}` declined this module and property."),
        why=("The budget ran out first; nothing is known about the space."
             if searched else
             "Nothing was searched, so this says what the route does not "
             "take rather than anything about the module's difficulty."),
    )


def _note_unrenderable(store, route, e: Refused) -> None:
    """Leave in `artifacts/` that a route's answer has no Lean form.

    Not a refusal by the route: it answered, and what failed was printing
    the answer as Lean, after the route had returned. So no note of the
    route's own can say this, and without one the next run reads a found
    predicate in `artifacts/` and nothing about why no certificate came of
    it -- and an LLM route resuming from it proposes the same shape again.
    """
    note(
        store, f"{e}\n",
        status="unsupported",
        what=f"An answer by `--infer {route.name}` that has no Lean form.",
        why=("The route answered; what failed is writing that answer as "
             "Lean, so the same shape proposed again fails the same way."),
    )


def _infer(
    settings: Settings, module, handle: ProjectHandle, claim: Claim,
    given: Certificate, prechecked, smt=None,
    problem: "inputs.Problem | None" = None,
) -> "InferResult":
    """Run the selected route against the generated project.

    Returns what it handed back: predicates, or -- for a route whose row
    says `returns="files"` -- whole files, with no predicates for the
    project to be regenerated from.

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
                claim=claim,
                given=given,
                opts=settings.route_opts,
                config=settings.route_config,
                prechecked=prechecked,
                budget=settings.budget,
                model=settings.model,
                base_url=settings.base_url,
                log=print,
                smt=smt,
                source=(problem.source
                        if route.reads_source and problem else None),
                names=(problem.names
                       if route.reads_source and problem else None),
            )
        )
    except (Refused, ImportError) as e:
        # A missing package and a missing key are both about how the route
        # was asked for, not about the module.
        _note_refusal(handle, route, e)
        raise _refused(settings, e) from e

    # The row *declares* what comes back and the result carries it, so the
    # two are checked against each other once rather than each reader
    # trusting whichever is nearer to hand.
    agrees = {
        "smt": not result.cert.empty and result.lean is None,
        "lean": result.lean is not None,
        "files": bool(result.files) and result.cert.empty
                 and result.lean is None,
    }[route.returns]
    if not agrees:
        if result.cert.empty and result.lean is None:
            handed = ("files and no predicates" if result.files
                      else "no predicates")
        else:
            handed = "predicates"
            handed += " in Lean" if result.lean is not None else " in SMT-LIB"
            if result.files:
                handed += " and files"
        raise SystemExit(
            f"error: --infer {route.name}: the route's row declares "
            f"returns={route.returns!r} but it handed back {handed}"
        )
    for rel in result.files:
        if not handle.owns_path(rel):
            raise SystemExit(
                f"error: --infer {route.name}: the route handed back {rel}, "
                f"which its row does not own"
            )
    return result


def _load(settings: Settings) -> "inputs.Problem":
    """The module the run is about, and the input it was read from.

    The loader refuses a path that is not there, a file that is not a
    module and a `-d` the file does not define. All three are the user's
    input, so they are reported rather than raised.
    """
    try:
        return inputs.load(settings.module_file, settings.input_opts,
                           format=settings.input_format)
    except (OSError, AttributeError, RuntimeError, inputs.ClaimError,
            inputs.InputError) as e:
        raise SystemExit(f"error: {e}") from e


def _claim(settings: Settings,
           names: "inputs.Names") -> "tuple[Claim | None, Certificate]":
    """What the flags ask to be proved, and what of its certificate they
    supply. No claim when no predicate flag was passed.

    The flags are in the module's own terms (`inputs.names`); the claim
    is in the internal names every later stage reads, and each flag is
    printed in them once, since that is how the logs, the artifacts and
    any refusal will spell it."""
    if not (settings.prp or settings.pre or settings.invariant
            or settings.ranking):
        return None, Certificate()
    prp_flag = "--safety" if settings.is_safety else "--buchi"

    def internal(text: "str | None", flag: str) -> "str | None":
        if not text:
            return None
        try:
            spelled = inputs.to_internal(text, names, flag)
        except inputs.ClaimError as e:
            raise SystemExit(f"error: {e}") from e
        print(f".. {flag}, in internal names: {spelled}")
        return spelled

    prp = internal(settings.prp, prp_flag)
    pre = internal(settings.pre, "--pre")
    claim = Claim(prp=prp, kind=settings.kind, init_pre=pre, update_pre=pre)
    return claim, Certificate(internal(settings.invariant, "--invariant"),
                              internal(settings.ranking, "--ranking"))


def _route_precheck(settings: Settings, module, claim: "Claim | None", smt):
    """The route's own shape check, before a line of Lean is written; what
    it computed reaches the route as `InferInput.prechecked`.

    Left until the route runs, a module its encoding cannot express is met
    first by `create_project` -- which fails about the functional encoding,
    in a traceback, for a route that was never going to use it. The check
    reads the claim as well: `--pre` on `--infer fbk-proveit` is answerable
    only by the encoding, so it is answered where the encoding is made.
    """
    route = settings.route
    if route is None or route.precheck is None:
        return None
    try:
        return route.precheck(
            module, settings.route_opts, settings.route_config, claim, smt
        )
    except (Refused, ImportError) as e:
        raise _refused(settings, e) from e


def _lean_context(settings: Settings, module) -> LeanContext:
    """The module's `LeanContext`, refusing a module with no Lean form.

    After the route's own check, so a route with an opinion about this
    module states it in its own words. An operator with no Lean form is a
    fact about the module, said here once, before any of the project
    exists, rather than as a missing table cell in the middle of a file.

    Built once and threaded: its discovery pass interns every init/update
    term, and codegen and the rewrites after inference need the same one.
    """
    ctx = LeanContext(module)
    gaps = lean_gaps(ctx)
    if gaps:
        raise _refused(settings, Refused(
            "this module has no Lean form: "
            + "; also ".join(gaps)
        ))
    return ctx


def _generate(settings: Settings, module, ctx: LeanContext,
              lean_cert: "LeanCertificate | None",
              source: "inputs.SourceText") -> Path:
    """The Lean project, with the certificate as it stands before inference."""
    print(".. Generating lean code")
    try:
        return create_project(
            output_dir=settings.output_dir,
            module=module,
            project_name=settings.project_name,
            executable=settings.executable,
            lean_cert=lean_cert,
            source=source,
            lakefile=(
                settings.route.lakefile(
                    settings.route_config, settings.project_name)
                if settings.route is not None and settings.route.lakefile
                else None
            ),
            # `Certificate/` is globbed by the build, so a copy of a file a
            # route writes, left by an earlier run in this `-o`, would be
            # built against this run's module. Directories are left alone:
            # nothing globs them until a route declares a lean_lib for them.
            remove=owned_files(),
            ctx=ctx,
        )
    except (Refused, OSError) as e:
        raise SystemExit(f"error: {e}") from e


def _regenerate(settings: Settings, smt, handle: ProjectHandle, store,
                claim: Claim, result: InferResult) -> LeanCertificate:
    """The project's certificate, rewritten from the route's answer.

    The certificate is that answer alone, under the claim the run was given,
    and its tactic plan is restated over it. Both files are rewritten --
    `Data.lean` holds the definitions and `Certificate.lean`'s tactics are
    generated from their shape -- through the handle, so a route that
    writes more of the project and this pipeline go through one writer.
    """
    cert, lean = result.cert, result.lean
    if settings.pre_check == "cvc5":
        pre_check(smt, claim, lean or cert, settings.budget,
                  artifacts=store)
    _record_predicates(store, claim, cert, inferred=True)
    try:
        lean_cert = _render(settings, smt, claim, cert,
                            hints=lean is None, restated=True, verbatim=lean)
    except Refused as e:
        _note_unrenderable(store, settings.route, e)
        raise _refused(settings, e) from e
    handle.write_predicates(lean_cert)
    return lean_cert


def _show_certificate(cert: Certificate, names: "inputs.Names") -> None:
    """The route's answer in the input's names, as `--invariant` and
    `--ranking` take it: what one run found, the next can be given."""
    shown = [(flag, text) for flag, text in (("--invariant", cert.inv),
                                             ("--ranking", cert.ranking))
             if text]
    if not shown:
        return
    print(".. The certificate, in the input's names:")
    for flag, text in shown:
        print(f"   {flag} {shlex.quote(inputs.to_surface(text, names))}")


def _stated_claim(settings: Settings, problem: "inputs.Problem",
                  argv: list[str]) -> Settings:
    """The command line, with the property the input states when it gives
    none (an SMV INVARSPEC or LTLSPEC).

    The stated property is the claim exactly as if it had been passed as
    `--safety`/`--buchi`: the command line is parsed again with it added,
    so every rule about the flags applies to it unchanged. Given both, the
    run would prove one and ignore the other, so that is refused.
    """
    stated = problem.claims
    chosen = settings.input_opts.get("spec")
    listing = "; ".join(f"{i}: {c.where}" for i, c in enumerate(stated, 1))
    if settings.prp:
        if chosen is not None:
            raise SystemExit(
                "error: --spec chooses one of the input's properties, and "
                "--safety/--buchi already gives the claim"
            )
        if stated:
            raise SystemExit(
                f"error: {problem.source.path} states its own property "
                f"({listing}); pass no --safety/--buchi to prove it, or "
                "remove it to prove another"
            )
        return settings
    if not settings.claim_deferred:
        return settings
    if not stated:
        if chosen is not None:
            raise SystemExit("error: --spec: the input states no property")
        return parse_args(argv, defer_claim=False)
    if chosen is None:
        if len(stated) > 1:
            raise SystemExit(
                f"error: the input states {len(stated)} properties "
                f"({listing}); choose one with --spec N"
            )
        chosen = 1
    if not 1 <= chosen <= len(stated):
        raise SystemExit(
            f"error: --spec {chosen}: the input states {len(stated)} "
            f"propert{'y' if len(stated) == 1 else 'ies'} ({listing})"
        )
    claim = stated[chosen - 1]
    print(f".. the claim is the input's {claim.where} (--{claim.kind})")
    return parse_args(
        [*argv, f"--{claim.kind}", inputs.to_surface(claim.text, problem.names)],
        defer_claim=False,
    )


def main():
    argv = sys.argv[1:]
    settings = parse_args(argv)

    if settings.hammer_file:
        _hammer_only(settings)
        return

    problem = _load(settings)
    settings = _stated_claim(settings, problem, argv)
    module = problem.module
    print(module)
    # The run's one cvc5 encoding, built by whichever stage asks first.
    smt = lazy(module)
    claim, given = _claim(settings, problem.names)
    prechecked = _route_precheck(settings, module, claim, smt)
    ctx = _lean_context(settings, module)

    # The workspace belongs to the project, not to the route: `--artifacts
    # reset` is what to pass when a bad artifact is being inherited by every
    # run, and a run that selects no route is one of those runs. Built
    # before the pre-check, the run's first encoding, so what that encodes
    # is kept.
    store = (
        None if settings.cert_file
        else _workspace(settings, module,
                        settings.output_dir / settings.project_name, claim)
    )
    _record_predicates(store, claim, given)

    # With a route the certificate is the inferred one, and `_regenerate`
    # checks that instead.
    if settings.pre_check == "cvc5" and settings.route is None:
        if claim is None:
            print(".. SMT pre-check: nothing to check (no certificate data)")
        else:
            pre_check(smt, claim, given, settings.budget, artifacts=store)

    # With a route this rendering is a placeholder the answer replaces, and
    # no route reads `Certificate/`, so cvc5 is asked for hints only about
    # the certificate the project ends up with.
    try:
        lean_cert = _render(settings, smt, claim, given,
                            hints=settings.route is None)
    except Refused as e:
        raise SystemExit(f"error: {e}") from e

    if settings.cert_file:
        _standalone(settings, module, lean_cert)
        return

    project_dir = _generate(settings, module, ctx, lean_cert, problem.source)

    if settings.route is not None:
        print(f".. Inferring the certificate with --infer {settings.route.name}")
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
        result = _infer(settings, module, handle, claim, given, prechecked,
                        smt=smt, problem=problem)
        # `files`: the certificate is the route's own, so there is nothing
        # of ours to be complete about and nothing to regenerate from.
        lean_cert = (
            None if settings.route.returns == "files"
            else _regenerate(settings, smt, handle, store, claim, result)
        )
        handle.write_files(result.files)
        _show_certificate(result.cert, problem.names)

    _build_cert(settings, project_dir, lean_cert)

    print(f"\nProject ready at: {project_dir}")


if __name__ == "__main__":
    main()
