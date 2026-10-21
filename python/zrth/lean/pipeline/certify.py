"""The certificate: the module checked for a Lean form and for the route's
shape (`lean_context`, `route_precheck`), the predicates recorded in
`artifacts/` and rendered as codegen takes them (`record_predicates`,
`render`), the route run (`run_route`), and the certificate rewritten
from its answer (`regenerate`). A refusal is reported under the flag it
is about (`refused`).
"""

from __future__ import annotations


from ..artifacts import ARTIFACTS_DIR, ArtifactStore, module_digest
from ..artifacts.record import note
from ..cli import Settings
from ..codegen.context import LeanContext
from ..codegen.terms import lean_gaps
from ..core.certificate import Certificate, Claim, LeanAnswer
from ..core.errors import Refused
from .. import inputs
from ..infer import InferInput, InferResult, InferRoute, ProjectHandle
from ..proof.facts import solver_hints
from ..proof.render import LeanCertificate, prepare
from ..smt.limits import SmtBudget
from ..smt.precheck import pre_check, predicate_terms


def refused(route: "InferRoute | None", error: Exception) -> SystemExit:
    """A refusal, under the flag it is about.

    Most routes are asked for with `--infer`, so that is what a failure of
    one is reported under. The proveit route's own errors already name
    `--fbk-proveit`, `--ic3ia` or `--safety`, and prefixing those again
    would name two flags for one problem. With no route there is no flag to
    name: the refusal is about the module.
    """
    if route is None or route.errors_self_named:
        return SystemExit(f"error: {error}")
    return SystemExit(f"error: --infer: {error}")


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


def record_predicates(store, claim: "Claim | None", cert: Certificate, *,
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


def workspace(settings: Settings, module,
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
        dir=settings.output_dir / settings.project_name / ARTIFACTS_DIR,
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


def hint_budget(settings: Settings) -> "SmtBudget | None":
    """What cvc5's solver hints are asked under: the run's budget with
    `--smt-tactics cvc5`, and `None`, no hints, without it."""
    return settings.budget if settings.smt_tactics == "cvc5" else None


def render(smt, claim: "Claim | None", cert: Certificate, *,
           hints: "SmtBudget | None", restated: bool = False,
           verbatim: "LeanAnswer | None" = None) -> "LeanCertificate | None":
    """The certificate as codegen takes it. `None` without a claim: a bare
    run asks cvc5 nothing. `smt` is the run's `smt.encoding.lazy` encoding.

    The predicates are parsed once, and that one parse is printed as Lean,
    read for its shape (`predicate_facts`) and, given a `hints` budget, put
    to `solver_hints` under it (`proof.render.prepare` says how the two
    meet). `verbatim` is an answer that is Lean already.
    """
    if claim is None:
        return None
    q = predicate_terms(smt, claim, cert)
    h = None
    if hints is not None:
        restating = ", restated over the inferred predicates" if restated else ""
        print(
            f".. SMT-informed tactics (cvc5){restating}: "
            f"<={hints.per_call_ms} ms per query, "
            f"<={hints.phase_ms} ms total"
        )
        h = solver_hints(q, hints)
    lean = prepare(q, h, verbatim)
    if h is not None:
        reasons = [ln.removeprefix("-- cvc5: ")
                   for ln in lean.plan.hints.why().splitlines()]
        for reason in reasons or ["nothing cvc5 can add to the plan"]:
            print(f"   {reason}")
    return lean


def note_refusal(handle: ProjectHandle, route, e: Refused) -> None:
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

    A `setup` refusal is not about the module at all -- a package or a key
    is missing -- so it is left unrecorded rather than filed against a run
    that never started.
    """
    store = handle.artifacts
    if store is None or store.notes or e.setup:
        return
    searched = e.searched
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


def run_route(
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
    except Refused as e:
        note_refusal(handle, route, e)
        raise refused(route, e) from e

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


def route_precheck(settings: Settings, module, claim: "Claim | None", smt):
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
    except Refused as e:
        raise refused(route, e) from e


def lean_context(settings: Settings, module) -> LeanContext:
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
        raise refused(settings.route, Refused(
            "this module has no Lean form: "
            + "; also ".join(gaps)
        ))
    return ctx


def regenerate(settings: Settings, smt, handle: ProjectHandle,
               claim: Claim, result: InferResult) -> LeanCertificate:
    """The project's certificate, rewritten from the route's answer.

    The certificate is that answer alone, under the claim the run was given,
    and its tactic plan is restated over it. Both files are rewritten --
    `Data.lean` holds the definitions and `Certificate.lean`'s tactics are
    generated from their shape -- through the handle, so a route that
    writes more of the project and this pipeline go through one writer.
    """
    cert, lean = result.cert, result.lean
    store = handle.artifacts
    if settings.pre_check == "cvc5":
        pre_check(smt, claim, lean or cert, settings.budget,
                  artifacts=store)
    record_predicates(store, claim, cert, inferred=True)
    try:
        lean_cert = render(smt, claim, cert,
                           hints=hint_budget(settings) if lean is None else None,
                           restated=True, verbatim=lean)
    except Refused as e:
        _note_unrenderable(store, settings.route, e)
        raise refused(settings.route, e) from e
    handle.write_predicates(lean_cert)
    return lean_cert
