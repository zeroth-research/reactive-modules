from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from zrth.lean.native import _product_type
from zrth.lean.certificate import Certificate, Claim, LeanAnswer
from zrth.lean.errors import Refused
from zrth.lean.context import LeanContext
from zrth.lean.lean_names import MAT_SIMP, rm_def
from zrth.lean.tactics import plan_for
from zrth.lean.template_env import render

if TYPE_CHECKING:
    from zrth import Module

    from zrth.lean.smt_query import ModuleQueries, PredicateFacts, SolverHints

def generate_zeroth_hammer_lean() -> str:
    """Generate a standalone ZerothHammer.lean with only the zeroth_hammer tactic."""
    return render("project/ZerothHammer.lean.j2")


# ══════════════════════════════════════════════════════════════════════════
# The certificate as Lean
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LeanPredicates:
    """The certificate's definitions as Lean text, `fun s => ...` over the
    state and `fun e => ...` over the inputs; `None` where the template's
    placeholder stands. Built by `render_certificate` alone."""

    kind: str = "buchi"
    prp: str | None = None
    inv: str | None = None
    ranking: str | None = None
    init_pre: str | None = None
    update_pre: str | None = None

    @property
    def is_safety(self) -> bool:
        return self.kind == "safety"


@dataclass(frozen=True)
class PlanInputs:
    """What the tactic plan reads besides the printed predicates: their
    shape as cvc5 parsed it, and what cvc5 established about them."""

    facts: PredicateFacts | None = None
    hints: SolverHints | None = None
    # The Lean of the predicates taken as they were given (a `LeanAnswer`):
    # no term stands behind them, so the facts cannot speak for them and
    # the plan reads their text instead.
    verbatim: str = ""


@dataclass(frozen=True)
class LeanCertificate:
    """Everything codegen needs of a certificate."""

    defs: LeanPredicates = field(default_factory=LeanPredicates)
    plan: PlanInputs = field(default_factory=PlanInputs)


def predicate_terms(
    smt, claim: Claim, cert: Certificate = Certificate()
) -> "ModuleQueries":
    """`claim` and `cert` parsed against the module: what the Lean is printed
    from, and what `predicate_facts` and `solver_hints` read. `smt` is the
    module, its encoding or a `smt.run.lazy` factory for one.

    A predicate that does not parse is refused naming its flag; anything
    else is the module's, and refused as that.
    """
    from .smt.run import encoding_of

    try:
        return encoding_of(smt).queries(claim, cert)
    except Refused:
        raise
    except Exception as e:
        raise Refused(
            f"the certificate predicates cannot be read for this module: {e}"
        ) from e


def render_certificate(
    q: "ModuleQueries", *, share: bool = True,
    verbatim: LeanAnswer | None = None,
) -> LeanPredicates:
    """The one place certificate predicates are printed as Lean.

    `share=False` prints repeated subterms in full instead of `let`-binding
    them: a condition `--smt-tactics` settled is spliced into a tactic
    expanded, and a `have` outside the definition cannot name a `let`
    inside it. `verbatim` is an answer that is Lean already; its `inv` and
    `ranking` are taken as they are.
    """
    from .smt_to_lean import smt_to_lean, smt_to_lean_nat

    msmt = q.msmt

    def translate(term, mode: str) -> str | None:
        if term is None:
            return None
        if mode in ("property", "invariant"):
            return smt_to_lean(term, msmt.ctrl_next, param_name="s", share=share)
        if mode == "ranking":
            return smt_to_lean_nat(term, msmt.ctrl_next, param_name="s", share=share)
        # preconditions
        return smt_to_lean(
            term,
            state_wires=[],
            param_name="e",
            extra=[
                ("e", "e.2", msmt.extl_next),
                ("el", "e.1", msmt.extl_latched),
            ],
            share=share,
        )

    def named(flag: str, term, mode: str) -> str | None:
        """`translate`, with the flag the source came from on any refusal.

        The translator walks a cvc5 term and knows nothing of the command
        line; `--pre (= s0 0)` parses and then fails on a name preconditions
        do not bind, and "Unknown free variable `s0`" alone does not say
        which of five sources asked for it.
        """
        try:
            return translate(term, mode)
        except Refused as e:
            raise Refused(f"{flag}: {e}") from e

    prp_flag = "--safety" if q.kind == "safety" else "--buchi"
    prp = named(prp_flag, q.prp, "property")
    init_pre = named("--pre", q.init_pre, "pre")
    update_pre = named("--pre", q.update_pre, "pre")
    if verbatim is not None:
        inv, ranking = verbatim.inv, verbatim.ranking
    else:
        inv = named("--invariant", q.inv, "invariant")
        ranking = named("--ranking", q.ranking, "ranking")
    return LeanPredicates(
        kind=q.kind, prp=prp, inv=inv, ranking=ranking,
        init_pre=init_pre, update_pre=update_pre,
    )


def prepare(
    q: "ModuleQueries", hints: "SolverHints | None" = None,
    verbatim: LeanAnswer | None = None,
) -> LeanCertificate:
    """`q`'s certificate as codegen takes it: the predicates printed, the
    facts their terms give, `hints`, and the text of a `verbatim` answer.

    The one place the printing is chosen. A condition the hints settle is
    spliced into a tactic in expanded form, and a `have` outside a
    definition cannot name a `let` inside it, so with such hints every
    definition is printed unshared -- a condition can come from any of
    them.
    """
    from .smt_query import predicate_facts

    return LeanCertificate(
        render_certificate(q, share=not (hints and hints.determined),
                           verbatim=verbatim),
        PlanInputs(
            predicate_facts(q), hints,
            verbatim="\n".join(t for t in (verbatim.inv, verbatim.ranking) if t)
            if verbatim is not None else "",
        ),
    )


def lean_certificate(
    module: "Module", claim: Claim, cert: Certificate = Certificate(),
    *, share: bool = True,
) -> LeanCertificate:
    """`claim` and `cert` printed for codegen, with the facts their terms
    give and no solver hints."""
    from .smt_query import predicate_facts

    q = predicate_terms(module, claim, cert)
    return LeanCertificate(
        render_certificate(q, share=share), PlanInputs(facts=predicate_facts(q))
    )


def _cert_def_context(ctx: LeanContext, lean_cert: LeanCertificate) -> dict:
    """Build a Jinja2 context dict for the Data.lean.j2 template.

    Each predicate is an ``_expr`` (a Lean `fun` from `render_certificate`)
    or `None`, in which case the template uses the placeholder value.
    """
    d = lean_cert.defs
    extl_native = (f"({_product_type(ctx.extl_latched)}) × "
                   f"({_product_type(ctx.extl_next)})")
    return dict(
        extl_type=extl_native,
        ctrl_type=_product_type(ctx.ctrl_next),
        noncomp="noncomputable " if ctx.uses_real else "",
        is_safety=d.is_safety,
        init_pre_expr=d.init_pre,
        update_pre_expr=d.update_pre,
        inv_expr=d.inv,
        p_expr=d.prp,
        ranking_expr=None if d.is_safety else d.ranking,
    )


def _predicate_text(ctx: LeanContext, lean_cert: LeanCertificate) -> str:
    """The Lean text of every certificate predicate, for shape detection."""
    c = _cert_def_context(ctx, lean_cert)
    return "\n".join(str(v) for k, v in c.items() if v and k.endswith("_expr"))


def generate_data_lean(
    ctx: LeanContext,
    lean_cert: LeanCertificate | None = None,
) -> str:
    """Generate Certificate/Data.lean: init_pre, update_pre, inv, P, ranking.

    Imports only ``Core.Basic``.  ``Certificate.lean`` imports this file so that
    only the data file needs regeneration when the certificate data changes
    (e.g. after --infer).

    It sits under ``Certificate/`` rather than ``System/`` because that is what
    it is about: ``System/`` is the module, which is the same whatever is being
    proved about it, while these five definitions are the certificate's own and
    change with every ``--infer``.  The two files a route rewrites are then the
    two files in one directory.
    """
    return render(
        "project/Certificate/Data.lean.j2",
        **_cert_def_context(ctx, lean_cert or LeanCertificate()),
    )


def generate_certificate_lean(
    ctx: LeanContext,
    lean_cert: LeanCertificate | None = None,
    *,
    module_inline: str | None = None,
) -> str:
    """Generate Certificate.lean.

    Args:
        module_inline: When set, inline this Lean source (the ``init`` /
            ``update`` functions) for standalone certs instead of importing
            ``System.System``.  Should be the output of
            ``ModuleToLean4.to_lean_functional()``.
    """
    lean_cert = lean_cert or LeanCertificate()
    defs = lean_cert.defs

    extl_latched = ctx.extl_latched
    extl_next = ctx.extl_next
    ctrl_next = ctx.ctrl_next

    extl_native = f"({_product_type(extl_latched)}) × ({_product_type(extl_next)})"
    ctrl_native = _product_type(ctrl_next)

    const_names = ctx.constants.names()
    const_list = ", ".join(const_names) if const_names else ""


    # Inline definitions when used standalone (no separate Certificate/Data.lean).
    inline_defs = (
        render("project/Certificate/_defs.lean.j2",
               **_cert_def_context(ctx, lean_cert)).strip("\n")
        if module_inline is not None else None
    )
    # A Buchi certificate always defines `ranking` (as `sorry` when nothing
    # supplied one); a safety certificate defines none, and naming it in a
    # simp set would be an unknown identifier rather than an unused one.
    has_ranking = not defs.is_safety

    all_defs = "RM, init, update, inv, init_pre, update_pre, P"
    if has_ranking:
        all_defs += ", ranking"
    if const_list:
        all_defs += f", {const_list}"

    plan = plan_for(
        ctx,
        _predicate_text(ctx, lean_cert),
        facts=lean_cert.plan.facts,
        hints=lean_cert.plan.hints,
        verbatim=lean_cert.plan.verbatim,
    )

    return render(
        "project/Certificate/Certificate.lean.j2",
        module_inline=module_inline,
        inline_defs=inline_defs,
        noncomp="noncomputable " if ctx.uses_real else "",
        rm_def=rm_def(extl_native, ctrl_native, noncomputable=ctx.uses_real),
        all_defs=all_defs,
        mat_simp=", ".join(MAT_SIMP),
        is_safety=defs.is_safety,
        plan=plan,
    )
