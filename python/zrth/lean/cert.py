from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zrth.lean.native import _product_type, _translate_terms
from zrth.lean.common import LeanContext, _bind_wires
from zrth.lean.tactics import plan_for
from zrth.lean.template_env import render
from ..expr import Expr

if TYPE_CHECKING:
    from zrth import Module

def generate_zeroth_hammer_lean() -> str:
    """Generate a standalone ZerothHammer.lean with only the zeroth_hammer tactic."""
    return render("project/ZerothHammer.lean.j2")


@dataclass
class CertificateData:
    """Data needed to generate a Lean certificate."""

    prp: str | Expr | None = None
    inv: Expr | str | None = None

    # Which proof rule the certificate is built on, and so what `prp` means:
    #
    #   "buchi"  -- `G (F P)`, `rule_buchi`: an invariant *and* a ranking
    #              function that decreases off `P`. `--buchi`.
    #   "safety" -- `G P`, `rule_globally`: an invariant that implies `P`,
    #              and no ranking at all -- there is nothing to rank when
    #              the property never has to be re-reached. `--safety`.
    kind: str = "buchi"

    init_pre: Expr | str | None = None
    update_pre: Expr | str | None = None
    ranking: Expr | str | None = None

    # SMT-LIB (or Python-expression) source for `inv` / `ranking` when those
    # fields hold Lean text printed from a cvc5 term -- the `--infer ai-cegar`
    # route. The project needs the Lean; cvc5 needs its own input back, so
    # `--pre-check` restates the obligations from these.
    inv_smt: str | None = None
    ranking_smt: str | None = None

    # Shape of the predicates as cvc5 sees them, when they came from SMT-LIB
    # and cvc5 could parse them. `None` everywhere else, and the tactic plan
    # falls back to reading the rendered Lean.
    facts: "object | None" = None

    # `smt_query.SolverHints` when `--smt-tactics=cvc5` ran. `None`
    # otherwise, and the plan is exactly the one it would have emitted.
    hints: "object | None" = None

    @property
    def is_safety(self) -> bool:
        return self.kind == "safety"


def _cert_def_lines(
    ctx: LeanContext,
    cert_data: CertificateData,
) -> list[str]:
    """Emit the five certificate definition lines as Lean source strings.

    Used by the inline path of ``generate_certificate_lean`` (standalone certs).
    Delegates field parsing to ``_cert_def_context`` to avoid duplicating logic.
    """
    c = _cert_def_context(ctx, cert_data)
    extl_native = c["extl_type"]
    ctrl_native = c["ctrl_type"]
    lines: list[str] = []

    def _prop_def(name: str, ty: str, param: str, body: "str | None", expr: "str | None", default: str) -> None:
        if body is not None:
            lines.append(f"def {name} ({param} : {ty}) : Prop :=")
            lines.append(body)
        elif expr is not None:
            lines.append(f"def {name} : {ty} → Prop := {expr}")
        else:
            lines.append(f"def {name} ({param} : {ty}) : Prop := {default}")
        lines.append("")

    _prop_def("init_pre",   extl_native, "e", c["init_pre_body"],  c["init_pre_expr"],  "True")
    _prop_def("update_pre", extl_native, "e", c["update_pre_body"], c["update_pre_expr"], "True")
    _prop_def("inv",        ctrl_native, "s", c["inv_body"],        c["inv_expr"],        "True")
    _prop_def("P",          ctrl_native, "s", c["p_body"],          c["p_expr"],          "sorry")

    # DecidablePred P (noncomputable for Real-typed state: Real.decidableLE
    # and friends are classical, so the instance cannot be compiled)
    noncomp = c["noncomp"]
    if c["p_body"] is not None or c["p_expr"] is not None:
        lines.append(
            f"{noncomp}instance : DecidablePred P := fun s => by unfold P; first | infer_instance | dsimp; infer_instance"
        )
    else:
        lines.append(f"{noncomp}instance : DecidablePred P := sorry")
    lines.append("")

    # ranking (Nat, not Prop). Noncomputable for a Real state: `to_int`
    # emits `⌊·⌋`, which goes through the noncomputable `Real.instFloorRing`.
    # `rule_globally` takes no ranking, so a safety certificate has none to
    # define -- and defining it as `sorry` would put a `sorry` in a file
    # whose whole point is that it has none.
    if c["is_safety"]:
        return lines
    if c["ranking_body"] is not None:
        lines.append(f"{noncomp}def ranking (s : {ctrl_native}) : Nat :=")
        lines.append(c["ranking_body"])
    elif c["ranking_expr"] is not None:
        lines.append(f"{noncomp}def ranking : {ctrl_native} → Nat := {c['ranking_expr']}")
    else:
        lines.append(f"{noncomp}def ranking (s : {ctrl_native}) : Nat := sorry")
    lines.append("")

    return lines


def _cert_def_context(ctx: LeanContext, cert_data: CertificateData) -> dict:
    """Build a Jinja2 context dict for the Data.lean.j2 template.

    Each field yields either a ``_body`` (multi-line Lean from compiled terms),
    an ``_expr`` (raw Lean expression string from SMT translation), or neither
    (in which case the template uses the placeholder value).
    """
    def _as_terms(v):
        return v if isinstance(v, list) else None

    extl_latched = ctx.extl_latched
    extl_next = ctx.extl_next
    ctrl_next = ctx.ctrl_next

    extl_native = f"({_product_type(extl_latched)}) × ({_product_type(extl_next)})"
    ctrl_native = _product_type(ctrl_next)

    e_bindings = {
        **_bind_wires([("e.1", extl_latched)]),
        **_bind_wires([("e.2", extl_next)]),
    }
    s_bindings = _bind_wires([("s", ctrl_next)])

    def _body(terms, bindings):
        output = [terms[-1].write[0]]
        return _translate_terms(terms, bindings, output, ctx.constants)

    def _field(value, bindings):
        """Return (body, expr) where at most one is set."""
        terms = _as_terms(value)
        if terms is not None:
            return _body(terms, bindings), None
        if isinstance(value, str):
            return None, value
        return None, None

    ip_body, ip_expr = _field(cert_data.init_pre, e_bindings)
    up_body, up_expr = _field(cert_data.update_pre, e_bindings)
    inv_body, inv_expr = _field(cert_data.inv, s_bindings)
    p_body, p_expr = _field(cert_data.prp, s_bindings)
    rank_body, rank_expr = _field(cert_data.ranking, s_bindings)
    if cert_data.is_safety:
        rank_body = rank_expr = None

    return dict(
        extl_type=extl_native,
        ctrl_type=ctrl_native,
        noncomp="noncomputable " if ctx.uses_real else "",
        is_safety=cert_data.is_safety,
        init_pre_body=ip_body,
        init_pre_expr=ip_expr,
        update_pre_body=up_body,
        update_pre_expr=up_expr,
        inv_body=inv_body,
        inv_expr=inv_expr,
        p_body=p_body,
        p_expr=p_expr,
        ranking_body=rank_body,
        ranking_expr=rank_expr,
    )


def _predicate_text(ctx: LeanContext, cert_data: CertificateData) -> str:
    """The Lean text of every certificate predicate, for shape detection.

    Fields reach us as compiled term lists, as already-translated Lean
    expression strings, or not at all; `_cert_def_context` has normalised all
    three into `_body` / `_expr` entries by the time we look.
    """
    c = _cert_def_context(ctx, cert_data)
    return "\n".join(
        str(v)
        for k, v in c.items()
        if v and (k.endswith("_body") or k.endswith("_expr"))
    )


def generate_data_lean(
    ctx: LeanContext,
    cert_data: CertificateData | None = None,
) -> str:
    """Generate System/Data.lean content: init_pre, update_pre, inv, P, ranking.

    Imports only ``Core.Basic``.  ``Certificate.lean`` imports this file so that
    only the data file needs regeneration when the certificate data changes
    (e.g. after --infer).
    """
    if cert_data is None:
        cert_data = CertificateData()
    return render("project/System/Data.lean.j2", **_cert_def_context(ctx, cert_data))


def generate_certificate_lean(
    ctx: LeanContext,
    cert_data: CertificateData | None = None,
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
    if cert_data is None:
        cert_data = CertificateData()

    extl_latched = ctx.extl_latched
    extl_next = ctx.extl_next
    ctrl_next = ctx.ctrl_next

    extl_native = f"({_product_type(extl_latched)}) × ({_product_type(extl_next)})"
    ctrl_native = _product_type(ctrl_next)

    const_names = ctx.constants.names()
    const_list = ", ".join(const_names) if const_names else ""

    rm_noncomp = "noncomputable " if ctx.uses_real else ""

    # Inline definitions when used standalone (no separate System/Data.lean).
    if module_inline is not None:
        inline_defs = "\n".join(_cert_def_lines(ctx, cert_data))
        has_ranking = cert_data.ranking is not None
    else:
        inline_defs = None
        has_ranking = True
    # A safety certificate defines no ranking at all, so naming it in a simp
    # set would be an unknown identifier rather than an unused one.
    has_ranking = has_ranking and not cert_data.is_safety

    all_defs = "RM, init, update, inv, init_pre, update_pre, P"
    if has_ranking:
        all_defs += ", ranking"
    if const_list:
        all_defs += f", {const_list}"

    plan = plan_for(
        ctx,
        _predicate_text(ctx, cert_data),
        facts=cert_data.facts,
        hints=cert_data.hints,
    )

    return render(
        "project/Certificate/Certificate.lean.j2",
        module_inline=module_inline,
        inline_defs=inline_defs,
        noncomp=rm_noncomp,
        extl_type=extl_native,
        ctrl_type=ctrl_native,
        all_defs=all_defs,
        is_safety=cert_data.is_safety,
        plan=plan,
    )


def smt_predicates_to_lean(
    cert_data: CertificateData, module: "Module", *, share: bool = True
) -> CertificateData:
    """Translate SMT-LIB string fields in *cert_data* to Lean expression strings.

    None and compiled term-list fields pass through unchanged.

    `share=False` prints repeated subterms in full instead of `let`-binding
    them. The caller wants that when `--smt-tactics` settled a branch
    condition: `cert_facts` states the condition expanded, because a `have`
    outside the definition cannot name a `let` inside it, and
    `simp only [if_pos …]` then has nothing to match against a shared
    `if (u3 ≥ 0) …`.
    """
    fields = (
        cert_data.prp,
        cert_data.inv,
        cert_data.init_pre,
        cert_data.update_pre,
        cert_data.ranking,
    )
    if not any(isinstance(f, str) for f in fields):
        return cert_data

    # Lazy imports keep cvc5 off the critical path when this function is unused.
    from .smt_query import ModuleQueries, predicate_facts
    from .smt_to_lean import smt_to_lean, smt_to_lean_nat

    q = ModuleQueries.build(module, cert_data)
    if q is None:
        # cvc5 could not parse or encode; leave the SMT sources in place and
        # let the codegen path fail with its own message.
        return cert_data
    msmt = q.msmt

    def translate(term, mode: str, original) -> str | None:
        if term is None:
            return original
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

    return CertificateData(
        prp=translate(q.prp, "property", cert_data.prp),
        init_pre=translate(q.init_pre, "pre", cert_data.init_pre),
        update_pre=translate(q.update_pre, "pre", cert_data.update_pre),
        inv=translate(q.inv, "invariant", cert_data.inv),
        ranking=translate(q.ranking, "ranking", cert_data.ranking),
        # The rendering changes the predicates, not what they are a
        # certificate *of*, nor what cvc5 was given to read them from.
        kind=cert_data.kind,
        inv_smt=cert_data.inv_smt,
        ranking_smt=cert_data.ranking_smt,
        # The terms are already parsed, so reading their shape is free. The
        # tactic plan uses it in place of guessing from the printed Lean.
        facts=predicate_facts(q),
    )
