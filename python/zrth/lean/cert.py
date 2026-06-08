from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zrth.lean.native import _product_type, _translate_terms
from zrth.lean.common import LeanContext, _bind_wires
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

    init_pre: Expr | str | None = None
    update_pre: Expr | str | None = None
    ranking: Expr | str | None = None


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

    # DecidablePred P
    if c["p_body"] is not None or c["p_expr"] is not None:
        lines.append(
            "instance : DecidablePred P := fun s => by unfold P; first | infer_instance | dsimp; infer_instance"
        )
    else:
        lines.append("instance : DecidablePred P := sorry")
    lines.append("")

    # ranking (Nat, not Prop)
    if c["ranking_body"] is not None:
        lines.append(f"def ranking (s : {ctrl_native}) : Nat :=")
        lines.append(c["ranking_body"])
    elif c["ranking_expr"] is not None:
        lines.append(f"def ranking : {ctrl_native} → Nat := {c['ranking_expr']}")
    else:
        lines.append(f"def ranking (s : {ctrl_native}) : Nat := sorry")
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

    return dict(
        extl_type=extl_native,
        ctrl_type=ctrl_native,
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

    all_defs = "RM, init, update, inv, init_pre, update_pre, P"
    if has_ranking:
        all_defs += ", ranking"
    if const_list:
        all_defs += f", {const_list}"

    return render(
        "project/Certificate/Certificate.lean.j2",
        module_inline=module_inline,
        inline_defs=inline_defs,
        noncomp=rm_noncomp,
        extl_type=extl_native,
        ctrl_type=ctrl_native,
        all_defs=all_defs,
    )


def smt_predicates_to_lean(
    cert_data: CertificateData, module: "Module"
) -> CertificateData:
    """Translate SMT-LIB string fields in *cert_data* to Lean expression strings.

    None and compiled term-list fields pass through unchanged.
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
    import cvc5

    from .smt_module import ModuleSMT
    from .smt_prompt import CegarPromptEnv, parse_predicate
    from .smt_to_lean import smt_to_lean, smt_to_lean_nat

    tm = cvc5.TermManager()
    msmt = ModuleSMT(tm=tm, module=module)
    env = CegarPromptEnv(msmt)

    def translate(smt_src: str | None, mode: str) -> str | None:
        if not isinstance(smt_src, str):
            return smt_src
        term = parse_predicate(env, smt_src)
        if mode in ("property", "invariant"):
            return smt_to_lean(term, msmt.ctrl_next, param_name="s")
        if mode == "ranking":
            return smt_to_lean_nat(term, msmt.ctrl_next, param_name="s")
        # preconditions
        return smt_to_lean(
            term,
            state_wires=[],
            param_name="e",
            extra=[
                ("e", "e.2", msmt.extl_next),
                ("el", "e.1", msmt.extl_latched),
            ],
        )

    return CertificateData(
        prp=translate(cert_data.prp, "property"),
        init_pre=translate(cert_data.init_pre, "pre"),
        update_pre=translate(cert_data.update_pre, "pre"),
        inv=translate(cert_data.inv, "invariant"),
        ranking=translate(cert_data.ranking, "ranking"),
    )
