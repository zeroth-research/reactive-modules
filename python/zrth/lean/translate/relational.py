"""The two provable relational encodings: `Rel` (matrix) and `ScalarRel` (flat).

Both are the skeleton in `_skeleton.py` — a body def, a relation and a
conjunction per slot — instantiated on the three axes it is parameterised on:
the type builder (`_product_type` vs `_product_type_scalar`), the body
producer (`_translate_terms` vs `_translate_terms_scalar`) and the projection
(a product accessor vs a re-tupled slice of the flattened element tuple).

What the two do *not* share is the closing theorem. `Rel` relates its
conjunction to the matrix `update`/`init` directly. `ScalarRel` relates its
own to `Scalar.update`/`Scalar.init`, and then bridges to the matrix form
through `Scalar.pack ∘ Scalar.unpack_ctrl` — the round trip the two state
types differ by.
"""

from dataclasses import dataclass

from zrth.lean.native import (
    _product_type,
    _product_type_scalar,
    _translate_terms,
    _translate_terms_scalar,
    _reachable_terms,
)
from zrth.lean.common import (
    LeanContext,
    _accessor,
    _bind_wires,
    dtype_to_lean_type,
    flat_layout,
)
from zrth.lean.translate._shared import (
    _scalar_bindings_with_recon,
    _prepend_recon,
    _flat_slice,
    _effect_type,
)
from zrth.lean.translate._skeleton import RelBlock, RelSyntax, emit_rel_block


@dataclass(frozen=True)
class _Domain:
    """The binders one relational encoding quantifies over."""

    state_ty: str
    update_binders: str
    update_args: str
    extl_binders: str
    extl_args: str
    extl_n_binder: str


def _domain(ctx: LeanContext, ty) -> _Domain:
    """The binders of a relational encoding, in one type builder.

    `ty` is the only thing the matrix and scalar domains disagree about here:
    `_product_type` gives each wire its own `Mat`, `_product_type_scalar` one
    component per element. Both answer "Unit" for an empty group, so a module
    with no external inputs still gets the binder written out rather than
    dropped — the call sites are built from the same lists.
    """
    update_groups = [
        ("ctrl", ctx.ctrl_latched),
        ("extl_l", ctx.extl_latched),
        ("extl_n", ctx.extl_next),
    ]
    extl_groups = update_groups[1:]

    def binders(groups):
        return " ".join(f"({p} : {ty(w)})" for p, w in groups)

    def args(groups):
        return " ".join(p for p, _ in groups)

    return _Domain(
        state_ty=ty(ctx.ctrl_next),
        update_binders=binders(update_groups),
        update_args=args(update_groups),
        extl_binders=binders(extl_groups),
        extl_args=args(extl_groups),
        extl_n_binder=f"(extl_n : {ty(ctx.extl_next)})",
    )


def _conj_eq_theorem(
    name: str,
    blk: RelBlock,
    n: int,
    rhs: str,
) -> "list[str]":
    """`conj … ↔ state = ref …`: the conjunction is the functional form.

    Unfolding the relations and rewriting each slot by its own `*_eq` leaves
    a conjunction of component equalities against a tuple equality, which is
    `Prod.ext_iff`; `tauto` finishes when the state is a single component and
    there is nothing to split.
    """
    simp = (
        [blk.conj_name]
        + [f"{blk.rel_name}_{i}" for i in range(n)]
        + [f"{blk.body_name}_{i}_eq" for i in range(n)]
        + ["Prod.ext_iff"]
    )
    return [
        f"theorem {name} : ∀ {blk.state_binders} {blk.extl_binders},",
        f"    {blk.conj_name} {blk.state_args} {blk.extl_args} ↔ {rhs} := by",
        f"  intro {blk.state_args} {blk.extl_args}",
        f"  simp only [{', '.join(simp)}]",
        "  try tauto",
        "",
    ]


def _update_block(dom: _Domain, n: int, ref: str) -> RelBlock:
    """The transition half: the state twice, the inputs alongside it.

    `ref` is the functional counterpart each slot is proved equal to —
    `update` in the matrix domain, `Scalar.update` in the flat one. It is the
    only thing the two encodings disagree about here; the body producer they
    disagree about is not a parameter of the block.
    """
    return RelBlock(
        body_name="effect",
        rel_name="R",
        conj_name="TransRel",
        state_binders=f"(old new : {dom.state_ty})",
        state_args="old new",
        target="new",
        binders=[f" {dom.update_binders}"] * n,
        args=[f" old {dom.extl_args}"] * n,
        extl_binders=dom.extl_binders,
        extl_args=dom.extl_args,
        ref=ref,
        ref_binders=dom.update_binders,
        ref_args=dom.update_args,
    )


def _init_block(dom: _Domain, n: int, ref: str) -> RelBlock:
    """The initial half: one state, and only the next-input group in scope."""
    return RelBlock(
        body_name="init",
        rel_name="Init",
        conj_name="InitCond",
        state_binders=f"(s : {dom.state_ty})",
        state_args="s",
        target="s",
        binders=[f" {dom.extl_n_binder}"] * n,
        args=[" extl_n"] * n,
        extl_binders=dom.extl_n_binder,
        extl_args="extl_n",
        ref=ref,
        ref_binders=dom.extl_n_binder,
        ref_args="extl_n",
    )


def _pack_roundtrip(var: str) -> "list[str]":
    """Tactics for ``var = Scalar.pack (Scalar.unpack_ctrl var)``.

    The functional and scalar states differ by exactly this round trip.
    Widest tactic first, as in `scalar.py`: a multi-element wire leaves a
    `match i, j with` on a symbolic `i` that only `fin_cases` clears, and
    once `i`/`j` are introduced a later `funext` can no longer fire.
    """
    return [
        f"    have hpack : {var} = Scalar.pack (Scalar.unpack_ctrl {var}) := by",
        "      simp only [Scalar.pack, Scalar.unpack_ctrl, ← Mat_1_1_eq, Prod.eta]",
        "      try rfl",
        "      try (apply Prod.ext <;> funext i j <;> fin_cases i <;> fin_cases j <;> simp)",
        "      try (funext i j <;> fin_cases i <;> fin_cases j <;> simp)",
        "      try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])",
        "      try (funext i j; simp [Fin.fin_one_eq_zero])",
    ]


def _pack_bridge_theorem(
    name: str, binders: str, intro: str, lhs: str, rhs: str, rewrites: "list[str]"
) -> "list[str]":
    """The same statement as `_conj_eq_theorem`, one domain up.

    `ScalarRel`'s conjunction is already known to be the scalar functional
    form (`*_scalar_eq`), and `scalar.py` proves that to be the matrix one
    (`update_scalar_eq` / `init_scalar_eq`); `rewrites` chains the two. What
    is left is the `pack ∘ unpack` round trip the two state types differ by
    -- forwards by rewriting with it, backwards by `simp` on its two
    definitions.
    """
    return [
        f"theorem {name} : ∀ {binders},",
        f"    {lhs} ↔ {rhs} := by",
        f"  intro {intro}",
        f"  rw [{', '.join(rewrites)}]",
        "  constructor",
        "  · intro h",
        *_pack_roundtrip("ctrl'"),
        "    rw [h] at hpack; exact hpack",
        "  · intro h",
        "    simp [Scalar.pack, Scalar.unpack_ctrl, h]",
        "",
    ]


def _unpack(param: str, wires, var: str = "") -> str:
    """`var`, read back as a matrix by `Scalar.unpack_<param>`.

    Left alone when the group is empty: `unpack_extl_l` of no wires is `Unit`
    to `Unit`, and `scalar.py` does not emit it. `var` defaults to the
    parameter's own name; the two init/trans theorems quantify the *next*
    ctrl state as `ctrl'` but still read it back with `unpack_ctrl`.
    """
    var = var or param
    return f"(Scalar.unpack_{param} {var})" if wires else var


def atom_to_lean_mat_rel(ctx: LeanContext) -> str:
    """Generate the matrix-domain relational encoding inside ``namespace Rel``."""
    noncomp = "noncomputable " if ctx.uses_real else ""

    n_ctrl = len(ctx.ctrl_next)
    dom = _domain(ctx, _product_type)

    syn = RelSyntax(
        body_decl=f"@[simp] {noncomp}def",
        rel_decl="def",
        prop="Prop",
        eq="=",
        conj="∧",
        slot_ty=lambda i: dtype_to_lean_type(ctx.ctrl_next[i]),
        project=lambda e, i: f"{e}{_accessor(i, n_ctrl)}",
    )

    update_bindings = _bind_wires(
        [
            ("ctrl", ctx.ctrl_latched),
            ("extl_l", ctx.extl_latched),
            ("extl_n", ctx.extl_next),
        ]
    )
    init_bindings = _bind_wires([("extl_n", ctx.extl_next)])

    def _bodies(terms, bindings) -> "list[str]":
        return [
            _translate_terms(
                _reachable_terms(terms, [w]), bindings, [w], ctx.constants
            )
            for w in ctx.ctrl_next
        ]

    lines = ["namespace Rel", ""]

    if list(ctx.atom.update):
        blk = _update_block(dom, n_ctrl, "update")
        lines += emit_rel_block(
            syn, blk, _bodies(ctx.atom.update, update_bindings)
        )
        lines += _conj_eq_theorem(
            "TransRel_func_eq", blk, n_ctrl, f"new = update old {dom.extl_args}"
        )

    if list(ctx.atom.init):
        blk = _init_block(dom, n_ctrl, "init")
        lines += emit_rel_block(syn, blk, _bodies(ctx.atom.init, init_bindings))
        lines += _conj_eq_theorem(
            "InitCond_func_eq", blk, n_ctrl, "s = init extl_n"
        )

    lines.append("end Rel")
    return "\n".join(lines)


def atom_to_lean_rel(ctx: LeanContext) -> str:
    """Generate the relational encoding inside ``namespace ScalarRel``."""
    noncomp = "noncomputable " if ctx.uses_real else ""

    # One `effect_i`/`R_i` per ctrl *wire*, but one state component per
    # *element*: `Scalar.update`'s codomain is the flattened tuple, so wire
    # `i`'s data sits at its own span in it, not at position `i`.
    layout = flat_layout(ctx.ctrl_next)
    n_ctrl = len(ctx.ctrl_next)
    dom = _domain(ctx, _product_type_scalar)

    syn = RelSyntax(
        body_decl=f"@[simp] {noncomp}def",
        rel_decl="def",
        prop="Prop",
        eq="=",
        conj="∧",
        slot_ty=lambda i: _effect_type(ctx.ctrl_next[i]),
        project=lambda e, i: _flat_slice(e, layout, ctx.ctrl_next[i]),
        eq_simp=("Fin.cons_zero", "Fin.cons_succ"),
    )

    update_recon, update_bindings, update_flat = _scalar_bindings_with_recon(
        [
            ("ctrl", ctx.ctrl_latched),
            ("extl_l", ctx.extl_latched),
            ("extl_n", ctx.extl_next),
        ]
    )
    init_recon, init_bindings, init_flat = _scalar_bindings_with_recon(
        [("extl_n", ctx.extl_next)]
    )

    def _bodies(terms, recon, bindings, flat) -> "list[str]":
        return [
            _prepend_recon(
                recon,
                _translate_terms_scalar(
                    _reachable_terms(terms, [w]),
                    bindings,
                    [w],
                    ctx.constants,
                    flat_slots=flat,
                    flatten_outputs=True,
                ),
            )
            for w in ctx.ctrl_next
        ]

    lines = ["namespace ScalarRel", ""]

    # The matrix domain of the same module: what the two bridging theorems
    # quantify over, since they state the scalar relation in matrix terms.
    mat = _domain(ctx, _product_type)

    has_update = bool(list(ctx.atom.update))
    has_init = bool(list(ctx.atom.init))

    # Described up front because its two closing theorems are emitted after
    # the init block, not with the block itself: `TransRel_func_eq` rewrites
    # by `update_scalar_eq`, and the emitted order is the one the generated
    # `ScalarRel.lean` has always had.
    update_blk = _update_block(dom, n_ctrl, "Scalar.update")

    if has_update:
        lines += emit_rel_block(
            syn,
            update_blk,
            _bodies(ctx.atom.update, update_recon, update_bindings, update_flat),
        )

    if has_init:
        init_blk = _init_block(dom, n_ctrl, "Scalar.init")
        lines += emit_rel_block(
            syn,
            init_blk,
            _bodies(ctx.atom.init, init_recon, init_bindings, init_flat),
        )
        lines += _conj_eq_theorem(
            "InitCond_scalar_eq", init_blk, n_ctrl, "s = Scalar.init extl_n"
        )

        new_unpack = _unpack("ctrl", ctx.ctrl_next, "ctrl'")
        extl_n_unpack = _unpack("extl_n", ctx.extl_next)

        lines += _pack_bridge_theorem(
            "InitCond_func_eq",
            f"(ctrl' : {mat.state_ty}) {mat.extl_n_binder}",
            "ctrl' extl_n",
            f"InitCond {new_unpack} {extl_n_unpack}",
            "ctrl' = init extl_n",
            ["InitCond_scalar_eq", "init_scalar_eq"],
        )

    if has_update:
        lines += _conj_eq_theorem(
            "TransRel_scalar_eq",
            update_blk,
            n_ctrl,
            f"new = Scalar.update old {dom.extl_args}",
        )

        extl_groups = [("extl_l", ctx.extl_latched), ("extl_n", ctx.extl_next)]

        old_unpack = _unpack("ctrl", ctx.ctrl_latched)
        new_unpack = _unpack("ctrl", ctx.ctrl_next, "ctrl'")
        extl_unpacks = " ".join(_unpack(p, w) for p, w in extl_groups)

        lines += _pack_bridge_theorem(
            "TransRel_func_eq",
            f"(ctrl ctrl' : {_product_type(ctx.ctrl_latched)}) {mat.extl_binders}",
            f"ctrl ctrl' {mat.extl_args}",
            f"TransRel {old_unpack} {new_unpack} {extl_unpacks}",
            f"ctrl' = update ctrl {mat.extl_args}",
            ["TransRel_scalar_eq", "update_scalar_eq"],
        )

    lines.append("end ScalarRel")
    return "\n".join(lines)
