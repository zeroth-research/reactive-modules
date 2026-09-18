"""The shape every relational encoding has, written once.

`Rel` (matrix domain), `ScalarRel` (flat scalars) and the NA model's
`Definition` (flat scalars, Bool) all emit the same three things in the same
order: a per-slot body def (`effect_i` / `init_i`), a per-slot relation
(`R_i` / `Init_i`) pinning one slot of the next state to that body, and a
conjunction over them (`TransRel` / `INIT`).

They vary on three axes, and this module is parameterised on exactly those:

1. **type builder** — what a state value is: a product of `Mat`s
   (`_product_type`), one component per *element* (`_product_type_scalar`),
   or the NA model's per-slot element type.  :attr:`RelSyntax.slot_ty` is the
   half of it a slot needs.
2. **body producer** — `_translate_terms`, `_translate_terms_scalar`, or
   `smt_encode` + `smt_to_lean_bool`.  This is not a parameter: each emitter
   hands :func:`emit_rel_block` the body text it produced.  That is also what
   lets the three disagree about what a *slot is* — one per wire for
   `Rel`/`ScalarRel`, one per element for the NA model — since the skeleton
   only ever counts the bodies it was given.
3. **projection** — how slot `i` is read out of a state value
   (:attr:`RelSyntax.project`): a product accessor, a re-tupled slice of the
   flattened element tuple, or `var_i state`.

The per-slot `*_eq` theorems are optional (:attr:`RelBlock.ref`), emitted
only by the two encodings that can prove them *by `rfl`*: both sides come
from the same body producer over the same reachable terms, so they are
syntactically equal.  A cvc5-printed body is the same transition spelled
differently — `-1 + x` for `x - 1`, an equality for an `ite` — so the NA
model emits no theorem block here and has its slots related to the module
semantically instead, which is what `fbk_bridge.py` discharges.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RelSyntax:
    """How one encoding spells the shape all of them share."""

    body_decl: str  # opens a per-slot body def: `@[simp] def` / `abbrev`
    rel_decl: str  # opens a relation or the conjunction: `def` / `abbrev`
    prop: str  # what a relation returns: `Prop` / `Bool`
    eq: str  # how a relation compares its two sides: `=` / `==`
    conj: str  # how the conjunction joins its relations: `∧` / `&&`
    slot_ty: Callable[[int], str]  # codomain of slot `i`'s body
    project: Callable[[str, int], str]  # slot `i` of a state expression
    # Extra `simp only` lemmas the `*_eq` proofs need. The scalar domain's
    # bodies come out of `Fin.cons`, the matrix domain's do not.
    eq_simp: "tuple[str, ...]" = ()


@dataclass(frozen=True)
class RelBlock:
    """One half of a relational encoding: the transition, or the init condition.

    `binders` and `args` are per slot rather than shared because the NA model
    leaves the `state` binder off a slot whose next value is a closed term,
    rather than writing one it does not use.
    """

    body_name: str  # `effect` / `init`
    rel_name: str  # `R` / `Init`
    conj_name: str  # `TransRel` / `INIT`
    state_binders: str  # what quantifies the state(s): `(old new : S)`
    state_args: str  # and their names: `old new`
    target: str  # the state each slot is projected out of: `new`
    binders: "list[str]"  # per body, what its body def binds (leading space)
    args: "list[str]"  # per body, what the relation applies it to (leading space)
    # Which slot each body belongs to. Empty means the bodies *are* the
    # slots, which is every case but one: the NA model's `INIT` omits a slot
    # the module starts at an unconstrained input, and the slots it does
    # pin keep their own numbers, because `Init_k` names `var_k`.
    slots: "tuple[int, ...]" = ()
    # Conjuncts that are not a slot, already applied to their arguments, and
    # joined after the per-slot ones. The NA model's `INIT` carries `--pre`
    # here: it constrains the state without pinning any one slot to a body,
    # so it is a conjunct and nothing else in this shape.
    extra: "tuple[str, ...]" = ()
    extl_binders: str = ""  # the external inputs, carried by every declaration
    extl_args: str = ""
    # The functional counterpart to relate each slot back to. Empty means no
    # `*_eq` theorem block -- the NA model proves nothing about its bodies.
    ref: str = ""
    ref_binders: str = ""
    ref_args: str = ""


def _decl(opener: str, name: str, *binders: str, ty: str) -> str:
    """`opener name binders… : ty :=`, skipping the binder groups that are empty.

    The NA model has no external inputs at all (`check_na_supported` rejects
    a module with any), so its declarations are one binder group short of the
    other two encodings'.
    """
    parts = [opener, name, *[b for b in binders if b]]
    return f"{' '.join(parts)} : {ty} :="


def _apply(head: str, *args: str) -> str:
    """`head args…`, skipping the argument groups that are empty."""
    return " ".join([head, *[a for a in args if a]])


def _eq_theorem(syn: RelSyntax, blk: RelBlock, i: int) -> "list[str]":
    """`body_i … = (ref …)[slot i]`, the theorem that ties one slot to its source.

    `rfl` closes it whenever the two sides were produced by the same body
    producer over the same terms; the tactics after it are for the shapes
    where a `Mat 1 1` meets a scalar and the equality is only propositional.
    """
    name = f"{blk.body_name}_{i}"
    lemmas = ", ".join([blk.ref, name, *syn.eq_simp])
    return [
        f"theorem {name}_eq : ∀ {blk.ref_binders},",
        f"    {name} {blk.ref_args} = "
        f"{syn.project(f'({blk.ref} {blk.ref_args})', i)} := by",
        f"  intro {blk.ref_args}",
        f"  simp only [{lemmas}]",
        "  try rfl",
        "  try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])",
        "  try simp [Fin.fin_one_eq_zero]",
        "  try omega",
        "",
    ]


def emit_rel_block(
    syn: RelSyntax, blk: RelBlock, bodies: "list[str]"
) -> "list[str]":
    """Emit one block: the bodies, their `*_eq` theorems, the relations, the conjunction.

    `bodies` is the body producer's output, one entry per body, already
    indented the way that producer indents. Which slot each belongs to is
    `blk.slots`, and by default that is simply its position.
    """
    n = len(bodies)
    slots = blk.slots or tuple(range(n))
    lines: "list[str]" = []

    for pos, body in enumerate(bodies):
        i = slots[pos]
        lines.append(
            f"{syn.body_decl} {blk.body_name}_{i}{blk.binders[pos]} "
            f": {syn.slot_ty(i)} :="
        )
        lines.append(body)
        lines.append("")

    if blk.ref:
        for i in slots:
            lines.extend(_eq_theorem(syn, blk, i))

    for pos, i in enumerate(slots):
        lines.append(
            _decl(
                syn.rel_decl,
                f"{blk.rel_name}_{i}",
                blk.state_binders,
                blk.extl_binders,
                ty=syn.prop,
            )
        )
        lines.append(
            f"  {syn.project(blk.target, i)} {syn.eq} "
            f"{blk.body_name}_{i}{blk.args[pos]}"
        )
        lines.append("")

    calls = [
        _apply(f"{blk.rel_name}_{i}", blk.state_args, blk.extl_args)
        for i in slots
    ] + list(blk.extra)
    lines.append(
        _decl(
            syn.rel_decl,
            blk.conj_name,
            blk.state_binders,
            blk.extl_binders,
            ty=syn.prop,
        )
    )
    # An empty conjunction is `true`, and it is reachable: a module whose
    # every slot starts at an unconstrained input constrains nothing at
    # time 0, which is a state set, not a missing one.
    empty = "true" if syn.prop == "Bool" else "True"
    lines.append("  " + (f" {syn.conj}\n  ".join(calls) if calls else empty))
    lines.append("")

    return lines
