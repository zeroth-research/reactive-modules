"""A module's transition as one scalar term per state slot: what the
fbk-proveit NA encoding (`translate/fbk`) prints, built on the run's
encoding. The state, latched and awaited inputs are the encoding's
constants `s*`, `el*` and `en*`, which is what the printer's accessor maps
are keyed by."""

from __future__ import annotations

from ..layout import flat_layout
from ..sorts import wire_shape


def scalar_element(tm, term, shape, i: int, j: int):
    """Element `[i][j]` of a matrix-valued term, as a *scalar* term.

    `mat_select` alone leaves `select k (…)` sitting on whatever built the
    matrix, and the two things that build one here are a tuple constructor
    and an `ite` over tuples. Pushing the selection through both is what
    turns a wire's transition into one scalar expression per slot; the
    rewrite is the tuple axiom in each case, and cvc5's own simplifier does
    the rest.
    """
    from cvc5 import Kind

    from .encode import mat_select

    if shape.is_scalar:
        return term
    idx = i * shape.n + j
    kind = term.getKind()
    if kind == Kind.APPLY_CONSTRUCTOR:
        return term[idx + 1]          # child 0 is the constructor
    if kind == Kind.ITE:
        return tm.mkTerm(
            Kind.ITE,
            term[0],
            scalar_element(tm, term[1], shape, i, j),
            scalar_element(tm, term[2], shape, i, j),
        )
    return mat_select(tm, term, shape, i, j)


def slot_terms(enc, ctrl_next, simplify: bool = True) -> "tuple[list, list]":
    """`(update, init)`: the term of every state slot, in `flat_layout`
    order, after one round and after tick 0.

    Simplified because the encoder builds a matrix and then takes one
    element of it: `m_vec32`'s affine layer is 97 KB of term before the
    rewriter folds the constant weights and 1.9 KB after. Never a
    Bool-sorted slot, though: the rewriter turns `ite c b (!b)` into the
    equality `b = c`, which moves the state variable from a branch into a
    condition -- and a condition is where `vmt2lean`'s certificate cannot
    follow it (`generalizeNatVar` retypes the slot to the unreduced
    `TypeMap` match, and the `Decidable`/`BEq` instance the condition needs
    can no longer be synthesised; `MixedBoolInt` failed its certificate
    with "Tactic `generalize` failed"). `simplify=False` (`--fbk-simplify
    none`) leaves every slot the shape the module's own terms give it.
    """
    tm, msmt = enc.tm, enc.msmt
    state = msmt.fresh_ctrl("s")
    nxt = msmt.update_state(state, msmt.fresh_extl_l("el"),
                            msmt.fresh_extl_n("en"))
    ini = msmt.init_state(msmt.fresh_extl_n("en"))

    def rewritten(term):
        if not simplify or term.getSort().isBoolean():
            return term
        return enc.simplify(term)

    update, init = [], []
    for i, r, c in flat_layout(ctrl_next).slots:
        shape = wire_shape(ctrl_next[i])
        update.append(rewritten(scalar_element(tm, nxt[i], shape, r, c)))
        init.append(rewritten(scalar_element(tm, ini[i], shape, r, c)))
    return update, init
