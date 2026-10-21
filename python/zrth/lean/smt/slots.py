"""A module's transition as one scalar term per state slot: what the
fbk-proveit NA encoding (`infer/fbk_proveit/na`) prints, built on the run's
encoding. The state, latched and awaited inputs are the encoding's
constants `s*`, `el*` and `en*`, which is what the printer's accessor maps
are keyed by."""

from __future__ import annotations

from ..lowering.elements import flat_layout
from ..lowering.sorts import wire_shape


def scalar_element(tm, term, shape, i: int, j: int):
    """Element `[i][j]` of a matrix-valued term, as a *scalar* term:
    `select_element`, which is what turns a wire's transition into one
    scalar expression per slot, and cvc5's own simplifier does the rest."""
    from .encode import select_element

    if shape.is_scalar:
        return term
    return select_element(tm, term, i * shape.n + j)


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
    tm = enc.tm
    # The encoding's transition, translated once for the run, over constants
    # named `s*`, `el*` and `en*` as the accessor maps expect.
    state, el, en = enc.fresh()
    nxt = enc.next_state(state, el, en)
    ini = enc.init_state(en)

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
