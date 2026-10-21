"""cvc5 terms as Bool-valued Lean, for the NA model `lean2vmt` reads.

`lean2vmt` (in `lean-ltl-certifying`) translates a model's transition and
property only in `Bool` -- `&&`, `!`, `decide (… ≤ …)` -- not in the `Prop`
that `proof.from_smt` prints for a certificate. `smt_to_lean_bool` prints
a term in that fragment, or refuses it.
"""

from __future__ import annotations

import cvc5
from cvc5 import Kind

from ...core.errors import Refused
from ...codegen.literals import lean_int
from ...smt.terms import selector_index


# `lean2vmt`'s `exprToSMT` reads `decide`'s *instance* argument, not its
# proposition, so a comparison only survives the round trip when the
# instance it picks is one of `Int.decLt` / `Int.decLe` / `Int.decEq`.
# Normalising `>`/`≥` into `<`/`≤` with swapped operands keeps it that way
# without relying on how `GT.gt` unfolds during instance synthesis.
_BOOL_CMP = {
    Kind.LT: ("<", False),
    Kind.LEQ: ("≤", False),
    Kind.GT: ("<", True),
    Kind.GEQ: ("≤", True),
}

_BOOL_ARITH = {
    Kind.ADD: " + ",
    Kind.SUB: " - ",
    Kind.MULT: " * ",
    # No `INTS_MODULUS`. `lean2vmt` does now translate `%` (as SMT-LIB
    # `mod`) and ic3ia proves such a property, but the *witness* comes back
    # with the mod eliminated: MathSAT rewrites `x mod 2 = 0` into
    # `x + (-2) * to_int ((1/2) * to_real x) = 0`, and `vmt2lean.py` has no
    # case for `to_real`/`to_int`/`/` -- rendering them would also drag Real
    # into a Bool-valued `INVAR`. Rejecting here keeps the failure at the
    # front of the pipeline, where the message can say why.
}


def smt_to_lean_bool(term: cvc5.Term, var_slots: dict[str, list[str]]) -> str:
    """Translate `term` into a **Bool**-valued Lean expression.

    `var_slots` maps each SMT constant name to the Lean expressions that read
    its *elements* -- one per flat element, so a 1×1 wire has a single entry
    (`s0` → `["(var_0 state)"]`) and a wider one has an entry per element,
    reached through the tuple selectors cvc5 builds. The constant itself is
    only usable where it has exactly one element: a whole tuple has no
    Bool/Int value to print.

    Unlike :func:`proof.from_smt.smt_to_lean`, which emits `Prop` (`∧`,
    `¬`, `≤`), this stays in `Bool` (`&&`, `!`, `decide (… ≤ …)`) because
    that is what `lean2vmt` can translate: `decide` applied to a compound
    proposition hands it `instDecidableAnd`, which it prints as an
    unapplied leaf.
    Raises `ValueError` on anything outside that fragment rather than
    emitting Lean that would silently mistranslate.
    """
    return _walk_bool(term, var_slots)


def _walk_bool(t: cvc5.Term, acc: dict[str, list[str]]) -> str:
    k = t.getKind()
    recur = lambda x: _walk_bool(x, acc)

    if k == Kind.CONST_BOOLEAN:
        return "true" if t.getBooleanValue() else "false"
    if k == Kind.CONST_INTEGER:
        return lean_int(t.getIntegerValue())
    if k == Kind.CONSTANT:
        name = t.getSymbol()
        if name in acc:
            slots = acc[name]
            if len(slots) == 1:
                return slots[0]
            raise Refused(
                f"`{name}` holds {len(slots)} elements, so it has no single "
                "value; select an element instead"
            )
        raise Refused(
            f"Unknown free variable `{name}` (known: {sorted(acc)})"
        )

    if k == Kind.APPLY_SELECTOR:
        obj = t[1]
        idx = selector_index(t)
        # A tuple built on the spot -- `smt.encode` packs every matrix-valued
        # wire, so a constant matrix arrives as `select k (tuple …)`. Project
        # it away here: a constructor has no form `lean2vmt` reads, and the
        # element is what the slot wanted anyway. Child 0 is the constructor,
        # so element `k` is child `k + 1`.
        if obj.getKind() == Kind.APPLY_CONSTRUCTOR:
            return recur(obj[idx + 1])
        # A wide wire reaches here as a tuple constant; its elements are
        # exactly the state slots, in the row-major order `mat_select` packs.
        if obj.getKind() != Kind.CONSTANT or obj.getSymbol() not in acc:
            raise Refused(
                f"tuple selector against a non-state term is not supported: {t}"
            )
        slots = acc[obj.getSymbol()]
        if idx >= len(slots):
            raise Refused(
                f"element {idx} of `{obj.getSymbol()}`, which has "
                f"{len(slots)}"
            )
        return slots[idx]

    if k == Kind.NOT:
        return f"(!{recur(t[0])})"
    if k == Kind.AND:
        return "(" + " && ".join(recur(c) for c in t) + ")"
    if k == Kind.OR:
        return "(" + " || ".join(recur(c) for c in t) + ")"
    if k == Kind.IMPLIES:
        # cvc5 keeps `=>` variadic and right-associated; `a → b` is `!a || b`.
        parts = [recur(c) for c in t]
        out = parts[-1]
        for p in reversed(parts[:-1]):
            out = f"((!{p}) || {out})"
        return out
    if k == Kind.XOR:
        return f"(!({recur(t[0])} == {recur(t[1])}))"

    if k == Kind.EQUAL:
        return f"({recur(t[0])} == {recur(t[1])})"
    if k == Kind.DISTINCT:
        return f"(!({recur(t[0])} == {recur(t[1])}))"

    if k in _BOOL_CMP:
        op, swap = _BOOL_CMP[k]
        lhs, rhs = (t[1], t[0]) if swap else (t[0], t[1])
        return f"(decide ({recur(lhs)} {op} {recur(rhs)}))"

    if k in _BOOL_ARITH:
        return "(" + _BOOL_ARITH[k].join(recur(c) for c in t) + ")"
    if k == Kind.NEG:
        return f"(-{recur(t[0])})"

    if k == Kind.ITE:
        return f"(if {recur(t[0])} then {recur(t[1])} else {recur(t[2])})"

    raise Refused(
        f"SMT→Bool Lean: unsupported kind {k} in {t}; "
        + (
            "lean2vmt does translate `mod`, and ic3ia proves such a "
            "property, but MathSAT eliminates the mod from the witness "
            "(via to_real/to_int) and vmt2lean cannot render that back"
            if k == Kind.INTS_MODULUS
            else "lean2vmt has no translation for it"
        )
    )
