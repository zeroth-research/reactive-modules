"""SMT (cvc5.Term) → Lean 4 expression translator.

Given a cvc5 `Term` over declared state consts `s0..sN-1`, emit a Lean
expression referring to a single `s` parameter (a left-nested state
tuple), using `.1, .2.1, .2.2.1, ...` to project each component and
`_ 0 0` to extract the scalar from a `Mat t 1 1` wrapper.
"""

from __future__ import annotations

import cvc5
from cvc5 import Kind

from zrth import Wire, Sort
from .common import _accessor, dtype_shape, _is_scalar_shape


# Binary / variadic operator maps
_INFIX_LOGIC = {
    Kind.AND: " ∧ ",
    Kind.OR: " ∨ ",
}
_INFIX_ARITH = {
    Kind.ADD: " + ",
    Kind.SUB: " - ",
    Kind.MULT: " * ",
    Kind.INTS_MODULUS: " % ",
}
_INFIX_CMP = {
    Kind.LT: " < ",
    Kind.LEQ: " ≤ ",
    Kind.GT: " > ",
    Kind.GEQ: " ≥ ",
    Kind.EQUAL: " = ",
}


def build_var_map(
    bindings: list[tuple[str, str, list[Wire]]],
) -> tuple[dict[str, str], dict[str, Wire]]:
    """Build Lean-accessor + wire maps for a list of (prefix, base, wires).

    For each binding, each wire at position `i` gets a variable name
    `<prefix><i>` mapped to a Lean accessor `<base><acc> [0 0]`. Scalar
    1×1 wires get the `0 0` scalar extraction; vector/matrix wires keep
    the raw function form.
    """
    var_accessor: dict[str, str] = {}
    var_wire: dict[str, Wire] = {}
    for prefix, base, wires in bindings:
        n = len(wires)
        for i, w in enumerate(wires):
            acc = _accessor(i, n)
            shape = dtype_shape(w.dtype)
            name = f"{prefix}{i}"
            if _is_scalar_shape(shape):
                var_accessor[name] = f"({base}{acc} 0 0)"
            else:
                var_accessor[name] = f"({base}{acc})"
            var_wire[name] = w
    return var_accessor, var_wire


def _state_access(state_wires: list[Wire]) -> tuple[dict[str, str], dict[str, Wire]]:
    """Convenience: bind `s0..sN-1` → `s.acc_i [0 0]`."""
    return build_var_map([("s", "s", state_wires)])


def smt_to_lean(
    term: cvc5.Term,
    state_wires: list[Wire],
    *,
    param_name: str = "s",
    extra: list[tuple[str, str, list[Wire]]] | None = None,
) -> str:
    """Translate `term` into a Lean 4 expression in a single parameter.

    `extra` lets the caller bind additional SMT variable names (e.g.
    `e0..eM-1` → `e.2.acc_i 0 0` for extl_next) against the Lean param.
    """
    bindings = [("s", param_name, state_wires)] + (extra or [])
    var_accessor, var_wire = build_var_map(bindings)
    body = _walk(term, var_accessor, var_wire)
    return f"fun {param_name} => {body}"


def smt_to_lean_nat(
    term: cvc5.Term,
    state_wires: list[Wire],
    *,
    param_name: str = "s",
    extra: list[tuple[str, str, list[Wire]]] | None = None,
) -> str:
    """Int-sorted term → Lean `Nat` expression, clamped via `Int.toNat`."""
    bindings = [("s", param_name, state_wires)] + (extra or [])
    var_accessor, var_wire = build_var_map(bindings)
    body = _walk(term, var_accessor, var_wire)
    return f"fun {param_name} => (({body} : Int)).toNat"


# ---------------------------------------------------------------------


# `ite (a ≥ b) a b` is `max a b`, `ite (a ≥ b) b a` is `min a b`, and so on
# for each strict/non-strict comparison. The flag says whether the branches
# are in the same order as the comparison's operands.
_MIN_MAX = {
    (Kind.GEQ, True): "max", (Kind.GT, True): "max",
    (Kind.LEQ, True): "min", (Kind.LT, True): "min",
    (Kind.GEQ, False): "min", (Kind.GT, False): "min",
    (Kind.LEQ, False): "max", (Kind.LT, False): "max",
}


def _fold_min_max(t: cvc5.Term, recur) -> str | None:
    """An `ite` that is really a `min`/`max`, emitted as one. None otherwise.

    Every ReLU unit arrives here as `(ite (>= e 0) e 0)`, because there is no
    `max` kind to translate from. Left as an `ite` it costs the certificate a
    `split_ifs` branch, and `hrank` mentions the ranking twice, so a k-unit
    net fans a single goal out into 2^(2k) of them -- 4096 for six units,
    each one paying for the whole prep chain. That is what makes a net
    ranking exhaust the heartbeat budget rather than fail to be provable.
    `omega` reasons about `min`/`max` over `Int` natively, with no split.

    Int only. A Real goal is closed by `linarith`, which has no `min`/`max`
    support, so there the `ite` and its `split_ifs` branch are still the way
    through.
    """
    if not t.getSort().isInteger():
        return None
    cond = t[0]
    kind = cond.getKind()
    if kind not in (Kind.GEQ, Kind.GT, Kind.LEQ, Kind.LT):
        return None
    lhs, rhs = cond[0], cond[1]
    then_, else_ = t[1], t[2]
    if lhs == then_ and rhs == else_:
        in_order = True
    elif lhs == else_ and rhs == then_:
        in_order = False
    else:
        return None
    return f"({_MIN_MAX[(kind, in_order)]} {recur(lhs)} {recur(rhs)})"


def _walk(
    t: cvc5.Term,
    var_accessor: dict[str, str],
    var_wire: dict[str, Wire],
) -> str:
    k = t.getKind()

    if k == Kind.CONST_BOOLEAN:
        return "true" if t.getBooleanValue() else "false"
    if k == Kind.CONST_INTEGER:
        v = t.getIntegerValue()
        return f"({v} : Int)" if v < 0 else str(v)
    if k == Kind.CONST_RATIONAL:
        # `str(t)` is SMT-LIB, not Lean: cvc5 prints 0.5 as `(/ 1 2)` and
        # -0.25 as `(/ (- 1) 4)`. Build the literal from the exact value.
        q = t.getRealValue()
        if q.denominator == 1:
            return f"({q.numerator} : Real)"
        return f"(({q.numerator} : Real) / {q.denominator})"
    if k == Kind.CONST_BITVECTOR:
        width = t.getSort().getBitVectorSize()
        value = int(t.getBitVectorValue(10))
        # see the note in common.py: `ofNat` rejects a negative literal
        return f"(BitVec.ofInt {width} ({value}))"
    if k == Kind.CONSTANT:
        name = t.getSymbol()
        if name in var_accessor:
            return var_accessor[name]
        raise ValueError(
            f"Unknown free variable `{name}` (known: {list(var_accessor)})"
        )

    recur = lambda x: _walk(x, var_accessor, var_wire)

    if k == Kind.NOT:
        return f"¬ ({recur(t[0])})"

    if k in _INFIX_LOGIC:
        parts = [recur(c) for c in t]
        return "(" + _INFIX_LOGIC[k].join(parts) + ")"

    if k == Kind.IMPLIES:
        # cvc5 keeps `=>` variadic and right-associated.
        parts = [recur(c) for c in t]
        return "(" + " → ".join(parts) + ")"

    if k in _INFIX_ARITH:
        parts = [recur(c) for c in t]
        return "(" + _INFIX_ARITH[k].join(parts) + ")"

    if k in _INFIX_CMP:
        return f"({recur(t[0])}{_INFIX_CMP[k]}{recur(t[1])})"

    if k == Kind.DISTINCT:
        return f"({recur(t[0])} ≠ {recur(t[1])})"

    if k == Kind.NEG:
        return f"(- {recur(t[0])})"

    if k == Kind.ITE:
        folded = _fold_min_max(t, recur)
        if folded is not None:
            return folded
        return f"(if {recur(t[0])} then {recur(t[1])} else {recur(t[2])})"

    if k == Kind.TO_INTEGER:
        return f"⌊{recur(t[0])}⌋"

    if k == Kind.APPLY_SELECTOR:
        sel = t[0]
        obj = t[1]
        if obj.getKind() != Kind.CONSTANT or obj.getSymbol() not in var_wire:
            raise ValueError(
                f"APPLY_SELECTOR against non-bound term not supported: {t}"
            )
        idx = _selector_index(sel)
        sname = obj.getSymbol()
        wire = var_wire[sname]
        shape = dtype_shape(wire.dtype)
        if len(shape) == 1:
            m, n = 1, shape[0]
        elif len(shape) == 2:
            m, n = shape
        else:
            raise ValueError(f"tuple selector on unexpected shape: {shape}")
        i, j = divmod(idx, n)
        base = var_accessor[sname]
        return f"({base} {i} {j})"

    raise ValueError(f"SMT→Lean: unsupported kind {k} in {t}")


def _selector_index(sel: cvc5.Term) -> int:
    """Extract the flat projection index from a tuple-selector term.

    cvc5 auto-generates selector symbols like `__cvc5_tuple_N_stor_K` where
    K is the field index. We parse K from the symbol.
    """
    sym = sel.getSymbol()
    # e.g. '__cvc5_tuple_2_stor_0' → 0
    if "_stor_" in sym:
        return int(sym.rsplit("_stor_", 1)[1])
    raise ValueError(f"Cannot extract selector index from `{sym}`")


# ---------------------------------------------------------------------
# Bool-valued output, for the NA encoding (`translate/na.py`)
# ---------------------------------------------------------------------

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
}


def smt_to_lean_bool(term: cvc5.Term, var_accessor: dict[str, str]) -> str:
    """Translate `term` into a **Bool**-valued Lean expression.

    `var_accessor` maps each SMT constant name to the Lean expression that
    reads it (for the NA encoding, `s0` → `(var_0 state)`).

    Unlike :func:`smt_to_lean`, which emits `Prop` (`∧`, `¬`, `≤`), this
    stays in `Bool` (`&&`, `!`, `decide (… ≤ …)`) because that is what
    `lean2vmt` can translate: `decide` applied to a compound proposition
    hands it `instDecidableAnd`, which it prints as an unapplied leaf.
    Raises `ValueError` on anything outside that fragment rather than
    emitting Lean that would silently mistranslate.
    """
    return _walk_bool(term, var_accessor)


def _walk_bool(t: cvc5.Term, acc: dict[str, str]) -> str:
    k = t.getKind()
    recur = lambda x: _walk_bool(x, acc)

    if k == Kind.CONST_BOOLEAN:
        return "true" if t.getBooleanValue() else "false"
    if k == Kind.CONST_INTEGER:
        return f"({t.getIntegerValue()} : Int)"
    if k == Kind.CONSTANT:
        name = t.getSymbol()
        if name in acc:
            return acc[name]
        raise ValueError(
            f"Unknown free variable `{name}` (known: {sorted(acc)})"
        )

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

    raise ValueError(
        f"SMT→Bool Lean: unsupported kind {k} in {t}; lean2vmt has no "
        "translation for it"
    )
