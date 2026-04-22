"""SMT (cvc5.Term) → Lean 4 expression translator.

Given a cvc5 `Term` over declared state consts `s0..sN-1`, emit a Lean
expression referring to a single `s` parameter (a left-nested state
tuple), using `.1, .2.1, .2.2.1, ...` to project each component and
`_ 0 0` to extract the scalar from a `Mat t 1 1` wrapper.
"""

from __future__ import annotations

import cvc5
from cvc5 import Kind

from zrth import Wire, DType
from .common import _accessor


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
            shape = w.dtype.shape
            name = f"{prefix}{i}"
            if shape in ([], [1]):
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
        return str(t)
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
        shape = wire.dtype.shape
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
