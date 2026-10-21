"""SMT (cvc5.Term) → Lean 4 expression translator.

Given a cvc5 `Term` over declared state consts `s0..sN-1`, emit a Lean
expression referring to a single `s` parameter (a left-nested state
tuple), using `.1, .2.1, .2.2.1, ...` to project each component and
`_ 0 0` to extract the scalar from a `Mat t 1 1` wrapper.
"""

from __future__ import annotations

import cvc5
from cvc5 import Kind

from zrth import Wire
from ..core.errors import Refused
from ..lowering.sorts import wire_shape
from ..lowering.elements import product_accessor
from ..codegen.literals import lean_bitvec, lean_int, lean_real
from ..smt.terms import min_max_of, selector_index, state_dependent


# Binary / variadic operator maps
_INFIX_LOGIC = {
    Kind.AND: " ∧ ",
    Kind.OR: " ∨ ",
}
_INFIX_ARITH = {
    Kind.ADD: " + ",
    Kind.SUB: " - ",
    Kind.MULT: " * ",
    # SMT-LIB's `div`/`mod` are Euclidean -- `a = b * (a div b) + (a mod b)`
    # with `0 <= a mod b < |b|` -- and so are Lean 4's `/` and `%` on `Int`:
    # `(-7 : Int) / 2` is `-4` and `(-7 : Int) % 2` is `1`, where truncating
    # division would give `-3` and `-1`. The pair goes across unchanged.
    Kind.INTS_DIVISION: " / ",
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
            acc = product_accessor(i, n)
            name = f"{prefix}{i}"
            if wire_shape(w).is_scalar:
                var_accessor[name] = f"({base}{acc} 0 0)"
            else:
                var_accessor[name] = f"({base}{acc})"
            var_wire[name] = w
    return var_accessor, var_wire


def smt_to_lean(
    term: cvc5.Term,
    state_wires: list[Wire],
    *,
    param_name: str = "s",
    extra: list[tuple[str, str, list[Wire]]] | None = None,
    share: bool = True,
) -> str:
    """Translate `term` into a Lean 4 expression in a single parameter.

    `extra` lets the caller bind additional SMT variable names (e.g.
    `e0..eM-1` → `e.2.acc_i 0 0` for extl_next) against the Lean param.
    """
    bindings = [("s", param_name, state_wires)] + (extra or [])
    var_accessor, var_wire = build_var_map(bindings)
    body, lets = _render(term, var_accessor, var_wire, share)
    return f"fun {param_name} => {_with_lets(body, lets)}"


def smt_to_lean_body(
    term: cvc5.Term,
    state_wires: list[Wire],
    *,
    param_name: str = "s",
) -> str:
    """The expression alone, with no `fun` and no `let`s wrapped round it.

    For splicing a subterm into a tactic -- a branch condition inside a
    `have`, say -- where a binder would be wrong and a `let` could not be
    referred to from the surrounding proof.
    """
    var_accessor, var_wire = build_var_map([("s", param_name, state_wires)])
    return _walk(term, var_accessor, var_wire)


def smt_to_lean_nat(
    term: cvc5.Term,
    state_wires: list[Wire],
    *,
    param_name: str = "s",
    extra: list[tuple[str, str, list[Wire]]] | None = None,
    share: bool = True,
) -> str:
    """Int-sorted term → Lean `Nat` expression, clamped via `Int.toNat`."""
    bindings = [("s", param_name, state_wires)] + (extra or [])
    var_accessor, var_wire = build_var_map(bindings)
    body, lets = _render(term, var_accessor, var_wire, share)
    return f"fun {param_name} => {_with_lets(f'(({body} : Int)).toNat', lets)}"


# ---------------------------------------------------------------------
# Sharing
#
# cvc5 hash-conses, so a predicate reaches us as a DAG; `_walk` alone prints
# it as a tree. That is not a cosmetic difference. A dense net whose hidden
# layer feeds the next one shares every unit, and the expansion is
# exponential in depth: a five-layer net that is 10 distinct ReLU units in
# the DAG printed as 62 `max` applications, and a sixth layer would have
# printed 126. `let`-binding each subterm that is reached more than once
# puts the emitted definition back in step with the term's real size.
#
# Only arithmetic is shared. Binding a Bool-sorted subterm would hide the
# predicate's logical structure behind a name, and `split_ifs` / `casesm*` /
# `not_and_or` in the prep chain all match on that structure.
# ---------------------------------------------------------------------


def _with_lets(body: str, lets: list[tuple[str, str]]) -> str:
    if not lets:
        return body
    binds = "".join(f"\n  let {name} := {expr}" for name, expr in lets)
    return f"{binds}\n  {body}"


def _shared_nodes(t: cvc5.Term) -> set[int]:
    """Ids of the arithmetic subterms the printer would emit more than once.

    The test is "reached by two or more distinct parent edges", which is
    exactly the set that needs its own binding: a node with a single parent
    is printed once *inside* that parent, however often the parent itself is
    printed.
    """
    edges: dict[int, int] = {}
    nodes: dict[int, cvc5.Term] = {}
    stack = [t]
    while stack:
        node = stack.pop()
        key = node.getId()
        nodes[key] = node
        edges[key] = edges.get(key, 0) + 1
        if edges[key] > 1:
            continue  # its children were already walked from the first edge
        stack.extend(node)
    return {k for k, n in edges.items() if n > 1 and _worth_binding(nodes[k])}


def _worth_binding(t: cvc5.Term) -> bool:
    """Compound, state-dependent, and not a proposition.

    Bool is excluded for the reason in the note above. A *ground* subterm --
    `(- 1)`, or any numeral arithmetic -- is excluded because binding it
    costs more text than repeating it, and the elaborator does not care
    about either.
    """
    if t.getSort().isBoolean() or t.getNumChildren() == 0:
        return False
    return state_dependent(t)


def _render(
    t: cvc5.Term,
    var_accessor: dict[str, str],
    var_wire: dict[str, Wire],
    share: bool,
) -> tuple[str, list[tuple[str, str]]]:
    """`(body, lets)`, with `lets` already in dependency order."""
    if not share:
        return _walk(t, var_accessor, var_wire), []
    plain = _walk(t, var_accessor, var_wire)
    shared = _shared_nodes(t)
    if not shared:
        return plain, []
    names: dict[int, str] = {}
    lets: list[tuple[str, str]] = []

    def emit(node: cvc5.Term) -> str:
        key = node.getId()
        if key in names:
            return names[key]
        text = _walk(node, var_accessor, var_wire, emit_child=emit)
        if key in shared:
            name = f"u{len(lets)}"
            # Children are emitted before the parent, so appending here is
            # already a topological order.
            lets.append((name, text))
            names[key] = name
            return name
        return text

    body = _walk(t, var_accessor, var_wire, emit_child=emit)
    # Sharing is only ever worth a `let` if it actually shrinks the output.
    # A shallow net repeats little, and there each binding costs more text
    # than the repetition it removes.
    if len(_with_lets(body, lets)) >= len(plain):
        return plain, []
    return body, lets


# ---------------------------------------------------------------------


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
    op = min_max_of(t)
    if op is None:
        return None
    return f"({op} {recur(t[0][0])} {recur(t[0][1])})"


def _walk(
    t: cvc5.Term,
    var_accessor: dict[str, str],
    var_wire: dict[str, Wire],
    emit_child=None,
) -> str:
    """Print `t`. `emit_child`, when given, prints each child instead of
    recursing directly, which is how `_render` interposes its `let` names."""
    k = t.getKind()

    if k == Kind.CONST_BOOLEAN:
        return "true" if t.getBooleanValue() else "false"
    if k == Kind.CONST_INTEGER:
        return lean_int(t.getIntegerValue())
    if k == Kind.CONST_RATIONAL:
        # `str(t)` is SMT-LIB, not Lean: cvc5 prints 0.5 as `(/ 1 2)`. The
        # literal is built from the exact value.
        return lean_real(t.getRealValue())
    if k == Kind.CONST_BITVECTOR:
        return lean_bitvec(t.getSort().getBitVectorSize(),
                           int(t.getBitVectorValue(10)))
    if k == Kind.CONSTANT:
        name = t.getSymbol()
        if name in var_accessor:
            return var_accessor[name]
        raise Refused(
            f"Unknown free variable `{name}` (known: {list(var_accessor)})"
        )

    recur = emit_child or (lambda x: _walk(x, var_accessor, var_wire))

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

    if k == Kind.BITVECTOR_UBV_TO_INT:
        # `bv2nat`/`ubv_to_int`: a bitvector read as the number it holds.
        # `--infer smt-linear` weighs a BitVec state component this way, so
        # one `Int` template covers a mixed state instead of a template per
        # sort. `Int.ofNat` rather than a coercion arrow: the surrounding
        # expression is `Int`-typed and an inserted `↑` there is a different
        # elaboration problem in each place it lands.
        return f"(Int.ofNat (BitVec.toNat {recur(t[0])}))"

    if k == Kind.TO_REAL:
        # cvc5 inserts `to_real` wherever an Int-sorted subterm meets a Real
        # one: an integer literal in the property, or the `Argmax` index that
        # an LRA module carries on a Real wire. An integer literal becomes a
        # Real literal outright; anything else elaborates at Int and is coerced.
        child = t[0]
        if child.getKind() == Kind.CONST_INTEGER:
            return lean_real(child.getIntegerValue())
        return f"((({recur(child)}) : Int) : Real)"

    if k == Kind.APPLY_SELECTOR:
        obj = t[1]
        if obj.getKind() != Kind.CONSTANT or obj.getSymbol() not in var_wire:
            raise Refused(
                f"APPLY_SELECTOR against non-bound term not supported: {t}"
            )
        idx = selector_index(t)
        sname = obj.getSymbol()
        wire = var_wire[sname]
        i, j = divmod(idx, wire_shape(wire).n)
        base = var_accessor[sname]
        return f"({base} {i} {j})"

    raise Refused(f"SMT→Lean: unsupported kind {k} in {t}")
