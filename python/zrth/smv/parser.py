"""SMV (NuSMV 2.1 subset) parser — Lark frontend + lowerer to zrth Module."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
import torch
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from lark import Lark, Transformer, Tree, Token

from ..zrth import Wire, Term, Module, BV, LIA, Var, X
from ..sort import Sort, BitVec, Bool, Int


def _bw(sort) -> int:
    """Bit-width of a BitVec Sort (Sorts expose no methods; match instead)."""
    match sort:
        case BitVec(bw, _):
            return bw
    raise TypeError(f"not a BitVec sort: {sort}")


def _bv_const(value) -> "BV.Const":
    """A scalar BV constant op from an int value (2-D initializer)."""
    return BV.Const(torch.tensor([[int(value)]], dtype=torch.int64))


# ---------------------------------------------------------------------------
# 1. Lark parser init
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_parser():
    grammar = (Path(__file__).parent / "grammar.lark").read_text()
    return Lark(grammar, parser="earley", propagate_positions=True)


# ---------------------------------------------------------------------------
# 2. Entry point
# ---------------------------------------------------------------------------


class SMVRefused(ValueError):
    """A model `parse_smv(strict=True)` would read differently from NuSMV."""


@dataclass(frozen=True)
class Spec:
    """A property the model states, as a claim: `kind` "safety"
    (`INVARSPEC p`, `LTLSPEC G p`) or "buchi" (`LTLSPEC G F p`). `wire`
    carries `p`, read at the state after a round -- every variable's next
    value -- and `terms` compute it from those values."""
    kind: str
    where: str          # e.g. "line 12: INVARSPEC"
    wire: Wire
    terms: tuple


def read_smv(text: str) -> "tuple[Module, dict[str, Var], list[Spec]]":
    """`parse_smv(text, strict=True)`, and the properties the model states.

    A spec that is not `INVARSPEC p`, `LTLSPEC G p` or `LTLSPEC G F p` --
    any other temporal operator, or one inside `p` -- is refused, as is
    a `p` that reads an input."""
    tree = _get_parser().parse(text)
    decls = _collect_declarations(tree)
    _split_assigns(decls, strict=True)
    _promote_constraints(decls, strict=True)
    theory = _theory(tree)
    _check_strict(decls, theory)
    return _build_module(decls, {}, strict=True, theory=theory, specs=True)


def parse_smv(
        text: str,
        overrides: dict[str, tuple[Wire, Wire]] | None = None,
        *,
        strict: bool = False,
) -> tuple[Module, dict[str, tuple[Wire, Wire]]]:
    """Parse an SMV source string and return ``(Module, name_map)``.

    *name_map* maps every variable name to its ``(latched, next)`` wire pair.
    *overrides* lets the caller supply pre-existing wires for specific
    variables (used for module composition via shared wires).

    With *strict*, the module means what NuSMV means by the text, or
    `SMVRefused` says why it cannot: a constraint that is not an
    assignment, a fairness constraint, a variable left free (no ``init``
    or no ``next``), an integer type in a model with words (it would lower
    to a wrapping word). A model with no words is lowered to LIA: integers,
    ranges and enum codes are Int, booleans Bool.
    Inputs are read in the round they label, ``next(e)`` reads ``e`` at the
    next state, and ``INVAR x = e`` / ``x := e`` hold in every state.
    Without it, such text is read leniently: constraints dropped, free
    variables started at 0 and held.
    """
    tree = _get_parser().parse(text)
    decls = _collect_declarations(tree)
    _split_assigns(decls, strict=strict)
    _promote_constraints(decls, strict=strict)
    theory = _theory(tree) if strict else "bv"
    if strict:
        _check_strict(decls, theory)
    return _build_module(decls, overrides or {}, strict=strict,
                         theory=theory)[:2]


# ---------------------------------------------------------------------------
# 3. Pass 1 — collect declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SMVType:
    """SMV variable type: BV Sort paired with its signedness.

    `kind` is the SMV type it came from: "boolean", "word", "enum",
    "integer" (`integer`, a range, or an enum of numbers only) or
    "mixed-enum" (an enum of names and numbers)."""
    dtype: Sort
    signed: bool
    kind: str = "word"


@dataclass(slots=True)
class _Decls:
    var_decls: list[tuple[str, _SMVType]] = field(default_factory=list)
    ivar_decls: list[tuple[str, _SMVType]] = field(default_factory=list)
    frozen_decls: list[tuple[str, _SMVType]] = field(default_factory=list)
    define_map: dict[str, Tree] = field(default_factory=dict)
    # ASSIGN statements in source order: (kind, name, rhs), kind one of
    # "init" (`init(x) :=`), "next" (`next(x) :=`), "invar" (`x :=`).
    assigns: list[tuple[str, str, Tree]] = field(default_factory=list)
    init_assigns: dict[str, Tree] = field(default_factory=dict)
    next_assigns: dict[str, Tree] = field(default_factory=dict)
    # Strict only: `x` equals the rhs in every state, the initial one included.
    invar_assigns: dict[str, Tree] = field(default_factory=dict)
    init_exprs: list[Tree] = field(default_factory=list)
    trans_exprs: list[Tree] = field(default_factory=list)
    invar_exprs: list[Tree] = field(default_factory=list)
    fairness: list[tuple[str, int | None]] = field(default_factory=list)
    # (keyword, formula) for each INVARSPEC / LTLSPEC, in source order.
    specs: list[tuple[str, Tree]] = field(default_factory=list)
    enum_values: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def _typed_children(section: Tree, data: str):
    return (c for c in section.children if isinstance(c, Tree) and c.data == data)


def _line(node) -> int | None:
    if isinstance(node, Token):
        return node.line
    if isinstance(node, Tree) and not node.meta.empty:
        return node.meta.line
    return None


def _at(node) -> str:
    line = _line(node)
    return f"line {line}: " if line is not None else ""


_ASSIGN_TARGETS = {
    "init_ref": "init",
    "next_ref": "next",
    "bare_assign": "invar",
}


def _var_collector(attr):
    def method(self, children):
        target = getattr(self.decls, attr)
        for child in children:
            if isinstance(child, tuple):
                target.append(child)

    return method


class _DeclCollector(Transformer):
    """Lark Transformer that collects SMV declarations into a _Decls struct."""

    def __init__(self):
        super().__init__()
        self.decls = _Decls()

    # --- variable declarations ----------------------------------------------

    def var_decl(self, children):
        return (str(children[0]), _resolve_type(children[1], self.decls.enum_values))

    var_section = _var_collector("var_decls")
    ivar_section = _var_collector("ivar_decls")
    frozenvar_section = _var_collector("frozen_decls")

    # --- defines ------------------------------------------------------------

    def define_decl(self, children):
        return (str(children[0]), children[1])

    def define_section(self, children):
        for child in children:
            if isinstance(child, tuple):
                self.decls.define_map[child[0]] = child[1]

    # --- assignments --------------------------------------------------------

    def assign_stmt(self, children):
        target_node, expr = children[0], children[1]
        if isinstance(target_node, Tree) and target_node.data in _ASSIGN_TARGETS:
            return (
                _ASSIGN_TARGETS[target_node.data],
                str(target_node.children[0]),
                expr,
            )
        return ("invar", str(target_node), expr)

    def assign_section(self, children):
        for child in children:
            if isinstance(child, tuple) and len(child) == 3:
                self.decls.assigns.append(child)

    # --- constraint sections ------------------------------------------------

    def init_section(self, children):
        self.decls.init_exprs.append(children[0])

    def trans_section(self, children):
        self.decls.trans_exprs.append(children[0])

    def invar_section(self, children):
        self.decls.invar_exprs.append(children[0])

    def fairness_section(self, children):
        self.decls.fairness.append(("FAIRNESS", _line(children[0])))

    def justice_section(self, children):
        self.decls.fairness.append(("JUSTICE", _line(children[0])))

    def compassion_section(self, children):
        self.decls.fairness.append(("COMPASSION", _line(children[0])))

    # --- specifications -----------------------------------------------------

    def invarspec_section(self, children):
        self.decls.specs.append(("INVARSPEC", children[-1]))

    def ltlspec_section(self, children):
        self.decls.specs.append(("LTLSPEC", children[-1]))


def _collect_declarations(tree: Tree) -> _Decls:
    collector = _DeclCollector()
    collector.transform(tree)
    return collector.decls


def _split_assigns(d: _Decls, *, strict: bool) -> None:
    """Sort the ASSIGN statements into `init_assigns` / `next_assigns`
    (and, when strict, `invar_assigns`).

    Lenient: `x := e` is an initial value, and a later assignment to the
    same target replaces an earlier one. Strict: `x := e` holds in every
    state, and a second assignment to the same target is a problem.
    """
    for kind, name, expr in d.assigns:
        if not strict:
            target = d.next_assigns if kind == "next" else d.init_assigns
            target[name] = expr
            continue
        target = {"init": d.init_assigns, "next": d.next_assigns,
                  "invar": d.invar_assigns}[kind]
        if name in target:
            d.problems.append(f"{_at(expr)}`{name}` is assigned twice")
            continue
        target[name] = expr


# ---------------------------------------------------------------------------
# 4. Constraint-to-assignment promotion
# ---------------------------------------------------------------------------


def _promote_loop(
        exprs, var_names, target: dict[str, Tree], *, expect_next=False,
        taken=(),
) -> list[Tree]:
    remaining = []
    for expr in exprs:
        result = _try_extract_assign(expr, var_names, expect_next=expect_next)
        if result and result[0] not in target and result[0] not in taken:
            target[result[0]] = result[1]
        else:
            remaining.append(expr)
    return remaining


def _promote_constraints(d: _Decls, *, strict: bool = False) -> None:
    var_names = {n for n, _ in d.var_decls} | {n for n, _ in d.frozen_decls}
    taken = d.invar_assigns

    # TRANS: next(var) = expr -> next_assigns
    d.trans_exprs = _promote_loop(
        d.trans_exprs, var_names, d.next_assigns, expect_next=True,
        taken=taken,
    )

    # INIT: var = expr -> init_assigns
    d.init_exprs = _promote_loop(d.init_exprs, var_names, d.init_assigns,
                                 taken=taken)

    # INVAR: var = expr -> define_map + next_assigns (strict: invar_assigns)
    parsed = [_try_extract_assign(expr, var_names) for expr in d.invar_exprs]
    counts = Counter(p[0] for p in parsed if p)

    remaining = []
    for expr, p in zip(d.invar_exprs, parsed):
        if strict:
            if (p and counts[p[0]] == 1 and p[0] not in d.next_assigns
                    and p[0] not in d.init_assigns
                    and p[0] not in d.invar_assigns):
                d.invar_assigns[p[0]] = p[1]
            else:
                remaining.append(expr)
        elif p and counts[p[0]] == 1 and p[0] not in d.next_assigns:
            d.define_map[p[0]] = p[1]
            d.next_assigns[p[0]] = p[1]
        else:
            remaining.append(expr)
    d.invar_exprs = remaining


def _check_strict(d: _Decls, theory: str) -> None:
    """Raise `SMVRefused` listing everything strict parsing would read
    differently from NuSMV, as far as the declarations show it. What only
    the expressions show (an undeclared name, an input read where NuSMV
    does not allow one) is refused while lowering them."""
    problems = list(d.problems)
    for keyword, line in d.fairness:
        at = f"line {line}: " if line is not None else ""
        problems.append(f"{at}{keyword} constraints are not read")
    for keyword, form, exprs in (
            ("INIT", "`x = e`", d.init_exprs),
            ("TRANS", "`next(x) = e`", d.trans_exprs),
            ("INVAR", "`x = e`", d.invar_exprs)):
        for expr in exprs:
            problems.append(
                f"{_at(expr)}{keyword} is not {form} for a variable with no "
                "other such value"
            )

    var_names = [n for n, _ in d.var_decls]
    frozen = [n for n, _ in d.frozen_decls]
    ivars = {n for n, _ in d.ivar_decls}
    declared = set(var_names) | set(frozen)
    for assigns in (d.init_assigns, d.next_assigns, d.invar_assigns):
        for name, expr in assigns.items():
            if name in ivars:
                problems.append(f"{_at(expr)}input variable `{name}` is assigned")
            elif name not in declared:
                problems.append(f"{_at(expr)}`{name}` is assigned but not declared")
    for name in var_names + frozen:
        if name in d.invar_assigns:
            if name in d.init_assigns or name in d.next_assigns:
                problems.append(
                    f"`{name}` is assigned in every state and also its "
                    "initial or next value"
                )
            continue
        if name not in d.init_assigns:
            problems.append(f"`{name}` has no initial value (NuSMV leaves it free)")
        if name in frozen:
            if name in d.next_assigns:
                problems.append(f"FROZENVAR `{name}` is assigned a next value")
        elif name not in d.next_assigns:
            problems.append(
                f"`{name}` has no next value (NuSMV leaves it free at every step)"
            )

    for name, t in d.var_decls + d.frozen_decls + d.ivar_decls:
        if t.kind == "integer" and theory == "bv":
            problems.append(
                f"`{name}` has an integer type in a model with words, so it "
                f"would lower to a {_DEFAULT_INT_BW}-bit word and wrap"
            )
        elif t.kind == "mixed-enum":
            problems.append(f"`{name}`'s type mixes names and numbers")
    clashes = sorted(set(d.enum_values) & (declared | ivars | set(d.define_map)))
    for name in clashes:
        problems.append(f"`{name}` is both an enum value and a variable or DEFINE")

    if problems:
        raise SMVRefused(
            "the model cannot be read as NuSMV reads it:\n  "
            + "\n  ".join(problems)
        )


# Single-child nodes that are operators, not wrappers: `_peel` stops at them.
_UNARY_OPS = frozenset({"next_expr", "neg", "abs_call"})


def _peel(tree):
    """Peel single-child wrappers down to a leaf or a unary operator."""
    while (isinstance(tree, Tree) and len(tree.children) == 1
           and tree.data not in _UNARY_OPS):
        tree = tree.children[0]
    return tree


def _extract_name(node) -> str | None:
    """Get variable name from a peeled Lark node (var_ref Tree or IDENT Token)."""
    if isinstance(node, Tree) and node.data == "var_ref":
        return str(node.children[0])
    if isinstance(node, Token) and node.type == "IDENT":
        return str(node)
    return None


def _try_extract_assign(
        expr, var_names: set[str], *, expect_next: bool = False
) -> tuple[str, Tree] | None:
    """Detect ``var = rhs`` (or ``next(var) = rhs`` when *expect_next*) in an expression tree."""
    node = _peel(expr)
    if not isinstance(node, Tree):
        return None

    children = node.children
    if len(children) != 3:
        return None

    op = children[1]
    if not isinstance(op, Token) or str(op) != "=":
        return None

    lhs = _peel(children[0])
    if expect_next:
        if not isinstance(lhs, Tree) or lhs.data != "next_expr":
            return None
        lhs = _peel(lhs.children[0])

    name = _extract_name(lhs)
    return (name, children[2]) if name in var_names else None


# ---------------------------------------------------------------------------
# 5. Type resolution
# ---------------------------------------------------------------------------

# A model with words, and every model read leniently, lowers to bit-vector
# wires and BV ops (strict parsing lowers a model without words to LIA:
# `_LIALowerer`). The default width below is the BV lowering's for SMV's
# unbounded `integer` (and enum / range types). Adjust freely; the tests
# don't depend on the specific value.
_DEFAULT_INT_BW = 32


def _resolve_type(tree: Tree, enum_values: dict[str, int]) -> _SMVType:
    if isinstance(tree, Tree):
        match tree.data:
            case "type_bool":
                return _SMVType(BitVec(1, [1, 1]), signed=False, kind="boolean")
            case "type_int" | "range_type":
                return _SMVType(BitVec(_DEFAULT_INT_BW, [1, 1]), signed=True,
                                kind="integer")
            case "type_spec":
                return _resolve_type(tree.children[0], enum_values)
            case "word_type":
                sign_tok = str(tree.children[0])
                width = int(str(tree.children[1]))
                return _SMVType(BitVec(width, [1, 1]), signed=(sign_tok == "signed"))
            case "enum_type":
                names = numbers = 0
                for code, ev in enumerate(_typed_children(tree, "enum_value")):
                    name = _extract_name(ev.children[0])
                    if name is not None:
                        names += 1
                        enum_values.setdefault(name, code)
                    else:
                        numbers += 1
                kind = ("enum" if not numbers else
                        "integer" if not names else "mixed-enum")
                return _SMVType(BitVec(_DEFAULT_INT_BW, [1, 1]), signed=False,
                                kind=kind)
    # Token fallback
    text = str(tree).strip()
    if text == "boolean":
        return _SMVType(BitVec(1, [1, 1]), signed=False, kind="boolean")
    if text == "integer":
        return _SMVType(BitVec(_DEFAULT_INT_BW, [1, 1]), signed=True,
                        kind="integer")
    raise ValueError(f"unsupported type: {tree}")


# ---------------------------------------------------------------------------
# 6. Lowerer
# ---------------------------------------------------------------------------

# Table-driven binary operator dispatch. Values are op-name strings into
# the `IType.BV` namespace (`_LIALowerer` maps them to LIA). Resolution is
# deferred to term-construction time so missing ops produce a useful error.
_BINOPS: dict[str, dict[str, str]] = {
    "iff": {"<->": "Xnor", "xnor": "Xnor"},
    "or_expr": {"|": "Or", "or": "Or", "xor": "Xor"},
    "and_expr": {"&": "And", "and": "And"},
    "arith_expr": {"+": "Add", "-": "Sub"},
    "mod_expr": {"mod": "UMod"},
    "term_expr": {"*": "Mul", "/": "UDiv"},
}

# Comparison ops split by sign; handled in _lower_cmp / _cmp_itype.
_CMP_FIXED: dict[str, str] = {"=": "Eq", "!=": "Ne"}
_CMP_ORDERED: dict[str, str] = {"<": "Lt", "<=": "Le", ">": "Gt", ">=": "Ge"}

_OP_TOKENS = {"NOT_OP"}

_NARY_OPS = {
    "ternary": "Ite",
    "implies_expr": "Implies",
    "not_expr": "Not",
    "neg": "Neg",
    "abs_call": "Abs",
}


def _bv_op(name: str):
    """Construct `BV.<name>()` (raises AttributeError if the op is missing).

    Xnor and Implies, which the grammar maps name, are not BV ops: the
    lowerer composes them (Not(Xor), Or(Not, b)) before asking here.
    """
    return getattr(BV, name)()


_METHODS = {
    "builtin_call": "_lower_builtin",
    "case_expr": "_lower_case",
    "cmp_expr": "_lower_cmp",
    "next_expr": "_lower_next",
    "word_lit": "_lower_word_lit",
    "paren_expr": "_lower_paren",
}

_BUILTIN_MAP = {
    "bool": lambda: BV.BVToBool(),
    "word1": lambda: BV.Id(),
    "unsigned": lambda: BV.Id(),
    "signed": lambda: BV.Id(),
}


def _const_out_dtype(itype):
    """Output Sort for a scalar const op, or None to defer to the target wire."""
    # BV lowering: const width defers to the target/default width.
    return None


def _builtin_out_dtype(fn_name: str, in_dtype):
    """Return the output DType for a BV cast/conversion builtin."""
    if fn_name in ("bool", "word1"):
        return BitVec(1, [1, 1])
    if fn_name in ("unsigned", "signed"):
        return BitVec(_bw(in_dtype), [1, 1])
    return None


class _Lowerer:
    """Recursive expression-to-Term lowerer."""

    def __init__(
            self,
            name_map: dict[str, tuple[Wire, Wire]],
            defines: dict[str, Tree],
            enum_values: dict[str, int],
            var_types: dict[str, _SMVType],
            is_init: bool = False,
            *,
            strict: bool = False,
            ivars: frozenset[str] = frozenset(),
    ):
        self.name_map = name_map
        self.defines = defines
        self.enum_values = enum_values
        self.var_types = var_types
        self.is_init = is_init
        # Strict only: what a name reads. The init block reads every
        # variable's initial value and no input; the update block reads a
        # variable's latched value and the input of its own round, and
        # under `next()` (`at_next`) a variable's next value and no input.
        self.strict = strict
        self.ivars = ivars
        self.at_next = False
        self.context = ""
        self.terms: list[Term] = []
        self._expanding: set[str] = set()
        self._wire_types: dict[int, _SMVType] = {}

    def refuse(self, why: str, node=None) -> "SMVRefused":
        where = f" (in {self.context})" if self.context else ""
        return SMVRefused(f"{_at(node) if node is not None else ''}{why}{where}")

    # --- public entry -------------------------------------------------------

    def lower(self, tree, target: Wire | None = None) -> Wire:
        """Lower *tree* into Term(s), return the wire carrying the result."""
        # Token leaf (from Lark inlining)
        if isinstance(tree, Token):
            if tree.type == "IDENT":
                return self._lower_name(str(tree), target)
            if tree.type == "NUMBER":
                return self._lower_number(tree, target)
            raise ValueError(f"unexpected token type: {tree.type}")

        name = tree.data

        if name == "mod_expr" and len(tree.children) > 1:
            return self._lower_mod(tree, target)

        # Table-driven binary ops
        if name in _BINOPS:
            return self._lower_binop(
                tree, _BINOPS[name], target, out_dtype_fn=lambda d: d
            )

        # Table-driven n-ary ops (unary / implies / ternary)
        if name in _NARY_OPS:
            return self._lower_nary(tree, _NARY_OPS[name], target)

        # Table-driven method dispatch
        if name in _METHODS:
            return getattr(self, _METHODS[name])(tree, target)

        if name == "var_ref":
            return self._lower_name(str(tree.children[0]), target)
        if name == "int_lit":
            return self._lower_number(tree.children[0], target)
        if name in ("true_lit", "false_lit"):
            return self._lower_bool_lit(name == "true_lit", target)

        # Fallback: descend into single child
        if tree.children:
            return self.lower(tree.children[0], target)
        raise ValueError(f"unexpected empty tree: {tree.data}")

    # --- binary ops ---------------------------------------------------------

    def _lower_binop(
            self, tree: Tree, op_map: dict, target: Wire | None, *, out_dtype_fn=None
    ) -> Wire:
        children = tree.children
        left = self.lower(children[0])
        i = 1
        while i < len(children):
            op_tok = str(children[i])
            rhs = self.lower(children[i + 1])
            if (self.strict and op_tok == "/"
                    and (self._is_signed(left) or self._is_signed(rhs))):
                raise self.refuse("`/` on signed words is not read", children[i])
            operands = [left, rhs]
            if op_map[op_tok] == "Xnor":
                # No BV op: `a <-> b` is `!(a xor b)`.
                x = self._fresh(None, left.dtype)
                self.terms.append(Term(BV.Xor(), [x], operands))
                itype, operands = BV.Not(), [x]
            else:
                itype = _bv_op(op_map[op_tok])
            out = target if i + 2 >= len(children) else None
            out_dtype = (
                out_dtype_fn(left.dtype) if (out_dtype_fn and out is None) else None
            )
            w = self._fresh(out, out_dtype)
            self.terms.append(Term(itype, [w], operands))
            self._propagate_type(w, left, rhs)
            left = w
            i += 2
        return self._enforce(left, target)

    # --- comparison ops (sign-aware) ----------------------------------------

    def _cmp_itype(self, op_tok: str, left: Wire):
        if op_tok in _CMP_FIXED:
            return _bv_op(_CMP_FIXED[op_tok])
        prefix = "S" if self._is_signed(left) else "U"
        return _bv_op(prefix + _CMP_ORDERED[op_tok])

    def _lower_cmp(self, tree: Tree, target: Wire | None) -> Wire:
        children = tree.children
        left = self.lower(children[0])
        i = 1
        while i < len(children):
            op_tok = str(children[i])
            rhs = self.lower(children[i + 1])
            itype = self._cmp_itype(op_tok, left)
            out = target if i + 2 >= len(children) else None
            w = self._fresh(out, BitVec(1, [1, 1]))
            self.terms.append(Term(itype, [w], [left, rhs]))
            left = w
            i += 2
        return self._enforce(left, target)

    # --- n-ary (unary / implies / ternary) ----------------------------------

    def _lower_nary(self, tree: Tree, op_name: str, target: Wire | None) -> Wire:
        inputs = [
            self.lower(c)
            for c in tree.children
            if not (isinstance(c, Token) and c.type in _OP_TOKENS)
        ]
        if op_name == "Implies":
            # No BV op: `a -> b` is `!a | b`.
            not_a = self._fresh(None, inputs[0].dtype)
            self.terms.append(Term(BV.Not(), [not_a], [inputs[0]]))
            itype, inputs = BV.Or(), [not_a, inputs[1]]
        else:
            itype = _bv_op(op_name)
        # Ite: output matches then-branch (inputs[1]); others match first input
        out_dtype = (
            inputs[1].dtype
            if (op_name == "Ite" and len(inputs) >= 2)
            else inputs[0].dtype
        )
        w = self._fresh(target, out_dtype)
        self.terms.append(Term(itype, [w], inputs))
        return self._enforce(w, target)

    # --- builtin_call -------------------------------------------------------

    def _lower_builtin(self, tree: Tree, target: Wire | None) -> Wire:
        fn_name = str(tree.children[0])
        exprs = [
            c
            for c in tree.children[1:]
            if not (isinstance(c, Token) and c.type == "BUILTIN")
        ]
        arg_w = self.lower(exprs[0])

        if fn_name in _BUILTIN_MAP:
            itype = _BUILTIN_MAP[fn_name]()
            out_dtype = _builtin_out_dtype(fn_name, arg_w.dtype)
        elif fn_name == "extend":
            extra = int(str(_peel(exprs[1])))
            in_bw = _bw(arg_w.dtype)
            out_bw = in_bw + extra
            itype = BV.Extend(extra=extra)
            out_dtype = BitVec(out_bw, [1, 1])
        else:
            raise ValueError(f"unknown builtin: {fn_name}")

        w = self._fresh(target, out_dtype)
        self.terms.append(Term(itype, [w], [arg_w]))
        if fn_name in ("signed", "unsigned"):
            self._wire_types[id(w)] = _SMVType(w.dtype, signed=(fn_name == "signed"))
        return self._enforce(w, target)

    # --- case..esac ---------------------------------------------------------

    def _lower_case(self, tree: Tree, target: Wire | None) -> Wire:
        branches = [
            c for c in tree.children if isinstance(c, Tree) and c.data == "case_branch"
        ]
        return self._lower_case_rec(branches, 0, target)

    def _lower_case_rec(self, branches, idx, target):
        if idx >= len(branches):
            return self._emit_const(_bv_const(0), target)
        cond_tree, val_tree = branches[idx].children
        if idx == len(branches) - 1:
            # The last branch is taken whatever its condition: only right
            # when that condition is TRUE (NuSMV has no value otherwise).
            if self.strict and not _is_true(cond_tree):
                raise self.refuse(
                    "the last branch of a `case` must have condition TRUE",
                    cond_tree,
                )
            return self.lower(val_tree, target)
        cond = self.lower(cond_tree)
        then_w = self.lower(val_tree)
        else_w = self._lower_case_rec(branches, idx + 1, None)
        return self._ite(cond, then_w, else_w, target, cond_tree)

    def _ite(self, cond: Wire, a: Wire, b: Wire, target: Wire | None,
             node) -> Wire:
        w = self._fresh(target, a.dtype)
        self.terms.append(Term(BV.Ite(), [w], [cond, a, b]))
        return self._enforce(w, target)

    # --- next(expr) ---------------------------------------------------------

    def _lower_next(self, tree: Tree, target: Wire | None) -> Wire:
        inner = tree.children[0]
        if self.strict:
            if self.is_init or self.at_next:
                raise self.refuse("`next()` is not allowed here", tree)
            with self.next_values():
                return self.lower(inner, target)
        name = _extract_name(_peel(inner))
        if name and name in self.name_map:
            _, next_w = self.name_map[name]
            return self._enforce(next_w, target)
        return self.lower(inner, target)

    @contextmanager
    def next_values(self):
        """Names read their next values while inside (strict only)."""
        self.at_next = True
        try:
            yield
        finally:
            self.at_next = False

    # --- mod ----------------------------------------------------------------

    def _lower_mod(self, tree: Tree, target: Wire | None) -> Wire:
        # `a mod b mod c`: the grammar keeps no operator tokens here.
        operands = tree.children
        left = self.lower(operands[0])
        for i, operand in enumerate(operands[1:], start=1):
            rhs = self.lower(operand)
            if self.strict and (self._is_signed(left) or self._is_signed(rhs)):
                raise self.refuse("`mod` on signed words is not read", tree)
            out = target if i == len(operands) - 1 else None
            w = self._fresh(out, None if out is not None else left.dtype)
            self.terms.append(Term(BV.UMod(), [w], [left, rhs]))
            self._propagate_type(w, left, rhs)
            left = w
        return self._enforce(left, target)

    # --- numbers ------------------------------------------------------------

    def _lower_number(self, token, target: Wire | None) -> Wire:
        if self.strict:
            # Every type that holds a number is refused, so a bare number
            # is a word of whatever width it lands in.
            raise self.refuse(
                f"the number {token} is not a word constant (write e.g. "
                f"0ud8_{token})",
                token,
            )
        return self._emit_const(_bv_const(int(str(token))), target)

    # --- word literal -------------------------------------------------------

    def _lower_word_lit(self, tree: Tree, target: Wire | None) -> Wire:
        lit_tok = str(tree.children[0])
        # Parse e.g. "0ud16_0" or "0sd32_300"
        signed = lit_tok[1] == "s"
        rest = lit_tok[3:]  # skip "0ud" or "0sd"
        upos = rest.index("_")
        width = int(rest[:upos])
        value = int(rest[upos + 1:])

        if signed and value < 0:
            value = value + (1 << width)
        dtype = BitVec(width, [1, 1])
        w = Wire(dtype)
        self._wire_types[id(w)] = _SMVType(dtype, signed=signed)
        # Word literals produce a BV constant, not an LIA Const.
        self.terms.append(Term(_bv_const(value), [w]))
        return self._maybe_bit_select(tree, w, 1, target)

    # --- paren_expr ---------------------------------------------------------

    def _lower_paren(self, tree: Tree, target: Wire | None) -> Wire:
        inner = self.lower(
            tree.children[0], target if len(tree.children) == 1 else None
        )
        return self._maybe_bit_select(tree, inner, 1, target)

    # --- name (ident / define / enum) ---------------------------------------

    def _lower_name(self, name: str, target: Wire | None) -> Wire:
        # Check DEFINE
        if name in self.defines:
            if name in self._expanding:
                raise ValueError(f"circular DEFINE: {name}")
            self._expanding.add(name)
            result = self.lower(self.defines[name], target)
            self._expanding.discard(name)
            return result
        # Check enum
        if name in self.enum_values:
            return self._lower_code(self.enum_values[name], target)
        # Wire lookup
        if name in self.name_map:
            var = self.name_map[name]
            w = self._strict_read(name, var) if self.strict else (
                X(var) if self.is_init else var)
            if name in self.var_types:
                self._wire_types[id(w)] = self.var_types[name]
            return self._enforce(w, target)
        if self.strict:
            raise self.refuse(f"`{name}` is not declared")
        # Unknown name — emit zero with warning
        warnings.warn(f"undeclared variable '{name}', defaulting to 0")
        return self._emit_const(_bv_const(0), target)

    def _strict_read(self, name: str, var: Var) -> Wire:
        if name in self.ivars:
            if self.is_init or self.at_next:
                raise self.refuse(
                    f"input variable `{name}` is read where NuSMV allows no "
                    "input (an initial value, an INVAR or `x :=`, or under "
                    "`next()`)"
                )
            return X(var)
        return X(var) if (self.is_init or self.at_next) else var

    # --- helpers ------------------------------------------------------------

    def _smv_type(self, wire: Wire) -> _SMVType | None:
        return self._wire_types.get(id(wire))

    def _is_signed(self, wire: Wire) -> bool:
        t = self._smv_type(wire)
        return t.signed if t is not None else False

    def _propagate_type(self, result: Wire, *inputs: Wire) -> None:
        """Set result's _SMVType to signed if any input is signed."""
        signed = any(self._is_signed(w) for w in inputs)
        if signed or any(self._smv_type(w) is not None for w in inputs):
            self._wire_types[id(result)] = _SMVType(result.dtype, signed=signed)

    def _fresh(self, target: Wire | None, dtype=None) -> Wire:
        if target is not None:
            return target
        return Wire(dtype if dtype is not None else BitVec(_DEFAULT_INT_BW, [1, 1]))

    def _lower_bool_lit(self, value: bool, target: Wire | None) -> Wire:
        # Boolean literals live in BV<1>; the parser's `_DEFAULT_INT_BW`
        # fallback would otherwise widen them to BV<32>.
        w = self._fresh(target, BitVec(1, [1, 1]))
        self.terms.append(Term(_bv_const(1 if value else 0), [w]))
        return self._enforce(w, target)

    def _lower_code(self, code: int, target: Wire | None) -> Wire:
        return self._emit_const(_bv_const(code), target)

    def id_op(self):
        return BV.Id()

    def _emit_const(self, itype, target: Wire | None) -> Wire:
        out_dtype = _const_out_dtype(itype)
        w = self._fresh(target, out_dtype)
        self.terms.append(Term(itype, [w]))
        return self._enforce(w, target)

    def _enforce(self, wire: Wire, target: Wire | None) -> Wire:
        if target is not None and target != wire:
            self.terms.append(Term(self.id_op(), [target], [wire]))
            t = self._smv_type(wire)
            if t is not None:
                self._wire_types[id(target)] = t
            return target
        return wire

    def _maybe_bit_select(
            self, tree: Tree, base_w: Wire, idx: int, target: Wire | None
    ) -> Wire:
        if (
                len(tree.children) > idx
                and isinstance(tree.children[idx], Tree)
                and tree.children[idx].data == "bit_select"
        ):
            bs = tree.children[idx]
            high, low = int(str(bs.children[0])), int(str(bs.children[1]))
            out_dtype = BitVec(high - low + 1, [1, 1])
            w = self._fresh(target, out_dtype)
            self.terms.append(Term(BV.BitSelect(high=high, low=low), [w], [base_w]))
            return self._enforce(w, target)
        return self._enforce(base_w, target)


# ---------------------------------------------------------------------------
# 7. Pass 2 — build Module
# ---------------------------------------------------------------------------


_INT = Int([1, 1])
_BOOL = Bool([1, 1])
_LIA_CMP = {"=": LIA.Eq, "!=": LIA.Ne, "<": LIA.Lt, "<=": LIA.Le,
            ">": LIA.Gt, ">=": LIA.Ge}


def _lia_sort(t: _SMVType) -> Sort:
    return _BOOL if t.kind == "boolean" else _INT


def _sort_name(sort: Sort) -> str:
    match sort:
        case Bool(_):
            return "boolean"
        case Int(_):
            return "integer"
    return str(sort)


class _LIALowerer(_Lowerer):
    """Strict lowering of a model with no words, in LIA: integers, ranges
    and enum codes are Int, booleans Bool.

    NuSMV's integers are unbounded, so this is the reading that does not
    wrap. LIA multiplies only by a constant and has no division, so `*` of
    two non-constants, `/` and `mod` are refused, as is an operand of the
    wrong type (NuSMV refuses those too)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._int_consts: dict[Wire, int] = {}

    def id_op(self):
        return LIA.Id()

    def _fresh(self, target: Wire | None, dtype=None) -> Wire:
        return target if target is not None else Wire(dtype)

    def _emit(self, op, sort: Sort, inputs: list[Wire], target: Wire | None,
              node=None) -> Wire:
        if target is not None and _sort_name(target.dtype) != _sort_name(sort):
            raise self.refuse(
                f"a {_sort_name(sort)} where a {_sort_name(target.dtype)} "
                "is needed", node)
        w = self._fresh(target, sort)
        self.terms.append(Term(op, [w], inputs))
        return w

    def _need(self, w: Wire, sort: Sort, node) -> None:
        if _sort_name(w.dtype) != _sort_name(sort):
            raise self.refuse(
                f"a {_sort_name(w.dtype)} where a {_sort_name(sort)} is needed",
                node)

    def _enforce(self, wire: Wire, target: Wire | None) -> Wire:
        if target is not None and target != wire:
            return self._emit(LIA.Id(), wire.dtype, [wire], target)
        return wire

    # --- constants ------------------------------------------------------

    def _int_const(self, value: int, target: Wire | None, node=None) -> Wire:
        op = LIA.Int(torch.tensor([[int(value)]], dtype=torch.int64))
        w = self._emit(op, _INT, [], target, node)
        self._int_consts[w] = int(value)
        return w

    def _lower_number(self, token, target: Wire | None) -> Wire:
        return self._int_const(int(str(token)), target, token)

    def _lower_code(self, code: int, target: Wire | None) -> Wire:
        return self._int_const(code, target)

    def _lower_bool_lit(self, value: bool, target: Wire | None) -> Wire:
        op = LIA.Bool(torch.tensor([[bool(value)]], dtype=torch.bool))
        return self._emit(op, _BOOL, [], target)

    # --- operators ------------------------------------------------------

    def _scale(self, k: int, x: Wire, target: Wire | None, node) -> Wire:
        op = LIA.Linear(torch.tensor([[k]], dtype=torch.int64),
                        torch.zeros(1, 1, dtype=torch.int64))
        return self._emit(op, _INT, [x], target, node)

    def _lower_binop(self, tree: Tree, op_map: dict, target: Wire | None,
                     *, out_dtype_fn=None) -> Wire:
        children = tree.children
        left = self.lower(children[0])
        for i in range(1, len(children), 2):
            tok = children[i]
            rhs = self.lower(children[i + 1])
            out = target if i + 2 >= len(children) else None
            left = self._binop(op_map[str(tok)], left, rhs, out, tok)
        return self._enforce(left, target)

    def _binop(self, name: str, a: Wire, b: Wire, out: Wire | None,
               tok) -> Wire:
        if name in ("And", "Or", "Xor", "Xnor"):
            self._need(a, _BOOL, tok)
            self._need(b, _BOOL, tok)
            if name == "Xnor":
                x = self._emit(LIA.Xor(), _BOOL, [a, b], None)
                return self._emit(LIA.Not(), _BOOL, [x], out, tok)
            return self._emit(getattr(LIA, name)(), _BOOL, [a, b], out, tok)
        self._need(a, _INT, tok)
        self._need(b, _INT, tok)
        if name in ("Add", "Sub"):
            return self._emit(getattr(LIA, name)(), _INT, [a, b], out, tok)
        if name == "Mul":
            for c, x in ((a, b), (b, a)):
                if c in self._int_consts:
                    return self._scale(self._int_consts[c], x, out, tok)
            raise self.refuse(
                "`*` of two non-constants is not linear integer arithmetic",
                tok)
        raise self.refuse(f"`{tok}` is not linear integer arithmetic", tok)

    def _lower_mod(self, tree: Tree, target: Wire | None) -> Wire:
        raise self.refuse("`mod` is not linear integer arithmetic", tree)

    def _lower_cmp(self, tree: Tree, target: Wire | None) -> Wire:
        children = tree.children
        left = self.lower(children[0])
        for i in range(1, len(children), 2):
            tok = children[i]
            op = str(tok)
            rhs = self.lower(children[i + 1])
            out = target if i + 2 >= len(children) else None
            if op in ("=", "!=") and _sort_name(left.dtype) == "boolean":
                self._need(rhs, _BOOL, tok)
                if op == "!=":
                    left = self._emit(LIA.Xor(), _BOOL, [left, rhs], out, tok)
                else:
                    x = self._emit(LIA.Xor(), _BOOL, [left, rhs], None)
                    left = self._emit(LIA.Not(), _BOOL, [x], out, tok)
            else:
                self._need(left, _INT, tok)
                self._need(rhs, _INT, tok)
                left = self._emit(_LIA_CMP[op](), _BOOL, [left, rhs], out, tok)
        return self._enforce(left, target)

    def _lower_nary(self, tree: Tree, op_name: str, target: Wire | None) -> Wire:
        inputs = [
            self.lower(c)
            for c in tree.children
            if not (isinstance(c, Token) and c.type in _OP_TOKENS)
        ]
        if op_name == "Ite":
            return self._ite(*inputs, target, tree)
        if op_name == "Not":
            self._need(inputs[0], _BOOL, tree)
            return self._emit(LIA.Not(), _BOOL, inputs, target, tree)
        if op_name == "Implies":
            a, b = inputs
            self._need(a, _BOOL, tree)
            self._need(b, _BOOL, tree)
            not_a = self._emit(LIA.Not(), _BOOL, [a], None)
            return self._emit(LIA.Or(), _BOOL, [not_a, b], target, tree)
        (x,) = inputs
        self._need(x, _INT, tree)
        if op_name == "Neg":
            return self._scale(-1, x, target, tree)
        # Abs
        negative = self._emit(LIA.Lt(), _BOOL, [x, self._int_const(0, None)],
                              None)
        return self._ite(negative, self._scale(-1, x, None, tree), x, target,
                         tree)

    def _ite(self, cond: Wire, a: Wire, b: Wire, target: Wire | None,
             node) -> Wire:
        self._need(cond, _BOOL, node)
        self._need(b, a.dtype, node)
        return self._emit(LIA.Ite(), a.dtype, [cond, a, b], target, node)

    def _lower_builtin(self, tree: Tree, target: Wire | None) -> Wire:
        raise self.refuse(f"`{tree.children[0]}()` needs a word", tree)


def _theory(tree: Tree) -> str:
    """"bv" for a model with words (a word type, a word constant, a word
    builtin or a bit selection), else "lia"."""
    words = {"word_type", "word_lit", "builtin_call", "bit_select"}
    return "bv" if any(t.data in words for t in tree.iter_subtrees()) else "lia"


def _is_true(tree) -> bool:
    node = _peel(tree)
    return isinstance(node, Tree) and node.data == "true_lit"


def _build_module(
        d: _Decls,
        overrides: dict[str, Var],
        *,
        strict: bool = False,
        theory: str = "bv",
        specs: bool = False,
) -> "tuple[Module, dict[str, Var], list[Spec]]":
    # Prepend frozen vars before regular state vars
    all_var_decls = list(d.frozen_decls) + list(d.var_decls)

    # Combine: state vars first, then ivars
    all_decls = all_var_decls + list(d.ivar_decls)
    # Create wire pairs and type map
    name_map: dict[str, Var] = {}
    var_types: dict[str, _SMVType] = {}
    for name, smv_type in all_decls:
        var_types[name] = smv_type
        name_map[name] = (
            overrides[name]
            if name in overrides
            else Var(_lia_sort(smv_type) if theory == "lia" else smv_type.dtype)
        )

    ivars = frozenset(n for n, _ in d.ivar_decls)
    lowerer = _LIALowerer if theory == "lia" else _Lowerer
    init_low = lowerer(name_map, d.define_map, d.enum_values, var_types,
                       is_init=True, strict=strict, ivars=ivars)
    update_low = lowerer(name_map, d.define_map, d.enum_values, var_types,
                         is_init=False, strict=strict, ivars=ivars)

    for name, smv_type in all_var_decls:
        var = name_map[name]

        if strict:
            _lower_strict(d, name, var, init_low, update_low)
            continue

        # --- INIT ---
        init_expr = d.init_assigns.get(name)
        if init_expr is not None:
            init_low.lower(init_expr, X(var))
        else:
            init_low.terms.append(Term(_bv_const(0), [X(var)]))

        # --- UPDATE ---
        next_expr = d.next_assigns.get(name)
        if next_expr is not None:
            update_low.lower(next_expr, X(var))
        else:
            update_low.terms.append(Term(BV.Id(), [X(var)], [var]))

    # Build obs_pairs: all variables (state + ivar)
    obs_pairs = [name_map[n] for n, _ in all_decls]

    init_terms, update_terms = init_low.terms, update_low.terms
    if strict:
        init_terms = _in_dataflow_order(init_terms, "the initial state")
        update_terms = _in_dataflow_order(update_terms, "the next state")
    module = Module.sequential(obs_pairs, init_terms, update_terms)
    stated = []
    if specs:
        stated = [
            _lower_spec(keyword, formula, lowerer(
                name_map, d.define_map, d.enum_values, var_types,
                is_init=True, strict=True, ivars=ivars))
            for keyword, formula in d.specs
        ]
    return module, name_map, stated


def _lower_spec(keyword: str, formula, low: "_Lowerer") -> Spec:
    """`formula`'s claim, lowered by `low`: an init-block reader, so a
    variable reads its value after the round and an input is refused."""
    where = f"{_at(formula)}{keyword}"
    ops: list[str] = []
    body = formula
    if keyword == "LTLSPEC":
        while isinstance(body, Tree) and body.data == "ltl_op":
            ops.append(str(body.children[0]))
            body = body.children[1]
    kind = {(): "safety", ("G",): "safety", ("G", "F"): "buchi"}.get(
        tuple(ops) if keyword == "LTLSPEC" else (), None)
    if keyword == "LTLSPEC" and not ops:
        kind = None
    nested = isinstance(body, Tree) and any(
        t.data == "ltl_op" for t in body.iter_subtrees())
    if kind is None or nested:
        raise SMVRefused(
            f"{where}: only `INVARSPEC p`, `LTLSPEC G p` and `LTLSPEC G F p` "
            "are read as a claim"
        )
    low.context = where
    wire = low.lower(body)
    if _sort_name(wire.dtype) != "boolean" and wire.dtype != BitVec(1, [1, 1]):
        raise SMVRefused(f"{where}: the property is not a boolean")
    return Spec(kind, where, wire, tuple(low.terms))


def _in_dataflow_order(terms: list[Term], block: str) -> list[Term]:
    """`terms` with each wire written before it is read.

    A strict block reads next values (an INVAR, `next()`), so one
    variable's term may read a wire another variable's term writes later
    in declaration order."""
    writer = {w: i for i, t in enumerate(terms) for w in t.write}
    deps = [{writer[w] for w in t.read if w in writer} for t in terms]
    order: list[Term] = []
    state = [0] * len(terms)  # 0 unvisited, 1 on the stack, 2 placed
    for root in range(len(terms)):
        if state[root]:
            continue
        state[root] = 1
        stack = [(root, iter(sorted(deps[root])))]
        while stack:
            i, it = stack[-1]
            j = next(it, None)
            if j is None:
                stack.pop()
                state[i] = 2
                order.append(terms[i])
            elif state[j] == 1:
                raise SMVRefused(
                    f"{block} depends on itself (a cycle through `next()` "
                    "or INVAR)"
                )
            elif state[j] == 0:
                state[j] = 1
                stack.append((j, iter(sorted(deps[j]))))
    return order


def _lower_strict(d: _Decls, name: str, var: Var,
                  init_low: _Lowerer, update_low: _Lowerer) -> None:
    """`name`'s initial and next value, which `_check_strict` has made sure
    the text gives (a FROZENVAR's next value is its own)."""
    invar = d.invar_assigns.get(name)
    if invar is not None:
        # In every state, so the next value is the rhs at the next state.
        init_low.context = update_low.context = f"`{name}` in every state"
        init_low.lower(invar, X(var))
        with update_low.next_values():
            update_low.lower(invar, X(var))
        return
    init_low.context = f"init({name})"
    init_low.lower(d.init_assigns[name], X(var))
    update_low.context = f"next({name})"
    next_expr = d.next_assigns.get(name)
    if next_expr is not None:
        update_low.lower(next_expr, X(var))
    else:
        update_low.terms.append(Term(update_low.id_op(), [X(var)], [var]))
