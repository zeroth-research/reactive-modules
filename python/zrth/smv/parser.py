"""SMV (NuSMV 2.1 subset) parser — Lark frontend + lowerer to zrth Module."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from lark import Lark, Transformer, Tree, Token

from ..zrth import Wire, DType, Term, Module
from .. import IType

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


def parse_smv(
    text: str,
    overrides: dict[str, tuple[Wire, Wire]] | None = None,
) -> tuple[Module, dict[str, tuple[Wire, Wire]]]:
    """Parse an SMV source string and return ``(Module, name_map)``.

    *name_map* maps every variable name to its ``(latched, next)`` wire pair.
    *overrides* lets the caller supply pre-existing wires for specific
    variables (used for module composition via shared wires).
    """
    tree = _get_parser().parse(text)
    decls = _collect_declarations(tree)
    _promote_constraints(decls)
    return _build_module(decls, overrides or {})


# ---------------------------------------------------------------------------
# 3. Pass 1 — collect declarations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Decls:
    var_decls: list[tuple[str, DType]] = field(default_factory=list)
    ivar_decls: list[tuple[str, DType]] = field(default_factory=list)
    frozen_decls: list[tuple[str, DType]] = field(default_factory=list)
    define_map: dict[str, Tree] = field(default_factory=dict)
    init_assigns: dict[str, Tree] = field(default_factory=dict)
    next_assigns: dict[str, Tree] = field(default_factory=dict)
    init_exprs: list[Tree] = field(default_factory=list)
    trans_exprs: list[Tree] = field(default_factory=list)
    invar_exprs: list[Tree] = field(default_factory=list)
    enum_values: dict[str, int] = field(default_factory=dict)


def _typed_children(section: Tree, data: str):
    return (c for c in section.children if isinstance(c, Tree) and c.data == data)


_ASSIGN_TARGETS = {
    "init_ref": "init_assigns",
    "next_ref": "next_assigns",
    "bare_assign": "init_assigns",
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
        return ("init_assigns", str(target_node), expr)

    def assign_section(self, children):
        for child in children:
            if isinstance(child, tuple) and len(child) == 3:
                attr, name, expr = child
                getattr(self.decls, attr)[name] = expr

    # --- constraint sections ------------------------------------------------

    def init_section(self, children):
        self.decls.init_exprs.append(children[0])

    def trans_section(self, children):
        self.decls.trans_exprs.append(children[0])

    def invar_section(self, children):
        self.decls.invar_exprs.append(children[0])


def _collect_declarations(tree: Tree) -> _Decls:
    collector = _DeclCollector()
    collector.transform(tree)
    return collector.decls


# ---------------------------------------------------------------------------
# 4. Constraint-to-assignment promotion
# ---------------------------------------------------------------------------


def _promote_loop(
    exprs, var_names, target: dict[str, Tree], *, expect_next=False
) -> list[Tree]:
    remaining = []
    for expr in exprs:
        result = _try_extract_assign(expr, var_names, expect_next=expect_next)
        if result and result[0] not in target:
            target[result[0]] = result[1]
        else:
            remaining.append(expr)
    return remaining


def _promote_constraints(d: _Decls) -> None:
    var_names = {n for n, _ in d.var_decls} | {n for n, _ in d.frozen_decls}

    # TRANS: next(var) = expr -> next_assigns
    d.trans_exprs = _promote_loop(
        d.trans_exprs, var_names, d.next_assigns, expect_next=True
    )

    # INIT: var = expr -> init_assigns
    d.init_exprs = _promote_loop(d.init_exprs, var_names, d.init_assigns)

    # INVAR: var = expr -> define_map + next_assigns
    parsed = [_try_extract_assign(expr, var_names) for expr in d.invar_exprs]
    counts = Counter(p[0] for p in parsed if p)

    remaining = []
    for expr, p in zip(d.invar_exprs, parsed):
        if p and counts[p[0]] == 1 and p[0] not in d.next_assigns:
            d.define_map[p[0]] = p[1]
            d.next_assigns[p[0]] = p[1]
        else:
            remaining.append(expr)
    d.invar_exprs = remaining


def _peel(tree):
    """Peel single-child wrappers down to a leaf."""
    while isinstance(tree, Tree) and len(tree.children) == 1:
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

# The SMV parser is BV-only: every SMV value lowers to a bit-vector wire,
# every emitted op is in the BV theory. The default width below is used for
# SMV's unbounded `integer` (and enum / range types). Adjust freely; the
# tests don't depend on the specific value.
_DEFAULT_INT_BW = 32


def _resolve_type(tree: Tree, enum_values: dict[str, int]) -> DType:
    if isinstance(tree, Tree):
        match tree.data:
            case "type_bool":
                return DType.BV(1)
            case "type_int" | "range_type":
                return DType.BV(_DEFAULT_INT_BW)
            case "type_spec":
                return _resolve_type(tree.children[0], enum_values)
            case "word_type":
                width = int(str(tree.children[1]))
                return DType.BV(width)
            case "enum_type":
                for code, ev in enumerate(_typed_children(tree, "enum_value")):
                    name = _extract_name(ev.children[0])
                    if name is not None:
                        enum_values.setdefault(name, code)
                return DType.BV(_DEFAULT_INT_BW)
    # Token fallback
    text = str(tree).strip()
    if text == "boolean":
        return DType.BV(1)
    if text == "integer":
        return DType.BV(_DEFAULT_INT_BW)
    raise ValueError(f"unsupported type: {tree}")


# ---------------------------------------------------------------------------
# 6. Lowerer
# ---------------------------------------------------------------------------

# Table-driven binary operator dispatch. Values are op-name strings into
# the `IType.BV` namespace (the SMV parser is BV-only). Resolution is
# deferred to term-construction time so missing ops produce a useful error.
_BINOPS: dict[str, dict[str, str]] = {
    "iff": {"<->": "Xnor", "xnor": "Xnor"},
    "or_expr": {"|": "Or", "or": "Or", "xor": "Xor"},
    "and_expr": {"&": "And", "and": "And"},
    "cmp_expr": {
        "=": "Eq",
        "!=": "Ne",
        "<": "Lt",
        "<=": "Le",
        ">": "Gt",
        ">=": "Ge",
    },
    "arith_expr": {"+": "Add", "-": "Sub"},
    "mod_expr": {"mod": "UMod"},
    "term_expr": {"*": "Mul", "/": "UDiv"},
}

_OP_TOKENS = {"NOT_OP"}

_NARY_OPS = {
    "ternary": "Ite",
    "implies_expr": "Implies",
    "not_expr": "Not",
    "neg": "Neg",
    "abs_call": "Abs",
}


def _bv_op(name: str):
    """Look up `IType.BV.<name>` (raises AttributeError if missing)."""
    return getattr(IType.BV, name)

_METHODS = {
    "builtin_call": "_lower_builtin",
    "case_expr": "_lower_case",
    "next_expr": "_lower_next",
    "word_lit": "_lower_word_lit",
    "paren_expr": "_lower_paren",
}

_BUILTIN_MAP = {
    "bool": lambda: IType.BV.BVToBool,
    "word1": lambda: IType.BV.Id,
    "unsigned": lambda: IType.BV.Id,
    "signed": lambda: IType.BV.Id,
}


def _const_out_dtype(itype):
    """Return the output DType for a scalar const IType, or None to defer to target."""
    # BV-only parser: every Const is BV. A boolean-valued constant is BV<1>;
    # otherwise the constant defaults to the integer width.
    if itype.op_name != "Const":
        return None
    return None


def _builtin_out_dtype(fn_name: str, in_dtype):
    """Return the output DType for a BV cast/conversion builtin."""
    if fn_name in ("bool", "word1"):
        return DType.BV(1)
    if fn_name in ("unsigned", "signed"):
        return DType.BV(in_dtype.bv_bitwidth())
    return None


class _Lowerer:
    """Recursive expression-to-Term lowerer."""

    def __init__(
        self,
        name_map: dict[str, tuple[Wire, Wire]],
        defines: dict[str, Tree],
        enum_values: dict[str, int],
        is_init: bool = False,
    ):
        self.name_map = name_map
        self.defines = defines
        self.enum_values = enum_values
        self.is_init = is_init
        self.terms: list[Term] = []
        self._expanding: set[str] = set()

    # --- public entry -------------------------------------------------------

    def lower(self, tree, target: Wire | None = None) -> Wire:
        """Lower *tree* into Term(s), return the wire carrying the result."""
        # Token leaf (from Lark inlining)
        if isinstance(tree, Token):
            if tree.type == "IDENT":
                return self._lower_name(str(tree), target)
            if tree.type == "NUMBER":
                return self._emit_const(IType.BV.Const(int(str(tree))), target)
            raise ValueError(f"unexpected token type: {tree.type}")

        name = tree.data

        # Table-driven binary ops
        if name in _BINOPS:
            out_dtype_fn = (
                (lambda d: DType.BV(1)) if name == "cmp_expr" else (lambda d: d)
            )
            return self._lower_binop(
                tree, _BINOPS[name], target, out_dtype_fn=out_dtype_fn
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
            return self._emit_const(IType.BV.Const(int(str(tree.children[0]))), target)
        if name in ("true_lit", "false_lit"):
            # Boolean literals live in BV<1>; the parser's `_DEFAULT_INT_BW`
            # fallback would otherwise widen them to BV<32>.
            w = self._fresh(target, DType.BV(1))
            self.terms.append(
                Term(IType.BV.Const(1 if name == "true_lit" else 0), [w])
            )
            return self._enforce(w, target)

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
            itype = _bv_op(op_map[op_tok])
            out = target if i + 2 >= len(children) else None
            out_dtype = (
                out_dtype_fn(left.dtype) if (out_dtype_fn and out is None) else None
            )
            w = self._fresh(out, out_dtype)
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
            in_bw = arg_w.dtype.bv_bitwidth()
            out_bw = in_bw + extra
            itype = IType.BV.Extend(extra)
            out_dtype = DType.BV(out_bw)
        else:
            raise ValueError(f"unknown builtin: {fn_name}")

        w = self._fresh(target, out_dtype)
        self.terms.append(Term(itype, [w], [arg_w]))
        return self._enforce(w, target)

    # --- case..esac ---------------------------------------------------------

    def _lower_case(self, tree: Tree, target: Wire | None) -> Wire:
        branches = [
            c for c in tree.children if isinstance(c, Tree) and c.data == "case_branch"
        ]
        return self._lower_case_rec(branches, 0, target)

    def _lower_case_rec(self, branches, idx, target):
        if idx >= len(branches):
            return self._emit_const(IType.BV.Const(0), target)
        cond_tree, val_tree = branches[idx].children
        if idx == len(branches) - 1:
            return self.lower(val_tree, target)
        cond = self.lower(cond_tree)
        then_w = self.lower(val_tree)
        else_w = self._lower_case_rec(branches, idx + 1, None)
        w = self._fresh(target, then_w.dtype)
        self.terms.append(Term(IType.BV.Ite, [w], [cond, then_w, else_w]))
        return self._enforce(w, target)

    # --- next(expr) ---------------------------------------------------------

    def _lower_next(self, tree: Tree, target: Wire | None) -> Wire:
        inner = tree.children[0]
        name = _extract_name(_peel(inner))
        if name and name in self.name_map:
            _, next_w = self.name_map[name]
            return self._enforce(next_w, target)
        return self.lower(inner, target)

    # --- word literal -------------------------------------------------------

    def _lower_word_lit(self, tree: Tree, target: Wire | None) -> Wire:
        lit_tok = str(tree.children[0])
        # Parse e.g. "0ud16_0" or "0sd32_300"
        signed = lit_tok[1] == "s"
        rest = lit_tok[3:]  # skip "0ud" or "0sd"
        upos = rest.index("_")
        width = int(rest[:upos])
        value = int(rest[upos + 1 :])

        if signed and value < 0:
            value = value + (1 << width)
        dtype = DType.BV(width)
        w = Wire(dtype)
        # Word literals produce a BV constant, not an LIA ConstInt.
        self.terms.append(Term(IType.BV.Const(value), [w]))
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
            return self._emit_const(IType.BV.Const(self.enum_values[name]), target)
        # Wire lookup
        if name in self.name_map:
            latched, next_w = self.name_map[name]
            w = next_w if self.is_init else latched
            return self._enforce(w, target)
        # Unknown name — emit zero with warning
        warnings.warn(f"undeclared variable '{name}', defaulting to 0")
        return self._emit_const(IType.BV.Const(0), target)

    # --- helpers ------------------------------------------------------------

    def _fresh(self, target: Wire | None, dtype=None) -> Wire:
        if target is not None:
            return target
        return Wire(dtype if dtype is not None else DType.BV(_DEFAULT_INT_BW))

    def _emit_const(self, itype, target: Wire | None) -> Wire:
        out_dtype = _const_out_dtype(itype)
        w = self._fresh(target, out_dtype)
        self.terms.append(Term(itype, [w]))
        return self._enforce(w, target)

    def _enforce(self, wire: Wire, target: Wire | None) -> Wire:
        if target is not None and target != wire:
            self.terms.append(Term(IType.BV.Id, [target], [wire]))
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
            out_dtype = DType.BV(high - low + 1)
            w = self._fresh(target, out_dtype)
            self.terms.append(Term(IType.BV.BitSelect(high, low), [w], [base_w]))
            return self._enforce(w, target)
        return self._enforce(base_w, target)


# ---------------------------------------------------------------------------
# 7. Pass 2 — build Module
# ---------------------------------------------------------------------------


def _build_module(
    d: _Decls,
    overrides: dict[str, tuple[Wire, Wire]],
) -> tuple[Module, dict[str, tuple[Wire, Wire]]]:
    # Prepend frozen vars before regular state vars
    all_var_decls = list(d.frozen_decls) + list(d.var_decls)

    # Combine: state vars first, then ivars
    all_decls = all_var_decls + list(d.ivar_decls)
    # Create wire pairs
    name_map: dict[str, tuple[Wire, Wire]] = {}
    for name, dtype in all_decls:
        name_map[name] = (
            overrides[name] if name in overrides else (Wire(dtype), Wire(dtype))
        )

    init_low = _Lowerer(name_map, d.define_map, d.enum_values, is_init=True)
    update_low = _Lowerer(name_map, d.define_map, d.enum_values, is_init=False)

    for name, dtype in all_var_decls:
        latched, next_w = name_map[name]

        # --- INIT ---
        init_expr = d.init_assigns.get(name)
        if init_expr is not None:
            init_low.lower(init_expr, next_w)
        else:
            init_low.terms.append(Term(IType.BV.Const(0), [next_w]))

        # --- UPDATE ---
        next_expr = d.next_assigns.get(name)
        if next_expr is not None:
            update_low.lower(next_expr, next_w)
        else:
            update_low.terms.append(Term(IType.BV.Id, [next_w], [latched]))

    # Build obs_pairs: all variables (state + ivar)
    obs_pairs = [list(name_map[n]) for n, _ in all_decls]

    module = Module.sequential(init_low.terms, update_low.terms, obs=obs_pairs)
    return module, name_map
