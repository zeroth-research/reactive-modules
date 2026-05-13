"""SMV (NuSMV 2.1 subset) parser — Lark frontend + lowerer to zrth Module."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from lark import Lark, Transformer, Tree, Token

from ..zrth import Arith, Wire, DType, IType, Term, Module


def _arith_itype(dtype, op):
    """Return the appropriate theory IType for an Arith op and a dtype."""
    return IType.mk(op, dtype)

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


_ASSIGN_TARGETS = {"init_ref": "init_assigns", "next_ref": "next_assigns", "bare_assign": "init_assigns"}


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
            return (_ASSIGN_TARGETS[target_node.data], str(target_node.children[0]), expr)
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

def _promote_loop(exprs, var_names, target: dict[str, Tree], *, expect_next=False) -> list[Tree]:
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
    d.trans_exprs = _promote_loop(d.trans_exprs, var_names, d.next_assigns, expect_next=True)

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

_SIMPLE_TYPES = {"type_bool": DType.Bool, "type_int": DType.Int, "range_type": DType.Int}


def _resolve_type(tree: Tree, enum_values: dict[str, int]) -> DType:
    if isinstance(tree, Tree):
        if tree.data in _SIMPLE_TYPES:
            return _SIMPLE_TYPES[tree.data]([1])
        match tree.data:
            case "type_spec":
                return _resolve_type(tree.children[0], enum_values)
            case "word_type":
                sign = str(tree.children[0])
                width = int(str(tree.children[1]))
                return DType.SWord(width) if sign == "signed" else DType.UWord(width)
            case "enum_type":
                for code, ev in enumerate(_typed_children(tree, "enum_value")):
                    name = _extract_name(ev.children[0])
                    if name is not None:
                        enum_values.setdefault(name, code)
                return DType.Int([1])
    # Token fallback
    text = str(tree).strip()
    if text == "boolean":
        return DType.Bool([1])
    if text == "integer":
        return DType.Int([1])
    raise ValueError(f"unsupported type: {tree}")


# ---------------------------------------------------------------------------
# 6. Lowerer
# ---------------------------------------------------------------------------

# Table-driven binary operator dispatch.
# Fixed-dtype ops store Ops directly; arith ops store lambda(dtype) -> Ops.
_BINOPS: dict[str, dict[str, object]] = {
    "iff":        {"<->": IType.Bool.Xnor, "xnor": IType.Bool.Xnor},
    "or_expr":    {"|": IType.Bool.Or, "or": IType.Bool.Or, "xor": IType.Bool.Xor},
    "and_expr":   {"&": IType.Bool.And, "and": IType.Bool.And},
    "cmp_expr":   {"=": IType.Cmp.Eq, "!=": IType.Cmp.Ne, "<": IType.Cmp.Lt,
                   "<=": IType.Cmp.Le, ">": IType.Cmp.Gt, ">=": IType.Cmp.Ge},
    "arith_expr": {"+": lambda d: _arith_itype(d, Arith.Add), "-": lambda d: _arith_itype(d, Arith.Sub)},
    "mod_expr":   {"mod": lambda d: _arith_itype(d, Arith.Mod)},
    "term_expr":  {"*": lambda d: _arith_itype(d, Arith.Mul), "/": lambda d: _arith_itype(d, Arith.Div)},
}

_OP_TOKENS = {"NOT_OP"}

_NARY_OPS = {
    "ternary": IType.Ite,
    "implies_expr": IType.Bool.Implies,
    "not_expr": IType.Bool.Not,
    "neg": lambda d: _arith_itype(d, Arith.Neg),
    "abs_call": lambda d: _arith_itype(d, Arith.Abs),
}

_METHODS = {
    "builtin_call": "_lower_builtin", "case_expr": "_lower_case",
    "next_expr": "_lower_next", "word_lit": "_lower_word_lit",
    "paren_expr": "_lower_paren",
}

_BUILTIN_MAP = {"bool": IType.BVToBool, "word1": IType.BVToWord1, "unsigned": IType.ToUnsigned, "signed": IType.ToSigned}


def _const_out_dtype(itype):
    """Return the 1x1 output DType for a scalar const IType, or None to defer to target."""
    if not itype.is_const:
        return None
    r = repr(itype)
    if r.startswith("Bool("):
        return DType.Bool([1])
    if r.startswith("Int("):
        return DType.Int([1])
    if r.startswith("Float("):
        return DType.Float([1])
    if r.startswith("Real("):
        return DType.Real([1])
    return None


def _builtin_out_dtype(itype, in_dtype):
    """Return the output DType for a BV cast/conversion builtin, or None."""
    shape = in_dtype.shape
    if itype is IType.BVToBool:
        return DType.Bool(shape)
    if itype is IType.BVToWord1:
        return DType.UWord(1).reshape(shape)
    if itype is IType.ToUnsigned:
        return DType.UWord(in_dtype.bv_bw()).reshape(shape)
    if itype is IType.ToSigned:
        return DType.SWord(in_dtype.bv_bw()).reshape(shape)
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
                return self._emit_const(IType.ConstInt(int(str(tree))), target)
            raise ValueError(f"unexpected token type: {tree.type}")

        name = tree.data

        # Table-driven binary ops
        if name in _BINOPS:
            out_dtype_fn = (lambda d: DType.Bool(d.shape)) if name == "cmp_expr" else (lambda d: d)
            return self._lower_binop(tree, _BINOPS[name], target, out_dtype_fn=out_dtype_fn)

        # Table-driven n-ary ops (unary / implies / ternary)
        if name in _NARY_OPS:
            return self._lower_nary(tree, _NARY_OPS[name], target)

        # Table-driven method dispatch
        if name in _METHODS:
            return getattr(self, _METHODS[name])(tree, target)

        if name == "var_ref":
            return self._lower_name(str(tree.children[0]), target)
        if name == "int_lit":
            return self._emit_const(IType.ConstInt(int(str(tree.children[0]))), target)
        if name in ("true_lit", "false_lit"):
            return self._emit_const(IType.ConstBool(name == "true_lit"), target)

        # Fallback: descend into single child
        if tree.children:
            return self.lower(tree.children[0], target)
        raise ValueError(f"unexpected empty tree: {tree.data}")

    # --- binary ops ---------------------------------------------------------

    def _lower_binop(self, tree: Tree, op_map: dict, target: Wire | None, *, out_dtype_fn=None) -> Wire:
        children = tree.children
        left = self.lower(children[0])
        i = 1
        while i < len(children):
            op_tok = str(children[i])
            rhs = self.lower(children[i + 1])
            itype_entry = op_map[op_tok]
            itype = itype_entry(left.dtype) if callable(itype_entry) else itype_entry
            out = target if i + 2 >= len(children) else None
            out_dtype = out_dtype_fn(left.dtype) if (out_dtype_fn and out is None) else None
            w = self._fresh(out, out_dtype)
            self.terms.append(Term(itype, [w], [left, rhs]))
            left = w
            i += 2
        return self._enforce(left, target)

    # --- n-ary (unary / implies / ternary) ----------------------------------

    def _lower_nary(self, tree: Tree, itype_or_fn, target: Wire | None) -> Wire:
        inputs = [self.lower(c) for c in tree.children if not (isinstance(c, Token) and c.type in _OP_TOKENS)]
        itype = itype_or_fn(inputs[0].dtype) if callable(itype_or_fn) else itype_or_fn
        # Ite: output matches then-branch (inputs[1]); others match first input
        out_dtype = inputs[1].dtype if (itype is IType.Ite and len(inputs) >= 2) else inputs[0].dtype
        w = self._fresh(target, out_dtype)
        self.terms.append(Term(itype, [w], inputs))
        return self._enforce(w, target)

    # --- builtin_call -------------------------------------------------------

    def _lower_builtin(self, tree: Tree, target: Wire | None) -> Wire:
        fn_name = str(tree.children[0])
        exprs = [c for c in tree.children[1:] if not (isinstance(c, Token) and c.type == "BUILTIN")]
        arg_w = self.lower(exprs[0])

        if fn_name in _BUILTIN_MAP:
            itype = _BUILTIN_MAP[fn_name]
            out_dtype = _builtin_out_dtype(itype, arg_w.dtype)
        elif fn_name == "extend":
            width = int(str(_peel(exprs[1])))
            itype = IType.Extend(width)
            in_bw = arg_w.dtype.bv_bw()
            in_signed = arg_w.dtype.bv_signed()
            out_bw = in_bw + width
            out_base = DType.SWord(out_bw) if in_signed else DType.UWord(out_bw)
            out_dtype = out_base.reshape(arg_w.dtype.shape)
        else:
            raise ValueError(f"unknown builtin: {fn_name}")

        w = self._fresh(target, out_dtype)
        self.terms.append(Term(itype, [w], [arg_w]))
        return self._enforce(w, target)

    # --- case..esac ---------------------------------------------------------

    def _lower_case(self, tree: Tree, target: Wire | None) -> Wire:
        branches = [c for c in tree.children if isinstance(c, Tree) and c.data == "case_branch"]
        return self._lower_case_rec(branches, 0, target)

    def _lower_case_rec(self, branches, idx, target):
        if idx >= len(branches):
            return self._emit_const(IType.ConstInt(0), target)
        cond_tree, val_tree = branches[idx].children
        if idx == len(branches) - 1:
            return self.lower(val_tree, target)
        cond = self.lower(cond_tree)
        then_w = self.lower(val_tree)
        else_w = self._lower_case_rec(branches, idx + 1, None)
        w = self._fresh(target, then_w.dtype)
        self.terms.append(Term(IType.Ite, [w], [cond, then_w, else_w]))
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
        value = int(rest[upos + 1:])

        dtype = DType.SWord(width) if signed else DType.UWord(width)
        w = Wire(dtype)
        self.terms.append(Term(IType.BV.Const([[value]]), [w]))
        return self._maybe_bit_select(tree, w, 1, target)

    # --- paren_expr ---------------------------------------------------------

    def _lower_paren(self, tree: Tree, target: Wire | None) -> Wire:
        inner = self.lower(tree.children[0], target if len(tree.children) == 1 else None)
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
            return self._emit_const(IType.ConstInt(self.enum_values[name]), target)
        # Wire lookup
        if name in self.name_map:
            latched, next_w = self.name_map[name]
            w = next_w if self.is_init else latched
            return self._enforce(w, target)
        # Unknown name — emit zero with warning
        warnings.warn(f"undeclared variable '{name}', defaulting to 0")
        return self._emit_const(IType.ConstInt(0), target)

    # --- helpers ------------------------------------------------------------

    def _fresh(self, target: Wire | None, dtype=None) -> Wire:
        if target is not None:
            return target
        return Wire(dtype if dtype is not None else DType.Int([1]))

    def _emit_const(self, itype, target: Wire | None) -> Wire:
        out_dtype = _const_out_dtype(itype)
        w = self._fresh(target, out_dtype)
        self.terms.append(Term(itype, [w]))
        return self._enforce(w, target)

    def _enforce(self, wire: Wire, target: Wire | None) -> Wire:
        if target is not None and target != wire:
            self.terms.append(Term(IType.Id, [target], [wire]))
            return target
        return wire

    def _maybe_bit_select(self, tree: Tree, base_w: Wire, idx: int, target: Wire | None) -> Wire:
        if len(tree.children) > idx and isinstance(tree.children[idx], Tree) and tree.children[idx].data == "bit_select":
            bs = tree.children[idx]
            high, low = int(str(bs.children[0])), int(str(bs.children[1]))
            out_dtype = DType.UWord(high - low + 1).reshape(base_w.dtype.shape)
            w = self._fresh(target, out_dtype)
            self.terms.append(Term(IType.BitSelect(high, low), [w], [base_w]))
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
        name_map[name] = overrides[name] if name in overrides else (Wire(dtype), Wire(dtype))

    init_low = _Lowerer(name_map, d.define_map, d.enum_values, is_init=True)
    update_low = _Lowerer(name_map, d.define_map, d.enum_values, is_init=False)

    for name, dtype in all_var_decls:
        latched, next_w = name_map[name]

        # --- INIT ---
        init_expr = d.init_assigns.get(name)
        if init_expr is not None:
            init_low.lower(init_expr, next_w)
        else:
            if dtype.is_bool():
                default = IType.ConstBool(False)
            elif dtype.is_bv():
                default = IType.BV.Const([[0]])
            else:
                default = IType.ConstInt(0)
            init_low.terms.append(Term(default, [next_w]))

        # --- UPDATE ---
        next_expr = d.next_assigns.get(name)
        if next_expr is not None:
            update_low.lower(next_expr, next_w)
        else:
            update_low.terms.append(Term(IType.Id, [next_w], [latched]))

    # Build obs_pairs: all variables (state + ivar)
    obs_pairs = [list(name_map[n]) for n, _ in all_decls]

    module = Module.sequential(init_low.terms, update_low.terms, obs=obs_pairs)
    return module, name_map
