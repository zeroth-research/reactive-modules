"""Predicates read against a module: SMT-LIB, or the Python DSL.

`PredicateParser` declares the module's symbols in cvc5 once -- `s0..sN-1`
for the ctrl-next state components (sorted per wire), `e0..` for the
extl-next inputs and `el0..` for the extl-latched ones -- and parses an
SMT-LIB expression over them. `parse_predicate` reads a predicate in either
dialect: the Python-expression DSL first, which is walked rather than
`eval`-ed, then SMT-LIB. The run's `smt.encoding.ModuleEncoding` holds the
one parser every predicate is read with.
"""

from __future__ import annotations

import ast
import operator
import re

import cvc5
from cvc5 import Kind

from ..sorts import wire_shape
from .encode import wire_sort
from .module import ModuleSMT


def _wire_rc(wire) -> tuple[int, int]:
    """(rows, cols) of a wire's matrix shape (`sorts.wire_shape`)."""
    shape = wire_shape(wire)
    return shape.m, shape.n


# ---------------------------------------------------------------------
#   Python-expression DSL over cvc5 terms
# ---------------------------------------------------------------------


class _E:
    """Operator-overloading wrapper around a cvc5 Term.

    Lets the user write `s0 > 0`, `s0 == s4`, `a + b`, `a & b` etc. in
    Python, producing cvc5 Terms under the hood. Used as a convenience
    layer for `--safety` / `--buchi` / `--pre` / `--invariant` / `--ranking`.
    """

    __slots__ = ("_tm", "_t")

    def __init__(self, tm: cvc5.TermManager, t: cvc5.Term):
        self._tm = tm
        self._t = t

    @property
    def t(self) -> cvc5.Term:
        return self._t

    def _coerce(self, x):
        if isinstance(x, _E):
            return x._t
        if isinstance(x, bool):
            return self._tm.mkBoolean(x)
        if isinstance(x, int):
            # Against a BitVec-sorted term, a bare Python int must become a
            # same-width bitvector literal — cvc5 has no Int→BV coercion, so
            # the SMT-LIB default (arbitrary-precision Int) would mismatch.
            sort = self._t.getSort()
            if sort.isBitVector():
                return self._tm.mkBitVector(sort.getBitVectorSize(), x)
            return self._tm.mkInteger(x)
        if isinstance(x, float):
            return self._tm.mkReal(x)
        if isinstance(x, cvc5.Term):
            return x
        raise TypeError(f"Cannot coerce {type(x).__name__} to a cvc5 Term")

    def _bin(self, kind, other, rev=False):
        a, b = self._coerce(other), self._t
        if not rev:
            a, b = b, a
        return _E(self._tm, self._tm.mkTerm(kind, a, b))

    def __gt__(self, o):  return self._bin(Kind.GT, o)
    def __lt__(self, o):  return self._bin(Kind.LT, o)
    def __ge__(self, o):  return self._bin(Kind.GEQ, o)
    def __le__(self, o):  return self._bin(Kind.LEQ, o)
    def __eq__(self, o):  return self._bin(Kind.EQUAL, o)
    def __ne__(self, o):  return self._bin(Kind.DISTINCT, o)
    def __add__(self, o): return self._bin(Kind.ADD, o)
    def __radd__(self, o): return self._bin(Kind.ADD, o, rev=True)
    def __sub__(self, o): return self._bin(Kind.SUB, o)
    def __rsub__(self, o): return self._bin(Kind.SUB, o, rev=True)
    def __mul__(self, o): return self._bin(Kind.MULT, o)
    def __rmul__(self, o): return self._bin(Kind.MULT, o, rev=True)
    def __and__(self, o):
        kind = Kind.BITVECTOR_AND if self._t.getSort().isBitVector() else Kind.AND
        return self._bin(kind, o)
    def __rand__(self, o):
        kind = Kind.BITVECTOR_AND if self._t.getSort().isBitVector() else Kind.AND
        return self._bin(kind, o, rev=True)
    def __or__(self, o):
        kind = Kind.BITVECTOR_OR if self._t.getSort().isBitVector() else Kind.OR
        return self._bin(kind, o)
    def __ror__(self, o):
        kind = Kind.BITVECTOR_OR if self._t.getSort().isBitVector() else Kind.OR
        return self._bin(kind, o, rev=True)
    def __neg__(self):
        kind = Kind.BITVECTOR_NEG if self._t.getSort().isBitVector() else Kind.NEG
        return _E(self._tm, self._tm.mkTerm(kind, self._t))
    def __invert__(self):
        kind = Kind.BITVECTOR_NOT if self._t.getSort().isBitVector() else Kind.NOT
        return _E(self._tm, self._tm.mkTerm(kind, self._t))
    def __hash__(self):   return id(self._t)

    # Tuple (matrix) element selection: `e0[k]` on a tuple-sorted var.
    def __getitem__(self, k: int) -> "_E":
        if not isinstance(k, int):
            raise TypeError("index must be int")
        sort = self._t.getSort()
        if not sort.isTuple():
            raise TypeError(f"{self._t} is not a tuple; cannot index")
        ctor = sort.getDatatype()[0]
        sel = ctor[k].getTerm()
        return _E(self._tm, self._tm.mkTerm(Kind.APPLY_SELECTOR, sel, self._t))


class _Row:
    """Intermediate for 2-D indexing: `v[i]` → `_Row`, then `v[i][j]` → element."""

    __slots__ = ("_var", "_i")

    def __init__(self, var: "_MatVar", i: int):
        self._var, self._i = var, i

    def __getitem__(self, j: int) -> "_E":
        return self._var._element(self._i, j)


class _MatVar(_E):
    """A state/input variable that knows its matrix shape, so predicates can
    address elements as ``v[i][j]`` (or ``v[i, j]``) into the row-major flat
    tuple. ``v`` alone is still the whole matrix term; a scalar (1×1) var
    accepts only ``v[0][0]``.
    """

    __slots__ = ("_rows", "_cols")

    def __init__(self, tm: cvc5.TermManager, t: cvc5.Term, rows: int, cols: int):
        super().__init__(tm, t)
        self._rows, self._cols = rows, cols

    def _element(self, i: int, j: int) -> "_E":
        if not (0 <= i < self._rows and 0 <= j < self._cols):
            raise IndexError(
                f"index ({i}, {j}) out of bounds for {self._rows}×{self._cols} matrix"
            )
        idx = i * self._cols + j
        sort = self._t.getSort()
        if not sort.isTuple():
            # scalar (1×1) var — the term itself is the single element
            return _E(self._tm, self._t)
        ctor = sort.getDatatype()[0]
        sel = ctor[idx].getTerm()
        return _E(self._tm, self._tm.mkTerm(Kind.APPLY_SELECTOR, sel, self._t))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            return self._element(i, j)
        return _Row(self, key)


def _pyeval_namespace(env: "PredicateParser") -> dict:
    """Build the namespace exposed to Python-expression predicates."""
    tm = env.tm

    def _unwrap(x, ref_sort=None):
        if isinstance(x, _E):
            return x._t
        if isinstance(x, bool):
            return tm.mkBoolean(x)
        if isinstance(x, int):
            # Match a BitVec sibling's width — see `_E._coerce`.
            if ref_sort is not None and ref_sort.isBitVector():
                return tm.mkBitVector(ref_sort.getBitVectorSize(), x)
            return tm.mkInteger(x)
        if isinstance(x, float):
            return tm.mkReal(x)
        return x

    def _mk(kind, *args):
        ref_sort = None
        for a in args:
            if isinstance(a, _E):
                ref_sort = a._t.getSort()
                break
            if isinstance(a, cvc5.Term):
                ref_sort = a.getSort()
                break
        return _E(tm, tm.mkTerm(kind, *[_unwrap(a, ref_sort) for a in args]))

    ns = {
        "And": lambda *a: _mk(Kind.AND, *a),
        "Or": lambda *a: _mk(Kind.OR, *a),
        "Not": lambda a: _mk(Kind.NOT, a),
        "Ite": lambda c, t, e: _mk(Kind.ITE, c, t, e),
        "Implies": lambda a, b: _mk(Kind.IMPLIES, a, b),
        "Eq": lambda a, b: _mk(Kind.EQUAL, a, b),
        "Ne": lambda a, b: _mk(Kind.DISTINCT, a, b),
        "true": _E(tm, tm.mkBoolean(True)),
        "false": _E(tm, tm.mkBoolean(False)),
        "True": _E(tm, tm.mkBoolean(True)),
        "False": _E(tm, tm.mkBoolean(False)),
        "ToInt": lambda a: _mk(Kind.TO_INTEGER, a),
    }
    def _bind(prefix: str, vars_: list, wires: list) -> None:
        for i, v in enumerate(vars_):
            r, c = _wire_rc(wires[i])
            ns[f"{prefix}{i}"] = _MatVar(tm, v, r, c)
        # Convenience alias without the index when there is a single component,
        # so a lone state wire can be written `s[i][j]` instead of `s0[i][j]`.
        if len(vars_) == 1:
            ns[prefix] = ns[f"{prefix}0"]

    _bind("s", env.state_vars, env.msmt.ctrl_next)
    _bind("e", env.extl_next_vars, env.msmt.extl_next)
    _bind("el", env.extl_latched_vars, env.msmt.extl_latched)
    return ns


class _NotPredicate(ValueError):
    """Python that the predicate DSL does not read."""


# The operators `_E` overloads. No `**`: on two constants it is evaluated
# here, and its size is not bounded by the length of the source.
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}
_UNARYOPS = {ast.USub: operator.neg, ast.Invert: operator.invert}
_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _pyeval(src: str, ns: dict):
    """`src` as a predicate-DSL expression over `ns`, read without `eval`.

    The source can be an LLM reply, and no namespace makes `eval` safe on
    one: `().__class__.__mro__` reaches every class in the process. So the
    tree is walked over a whitelist -- numeric and Bool constants, names
    bound in `ns`, the operators `_E` overloads, calls to `ns`'s own
    connectives, and `v[i][j]` / `v[i, j]` with int indices -- and anything
    else raises `_NotPredicate`. Attribute access and `_`-names never
    resolve.

    `and`, `or`, `not`, `x if c else y` and a chain `a < b < c` mean `And`,
    `Or`, `Not`, `Ite` and a conjunction: Python's own reading would ask
    whether a symbolic term is truthy, which it always is.
    """
    # `ast.parse` reads a leading blank as an indent.
    return _pywalk(ast.parse(src.lstrip(" \t"), mode="eval").body, ns)


def _pywalk(node: ast.AST, ns: dict):
    def walk(n):
        return _pywalk(n, ns)

    if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
        return node.value
    if isinstance(node, ast.Name) and not node.id.startswith("_") and node.id in ns:
        return ns[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](walk(node.left), walk(node.right))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return ns["Not"](walk(node.operand))
        if type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](walk(node.operand))
    if isinstance(node, ast.BoolOp):
        conn = ns["And"] if isinstance(node.op, ast.And) else ns["Or"]
        return conn(*[walk(v) for v in node.values])
    if isinstance(node, ast.IfExp):
        return ns["Ite"](walk(node.test), walk(node.body), walk(node.orelse))
    if isinstance(node, ast.Compare) and all(type(op) in _CMPOPS for op in node.ops):
        operands = [walk(node.left)] + [walk(c) for c in node.comparators]
        links = [
            _CMPOPS[type(op)](a, b)
            for op, a, b in zip(node.ops, operands, operands[1:])
        ]
        return links[0] if len(links) == 1 else ns["And"](*links)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.func.id.startswith("_")
        and callable(ns.get(node.func.id))
        and not node.keywords
        and not any(isinstance(a, ast.Starred) for a in node.args)
    ):
        return ns[node.func.id](*[walk(a) for a in node.args])
    if isinstance(node, ast.Subscript):
        keys = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        index = [walk(k) for k in keys]
        if all(type(i) is int for i in index):
            target = walk(node.value)
            return target[tuple(index)] if len(index) > 1 else target[index[0]]
    raise _NotPredicate(f"`{ast.unparse(node)}` is not predicate syntax")


_BARE_NUMERAL_RE = re.compile(r"^-?\d+$")
_BARE_EQ_RE = re.compile(r"\((?P<op>=|distinct)\s+(?P<a>-?\w+)\s+(?P<b>-?\w+)\)")


def _coerce_bare_numerals(src: str, bv_widths: dict[str, int]) -> str:
    """Rewrite `(= sK 0)` / `(distinct 0 sK)` into `(= sK (_ bv0 W))` when `sK`
    is a known BitVec<W>-sorted symbol.

    Plain SMT-LIB has no Int→BitVec coercion (cvc5 rejects `(= s0 0)` if `s0`
    is `(_ BitVec W)` outright), so bare numerals written against BV state
    would otherwise always fail to parse. Only rewrites atomic `(= a b)` /
    `(distinct a b)` forms — one bare symbol, one bare numeral — not
    numerals nested inside larger subexpressions.
    """
    if not bv_widths:
        return src

    def repl(m: re.Match) -> str:
        op, a, b = m.group("op"), m.group("a"), m.group("b")
        a_is_num, b_is_num = bool(_BARE_NUMERAL_RE.match(a)), bool(
            _BARE_NUMERAL_RE.match(b)
        )
        if a_is_num == b_is_num:
            return m.group(0)  # both/neither numeral: nothing to coerce
        sym, num = (b, a) if a_is_num else (a, b)
        width = bv_widths.get(sym)
        if width is None:
            return m.group(0)
        val = int(num) % (1 << width)
        lit = f"(_ bv{val} {width})"
        new_a, new_b = (lit, b) if a_is_num else (a, lit)
        return f"({op} {new_a} {new_b})"

    return _BARE_EQ_RE.sub(repl, src)


def parse_predicate(env: "PredicateParser", src: str) -> cvc5.Term:
    """Parse `src` as a cvc5 predicate. Tries the Python-expression DSL
    first (`_pyeval`), falls back to SMT-LIB on failure."""
    ns = _pyeval_namespace(env)
    try:
        result = _pyeval(src, ns)
        if isinstance(result, _E):
            return result.t
        if isinstance(result, cvc5.Term):
            return result
        # Some other Python object — fall through to SMT-LIB
    except Exception:
        pass
    term = env.parse_expr(src)
    if term.isNull():
        # cvc5 parses an empty or all-blank source to a null term rather than
        # an error, and the first use of one -- `getSort()` -- raises from
        # inside the binding: an LLM that wrote `RANKING:` and nothing after
        # it crashed `--infer ai-cegis` instead of being told. Every caller
        # already turns a `RuntimeError` into "could not parse".
        raise RuntimeError("the expression is empty")
    return term


class PredicateParser:
    """Reusable parser context with module symbols declared once:
      * `s0..sN-1`     — ctrl-next state components
      * `e0..eM-1`     — extl-next inputs
      * `el0..elM-1`   — extl-latched inputs
    """

    def __init__(self, msmt: ModuleSMT):
        self.msmt = msmt
        self.tm = msmt.tm
        self.solver = cvc5.Solver(self.tm)
        self.solver.setLogic("ALL")
        self.sm = cvc5.SymbolManager(self.solver)
        self.parser = cvc5.InputParser(self.solver, self.sm)
        self.parser.setIncrementalStringInput(
            cvc5.InputLanguage.SMT_LIB_2_6, "cegis"
        )

        self.state_sorts = [wire_sort(self.tm, w) for w in msmt.ctrl_next]
        self.extl_next_sorts = [wire_sort(self.tm, w) for w in msmt.extl_next]
        self.extl_latched_sorts = [
            wire_sort(self.tm, w) for w in msmt.extl_latched
        ]

        decls = []
        for i, s in enumerate(self.state_sorts):
            decls.append(f"(declare-const s{i} {s})")
        for i, s in enumerate(self.extl_next_sorts):
            decls.append(f"(declare-const e{i} {s})")
        for i, s in enumerate(self.extl_latched_sorts):
            decls.append(f"(declare-const el{i} {s})")
        self._run_commands("\n".join(decls))

        names = (
            [f"s{i}" for i in range(len(self.state_sorts))]
            + [f"e{i}" for i in range(len(self.extl_next_sorts))]
            + [f"el{i}" for i in range(len(self.extl_latched_sorts))]
        )
        if names:
            self.parser.appendIncrementalStringInput(" ".join(names))
            parsed = [self.parser.nextTerm() for _ in names]
        else:
            parsed = []

        k = len(self.state_sorts)
        m = len(self.extl_next_sorts)
        self.state_vars: list[cvc5.Term] = parsed[:k]
        self.extl_next_vars: list[cvc5.Term] = parsed[k : k + m]
        self.extl_latched_vars: list[cvc5.Term] = parsed[k + m :]

        self._bv_widths: dict[str, int] = {
            name: sort.getBitVectorSize()
            for name, sort in zip(
                names, self.state_sorts + self.extl_next_sorts + self.extl_latched_sorts
            )
            if sort.isBitVector()
        }

    # --- internals ------------------------------------------------------

    def _run_commands(self, text: str) -> None:
        self.parser.appendIncrementalStringInput(text)
        while True:
            cmd = self.parser.nextCommand()
            if cmd.isNull():
                break
            cmd.invoke(self.solver, self.sm)

    def parse_expr(self, smt_src: str) -> cvc5.Term:
        """Parse a single SMT-LIB expression referring to s0..sN-1."""
        smt_src = _coerce_bare_numerals(smt_src, self._bv_widths)
        # Re-set the input source each call — the incremental stream ends
        # after a single nextTerm/nextCommand pair.
        self.parser.setIncrementalStringInput(
            cvc5.InputLanguage.SMT_LIB_2_6, "cegis"
        )
        self.parser.appendIncrementalStringInput(smt_src)
        return self.parser.nextTerm()
