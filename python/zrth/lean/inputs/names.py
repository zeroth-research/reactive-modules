"""Claims in the reactive module's own terms.

A claim -- `--safety`, `--buchi`, `--pre`, `--invariant`, `--ranking` --
names a variable the way the module's own update does: a variable is its
latched value, and `(X v)` (`X(v)` in the Python predicate DSL) the value
after the round. A variable is named by the input's own name for it, or
by position: `@intf.{i}` is interface (ctrl) variable `i`, `@extl.{i}`
external variable `i`. The `@` keeps a positional name from ever being a
source name.

The rest of the pipeline -- the SMT encoding, the Lean, the artifacts, the
LLM prompts -- reads positional internal names, so a claim is rewritten
here, once, on the way in, and a predicate shown to the user is rewritten
back on the way out:

    surface                     internal
    (X @intf.{i})  (X x)        s{i}     the state after the round
    (X @extl.{i})  (X u)        e{i}     the input read in the round
    @extl.{i}      u            el{i}    the input read in the round before
    @intf.{i}      x            --       (no claim reads it yet)

Both dialects `parse_predicate` reads are rewritten: SMT-LIB, and the
Python DSL (any text that parses as a Python expression once its `@` names
are set aside).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass

_INTERNAL = re.compile(r"(s|e|el)(\d+)?")
_POSITIONAL = re.compile(r"@(intf|extl)\.(\d+)")
_KINDS = {"intf": "ctrl", "extl": "extl"}
_RESERVED = frozenset({"X", "true", "false", "True", "False"})
_FORBIDDEN = re.compile(r"[\s()|\";]")
_SIMPLE_SYMBOL = re.compile(r"[A-Za-z~!$%^&*_+=<>.?/-][A-Za-z0-9~!@$%^&*_+=<>.?/-]*")


class ClaimError(ValueError):
    """A claim that names something the module does not have."""


def usable(name: str) -> bool:
    """Whether a claim can call a variable `name`: not `X`, a boolean
    literal or a positional name, and a single symbol."""
    return bool(name) and not (name in _RESERVED or name.startswith("@")
                               or _FORBIDDEN.search(name))


@dataclass(frozen=True)
class Names:
    """Source names of a module's variables, in positional order:
    `ctrl[i]` names interface variable `i`, `extl[i]` external variable
    `i`, `None` where the input gives it no name."""

    ctrl: tuple[str | None, ...]
    extl: tuple[str | None, ...]

    def __post_init__(self):
        seen: set[str] = set()
        for name in self.ctrl + self.extl:
            if name is None:
                continue
            if not usable(name):
                raise ClaimError(f"`{name}` cannot name a variable in a claim")
            if name in seen:
                raise ClaimError(f"two variables are named `{name}`")
            seen.add(name)

    @classmethod
    def positional(cls, module) -> "Names":
        """No source names: the module's variables are named by position."""
        return cls((None,) * len(module.ctrl), (None,) * len(module.extl))

    @classmethod
    def of(cls, module, names: Mapping[str, object]) -> "Names":
        """Names for `module` from `{name: Var}`; a variable not in the
        mapping keeps only its positional name."""
        ctrl: list[str | None] = [None] * len(module.ctrl)
        extl: list[str | None] = [None] * len(module.extl)
        # Each `module.ctrl` builds its list anew, so they are indexed once.
        index = [(slots, {v: i for i, v in enumerate(vars_)})
                 for slots, vars_ in ((ctrl, module.ctrl), (extl, module.extl))]
        for name, var in names.items():
            for slots, at in index:
                if var in at:
                    slots[at[var]] = name
                    break
            else:
                raise ClaimError(
                    f"`{name}` names a variable that is neither an interface "
                    "nor an external variable of the module"
                )
        return cls(tuple(ctrl), tuple(extl))

    def by_name(self) -> dict[str, tuple[str, int]]:
        out = {n: ("ctrl", i) for i, n in enumerate(self.ctrl) if n is not None}
        out.update({n: ("extl", i) for i, n in enumerate(self.extl)
                    if n is not None})
        return out


# --- surface -> internal ---------------------------------------------------


class _Reader:
    def __init__(self, names: Names, flag: str):
        self.names = names
        self.flag = flag
        self.by_name = names.by_name()

    def fail(self, why: str) -> ClaimError:
        return ClaimError(f"{self.flag}: {why}")

    def variable(self, name: str) -> tuple[str, int] | None:
        """The variable `name` spells, or `None` for any other symbol."""
        hit = self.by_name.get(name)
        if hit is not None:
            return hit
        m = _POSITIONAL.fullmatch(name)
        if m:
            kind, i = _KINDS[m.group(1)], int(m.group(2))
            have = len(getattr(self.names, kind))
            if i >= have:
                what = "interface" if kind == "ctrl" else "external"
                raise self.fail(
                    f"`{name}`: the module has {have} {what} variable(s)")
            return kind, i
        if name.startswith("@"):
            raise self.fail(
                f"`{name}` is not a positional name (`@intf.N` or `@extl.N`)")
        m = _INTERNAL.fullmatch(name)
        if m:
            raise self.fail(
                f"`{name}` is not a name here; positional names are "
                f"{_respell(m.group(1), int(m.group(2) or 0))}")
        return None

    def internal(self, name: str, *, after: bool) -> str | None:
        """`name`'s internal spelling, read latched or (`after`) as `X`
        reads it; `None` when `name` is no variable."""
        hit = self.variable(name)
        if hit is None:
            return None
        kind, i = hit
        if kind == "extl":
            return f"e{i}" if after else f"el{i}"
        if after:
            return f"s{i}"
        raise self.fail(
            f"`{name}` is interface variable {i}'s latched value, which a "
            f"claim cannot name yet; its value after the round is `(X {name})`")


def _respell(prefix: str, i: int) -> str:
    return {"s": f"`(X @intf.{i})` for `s{i}`",
            "e": f"`(X @extl.{i})` for `e{i}`",
            "el": f"`@extl.{i}` for `el{i}`"}[prefix]


_TOKEN = re.compile(r"""
    (?P<ws>\s+) | (?P<comment>;[^\n]*) | (?P<lp>\() | (?P<rp>\))
  | (?P<quoted>\|[^|]*\|) | (?P<string>"(?:[^"]|"")*")
  | (?P<atom>[^\s()|";]+)
""", re.X)


def _tokens(text: str) -> list[tuple[str, str]]:
    out = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:   # an unclosed `|` or `"`: leave the rest to the parser
            out.append(("atom", text[pos:]))
            break
        out.append((m.lastgroup, m.group()))
        pos = m.end()
    return out


def _symbol(kind: str, tok: str) -> str | None:
    if kind == "atom":
        return tok
    if kind == "quoted":
        return tok[1:-1]
    return None


def _smtlib_to_internal(text: str, r: _Reader) -> str:
    toks = _tokens(text)
    sig = [i for i, (k, _) in enumerate(toks) if k not in ("ws", "comment")]
    out: list[str] = []
    j = 0           # index into `sig`
    done = 0        # tokens before this are emitted
    while j < len(sig):
        i = sig[j]
        kind, tok = toks[i]
        # `( X v )`
        if (kind == "lp" and j + 3 < len(sig)
                and toks[sig[j + 1]] == ("atom", "X")):
            arg_kind, arg = toks[sig[j + 2]]
            name = _symbol(arg_kind, arg)
            if name is None or toks[sig[j + 3]][0] != "rp":
                raise r.fail("`X` applies to a variable: `(X v)`")
            spelled = r.internal(name, after=True)
            if spelled is None:
                raise r.fail(f"`(X {arg})`: `{arg}` is not a variable")
            out.extend(t for _, t in toks[done:i])
            out.append(spelled)
            done = sig[j + 3] + 1
            j += 4
            continue
        head = j > 0 and toks[sig[j - 1]][0] == "lp"
        name = _symbol(kind, tok)
        if name is not None and not head:
            spelled = r.internal(name, after=False)
            if spelled is not None:
                out.extend(t for _, t in toks[done:i])
                out.append(spelled)
                done = i + 1
        j += 1
    out.extend(t for _, t in toks[done:])
    return "".join(out)


_PLACEHOLDER = "__zrth_{}_{}__"
_PLACEHELD = re.compile(r"__zrth_(intf|extl)_(\d+)__")


def _hold(text: str) -> str:
    """`text` with each positional name a Python identifier, so that the
    Python DSL can be told from SMT-LIB by parsing it."""
    return _POSITIONAL.sub(lambda m: _PLACEHOLDER.format(*m.groups()), text)


def _unhold(text: str) -> str:
    return _PLACEHELD.sub(lambda m: f"@{m.group(1)}.{m.group(2)}", text)


def _python(text: str) -> "tuple[ast.Expression, bytes] | None":
    """The Python DSL's reading of `text` and the bytes its offsets index
    (positional names held), or `None` for SMT-LIB."""
    held = _hold(text).strip()
    try:
        return ast.parse(held, mode="eval"), held.encode()
    except SyntaxError:
        return None


def _splice(data: bytes, edits: list[tuple["ast.AST", str]]) -> str:
    """`data` with each node's span replaced: the rest of the text is
    kept as written, so what the SMT-LIB fallback reads is what was
    given."""
    starts = [0]
    for line in data.split(b"\n")[:-1]:
        starts.append(starts[-1] + len(line) + 1)

    def offset(line: int, col: int) -> int:
        return starts[line - 1] + col

    out = data
    for node, text in sorted(
            edits, key=lambda e: offset(e[0].lineno, e[0].col_offset),
            reverse=True):
        a = offset(node.lineno, node.col_offset)
        b = offset(node.end_lineno, node.end_col_offset)
        out = out[:a] + text.encode() + out[b:]
    return _unhold(out.decode())


def _python_to_internal(tree: ast.Expression, data: bytes, r: _Reader) -> str:
    edits: list[tuple[ast.AST, str]] = []

    class _Visit(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "X":
                if (len(node.args) != 1 or node.keywords
                        or not isinstance(node.args[0], ast.Name)):
                    raise r.fail("`X` applies to a variable: `X(v)`")
                name = _unhold(node.args[0].id)
                spelled = r.internal(name, after=True)
                if spelled is None:
                    raise r.fail(f"`X({name})`: `{name}` is not a variable")
                edits.append((node, spelled))
                return
            # A function's name is never a variable's.
            for a in node.args:
                self.visit(a)
            for k in node.keywords:
                self.visit(k.value)

        def visit_Name(self, node: ast.Name):
            spelled = r.internal(_unhold(node.id), after=False)
            if spelled is not None:
                edits.append((node, spelled))

    _Visit().visit(tree)
    return _splice(data, edits)


def to_internal(text: str, names: Names, flag: str) -> str:
    """`text`, a claim in surface names, in internal ones. Raises
    `ClaimError` (prefixed by `flag`) for a name the module does not have,
    a latched interface variable, `X` of anything but a variable, or the
    old bare positional spelling."""
    r = _Reader(names, flag)
    parsed = _python(text)
    if parsed is None:
        return _smtlib_to_internal(text, r)
    return _python_to_internal(*parsed, r)


# --- internal -> surface ---------------------------------------------------


def _surface_name(names: Names | None, kind: str, i: int) -> str:
    src = getattr(names, kind)[i] if names is not None and i < len(
        getattr(names, kind)) else None
    if src is not None:
        return src
    return f"@{'intf' if kind == 'ctrl' else 'extl'}.{i}"


def _surface(m: re.Match, names: Names | None) -> tuple[str, bool] | None:
    """(name, after) for an internal name, or None."""
    prefix, idx = m.group(1), m.group(2)
    i = int(idx or 0)
    kind = "ctrl" if prefix == "s" else "extl"
    return _surface_name(names, kind, i), prefix != "el"


def _quote(name: str) -> str:
    return name if (_SIMPLE_SYMBOL.fullmatch(name) or name.startswith("@")) \
        else f"|{name}|"


_EMBEDDED = re.compile(r"(?<![A-Za-z0-9_])(s|e|el)\d+")


def _smtlib_shaped(text: str, tree: ast.Expression) -> bool:
    """Whether text both dialects read is shown as SMT-LIB: a lone name,
    or a parenthesised form in which every internal name is a whole atom
    and none is in head position (where Python's `(s0 >= 0)` has one)."""
    if isinstance(tree.body, ast.Name):
        return True
    if not text.lstrip().startswith("("):
        return False
    toks = [t for t in _tokens(text) if t[0] not in ("ws", "comment")]
    for j, (kind, tok) in enumerate(toks):
        if kind != "atom" or not _EMBEDDED.search(tok):
            continue
        if not _INTERNAL.fullmatch(tok) or (j and toks[j - 1][0] == "lp"):
            return False
    return True


def to_surface(text: str, names: Names | None = None) -> str:
    """`text`, a predicate in internal names, in the surface ones a claim
    takes: the input's names where `names` has them, positional otherwise.
    With no `names`, every variable is spelled by position."""
    parsed = _python(text)
    if parsed is None or _smtlib_shaped(text, parsed[0]):
        out = []
        for kind, tok in _tokens(text):
            m = _INTERNAL.fullmatch(tok) if kind == "atom" else None
            if m and m.group(2) is not None:
                name, after = _surface(m, names)
                name = _quote(name)
                out.append(f"(X {name})" if after else name)
            else:
                out.append(tok)
        return "".join(out)

    tree, data = parsed
    edits: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            m = _INTERNAL.fullmatch(node.id)
            if m:
                name, after = _surface(m, names)
                edits.append((node, f"X({name})" if after else name))
    return _splice(data, edits)
