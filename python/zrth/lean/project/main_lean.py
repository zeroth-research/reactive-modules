"""`Main.lean`, which `--executable` adds: the module as a program that
reads its inputs from stdin and prints its state each round.
"""

from __future__ import annotations


from zrth import Module, Wire, X

from ..core.errors import Refused
from ..ir.sorts import flat_element_type, wire_shape
from ..codegen.terms import build_tuple, product_type
from .names import (
    LINTERS_OFF,
)


def _token_count(wire: Wire) -> int:
    """When reading a value from user, we need to know how many primitive values (Int/Bool)
    to read to get a single input value (which can be e.g., a matrix).
    This function computes this.
    """
    return wire_shape(wire).total


def _unsupported_io_sort(elem: str, wire: Wire) -> Refused:
    return Refused(
        f"generate_main_lean: no Lean IO for element type {elem} "
        f"(wire dtype {wire.dtype}); `Real` is noncomputable in Lean, so it "
        "cannot be parsed or printed by the generated executable"
    )


def _elem_parser(elem: str, wire: Wire) -> "tuple[str, str, str]":
    """`(bind, parser, push)` for reading one element of `elem` from a token.

    The emitted line is `let v <bind> <parser> tokens[...]!`, so `bind` is
    `:=` for a pure parser and `←` for one in IO; `push` is the expression
    stored into the array.
    """
    if elem == "Bool":
        return ":=", "parseBool", "v"
    if elem == "Int":
        return "←", "parseIntOrFail", "v"
    if elem.startswith("(BitVec "):
        width = elem[len("(BitVec ") : -1]
        return "←", "parseIntOrFail", f"(BitVec.ofInt {width} v)"
    raise _unsupported_io_sort(elem, wire)


def _elem_shower(elem: str, wire: Wire) -> str:
    """Lean function rendering one element of `elem` as a String."""
    if elem == "Bool":
        return "showBool"
    if elem == "Int":
        return "toString"
    if elem.startswith("(BitVec "):
        return "toString"
    raise _unsupported_io_sort(elem, wire)


def generate_main_lean(project_name: str, module: Module, module_name: str) -> str:
    """Generate Main.lean source that runs init/update in a stdin/stdout loop."""
    extl_next = [X(v) for v in module.extl]
    ctrl_next = [X(v) for v in module.ctrl]
    ctrl_latched = list(module.ctrl)

    total_extl_tokens = sum(_token_count(w) for w in extl_next)
    total_ctrl_tokens = sum(_token_count(w) for w in ctrl_next)

    lines: list[str] = []

    # Imports. `init`/`update` are emitted into System/System.lean, which the
    # `System` lean_lib's root module pulls in; `{project_name}.{module_name}`
    # named no file the generator ever writes.
    lines.append("import System")
    lines.append("")
    lines.append(LINTERS_OFF)
    lines.append("")
    lines.append("open Box")
    lines.append("")

    # Static helpers
    lines.append(
        'def parseBool (s : String) : Bool := s.trimAscii.toString == "true" || s.trimAscii.toString == "1"'
    )
    lines.append('def showBool (b : Bool) : String := if b then "1" else "0"')
    lines.append("")
    lines.append("def parseIntOrFail (s : String) : IO Int := do")
    lines.append("  match s.trimAscii.toString.toInt? with")
    lines.append("  | some n => pure n")
    lines.append(
        '  | none => throw (.userError s!"Invalid integer: {s.trimAscii.toString}")'
    )
    lines.append("")
    lines.append(
        "def showMat {t : Type} (m n : Nat) (f : t → String) (mat : Fin m → Fin n → t)"
        " : String :="
    )
    lines.append(
        "  let vals := (List.ofFn fun i => List.ofFn fun j => [f (mat i j)]).flatten.flatten"
    )
    lines.append('  String.intercalate " " vals')
    lines.append("")

    # parseExtl
    lines.append(
        f"def parseExtl (tokens : Array String) : IO ({product_type(extl_next)}) := do"
    )
    lines.append(
        f'  if tokens.size < {total_extl_tokens} then throw (.userError "Expected {total_extl_tokens} input values")'
    )

    offset = 0
    parse_vars: list[str] = []
    for i, w in enumerate(extl_next):
        elem = flat_element_type(w)
        var = f"e{i}"
        m, n = wire_shape(w).m, wire_shape(w).n
        # Every wire is a matrix here, so parse m*n tokens into an array of
        # the wire's own element type -- reading them all as Int made the
        # body disagree with the ascribed return type for every other sort.
        bind, parser, push = _elem_parser(elem, w)
        lines.append(f"  let mut arr{i} : Array {elem} := #[]")
        lines.append(f"  for k in List.range {m * n} do")
        lines.append(f"    let v {bind} {parser} tokens[{offset} + k]!")
        lines.append(f"    arr{i} := arr{i}.push {push}")
        lines.append(
            f"  let {var} : Fin {m} → Fin {n} → {elem} :="
            f" fun i j => arr{i}[i.val * {n} + j.val]!"
        )
        offset += m * n
        parse_vars.append(var)

    # `product_type` orders components as the wires are declared, and
    # `product_accessor` projects them the same way, so the literal follows suit.
    # (An earlier right-nested `ValTuple` encoding needed the reverse.)
    lines.append(f"  pure {build_tuple(parse_vars)}")
    lines.append("")

    # showCtrl
    lines.append(f"def showCtrl (v : {product_type(ctrl_next)}) : String :=")
    # Destructure
    destr_vars: list[str] = []
    for i in range(len(ctrl_next)):
        destr_vars.append(f"v{i}")
    # Same order here: v{i} must bind ctrl_next[i], since the formatters
    # below pick showBool/toString/showMat from that wire's type.
    lines.append(f"  let {build_tuple(destr_vars)} := v")

    # Format each variable. `dtype_to_lean_type` always returns a `Mat ...`
    # here, so dispatch on the element sort instead: comparing that string
    # against "Bool"/"Int" never matched and pinned every element to Int.
    show_parts: list[str] = []
    for i, w in enumerate(ctrl_next):
        elem = flat_element_type(w)
        var = f"v{i}"
        m, n = wire_shape(w).m, wire_shape(w).n
        show_parts.append(f"showMat {m} {n} {_elem_shower(elem, w)} {var}")

    if len(show_parts) == 1:
        lines.append(f"  {show_parts[0]}")
    else:
        # Use s!"..." interpolation to join with spaces
        interp = " ".join(f"{{{p}}}" for p in show_parts)
        lines.append(f'  s!"{interp}"')
    lines.append("")

    # main function. `update` is emitted curried as
    # `update (ctrl) (extl_l) (extl_n)`, so the three groups are applied
    # separately -- and `extl_l` is the *previous* step's input, which the
    # loop therefore has to carry.
    lines.append(f"""\
def main : IO Unit := do
  let stdin ← IO.getStdin
  let line0 ← stdin.getLine
  if line0.trimAscii.toString.isEmpty then return
  let extl0 ← parseExtl (line0.trimAscii.toString.splitOn " " |>.toArray)
  let mut state := init extl0
  let mut extlPrev := extl0
  IO.println (showCtrl state)
  repeat do
    let line ← stdin.getLine
    if line.trimAscii.toString.isEmpty then break
    let extl ← parseExtl (line.trimAscii.toString.splitOn " " |>.toArray)
    let state' := update state extlPrev extl
    IO.println (showCtrl state')
    state := state'
    extlPrev := extl
""")

    return "\n".join(lines)
