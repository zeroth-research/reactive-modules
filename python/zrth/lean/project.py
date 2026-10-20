"""
Build a Lean4 project with Mathlib, Cslib, and a custom git dependency,
copy template library files, and generate a diagram Lean file.
"""

from zrth.lean.errors import Refused
from zrth.lean.sorts import flat_element_type, wire_shape
from zrth.lean.context import LeanContext

from zrth.lean.proof.render import (
    generate_certificate_lean,
    generate_data_lean,
    generate_zeroth_hammer_lean,
    LeanCertificate,
)
from zrth.lean.template_env import render, STATIC_DIR, PROJECT_TEMPLATES_DIR

import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from zrth import Module, Wire, X
from .native import (
    product_type,
    build_tuple,
)

from .translate import ModuleToLean4
from .lean_names import (
    CERT_LIB, DATA, LINTERS_OFF, PROOF, SYSTEM_LAYOUT, Layout, cert_file,
)

if TYPE_CHECKING:
    from .inputs import SourceText


# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

MATHLIB_URL = "https://github.com/leanprover-community/mathlib4"
CSLIB_URL = "https://github.com/leanprover-community/cslib"
PROOFS_URL = "https://github.com/zeroth/proof-prototyping"

MATHLIB_REV = "v4.28.0"
CSLIB_REV = MATHLIB_REV
# Lean toolchain version — should match Mathlib's requirement
LEAN_TOOLCHAIN = f"leanprover/lean4:{MATHLIB_REV}"

# Static template files (copied as-is into the project)
TEMPLATE_DIR = STATIC_DIR
CORE_FILES = ["Basic.lean", "Box.lean", "LTL.lean", "Mat.lean"]
LEAN_AI_FILES = ["LeanAI.lean", "LeanAI"]

# ══════════════════════════════════════════════════════════════════════════
# Lean project files
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LeanLib:
    """A `[[lean_lib]]` a route adds: `name` is the module name its files are
    imported under, `src_dir` the project-relative directory they are in."""

    name: str
    src_dir: str
    comment: str = ""


@dataclass(frozen=True)
class Require:
    """A `[[require]]` a route adds, from a local checkout at `path`."""

    name: str
    path: str
    comment: str = ""


@dataclass(frozen=True)
class LakefileParts:
    """What an inference route adds to the lakefile: libraries after
    `System`, requires after `smt`, each preceded by its `comment`."""

    libs: tuple[LeanLib, ...] = ()
    requires: tuple[Require, ...] = ()


def generate_lakefile(
    project_name: str,
    executable: bool = False,
    parts: LakefileParts = LakefileParts(),
) -> str:
    """Render the project's lakefile, with `parts` added to the fixed ones.

    A route cannot amend the lakefile after the fact -- it is written while
    the project is created, before any route runs -- so what the route needs
    in it is given here, from its row's `lakefile` hook.
    """
    return render(
        "project/lakefile.toml.j2",
        project_name=project_name,
        executable=executable,
        cslib_rev=CSLIB_REV,
        parts=parts,
    )


class LakeBuildError(RuntimeError):
    """`lake` could not build the generated project's certificate."""


def stream(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    """Run `cmd` in `cwd`, echoing its output as it arrives.

    Returns the exit code *and* the output rather than raising, so each
    caller can say what went wrong in its own vocabulary -- `lake` and
    the tool a route runs do not fail in the same words. `FileNotFoundError`
    propagates: a missing executable is the caller's to name.
    """
    print(f"$ {' '.join(cmd)}")
    chunks: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        chunks.append(line)
        print(line, end="")
    return proc.wait(), "".join(chunks)


def build_certificate(project_dir: Path) -> None:
    """`lake build` the certificate of a generated project.

    `lake update` runs first: a freshly generated project has no manifest at
    all, and one left by an earlier run predates this run's lakefile -- a
    route's lakefile can add a require the manifest has never seen, and
    `lake build` resolves neither case.

    A build that succeeds while the certificate still holds `sorry` is not a
    proof, so that is an error too. `zeroth_hammer` leaves a `sorry` where it
    cannot close an obligation, and lake reports those as warnings and exits
    0 -- which would make "built" mean nothing.
    """
    for cmd in (["lake", "update"], ["lake", "build", CERT_LIB]):
        try:
            rc, out = stream(cmd, cwd=project_dir)
        except FileNotFoundError as e:
            raise LakeBuildError(f"cannot run {cmd[0]}: {e}") from e
        if rc != 0:
            raise LakeBuildError(
                f"`{' '.join(cmd)}` failed (exit {rc}) in {project_dir}\n"
                + _build_why(out)
            )

    unproved = [l for l in out.splitlines()
                if "sorry" in l and f"{CERT_LIB}/" in l]
    if unproved:
        raise LakeBuildError(
            f"the certificate compiles but is not proved: "
            f"{len(unproved)} obligation(s) left as `sorry`\n"
            + "\n".join(f"  {l.strip()}" for l in unproved[:5])
        )
    print(f"Certificate built: lake build Certificate in {project_dir}")


# A declaration whose elaboration fails is never added to the environment,
# so the next one that mentions it fails again -- in the kernel this time,
# as `(kernel) unknown constant 'hrank'` after `hrank` itself timed out at
# `whnf`. The second error is the first one's shadow: it names a symbol
# where the first named a cause, and it is the one that reads like a
# codegen bug rather than a certificate that could not be proved. Over the
# matrix every occurrence of it follows an earlier error.
_CASCADE = re.compile(r"\(kernel\) unknown constant")


def without_cascades(errors: list) -> list:
    """`errors`, without the ones an earlier error already accounts for.

    Only ever drops a *later* kernel `unknown constant`. A build whose
    first complaint is one is left alone: nothing came before it to be a
    shadow of, and then something really is missing.
    """
    out: list = []
    for e in errors:
        if out and _CASCADE.search(str(e)):
            continue
        out.append(e)
    return out


def _build_why(out: str) -> str:
    """The reason lake refused, restated where the user will see it.

    Its own diagnosis is accurate but scrolls past behind thousands of
    replayed Mathlib jobs, leaving a bare "failed (exit 1)" as the last line.
    """
    # Lake's own two wrapper lines say only that something failed, which is
    # what this is already replacing; the diagnostics are the lines between.
    noise = ("error: build failed", "error: Lean exited with code")
    errors = without_cascades([
        l
        for l in out.splitlines()
        if l.startswith("error:") and not l.startswith(noise)
    ])
    if not errors:
        return "  (see the output above)"
    return (
        "\n".join(f"  {l}" for l in errors[:5])
        + "\n  An unresolved import means the project cannot see what the "
        "certificate needs.\n  A failed tactic means an obligation "
        "`zeroth_hammer` could not close: the invariant\n  or the ranking "
        "function is wrong, or too weak to be inductive."
    )


# ══════════════════════════════════════════════════════════════════════════
# The emitted encodings
#
# One row per Lean file `ModuleToLean4` prints, and one writer for all of
# them. Both routes that emit encodings -- the project (`create_project`)
# and the standalone `--cert-file` -- walk this same table, so the header,
# the imports and the producer agree across the two by construction rather
# than by inspection. They differ only in how a row becomes a path and a
# Lean module name, which is what `Layout` supplies.
#
# Row order is emission order only: each producer prints its definitions
# and the theorems about them in one call, so no row reads state another
# left behind. Rows are coupled by Lean names -- `ScalarRel` rewrites by
# `Scalar`'s `init_scalar_eq`, every row quotes the base's `init`/`update`
# -- and `Encoding.needs`, which becomes each file's imports, records that.
# ══════════════════════════════════════════════════════════════════════════




@dataclass(frozen=True)
class Encoding:
    """One Lean file: what names it, what it says, what it needs, who prints it."""

    suffix: str                 # appended to the layout's base; "" is the base
    title: str                  # header comment, "<title> of reactive module `X`"
    core: str                   # the Core module the encoding is written against
    needs: tuple[str, ...]      # suffixes of the sibling encodings it imports
    producer: str               # the `ModuleToLean4` method that prints it
    cert_file: bool = True      # also written by the `--cert-file` route


ENCODINGS: tuple[Encoding, ...] = (
    # The functional encoding is the base every other row imports. The
    # `--cert-file` route gets its own copy inlined into the standalone
    # certificate by `generate_standalone_cert_lean`, so it is not written
    # again there; `Circ` has no consumer on that route at all.
    Encoding("", "Functional encoding", "Core.Box", (), "to_lean_functional",
             cert_file=False),
    Encoding("Circ", "Circuit encoding", "Core.Box", ("",), "to_lean_circ",
             cert_file=False),
    Encoding("Rel", "Matrix-domain relational encoding", "Core.Basic", ("",),
             "to_lean_mat_rel"),
    Encoding("Scalar", "Scalar encoding", "Core.Basic", ("",), "to_lean_scalar"),
    Encoding("ScalarRel", "Scalar-relational encoding", "Core.Basic", ("Scalar",),
             "to_lean_rel"),
)


def write_encoding(
    enc: Encoding, m2l: "ModuleToLean4", layout: Layout, module_name: str
) -> Path:
    """Print one encoding and write it where `layout` says it goes."""
    imports = [enc.core] + [layout.lean_module(s) for s in enc.needs]
    body = getattr(m2l, enc.producer)()
    path = layout.path(enc.suffix)
    path.write_text(
        f"/- {enc.title} of reactive module `{module_name}` -/\n"
        + "".join(f"import {i}\n" for i in imports)
        + f"\n{LINTERS_OFF}\n\n{body}\n"
    )
    return path


def generate_root() -> str:
    """Root module file that imports the `System.*` encodings.

    The import list is the encoding table, so a row added there reaches the
    root without a second edit.
    """
    return render(
        "project/Root.lean.j2",
        imports=[SYSTEM_LAYOUT.lean_module(e.suffix) for e in ENCODINGS],
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


def write_data_lean(
    project_dir: Path,
    project_name: str,
    module: Module,
    lean_cert: LeanCertificate | None,
    ctx: LeanContext | None = None,
) -> Path:
    """Write/overwrite Certificate/Data.lean with init_pre, update_pre, inv, P,
    ranking.

    Pass a pre-built ``ctx`` to avoid rebuilding LeanContext (e.g. when called
    from ``create_project`` which already has one).

    Beside ``Certificate.lean`` rather than under ``System/``: both files are
    about the certificate and both are rewritten whenever the predicates
    change, while ``System/`` holds the module, which does not. The directory
    is made here because this can be called on its own, after inference, and
    is not guaranteed to run after ``write_certificate_lean``.
    """
    if ctx is None:
        ctx = LeanContext(module)

    data_file = project_dir / cert_file(DATA)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text(generate_data_lean(ctx, lean_cert))
    print(f"Wrote {data_file}")
    return data_file


def write_certificate_lean(
    project_dir: Path,
    project_name: str,
    module: Module,
    lean_cert: LeanCertificate | None = None,
    ctx: LeanContext | None = None,
) -> Path:
    """Write/overwrite Certificate.lean (stable proof structure, imports Data).

    Pass a pre-built ``ctx`` to avoid rebuilding LeanContext (e.g. when called
    from ``create_project`` which already has one).

    The *definitions* live in ``Certificate/Data.lean`` beside it, but the
    proof tactics are generated from the shape of the predicates (see
    ``zrth.lean.proof.tactics``), so this file has to be rewritten whenever they
    change — after ``--infer``, for instance.
    """
    if ctx is None:
        ctx = LeanContext(module)

    path = project_dir / cert_file(PROOF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_certificate_lean(ctx, lean_cert))
    print(f"Wrote {path}")
    return path


def create_project(
    output_dir: Path,
    module: Module,
    project_name: str = "Certificate",
    template_dir: Path = TEMPLATE_DIR,
    executable: bool = False,
    lean_cert: LeanCertificate | None = None,
    source: "SourceText | None" = None,
    lakefile: LakefileParts | None = None,
    remove: Iterable[str] = (),
    ctx: "LeanContext | None" = None,
) -> Path:
    """
    Create a full Lean4 project.

    Args:
        `output_dir`:      Where to create the project folder.
        `diagram_sources`: Dict of {filename_stem: lean_source_code}
                           e.g. {"MixedDiagram": "import ...\\ndef mixedDiagram ..."}
        `template_dir`:    Optional path to directory containing template .lean files.
        `project_name`:    Optional name of the Lean package / library.
        `executable`:      If True, generate Main.lean and add [[lean_exe]] to lakefile.
        `lean_cert`:       The certificate's predicates, printed; placeholders
                           where it is `None`.
        `source`:          The input the module was read from, written to
                           `dbg/system<suffix>` as a debug artifact.
        `lakefile`:        What the inference route adds to the lakefile,
                           from its row's `lakefile` hook. Declared as data
                           rather than passed per route: the lakefile is
                           written here, before any route runs, so a route
                           cannot amend it afterwards.
        `remove`:          Project-relative files an earlier run in the same
                           `-o` may have left where a lean_lib globs; they
                           are deleted before anything is written there.
        `ctx`:             A `LeanContext` this run already built, so the
                           discovery pass runs once rather than once here
                           and again for every file rewritten after
                           inference.
    """
    project_dir = output_dir / project_name
    src_dir = project_dir / "System"
    dbg_dir = project_dir / "dbg"
    module_name = project_name

    # Create directory structure
    src_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created project directory: `{project_dir}`")

    # Debug artifacts: module representation and the input it was read from
    dbg_dir.mkdir(parents=True, exist_ok=True)
    (dbg_dir / "system.txt").write_text(str(module))
    print("Wrote system.txt")
    if source is not None:
        copy = f"system{source.suffix}"
        (dbg_dir / copy).write_text(source.text)
        print(f"Wrote {copy}")

    # Render and write project-level files from templates
    lakefile_path = project_dir / "lakefile.toml"
    lakefile_path.write_text(
        generate_lakefile(
            project_name, executable=executable,
            parts=lakefile or LakefileParts(),
        )
    )
    print(f"Wrote {lakefile_path}")

    toolchain = project_dir / "lean-toolchain"
    toolchain.write_text(
        render("project/lean-toolchain.j2", lean_toolchain=LEAN_TOOLCHAIN)
    )
    print(f"Wrote {toolchain}")

    hammer_file = project_dir / "ZerothHammer.lean"
    hammer_file.write_text(generate_zeroth_hammer_lean())
    print(f"Wrote {hammer_file}")

    (project_dir / "Certificate.lean").write_text(render("project/Certificate.lean.j2"))

    # The `Certificate` lean_lib globs its submodules, so every file in
    # `Certificate/` is built -- including one this run did not write. A file
    # an inference route writes is written *after* this, so anything already
    # there is a copy an earlier run left in the same `-o`: it is about a
    # different module, and lake would build it against this run's `System/`.
    # The caller names those files in `remove`.
    for rel in remove:
        (project_dir / rel).unlink(missing_ok=True)

    # Copy static files (Core/, LeanAI/)
    core_dir = project_dir / "Core"
    core_dir.mkdir(parents=True, exist_ok=True)
    for tmpl_name in CORE_FILES:
        src_path = template_dir / "Core" / tmpl_name
        dst_path = core_dir / tmpl_name
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"Copied {tmpl_name} -> Core/")
        else:
            raise RuntimeError(f"Template file `{tmpl_name}` not found at {src_path}")

    for tmpl_name in LEAN_AI_FILES:
        src_path = template_dir / tmpl_name
        dst_path = project_dir / tmpl_name
        if src_path.exists():
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"Copied template directory {tmpl_name} -> /")
            else:
                shutil.copy2(src_path, dst_path)
                print(f"Copied template file {tmpl_name} -> /")
        else:
            raise RuntimeError(
                f"Template file/dir `{tmpl_name}` not found at {src_path}"
            )

    # ----------------------------------------------------------
    # Generate reactive module (init and update)
    # ----------------------------------------------------------
    if ctx is None:
        ctx = LeanContext(module)
    m2l = ModuleToLean4(ctx)

    root_lean = project_dir / "System.lean"
    root_lean.write_text(generate_root())
    print(f"Wrote root module {root_lean}")

    # Every encoding the project carries, in table order.
    layout = replace(SYSTEM_LAYOUT, directory=src_dir)
    for enc in ENCODINGS:
        print(f"++ Generated {write_encoding(enc, m2l, layout, module_name)} ++")

    # -- certificate data (init_pre, inv, P, ranking — placeholders if no lean_cert) --
    write_data_lean(project_dir, project_name, module, lean_cert, ctx=ctx)

    # ----------------------------------------------------------
    # Generate Main.lean for executable
    # ----------------------------------------------------------
    if executable:
        main_lean = project_dir / "Main.lean"
        main_lean.write_text(generate_main_lean(project_name, module, module_name))
        print(f"Wrote {main_lean}")

    # ----------------------------------------------------------
    # Always write Certificate.lean (stable proof structure, imports XXXData)
    # ----------------------------------------------------------
    write_certificate_lean(project_dir, project_name, module, lean_cert, ctx=ctx)

    print()
    print(f"DONE: Project created at: {project_dir}")
    return project_dir


def generate_standalone_cert_lean(
    module: Module,
    lean_cert: LeanCertificate | None = None,
) -> str:
    """Generate a self-contained certificate Lean file with init/update inlined.

    Unlike :func:`write_certificate_lean`, the functional module code
    (init, update) is inlined directly rather than imported.  ZerothHammer
    is always imported from the project-level ``ZerothHammer.lean``.

    Suitable for placing a single .lean file inside an existing lake project
    such as ``tests/lean/Certs/``.
    """
    ctx = LeanContext(module)
    m2l = ModuleToLean4(ctx)
    module_code = m2l.to_lean_functional()
    return generate_certificate_lean(ctx, lean_cert, module_inline=module_code)
