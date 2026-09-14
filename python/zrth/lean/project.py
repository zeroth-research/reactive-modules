"""
Build a Lean4 project with Mathlib, Cslib, and a custom git dependency,
copy template library files, and generate a diagram Lean file.
"""

from zrth.lean.common import (
    LeanContext,
    dtype_shape,
    _flat_element_type,
)

from zrth.lean.cert import (
    generate_certificate_lean,
    generate_data_lean,
    generate_zeroth_hammer_lean,
    CertificateData,
)
from zrth.lean.template_env import render, STATIC_DIR, PROJECT_TEMPLATES_DIR

import shutil
import subprocess
from pathlib import Path

from zrth import Module, Wire, Env, X
from .native import (
    _product_type,
    _build_tuple,
)

from .translate import ModuleToLean4


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

# The `--fbk-proveit` route's subdirectory inside the generated project. Its
# NA model is both a `proveit.py` input and a lake target of the project, so
# the directory and the model's module name are needed on both sides.
PROVEIT_DIR = "ProveIt"


def na_module_name(project_name: str) -> str:
    """Lean module name of the NA model the `--fbk-proveit` route writes.

    `proveit.py` compiles an out-of-tree model into an olean and imports it
    under the file stem, so the file name *is* the module name the generated
    certificate carries -- and the name the project's lakefile has to map
    back onto ``PROVEIT_DIR``.
    """
    return f"{project_name}NA"


# ══════════════════════════════════════════════════════════════════════════
# Lean project files
# ══════════════════════════════════════════════════════════════════════════


def generate_lakefile(
    project_name: str,
    executable: bool = False,
    ltl_project: Path | str | None = None,
) -> str:
    """Render the project's lakefile.

    `ltl_project` is the `--fbk-proveit` checkout: given one, the lakefile
    also requires it and declares the NA model as a lean_lib, which is what
    makes the installed certificate -- `Certificate/Certificate.lean`, which
    the root module imports -- a buildable target rather than a file lake
    reaches and cannot elaborate.
    """
    return render(
        "project/lakefile.toml.j2",
        project_name=project_name,
        executable=executable,
        cslib_rev=CSLIB_REV,
        ltl_project=str(ltl_project) if ltl_project is not None else None,
        na_lib=na_module_name(project_name) if ltl_project is not None else None,
        proveit_dir=PROVEIT_DIR,
    )


class LakeBuildError(RuntimeError):
    """`lake` could not build the generated project's certificate."""


def stream(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    """Run `cmd` in `cwd`, echoing its output as it arrives.

    Returns the exit code *and* the output rather than raising, so each
    caller can say what went wrong in its own vocabulary -- `lake` and
    `proveit.py` do not fail in the same words. `FileNotFoundError`
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
    all, and one left by an earlier run predates this run's lakefile -- with
    `--fbk-proveit` that lakefile has a require the manifest has never seen,
    and `lake build` resolves neither case.

    A build that succeeds while the certificate still holds `sorry` is not a
    proof, so that is an error too. `zeroth_hammer` leaves a `sorry` where it
    cannot close an obligation, and lake reports those as warnings and exits
    0 -- which would make "built" mean nothing.
    """
    for cmd in (["lake", "update"], ["lake", "build", "Certificate"]):
        try:
            rc, out = stream(cmd, cwd=project_dir)
        except FileNotFoundError as e:
            raise LakeBuildError(f"cannot run {cmd[0]}: {e}") from e
        if rc != 0:
            raise LakeBuildError(
                f"`{' '.join(cmd)}` failed (exit {rc}) in {project_dir}\n"
                + _build_why(out)
            )

    unproved = [l for l in out.splitlines() if "sorry" in l and "Certificate/" in l]
    if unproved:
        raise LakeBuildError(
            f"the certificate compiles but is not proved: "
            f"{len(unproved)} obligation(s) left as `sorry`\n"
            + "\n".join(f"  {l.strip()}" for l in unproved[:5])
        )
    print(f"Certificate built: lake build Certificate in {project_dir}")


def _build_why(out: str) -> str:
    """The reason lake refused, restated where the user will see it.

    Its own diagnosis is accurate but scrolls past behind thousands of
    replayed Mathlib jobs, leaving a bare "failed (exit 1)" as the last line.
    """
    # Lake's own two wrapper lines say only that something failed, which is
    # what this is already replacing; the diagnostics are the lines between.
    noise = ("error: build failed", "error: Lean exited with code")
    errors = [
        l
        for l in out.splitlines()
        if l.startswith("error:") and not l.startswith(noise)
    ]
    if not errors:
        return "  (see the output above)"
    return (
        "\n".join(f"  {l}" for l in errors[:5])
        + "\n  An unresolved import means the project cannot see what the "
        "certificate needs.\n  A failed tactic means an obligation "
        "`zeroth_hammer` could not close: the invariant\n  or the ranking "
        "function is wrong, or too weak to be inductive."
    )


def generate_root(scalar: bool = True) -> str:
    """Root module file that imports System.* encodings."""
    return render("project/Root.lean.j2", scalar=scalar)


def _token_count(wire: Wire) -> int:
    """When reading a value from user, we need to know how many primitive values (Int/Bool)
    to read to get a single input value (which can be e.g., a matrix).
    This function computes this.
    """
    shape = dtype_shape(wire.dtype)
    if shape == []:
        return 1
    if len(shape) == 1:
        return shape[0]
    if len(shape) == 2:  # matrix
        return shape[0] * shape[1]
    raise ValueError("Unsupported DType for token count")


def _unsupported_io_sort(elem: str, wire: Wire) -> ValueError:
    return ValueError(
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
        f"def parseExtl (tokens : Array String) : IO ({_product_type(extl_next)}) := do"
    )
    lines.append(
        f'  if tokens.size < {total_extl_tokens} then throw (.userError "Expected {total_extl_tokens} input values")'
    )

    offset = 0
    parse_vars: list[str] = []
    for i, w in enumerate(extl_next):
        elem = _flat_element_type(w)
        var = f"e{i}"
        shape = dtype_shape(w.dtype)
        m = shape[0]
        n = shape[1] if len(shape) == 2 else 1
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

    # `_product_type` orders components as the wires are declared, and
    # `_accessor` projects them the same way, so the literal follows suit.
    # (An earlier right-nested `ValTuple` encoding needed the reverse.)
    lines.append(f"  pure {_build_tuple(parse_vars)}")
    lines.append("")

    # showCtrl
    lines.append(f"def showCtrl (v : {_product_type(ctrl_next)}) : String :=")
    # Destructure
    destr_vars: list[str] = []
    for i in range(len(ctrl_next)):
        destr_vars.append(f"v{i}")
    # Same order here: v{i} must bind ctrl_next[i], since the formatters
    # below pick showBool/toString/showMat from that wire's type.
    lines.append(f"  let {_build_tuple(destr_vars)} := v")

    # Format each variable. `dtype_to_lean_type` always returns a `Mat ...`
    # here, so dispatch on the element sort instead: comparing that string
    # against "Bool"/"Int" never matched and pinned every element to Int.
    show_parts: list[str] = []
    for i, w in enumerate(ctrl_next):
        elem = _flat_element_type(w)
        var = f"v{i}"
        shape = dtype_shape(w.dtype)
        m = shape[0]
        n = shape[1] if len(shape) == 2 else 1
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
    cert_data: CertificateData | None,
    ctx: LeanContext | None = None,
) -> Path:
    """Write/overwrite XXXData.lean with init_pre, update_pre, inv, P, ranking.

    Pass a pre-built ``ctx`` to avoid rebuilding LeanContext (e.g. when called
    from ``create_project`` which already has one).
    """
    module_name = project_name
    if ctx is None:
        cert_terms: list = []
        if cert_data is not None:
            for field in (
                cert_data.prp,
                cert_data.inv,
                cert_data.init_pre,
                cert_data.update_pre,
                cert_data.ranking,
            ):
                if isinstance(field, list):
                    cert_terms.extend(field)
        ctx = LeanContext(module, cert_terms=cert_terms)

    src_dir = project_dir / "System"
    data_file = src_dir / "Data.lean"
    data_file.write_text(generate_data_lean(ctx, cert_data))
    print(f"Wrote {data_file}")
    return data_file


def write_certificate_lean(
    project_dir: Path,
    project_name: str,
    module: Module,
    cert_data: CertificateData | None = None,
    ctx: LeanContext | None = None,
) -> Path:
    """Write/overwrite Certificate.lean (stable proof structure, imports XXXData).

    Pass a pre-built ``ctx`` to avoid rebuilding LeanContext (e.g. when called
    from ``create_project`` which already has one).

    The *definitions* still live in XXXData.lean, but ``cert_data`` is no
    longer inert here: the proof tactics are generated from the shape of the
    predicates (see ``zrth.lean.tactics``), so this file has to be rewritten
    whenever they change — after ``--infer``, for instance.
    """
    module_name = project_name
    if ctx is None:
        ctx = LeanContext(module)

    cert_dir = project_dir / "Certificate"
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_file = cert_dir / "Certificate.lean"
    cert_file.write_text(generate_certificate_lean(ctx, cert_data))
    print(f"Wrote {cert_file}")
    return cert_file


def create_project(
    output_dir: Path,
    module: Module,
    project_name: str = "Certificate",
    template_dir: Path = TEMPLATE_DIR,
    executable: bool = False,
    cert_data: CertificateData | None = None,
    module_file: Path | str | None = None,
    ltl_project: Path | str | None = None,
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
        `module_file`:     Path to the Python source file used to create the module
                           (written as a debug artifact alongside the project).
        `ltl_project`:     A `lean-ltl-certifying` checkout (the `--fbk-proveit`
                           route). The lakefile then requires it and declares
                           the NA model's lean_lib, so the certificate that
                           route installs is buildable here.
    """
    project_dir = output_dir / project_name
    src_dir = project_dir / "System"
    dbg_dir = project_dir / "dbg"
    module_name = project_name

    # Create directory structure
    src_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created project directory: `{project_dir}`")

    # Debug artifacts: module representation and Python source
    dbg_dir.mkdir(parents=True, exist_ok=True)
    (dbg_dir / "system.txt").write_text(str(module))
    print("Wrote system.txt")
    if module_file is not None:
        py_src = Path(module_file).read_text()
        (dbg_dir / "system.py").write_text(py_src)
        print("Wrote system.py")

    # Render and write project-level files from templates
    lakefile = project_dir / "lakefile.toml"
    lakefile.write_text(
        generate_lakefile(
            project_name, executable=executable, ltl_project=ltl_project
        )
    )
    print(f"Wrote {lakefile}")

    toolchain = project_dir / "lean-toolchain"
    toolchain.write_text(
        render("project/lean-toolchain.j2", lean_toolchain=LEAN_TOOLCHAIN)
    )
    print(f"Wrote {toolchain}")

    hammer_file = project_dir / "ZerothHammer.lean"
    hammer_file.write_text(generate_zeroth_hammer_lean())
    print(f"Wrote {hammer_file}")

    (project_dir / "Certificate.lean").write_text(render("project/Certificate.lean.j2"))

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
    cert_terms: list = []
    if cert_data is not None:
        for field in (
            cert_data.prp,
            cert_data.inv,
            cert_data.init_pre,
            cert_data.update_pre,
            cert_data.ranking,
        ):
            if isinstance(field, list):
                cert_terms.extend(field)

    ctx = LeanContext(module, cert_terms=cert_terms)
    m2l = ModuleToLean4(ctx)

    root_lean = project_dir / "System.lean"
    root_lean.write_text(generate_root(scalar=m2l._can_scalarize()))
    print(f"Wrote root module {root_lean}")

    mod_file = src_dir / "System.lean"
    print(f"Generating `{mod_file.absolute()}`")
    mod_file.write_text(f"""\
/- Functional encoding of reactive module `{module_name}` -/
import Core.Box

{m2l.to_lean_functional()}
""")
    assert mod_file.exists()
    print(f"++ Generated {mod_file} ++")

    mod_file = src_dir / "Circ.lean"
    print(f"Generating `{mod_file.absolute()}`")
    mod_file.write_text(f"""\
/- Circuit encoding of reactive module `{module_name}` -/
import Core.Box
import System.System

{m2l.to_lean_circ()}
""")
    assert mod_file.exists()
    print(f"++ Generated {mod_file} ++")

    mat_rel_file = src_dir / "Rel.lean"
    print(f"Generating `{mat_rel_file.absolute()}`")
    mat_rel_file.write_text(f"""\
/- Matrix-domain relational encoding of reactive module `{module_name}` -/
import Core.Basic
import System.System

{m2l.to_lean_mat_rel()}
""")
    assert mat_rel_file.exists()
    print(f"++ Generated {mat_rel_file} ++")

    scalar_file = src_dir / "Scalar.lean"
    print(f"Generating `{scalar_file.absolute()}`")
    scalar_file.write_text(f"""\
/- Scalar encoding of reactive module `{module_name}` -/
import Core.Basic
import System.System

{m2l.to_lean_scalar()}
""")
    assert scalar_file.exists()
    print(f"++ Generated {scalar_file} ++")

    scalar_rel_file = src_dir / "ScalarRel.lean"
    print(f"Generating `{scalar_rel_file.absolute()}`")
    scalar_rel_file.write_text(f"""\
/- Scalar-relational encoding of reactive module `{module_name}` -/
import Core.Basic
import System.Scalar

{m2l.to_lean_rel()}
""")
    assert scalar_rel_file.exists()
    print(f"++ Generated {scalar_rel_file} ++")

    # -- certificate data (init_pre, inv, P, ranking — placeholders if no cert_data) --
    write_data_lean(project_dir, project_name, module, cert_data, ctx=ctx)

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
    write_certificate_lean(project_dir, project_name, module, cert_data, ctx=ctx)

    print()
    print(f"DONE: Project created at: {project_dir}")
    return project_dir


def generate_standalone_cert_lean(
    module: Module,
    cert_data: CertificateData | None = None,
) -> str:
    """Generate a self-contained certificate Lean file with init/update inlined.

    Unlike :func:`write_certificate_lean`, the functional module code
    (init, update) is inlined directly rather than imported.  ZerothHammer
    is always imported from the project-level ``ZerothHammer.lean``.

    Suitable for placing a single .lean file inside an existing lake project
    such as ``tests/lean/Certs/``.
    """
    cert_terms: list = []
    if cert_data is not None:
        for field in (
            cert_data.prp,
            cert_data.inv,
            cert_data.init_pre,
            cert_data.update_pre,
            cert_data.ranking,
        ):
            if isinstance(field, list):
                cert_terms.extend(field)

    ctx = LeanContext(module, cert_terms=cert_terms)
    m2l = ModuleToLean4(ctx)
    module_code = m2l.to_lean_functional()
    return generate_certificate_lean(
        ctx, cert_data=cert_data, module_inline=module_code
    )


def load_module_from_file(filepath: str, module_def: str = "module") -> Module:
    """
    Load modules from an external Python file.

    The file must define either a function `module_def` (which defaults into `module`):

        def module() -> Module:
            ...

    returning the Module, or it may define a class that inherits from `Module` and
    has constructor without arguments that create `Module`.
    """
    import importlib.util

    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Diagram file not found: {path}")

    spec = importlib.util.spec_from_file_location("_user_diagrams", path)
    if spec is None:
        raise RuntimeError(
            f"Failed to load module from file: `{filepath}`, module: `{module_def}`"
        )

    module = importlib.util.module_from_spec(spec)
    assert spec.loader, "No loaded created"
    spec.loader.exec_module(module)

    if not hasattr(module, module_def):
        raise AttributeError(
            f"Module file {path} must define `{module_def}` which is a function returning `Module` or a sub-class of `Module` with contructor without arguments."
        )

    module_def = getattr(module, module_def)
    result = module_def()
    if isinstance(result, Module):
        return result
    # main merged the old `Wrapper` into `Env`; wrap non-Module results (e.g.
    # a gym environment) into a reactive Module via `Env`.
    return Env(result)
