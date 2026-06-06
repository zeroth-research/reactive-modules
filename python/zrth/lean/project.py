"""
Build a Lean4 project with Mathlib, Cslib, and a custom git dependency,
copy template library files, and generate a diagram Lean file.
"""

from zrth.lean.common import LeanContext, dtype_to_lean_type

from zrth.lean.cert import (
    generate_certificate_lean,
    generate_zeroth_hammer_lean,
    CertificateData,
)

import shutil
import textwrap
from pathlib import Path

from zrth import Module, Wire, Wrapper
from .native import (
    _product_type,
    _append_expr,
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

# Template files to copy into Core package
TEMPLATE_DIR = Path(__file__).parent / "templates"
CORE_FILES = ["Basic.lean", "Box.lean", "LTL.lean", "Mat.lean"]
LEAN_AI_FILES = ["LeanAI.lean", "LeanAI"]


# ══════════════════════════════════════════════════════════════════════════
# Lean project files
# ══════════════════════════════════════════════════════════════════════════


def generate_lakefile(project_name: str, executable: bool = False) -> str:
    base = textwrap.dedent(f"""\

        name = "{project_name}"
        version = "0.1.0"
        defaultTargets = ["{project_name}", "Certificate"]

        [[lean_lib]]
        name = "Core"

        [[lean_lib]]
        name = "LeanAI"

        [[lean_lib]]
        name = "ZerothHammer"

        [[lean_lib]]
        name = "Certificate"

        [[lean_lib]]
        name = "{project_name}"

        [[require]]
        name = "cslib"
        scope = "leanprover"
        rev = "{CSLIB_REV}"

        [[require]]
        name = "smt"
        git = "https://github.com/ufmg-smite/lean-smt.git"
        rev = "f58d19d5d0803fcccb5ccb1b4473774dd2ae9f9a"

       #[[require]]
       #name = "mylib"
       #git = "{PROOFS_URL}"
       #rev = "main"
    """)
    if executable:
        base += textwrap.dedent("""\

            [[lean_exe]]
            name = "main"
            root = "Main"
        """)
    return base


def generate_root(
    project_name: str, modules_names: list[str], scalar: bool = True
) -> str:
    """
    Root module file (src/<ProjectName>.lean) that imports everything.

    `diagram_names` are names of files with wiring diagrams (reactive modules)
                    that need to be imported in the root module.
    `scalar` controls whether the Scalar and Rel encodings are imported
             (should be False for modules with non-scalar / matrix wires).
    """
    lines = [
        # f"import {project_name}.Diagram",
    ]
    for dname in modules_names:
        lines.append(f"import {project_name}.{dname}")
        lines.append(f"import {project_name}.{dname}Circ")
        lines.append(f"import {project_name}.{dname}Rel")
        if scalar:
            lines.append(f"import {project_name}.{dname}Scalar")
            lines.append(f"import {project_name}.{dname}ScalarRel")
    return "\n".join(lines) + "\n"


def _token_count(wire: Wire) -> int:
    """When reading a value from user, we need to know how many primitive values (Int/Bool)
    to read to get a single input value (which can be e.g., a matrix).
    This function computes this.
    """
    shape = wire.dtype.shape
    if shape == []:
        return 1
    if len(shape) == 1:
        return shape[0]
    if len(shape) == 2:  # matrix
        return shape[0] * shape[1]
    raise ValueError("Unsupported DType for token count")


def generate_main_lean(project_name: str, module: Module, module_name: str) -> str:
    """Generate Main.lean source that runs init/update in a stdin/stdout loop."""
    extl_next = [pair[1] for pair in module.extl]
    ctrl_next = [pair[1] for pair in module.ctrl]
    ctrl_latched = [pair[0] for pair in module.ctrl]

    total_extl_tokens = sum(_token_count(w) for w in extl_next)
    total_ctrl_tokens = sum(_token_count(w) for w in ctrl_next)

    lines: list[str] = []

    # Imports
    lines.append(f"import {project_name}.{module_name}")
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
    lines.append("def showMat (m n : Nat) (mat : Fin m → Fin n → Int) : String :=")
    lines.append(
        "  let vals := (List.ofFn fun i => List.ofFn fun j => [toString (mat i j)]).flatten.flatten"
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
        ty = dtype_to_lean_type(w)
        var = f"e{i}"
        if ty == "Bool":
            lines.append(f"  let {var} := parseBool tokens[{offset}]!")
            offset += 1
        elif ty == "Int":
            lines.append(f"  let {var} ← parseIntOrFail tokens[{offset}]!")
            offset += 1
        else:
            # .mat m n
            dt = w.dtype
            shape = dt.shape
            m = shape[0]
            n = shape[1] if len(shape) == 2 else 1
            # Parse m*n tokens into a matrix
            lines.append(f"  let mut arr{i} : Array Int := #[]")
            lines.append(f"  for k in List.range {m * n} do")
            lines.append(f"    let v ← parseIntOrFail tokens[{offset} + k]!")
            lines.append(f"    arr{i} := arr{i}.push v")
            lines.append(
                f"  let {var} : Fin {m} → Fin {n} → Int := fun i j => arr{i}[i.val * {n} + j.val]!"
            )
            offset += m * n
        parse_vars.append(var)

    # Build ValTuple literal: (e0, (e1, ()))
    vt = f"({', '.join(reversed(parse_vars))})"
    lines.append(f"  pure {vt}")
    lines.append("")

    # showCtrl
    lines.append(f"def showCtrl (v : {_product_type(ctrl_next)}) : String :=")
    # Destructure
    destr_vars: list[str] = []
    for i in range(len(ctrl_next)):
        destr_vars.append(f"v{i}")
    # Build destructuring pattern: let (v0, v1) := v
    pat = f"({', '.join(reversed(destr_vars))})"
    lines.append(f"  let {pat} := v")

    # Format each variable
    show_parts: list[str] = []
    for i, w in enumerate(ctrl_next):
        ty = dtype_to_lean_type(w)
        var = f"v{i}"
        if ty == "Bool":
            show_parts.append(f"showBool {var}")
        elif ty == "Int":
            show_parts.append(f"toString {var}")
        else:
            dt = w.dtype
            shape = dt.shape
            m = shape[0]
            n = shape[1] if len(shape) == 2 else 1
            show_parts.append(f"showMat {m} {n} {var}")

    if len(show_parts) == 1:
        lines.append(f"  {show_parts[0]}")
    else:
        # Use s!"..." interpolation to join with spaces
        interp = " ".join(f"{{{p}}}" for p in show_parts)
        lines.append(f'  s!"{interp}"')
    lines.append("")

    # main function
    main_append = _append_expr("state", len(ctrl_next), "extl", len(extl_next))

    lines.append(f"""\
def main : IO Unit := do
  let stdin ← IO.getStdin
  let line0 ← stdin.getLine
  if line0.trimAscii.toString.isEmpty then return
  let extl0 ← parseExtl (line0.trimAscii.toString.splitOn " " |>.toArray)
  let mut state := init extl0
  IO.println (showCtrl state)
  repeat do
    let line ← stdin.getLine
    if line.trimAscii.toString.isEmpty then break
    let extl ← parseExtl (line.trimAscii.toString.splitOn " " |>.toArray)
    let state' := update {main_append}
    IO.println (showCtrl state')
    state := state'
""")

    return "\n".join(lines)


def write_certificate_lean(
    project_dir: Path,
    project_name: str,
    module: Module,
    cert_data: CertificateData | None,
) -> Path:
    """Write/overwrite the Certificate.lean file using `cert_data`.

    Builds a fresh `LeanContext` so term-typed cert fields are registered
    with the constants table consistently with codegen.  String fields are
    expected to be Lean expressions; call `smt_predicates_to_lean` first if
    they are SMT-LIB.
    """
    module_name = project_name
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

    cert_dir = project_dir / "Certificate"
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_file = cert_dir / "Certificate.lean"
    cert_file.write_text(
        generate_certificate_lean(
            project_name,
            module_name,
            ctx,
            cert_data=cert_data,
            hammer_import="ZerothHammer",
        )
    )
    print(f"Wrote {cert_file}")

    cert_wrapper = project_dir / "Certificate.lean"
    cert_wrapper.write_text("import Certificate.Certificate\n")
    return cert_file


def create_project(
    output_dir: Path,
    module: Module,
    project_name: str = "Certificate",
    template_dir: Path = TEMPLATE_DIR,
    executable: bool = False,
    cert_data: CertificateData | None = None,
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
    """
    project_dir = output_dir / project_name
    src_dir = project_dir / project_name
    module_name = project_name

    # Create directory structure
    src_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created project directory: `{project_dir}`")

    # Create lakefile.lean
    lakefile = project_dir / "lakefile.toml"
    lakefile.write_text(generate_lakefile(project_name, executable=executable))
    print(f"Wrote {lakefile}")

    # Write lean-toolchain
    toolchain = project_dir / "lean-toolchain"
    toolchain.write_text(LEAN_TOOLCHAIN + "\n")
    print(f"Wrote {toolchain}")

    # Write root module (placeholder; rewritten below after m2l is available)
    root_module = src_dir.parent / f"{project_name}.lean"

    # Write ZerothHammer.lean — standalone tactic, imported by Certificate
    hammer_file = project_dir / "ZerothHammer.lean"
    hammer_file.write_text(generate_zeroth_hammer_lean())
    print(f"Wrote {hammer_file}")

    # Copy template files to Core/ package
    # TODO: put these templates into `Core/` subfolder in `templates`
    core_dir = project_dir / "Core"
    core_dir.mkdir(parents=True, exist_ok=True)
    for tmpl_name in CORE_FILES:
        src_path = template_dir / tmpl_name
        dst_path = core_dir / tmpl_name
        if src_path.exists():
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"Copied template directory {tmpl_name} -> Core/")
            else:
                shutil.copy2(src_path, dst_path)
                print(f"Copied template file {tmpl_name} -> Core/")
        else:
            raise RuntimeError(
                f"WARNING: Template file/dir `{tmpl_name}` not found at {src_path}"
            )

    # Copy template files to Core/ package
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
                f"WARNING: Template file/dir `{tmpl_name}` not found at {src_path}"
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

    root_module.write_text(generate_root(project_name, [module_name], scalar=m2l._can_scalarize()))
    print(f"Wrote root module {root_module}")

    mod_file = src_dir / f"{module_name}.lean"
    print(f"Generating `{mod_file.absolute()}`")

    mod_file.write_text(f"""\
/- Code generated for reactive module `{module_name}` -/
import Core.Box

{m2l.to_lean_functional()}
""")
    assert mod_file.exists()
    print(f"++ Generated {mod_file} ++")

    mod_file = src_dir / f"{module_name}Circ.lean"
    print(f"Generating `{mod_file.absolute()}`")

    mod_file.write_text(f"""\
/- Code generated for reactive module `{module_name}` as circuit -/
import Core.Box
import {project_name}.{module_name}

{m2l.to_lean_circ()}
""")
    assert mod_file.exists()
    print(f"++ Generated {mod_file} ++")

    mat_rel_file = src_dir / f"{module_name}Rel.lean"
    print(f"Generating `{mat_rel_file.absolute()}`")
    mat_rel_file.write_text(f"""\
/- Relational encoding (matrix domain) for reactive module `{module_name}` -/
import Core.Basic
import {project_name}.{module_name}

{m2l.to_lean_mat_rel()}
""")
    assert mat_rel_file.exists()
    print(f"++ Generated {mat_rel_file} ++")

    scalar_file = src_dir / f"{module_name}Scalar.lean"
    print(f"Generating `{scalar_file.absolute()}`")
    scalar_file.write_text(f"""\
/- Scalar encoding for reactive module `{module_name}` -/
import Core.Basic
import {project_name}.{module_name}

{m2l.to_lean_scalar()}
""")
    assert scalar_file.exists()
    print(f"++ Generated {scalar_file} ++")

    rel_file = src_dir / f"{module_name}ScalarRel.lean"
    print(f"Generating `{rel_file.absolute()}`")
    rel_file.write_text(f"""\
/- Relational encoding for reactive module `{module_name}` -/
import Core.Basic
import {project_name}.{module_name}Scalar

{m2l.to_lean_rel()}
""")
    assert rel_file.exists()
    print(f"++ Generated {rel_file} ++")

    # -- empty file for data --
    mod_file = src_dir / f"{module_name}Data.lean"
    print(f"Generating `{mod_file.absolute()}`")

    mod_file.write_text("")
    assert mod_file.exists()
    print(f"++ Generated {mod_file} ++")

    # ----------------------------------------------------------
    # Generate Main.lean for executable
    # ----------------------------------------------------------
    if executable:
        main_lean = project_dir / "Main.lean"
        main_lean.write_text(generate_main_lean(project_name, module, module_name))
        print(f"Wrote {main_lean}")

    print()
    print(f"DONE: Project created at: {project_dir}")
    return project_dir


def generate_standalone_cert_lean(
    name: str,
    module: Module,
    cert_data: CertificateData | None = None,
    *,
    hammer_import: str = "ZerothHammer",
) -> str:
    """Generate a self-contained certificate Lean file with init/update inlined.

    Unlike :func:`write_certificate_lean`, no ``import <name>.<name>`` is
    emitted.  Instead the functional module code (init, update) is inlined
    directly and ``zeroth_hammer`` is imported from *hammer_import*.

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
        name,
        name,
        ctx,
        cert_data=cert_data,
        hammer_import=hammer_import,
        module_inline=module_code,
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
    return Wrapper(result)
