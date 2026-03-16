"""
Build a Lean4 project with Mathlib, Cslib, and a custom git dependency,
copy template library files, and generate a diagram Lean file.
"""

import shutil
import textwrap
from pathlib import Path

from zrth import Module, Wire, DType
from .diagram import ModuleToLean4, dtype_to_lean_ty


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

# Template files to copy into the project
TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_FILES = ["Basic.lean", "Box.lean", "LTL.lean"]


# ══════════════════════════════════════════════════════════════════════════
# Lean project files
# ══════════════════════════════════════════════════════════════════════════


def generate_lakefile(project_name: str, executable: bool = False) -> str:
    base = textwrap.dedent(f"""\

        name = "{project_name}"
        version = "0.1.0"
        defaultTargets = ["{project_name}"]

        [[lean_lib]]
        name = "{project_name}"

        [[require]]
        name = "cslib"
        scope = "leanprover"
        rev = "{CSLIB_REV}"

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


def generate_root(project_name: str, modules_names: list[str]) -> str:
    """
    Root module file (src/<ProjectName>.lean) that imports everything.

    `diagram_names` are names of files with wiring diagrams (reactive modules)
                    that need to be imported in the root module
    """
    lines = [
        # f"import {project_name}.Diagram",
    ]
    for dname in modules_names:
        lines.append(f"import {project_name}.{dname}")
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


def _lean_ty_list(wires: list[Wire]) -> str:
    """Render a Lean type list like '[.bool, .int]'."""
    return "[" + ", ".join(dtype_to_lean_ty(w) for w in wires) + "]"


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
    lines.append('  | none => throw (.userError s!"Invalid integer: {s.trimAscii.toString}")')
    lines.append("")
    lines.append("def showMat (m n : Nat) (mat : Fin m → Fin n → Int) : String :=")
    lines.append(
        "  let vals := (List.ofFn fun i => List.ofFn fun j => [toString (mat i j)]).flatten.flatten"
    )
    lines.append('  String.intercalate " " vals')
    lines.append("")

    # parseExtl
    lines.append(
        f"def parseExtl (tokens : Array String) : IO (ValTuple {_lean_ty_list(extl_next)}) := do"
    )
    lines.append(
        f'  if tokens.size < {total_extl_tokens} then throw (.userError "Expected {total_extl_tokens} input values")'
    )

    offset = 0
    parse_vars: list[str] = []
    for i, w in enumerate(extl_next):
        ty = dtype_to_lean_ty(w)
        var = f"e{i}"
        if ty == ".bool":
            lines.append(f"  let {var} := parseBool tokens[{offset}]!")
            offset += 1
        elif ty == ".int":
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
                f"  let {var} : Fin {m} → Fin {n} → Int := fun i j => arr{i}.get! (i.val * {n} + j.val)"
            )
            offset += m * n
        parse_vars.append(var)

    # Build ValTuple literal: (e0, (e1, ()))
    vt = "()"
    for var in reversed(parse_vars):
        vt = f"({var}, {vt})"
    lines.append(f"  pure {vt}")
    lines.append("")

    # showCtrl
    lines.append(f"def showCtrl (v : ValTuple {_lean_ty_list(ctrl_next)}) : String :=")
    # Destructure
    destr_vars: list[str] = []
    for i in range(len(ctrl_next)):
        destr_vars.append(f"v{i}")
    # Build destructuring pattern: let (v0, (v1, ())) := v
    pat = "()"
    for var in reversed(destr_vars):
        pat = f"({var}, {pat})"
    lines.append(f"  let {pat} := v")

    # Format each variable
    show_parts: list[str] = []
    for i, w in enumerate(ctrl_next):
        ty = dtype_to_lean_ty(w)
        var = f"v{i}"
        if ty == ".bool":
            show_parts.append(f"showBool {var}")
        elif ty == ".int":
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
    # Build the state type list for ctrl
    ctrl_ty_list = _lean_ty_list(ctrl_next)
    # For update: ValTuple.append needs the ctrl type list
    update_append_ty = _lean_ty_list(ctrl_latched)

    lines.append("def main : IO Unit := do")
    lines.append("  let stdin ← IO.getStdin")
    lines.append("  let line0 ← stdin.getLine")
    lines.append("  if line0.trimAscii.toString.isEmpty then return")
    lines.append('  let extl0 ← parseExtl (line0.trimAscii.toString.splitOn " " |>.toArray)')
    lines.append("  let mut state := init.fn extl0")
    lines.append("  IO.println (showCtrl state)")
    lines.append("  repeat do")
    lines.append("    let line ← stdin.getLine")
    lines.append("    if line.trimAscii.toString.isEmpty then break")
    lines.append('    let extl ← parseExtl (line.trimAscii.toString.splitOn " " |>.toArray)')
    lines.append(
        f"    let state' := update.fn (ValTuple.append {update_append_ty} state extl)"
    )
    lines.append("    IO.println (showCtrl state')")
    lines.append("    state := state'")
    lines.append("")

    return "\n".join(lines)


def generate_certificate_lean(
    project_name: str,
    module: Module,
    module_name: str,
    const_names: list[str],
    update_layer_count: int,
) -> str:
    """Generate Certificate.lean skeleton with sorry/TODO for user-provided parts."""
    extl_next = [pair[1] for pair in module.extl]
    ctrl_next = [pair[1] for pair in module.ctrl]

    extl_ty = "[" + ", ".join(dtype_to_lean_ty(w) for w in extl_next) + "]"
    ctrl_ty = "[" + ", ".join(dtype_to_lean_ty(w) for w in ctrl_next) + "]"
    ctrl_latched_ty = "[" + ", ".join(dtype_to_lean_ty(w) for w in [p[0] for p in module.ctrl]) + "]"

    lines: list[str] = []

    # Imports
    lines.append("import Mathlib.Algebra.BigOperators.Fin")
    lines.append(f"import {project_name}.Basic")
    lines.append(f"import {project_name}.Box")
    lines.append(f"import {project_name}.{module_name}")
    lines.append("")

    # User-provided definitions (with defaults / TODOs)
    lines.append("-- TODO: precondition on initial inputs, provided by user (defaults to `True`)")
    lines.append(f"def init_pre (e : ValTuple {extl_ty}) : Prop := True")
    lines.append("")
    lines.append("-- TODO: precondition on inputs in every update round, provided by user (defaults to `True`)")
    lines.append(f"def update_pre (e : ValTuple {extl_ty}) : Prop := True")
    lines.append("")
    lines.append("-- TODO: invariant of the system, provided by user")
    lines.append(f"def inv (s : ValTuple {ctrl_ty}) : Prop := True")
    lines.append("")
    lines.append("-- TODO: property to prove to hold infinitely often, provided by user")
    lines.append(f"def P (s : ValTuple {ctrl_ty}) : Prop := sorry")
    lines.append("")
    lines.append("-- TODO: ranking function on (¬P ∧ I)-states, provided by user")
    lines.append(f"def ranking (s : ValTuple {ctrl_ty}) : Nat := sorry")
    lines.append("")
    lines.append("")

    # ReactiveModule definition
    lines.append(f"def RM : ReactiveModule")
    lines.append(f"          (ValTuple {extl_ty}) (ValTuple {ctrl_ty})")
    lines.append(":= {")
    lines.append("    init := init.fn")
    lines.append(f"    update := fun x e => update.fn (ValTuple.append {ctrl_latched_ty} x e)")
    lines.append("    init_pre := init_pre")
    lines.append("    update_pre := update_pre")
    lines.append("}")
    lines.append("")

    # box_simp tactic
    const_list = ", ".join(const_names) if const_names else ""
    layer_list = ", ".join(f"L{i+1}" for i in range(update_layer_count))

    lines.append("-- tactic that essentially \"evaluates\" the boxes")
    lines.append('macro "box_simp" : tactic =>')
    lines.append("  `(tactic| (")
    lines.append("    simp only [Box.seq, Box.par, Box.dup, Box.id, Box.swap,")
    lines.append("               Box.add, Box.mul, Box.lt, Box.or, Box.min, Box.max,")
    lines.append("               Box.ite, Box.const, Box.destr,")
    lines.append("               Box.matMul, Box.matAdd, Box.matGet,")
    lines.append("               ValTuple.split, ValTuple.append,")
    lines.append("               Function.comp, Ty.denote,")
    lines.append("               List.nil_append, List.cons_append,")
    lines.append("               ValTuple.append_split, ValTuple.append_ite,")
    lines.append("               MatAdd_apply, MatMul_apply, MatZero_apply,")
    lines.append("               Bool.or_eq_true, decide_eq_true_eq,")
    lines.append("               Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue,")
    simp_extras = "init, update, inv"
    if const_list:
        simp_extras += f",\n               {const_list}"
    if layer_list:
        simp_extras += f",\n               {layer_list}"
    lines.append(f"               {simp_extras}")
    lines.append("               ]")
    lines.append("    <;> (try omega)")
    lines.append("    <;> (try (split <;> simp_all [MatAdd_apply, MatMul_apply, MatZero_apply,")
    if const_list:
        lines.append(f"                                    {const_list},")
    lines.append("                                    Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue]")
    lines.append("              <;> omega))")
    lines.append("    <;> (try (split <;> split <;> simp_all [MatAdd_apply, MatMul_apply, MatZero_apply,")
    if const_list:
        lines.append(f"                                              {const_list},")
    lines.append("                                              Fin.sum_univ_succ, Fin.sum_univ_zero,")
    lines.append("                                              Fin.isValue]")
    lines.append("              <;> omega))))")
    lines.append("")

    # init_inv theorem
    lines.append("theorem init_inv :")
    lines.append("  ∀ s, RM.init_pre s → inv (RM.init s) := by")
    lines.append("   intro s hpre")
    lines.append("   unfold inv")
    lines.append("   simp [RM]")
    lines.append("   try box_simp")
    lines.append("")

    # step_inv theorem
    lines.append("theorem step_inv :")
    lines.append("  ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e) := by")
    lines.append("   intro s e hpre")
    lines.append("   unfold inv")
    lines.append("   simp_all [RM]")
    lines.append("   try box_simp")
    lines.append("")
    lines.append("")

    # LTS section
    lines.append("section LTS")
    lines.append("")
    lines.append("def lts := RM.toLTS'")
    lines.append("")

    # DecidablePred P
    lines.append("-- TODO: make this automatic/generated")
    lines.append("instance : DecidablePred P := sorry")
    lines.append("")

    # hinv' theorem
    lines.append("theorem hinv' : lts.StateSet_isInductiveInitial inv := by")
    lines.append("  unfold LTS'.StateSet_isInductiveInitial")
    lines.append("  unfold LTS'.StateSet_isInductive")
    lines.append("  constructor")
    lines.append("  · intro s hs")
    lines.append("    unfold lts at hs")
    lines.append("    simp [ReactiveModule.toLTS', ReactiveModule.LTS_init, RM, init_pre] at hs")
    lines.append("    obtain ⟨x, h⟩ := hs")
    lines.append("    rw [← h]")
    lines.append("    simp [Membership.mem]")
    lines.append("    apply init_inv")
    lines.append("    simp [RM, init_pre]")
    lines.append("    sorry")
    lines.append("  · intro s s' ⟨hs, l, hstep⟩")
    lines.append("    unfold lts at hstep")
    lines.append("    simp only [ReactiveModule.toLTS', ReactiveModule.LTS_update] at hstep")
    lines.append("    rw [← hstep.2]")
    lines.append("    exact step_inv s l ⟨hstep.1, hs⟩")
    lines.append("")

    # hinv theorem
    lines.append("theorem hinv : lts.StateSet_isInvariant inv := by")
    lines.append("  apply LTS'.StateSet_ind_init_is_inv lts")
    lines.append("  exact hinv'")
    lines.append("")
    lines.append("")

    # Commented-out ranking/Buchi section
    lines.append("-- TODO: uncomment and fill in ranking proof once `P` and `ranking` are defined")
    lines.append("-- set_option maxHeartbeats 1300 in")
    lines.append("-- theorem hrank : ∀ s s', inv s → ¬(P s) → (∃ l, lts.Tr s l s') →")
    lines.append("--     ranking s' < ranking s := by")
    lines.append("--   sorry")
    lines.append("")
    lines.append("-- def buchi := rule_buchi")
    lines.append("--   lts")
    lines.append("--   P")
    lines.append("--   inv")
    lines.append("--   hinv")
    lines.append("--   ranking")
    lines.append("--   hrank")
    lines.append("")
    lines.append("end LTS")
    lines.append("")

    return "\n".join(lines)


def generate_module_file(
    src_dir: Path, project_name: str, module: Module, module_name: str
) -> tuple[Path, ModuleToLean4]:
    """
    Generate .lean file with reactive module. Returns (Path, ModuleToLean4).
    """
    out = src_dir / f"{module_name}.lean"
    print(f"Generating `{out.absolute()}`")

    m2l = ModuleToLean4(module)

    out.write_text(f"""\
/- Code generated for reactive module `{module_name}` -/
import {project_name}.Box

{m2l.to_lean()}
""")
    return out, m2l


def create_project(
    output_dir: Path,
    module: Module,
    module_name: str = "ReactiveModule",
    project_name: str = "Rea",
    template_dir: Path = TEMPLATE_DIR,
    executable: bool = False,
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

    # Copy template files
    for tmpl_name in TEMPLATE_FILES:
        src_path = template_dir / tmpl_name
        dst_path = src_dir / tmpl_name
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"Copied template {tmpl_name}")
        else:
            # If template doesn't exist yet, write a placeholder
            dst_path.write_text(
                f"-- TODO: replace with actual {tmpl_name}\n"
                f"-- Expected at: {src_path}\n"
            )
            print(
                f"WARNING: Template {tmpl_name} not found at {src_path}, wrote placeholder"
            )

    # Generate reactive module files
    mod_file, m2l = generate_module_file(src_dir, project_name, module, module_name)
    assert mod_file.exists()
    print(f"++ Generated {mod_file} ++")

    # Generate Certificate.lean skeleton
    cert_dir = project_dir / "Certificate"
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_file = cert_dir / "Certificate.lean"
    cert_file.write_text(
        generate_certificate_lean(
            project_name, module, module_name,
            m2l.const_names, m2l.update_layer_count,
        )
    )
    print(f"Wrote {cert_file}")

    # Write Certificate import wrapper
    cert_wrapper = project_dir / "Certificate.lean"
    cert_wrapper.write_text("import Certificate.Certificate\n")
    print(f"Wrote {cert_wrapper}")

    # Write root module
    root_module = src_dir.parent / f"{project_name}.lean"
    root_module.write_text(generate_root(project_name, [module_name]))
    print(f"Wrote root module {root_module}")

    # Generate Main.lean for executable
    if executable:
        main_lean = project_dir / "Main.lean"
        main_lean.write_text(generate_main_lean(project_name, module, module_name))
        print(f"Wrote {main_lean}")

    print()
    print(f"DONE: Project created at: {project_dir}")
    return project_dir


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════


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
    return module_def()
