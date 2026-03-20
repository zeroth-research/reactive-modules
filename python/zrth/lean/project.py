"""
Build a Lean4 project with Mathlib, Cslib, and a custom git dependency,
copy template library files, and generate a diagram Lean file.
"""

import shutil
import textwrap
from pathlib import Path

from zrth import Module, Wire
from .diagram import (
    ModuleToLean4,
    _translate_terms,
    _product_type,
    _append_expr,
    dtype_to_lean_native,
)


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
CORE_FILES = ["Basic.lean", "Box.lean", "LTL.lean"]


# ══════════════════════════════════════════════════════════════════════════
# Lean project files
# ══════════════════════════════════════════════════════════════════════════


def generate_lakefile(project_name: str, executable: bool = False) -> str:
    base = textwrap.dedent(f"""\

        name = "{project_name}"
        version = "0.1.0"
        defaultTargets = ["{project_name}"]

        [[lean_lib]]
        name = "Core"

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
        ty = dtype_to_lean_native(w)
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
        ty = dtype_to_lean_native(w)
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

    lines.append("def main : IO Unit := do")
    lines.append("  let stdin ← IO.getStdin")
    lines.append("  let line0 ← stdin.getLine")
    lines.append("  if line0.trimAscii.toString.isEmpty then return")
    lines.append(
        '  let extl0 ← parseExtl (line0.trimAscii.toString.splitOn " " |>.toArray)'
    )
    lines.append("  let mut state := init extl0")
    lines.append("  IO.println (showCtrl state)")
    lines.append("  repeat do")
    lines.append("    let line ← stdin.getLine")
    lines.append("    if line.trimAscii.toString.isEmpty then break")
    lines.append(
        '    let extl ← parseExtl (line.trimAscii.toString.splitOn " " |>.toArray)'
    )
    lines.append(f"    let state' := update {main_append}")
    lines.append("    IO.println (showCtrl state')")
    lines.append("    state := state'")
    lines.append("")

    return "\n".join(lines)


def generate_certificate_lean(
    project_name: str,
    module: Module,
    module_name: str,
    m2l: ModuleToLean4,
    inv_terms: list | None = None,
    init_pre_terms: list | None = None,
    update_pre_terms: list | None = None,
    ranking_terms: list | None = None,
    p_terms: list | None = None,
) -> str:
    """Generate Certificate.lean with compiled or placeholder definitions."""
    extl_next = [pair[1] for pair in module.extl]
    ctrl_next = [pair[1] for pair in module.ctrl]

    ctrl_latched = [p[0] for p in module.ctrl]
    extl_native = _product_type(extl_next)
    ctrl_native = _product_type(ctrl_next)
    append = _append_expr("x", len(ctrl_latched), "e", len(extl_next))

    # Extract constants from certificate term lists
    existing_const_count = len(m2l._const_defs)
    for terms in [inv_terms, init_pre_terms, update_pre_terms, ranking_terms, p_terms]:
        if terms is not None:
            m2l._extract_constants(terms)
    cert_const_defs = m2l._const_defs[existing_const_count:]
    const_names = list(m2l._constants.values())

    def _cert_body(terms, block_inputs, param_name):
        """Compile a certificate term list into a Lean function body."""
        output = [terms[-1].write[0]]
        return _translate_terms(terms, block_inputs, output, m2l._constants, param_name)

    lines: list[str] = []

    # Imports
    lines.append("import Mathlib.Algebra.BigOperators.Fin")
    lines.append("import Core.Basic")
    lines.append(f"import {project_name}.{module_name}")
    lines.append("")

    # Certificate-specific constants (if any)
    if cert_const_defs:
        for cdef in cert_const_defs:
            lines.append(cdef)
        lines.append("")

    # init_pre
    if init_pre_terms is not None:
        body = _cert_body(init_pre_terms, extl_next, "e")
        lines.append(f"def init_pre (e : {extl_native}) : Prop :=")
        lines.append(body)
    else:
        lines.append(f"def init_pre (e : {extl_native}) : Prop := True")
    lines.append("")

    # update_pre
    if update_pre_terms is not None:
        body = _cert_body(update_pre_terms, extl_next, "e")
        lines.append(f"def update_pre (e : {extl_native}) : Prop :=")
        lines.append(body)
    else:
        lines.append(f"def update_pre (e : {extl_native}) : Prop := True")
    lines.append("")

    # inv
    if inv_terms is not None:
        body = _cert_body(inv_terms, ctrl_next, "s")
        lines.append(f"def inv (s : {ctrl_native}) : Prop :=")
        lines.append(body)
    else:
        lines.append(f"def inv (s : {ctrl_native}) : Prop := True")
    lines.append("")

    # P
    if p_terms is not None:
        body = _cert_body(p_terms, ctrl_next, "s")
        lines.append(f"def P (s : {ctrl_native}) : Prop :=")
        lines.append(body)
    else:
        lines.append(f"def P (s : {ctrl_native}) : Prop := sorry")
    lines.append("")

    # DecidablePred P — must come after P and before ranking
    lines.append("instance : DecidablePred P := inferInstance")
    lines.append("")

    # ranking
    if ranking_terms is not None:
        body = _cert_body(ranking_terms, ctrl_next, "s")
        lines.append(f"def ranking (s : {ctrl_native}) : Nat :=")
        lines.append(body)
    else:
        lines.append(f"def ranking (s : {ctrl_native}) : Nat := sorry")
    lines.append("")
    lines.append("")

    # ReactiveModule definition
    lines.append("def RM : ReactiveModule")
    lines.append(f"          ({extl_native}) ({ctrl_native})")
    lines.append(":= {")
    lines.append("    init := init")
    lines.append(f"    update := fun x e => update {append}")
    lines.append("    init_pre := init_pre")
    lines.append("    update_pre := update_pre")
    lines.append("}")
    lines.append("")

    # simp_mod tactic — unfolds module definitions for proof automation
    const_list = ", ".join(const_names) if const_names else ""

    lines.append("-- tactic that unfolds module definitions and simplifies")
    lines.append('macro "simp_mod" : tactic =>')
    lines.append("  `(tactic| (")
    simp_lemmas = "init, update, inv"
    if const_list:
        simp_lemmas += f",\n               {const_list}"
    simp_lemmas += (
        ",\n               MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply"
    )
    simp_lemmas += ",\n               Bool.or_eq_true, decide_eq_true_eq"
    simp_lemmas += ",\n               Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue"
    lines.append(f"    simp only [{simp_lemmas}]")
    lines.append("    <;> (try omega)")
    lines.append("    <;> (try (split <;> simp_all <;> omega))")
    lines.append("    <;> (try (split <;> split <;> simp_all <;> omega))))")
    lines.append("")

    # simp_inv tactic — reduces module defs then solves CNF goals
    lines.append(
        "-- tactic that reduces module definitions and solves CNF invariant goals"
    )
    lines.append('macro "simp_inv" : tactic =>')
    lines.append("  `(tactic| (")
    inv_lemmas = "RM, ReactiveModule.init, ReactiveModule.update,\n"
    inv_lemmas += "               ReactiveModule.init_pre, ReactiveModule.update_pre,\n"
    inv_lemmas += "               init, update, inv, init_pre, update_pre"
    if const_list:
        inv_lemmas += f",\n               {const_list}"
    inv_lemmas += (
        ",\n               MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply"
    )
    inv_lemmas += ",\n               Bool.or_eq_true, decide_eq_true_eq"
    inv_lemmas += ",\n               Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue"
    lines.append(f"    simp only [{inv_lemmas}] at *")
    lines.append("    <;> first")
    lines.append("      | trivial")
    lines.append("      | omega")
    lines.append("      | (simp_all; omega)")
    lines.append("      | (repeat' constructor)")
    lines.append("        <;> first")
    lines.append("          | trivial | omega")
    lines.append("          | (simp_all; omega)")
    lines.append("          | (left; omega) | (right; omega)")
    lines.append("          | (left; simp_all; omega) | (right; simp_all; omega)")
    lines.append("          | (right; right; omega) | (right; right; simp_all; omega)")
    lines.append("      | (split <;> simp_all <;> omega)")
    lines.append("      | (split <;> split <;> simp_all <;> omega)))")
    lines.append("")

    # init_inv theorem
    lines.append("theorem init_inv :")
    lines.append("  ∀ s, RM.init_pre s → inv (RM.init s) := by")
    lines.append("   intro s hpre")
    lines.append("   simp_inv")
    lines.append("")

    # step_inv theorem
    lines.append("theorem step_inv :")
    lines.append("  ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e) := by")
    lines.append("   intro s e ⟨hpre, hinv⟩")
    lines.append("   simp_inv")
    lines.append("")
    lines.append("")

    # LTS section
    lines.append("section LTS")
    lines.append("")
    lines.append("def lts := RM.toLTS'")
    lines.append("")

    # hinv' theorem
    lines.append("theorem hinv' : lts.StateSet_isInductiveInitial inv := by")
    lines.append("  unfold LTS'.StateSet_isInductiveInitial")
    lines.append("  unfold LTS'.StateSet_isInductive")
    lines.append("  constructor")
    lines.append("  · intro s hs")
    lines.append("    unfold lts at hs")
    lines.append(
        "    simp [ReactiveModule.toLTS', ReactiveModule.LTS_init, RM, init_pre] at hs"
    )
    lines.append("    unfold inv")
    lines.append("    simp [Membership.mem]")
    lines.append("  · intro s s' ⟨hs, l, hstep⟩")
    lines.append("    unfold lts at hstep")
    lines.append(
        "    simp only [ReactiveModule.toLTS', ReactiveModule.LTS_update] at hstep"
    )
    lines.append("    rw [← hstep.2]")
    lines.append("    exact step_inv s l ⟨hstep.1, hs⟩")
    lines.append("")

    # hinv theorem
    lines.append("theorem hinv : lts.StateSet_isInvariant inv := by")
    lines.append("  apply LTS'.StateSet_ind_init_is_inv lts")
    lines.append("  exact hinv'")
    lines.append("")
    lines.append("")

    # hrank theorem
    lines.append("theorem hrank : ∀ s s', inv s → ¬(P s) → (∃ l, lts.Tr s l s') →")
    lines.append("    ranking s' < ranking s := by")
    lines.append("    intro s s' hi hP htr")
    lines.append("    unfold lts at htr")
    lines.append(
        "    simp only [ReactiveModule.toLTS', ReactiveModule.LTS_update] at htr"
    )
    lines.append("    obtain ⟨l, hpre, heq⟩ := htr")
    lines.append("    rw [← heq]")
    lines.append("    unfold ranking P at *")
    lines.append("    unfold inv at *")
    lines.append("    simp only [RM, ReactiveModule.update]")
    lines.append("    unfold update")
    lines.append("    simp_mod")
    lines.append("")
    lines.append("def buchi := rule_buchi")
    lines.append("  lts")
    lines.append("  P")
    lines.append("  inv")
    lines.append("  hinv")
    lines.append("  ranking")
    lines.append("  hrank")
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
import Core.Box

{m2l.to_lean()}
""")
    return out, m2l


def create_project(
    output_dir: Path,
    module: Module,
    project_name: str = "Certificate",
    template_dir: Path = TEMPLATE_DIR,
    executable: bool = False,
    inv_terms: list | None = None,
    init_pre_terms: list | None = None,
    update_pre_terms: list | None = None,
    ranking_terms: list | None = None,
    p_terms: list | None = None,
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

    # Copy core template files to Core/ package
    core_dir = project_dir / "Core"
    core_dir.mkdir(parents=True, exist_ok=True)
    for tmpl_name in CORE_FILES:
        src_path = template_dir / tmpl_name
        dst_path = core_dir / tmpl_name
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"Copied template {tmpl_name} -> Core/")
        else:
            dst_path.write_text(
                f"-- TODO: replace with actual {tmpl_name}\n"
                f"-- Expected at: {src_path}\n"
            )
            print(
                f"WARNING: Template {tmpl_name} not found at {src_path}, wrote placeholder"
            )

    # Generate reactive module files
    module_name = project_name
    mod_file, m2l = generate_module_file(src_dir, project_name, module, module_name)
    assert mod_file.exists()
    print(f"++ Generated {mod_file} ++")

    # Generate Certificate.lean skeleton
    cert_dir = project_dir / "Certificate"
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_file = cert_dir / "Certificate.lean"
    cert_file.write_text(
        generate_certificate_lean(
            project_name,
            module,
            module_name,
            m2l,
            inv_terms=inv_terms,
            init_pre_terms=init_pre_terms,
            update_pre_terms=update_pre_terms,
            ranking_terms=ranking_terms,
            p_terms=p_terms,
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
