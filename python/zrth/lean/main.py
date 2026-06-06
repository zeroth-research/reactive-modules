"""CLI entry point for generating a Lean4 project from a reactive module.

Entry point: ``uv run verith``

Quick start
-----------
A *module file* is a Python file that defines a ``module()`` function (or
class) returning a :class:`zrth.Module`.  Optionally it may also define plain
``init()`` / ``update()`` functions so that the AI inference (``--infer``) can
read the high-level logic.

Generate a bare Lean project (all certificate fields left as ``sorry``)::

    uv run verith mymodule.py -o out/ -p MyProject

Pass a property so the certificate knows what to prove::

    uv run verith mymodule.py -P "x == 0" -o out/ -p MyProject

Ask the AI to infer the invariant and ranking function automatically::

    uv run verith mymodule.py -P "x == 0" --infer -o out/ -p MyProject

Use a local LLM via Ollama instead of Claude::

    uv run verith mymodule.py -P "x == 0" --infer \\
        --model qwen3-coder --base-url http://localhost:11434/v1 \\
        -o out/ -p MyProject

AI inference requirements
-------------------------
* **Claude (default)**: set ``ANTHROPIC_API_KEY`` and install
  ``pip install zrth[ai]``.
* **Local LLM (Ollama, vLLM, …)**: install ``pip install zrth[ai-local]``
  and provide ``--base-url`` pointing to an OpenAI-compatible endpoint.

Module file format
------------------
The file must expose a callable named ``module`` (override with ``-d``) that
returns a :class:`zrth.Module`::

    # mymodule.py
    from zrth import Wire, Module, DType as dt
    from zrth.analyzer import convert_method

    def init():
        return 0

    def update(old_x):
        x = old_x + 1
        if x == 10:
            return 0
        return x

    def module() -> Module:
        state = (Wire(dt.Int([1])), Wire(dt.Int([1])))
        init_terms   = convert_method(init,   {},               [state[1]])
        update_terms = convert_method(update, {"old_x": state}, [state[1]])
        return Module.sequential(init_terms, update_terms, obs=[state])
"""

import argparse
from pathlib import Path

from .cert import CertificateData, generate_zeroth_hammer_lean, smt_predicates_to_lean
from .project import (
    create_project,
    generate_standalone_cert_lean,
    load_module_from_file,
    write_certificate_lean,
)
from .translate import ModuleToLean4



_EPILOG = """\
examples:
  # bare project — all certificate fields left as sorry
  uv run verith mymodule.py -o out/ -p MyProject

  # pass a property (certificate stub with known P)
  uv run verith mymodule.py -P "x == 0" -o out/ -p MyProject

  # AI inference with Claude (requires ANTHROPIC_API_KEY + pip install zrth[ai])
  uv run verith mymodule.py -P "x == 0" --infer -o out/ -p MyProject

  # AI inference with Ollama (requires pip install zrth[ai-local])
  uv run verith mymodule.py -P "x == 0" --infer \\
      --model qwen3-coder --base-url http://localhost:11434/v1 -o out/ -p MyProject
"""


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Lean4 certificate project from a Python reactive module. "
            "With --infer an LLM automatically finds the invariant and ranking function."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "module_file",
        nargs="?",
        help=(
            "Path to a Python file defining the module (must contain a callable that "
            "returns a Module). Optional when --hammer-file is used alone."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory where the Lean project will be created (default: current directory).",
    )
    parser.add_argument(
        "-n",
        "--module-name",
        default="ReactiveModule",
        help="Name for the generated Lean module file (default: ReactiveModule).",
    )
    parser.add_argument(
        "-p",
        "--project-name",
        default="Rea",
        help="Name of the Lean package / library (default: Rea).",
    )
    parser.add_argument(
        "-d",
        "--module-def",
        default="module",
        help="Name of the function or class in the Python file that produces the Module (default: module).",
    )
    parser.add_argument(
        "-x",
        "--executable",
        action="store_true",
        help="Generate Main.lean and add [[lean_exe]] to lakefile for a runnable binary.",
    )
    parser.add_argument(
        "-P",
        "--property",
        default=None,
        help=(
            "Property as an SMT-LIB 2 Bool expression over state vars "
            "`s0..sN-1` (ctrl-next components). "
            "Example: '(= s0 false)'. Required when using --infer."
        ),
    )
    parser.add_argument(
        "--pre",
        default=None,
        help=(
            "Precondition as an SMT-LIB 2 Bool expression over input vars "
            "`e0..eM-1` (extl-next) and `el0..elM-1` (extl-latched). "
            "Tuple (matrix) elements use '((_ tuple.select k) eK)'. "
            "Applied to both init_pre and update_pre. "
            "Example: '(>= ((_ tuple.select 0) e0) 1.0)'."
        ),
    )
    parser.add_argument(
        "--invariant",
        default=None,
        help=(
            "Invariant as an SMT-LIB 2 Bool expression over state vars "
            "`s0..sN-1`. When combined with --infer, only a ranking "
            "function is inferred (the invariant is fixed). "
            "Example: '(= s0 s5)'."
        ),
    )
    parser.add_argument(
        "--ranking",
        default=None,
        help=(
            "Ranking as an SMT-LIB 2 Int expression over state vars "
            "`s0..sN-1`. When combined with --infer, only an invariant "
            "is inferred. Must satisfy `ranking ≥ 0` under the invariant. "
            "Example: '(ite s5 (ite s4 1 2) 0)'."
        ),
    )
    parser.add_argument(
        "--infer",
        nargs="?",
        const="ai-cegar",
        default=None,
        choices=["ai", "ai-cegar"],
        help=(
            "Infer the invariant and ranking function for --property. "
            "`ai` uses plain LLM self-check; `ai-cegar` uses LLM + cvc5 "
            "counterexample-guided refinement (default when --infer is "
            "passed without a value)."
        ),
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="LLM model to use for inference (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL for local LLMs, e.g. http://localhost:11434/v1 for Ollama.",
    )
    parser.add_argument(
        "--cert-file",
        default=None,
        help=(
            "Write a standalone, self-contained certificate .lean file to this path "
            "instead of creating a full project.  The file inlines init/update and "
            "imports zeroth_hammer from --hammer-import (default: ZerothHammer)."
        ),
    )
    parser.add_argument(
        "--hammer-import",
        default="ZerothHammer",
        help=(
            "Lean module name to import zeroth_hammer from when using --cert-file "
            "(default: ZerothHammer)."
        ),
    )
    parser.add_argument(
        "--hammer-file",
        default=None,
        help="Write a standalone ZerothHammer.lean to this path.",
    )

    args = parser.parse_args()

    # --hammer-file: generate ZerothHammer.lean and exit (no module needed)
    if args.hammer_file:
        out = Path(args.hammer_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_zeroth_hammer_lean())
        print(f"Wrote {out}")
        return

    if not args.module_file:
        parser.error("module_file is required (unless --hammer-file is used alone)")

    if args.infer and not args.property:
        parser.error("--infer requires --property")

    cert_data: CertificateData | None = None
    if args.property or args.pre or args.invariant or args.ranking:
        cert_data = CertificateData(prp=args.property)
        if args.pre:
            cert_data.init_pre = args.pre
            cert_data.update_pre = args.pre
        if args.invariant:
            cert_data.inv = args.invariant
        if args.ranking:
            cert_data.ranking = args.ranking

    module = load_module_from_file(args.module_file, module_def=args.module_def)
    print(module)

    # Translate SMT-LIB predicates to Lean expression strings for codegen.
    # `cert_data` keeps the original SMT source so `magic` can parse it with
    # its own cvc5 context.
    project_cert_data = cert_data
    if cert_data is not None:
        project_cert_data = smt_predicates_to_lean(cert_data, module)

    # --cert-file: generate a standalone, self-contained certificate file and exit
    if args.cert_file:
        out = Path(args.cert_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        lean_src = generate_standalone_cert_lean(
            out.stem,
            module,
            project_cert_data,
            hammer_import=args.hammer_import,
        )
        out.write_text(lean_src)
        print(f"Wrote standalone certificate: {out}")
        m2l = ModuleToLean4(module)
        mat_rel_out = out.with_stem(out.stem + "Rel")
        mat_rel_out.write_text(f"""\
/- Relational encoding (matrix domain) for reactive module `{out.stem}` -/
import Core.Basic
import {out.stem}

{m2l.to_lean_mat_rel()}
""")
        print(f"Wrote matrix-domain relational encoding: {mat_rel_out}")
        scalar_out = out.with_stem(out.stem + "Scalar")
        scalar_out.write_text(f"""\
/- Scalar encoding for reactive module `{out.stem}` -/
import Core.Basic
import {out.stem}

{m2l.to_lean_scalar()}
""")
        print(f"Wrote scalar encoding: {scalar_out}")
        rel_out = out.with_stem(out.stem + "ScalarRel")
        rel_out.write_text(f"""\
/- Relational encoding for reactive module `{out.stem}` -/
import Core.Basic
import {out.stem}Scalar

{m2l.to_lean_rel()}
""")
        print(f"Wrote relational encoding: {rel_out}")
        return

    print(".. Generating lean code")
    project_dir = create_project(
        output_dir=Path(args.output_dir),
        module=module,
        project_name=args.project_name,
        executable=args.executable,
        cert_data=project_cert_data,
    )

    # FIXME: this is brittle, we rely on hard-coded staff in `create_project`
    lean_code = project_dir / f"{args.project_name}.lean"

    print(".. Doing TA2Magic")
    if args.infer:
        if args.infer == "ai":
            from .magic_ai import TA2MagicAI

            magic = TA2MagicAI(
                lean_code.read_text(), model=args.model, base_url=args.base_url
            )
        else:  # "ai-cegar"
            from .magic_cegar import TA2MagicCEGAR

            magic = TA2MagicCEGAR(
                lean_code.read_text(),
                module,
                model=args.model,
                base_url=args.base_url,
            )
        cert_data = magic.infer(cert_data)

        # Merge inferred inv/ranking into project_cert_data.
        if project_cert_data is None:
            project_cert_data = CertificateData()
        project_cert_data.inv = cert_data.inv
        project_cert_data.ranking = cert_data.ranking

        data_file = project_dir / f"{args.project_name}Data.lean"
        data_lines = [
            f"/- Inferred certificate data for `{args.project_name}` -/",
            f"import {args.project_name}.{args.project_name}",
            "",
        ]
        if isinstance(cert_data.prp, str):
            data_lines.append(f"-- Property: {cert_data.prp}")
            data_lines.append("")
        for field in ("inv", "init_pre", "update_pre", "ranking"):
            value = getattr(cert_data, field)
            if isinstance(value, str):
                data_lines.append(f"def {field} := {value}")
                data_lines.append("")
        data_file.write_text("\n".join(data_lines) + "\n")
        print(f"Wrote {data_file}")

    # Write certificate once — after inference if it ran, with placeholders otherwise.
    write_certificate_lean(project_dir, args.project_name, module, project_cert_data)

    print(f"\nProject ready at: {project_dir}")


if __name__ == "__main__":
    main()
