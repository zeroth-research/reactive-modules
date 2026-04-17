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
import ast
import linecache
from pathlib import Path

from .cert import CertificateData
from .project import create_project, load_module_from_file

from zrth import Wire, Bool, Int
from zrth.analyzer import convert_method


def _state_var_names(source: str) -> list[str]:
    """Extract state variable names from the update function's parameters.

    Strips the 'old_' prefix convention (e.g. old_x -> x).
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "update":
            params = [arg.arg for arg in node.args.args if arg.arg != "self"]
            return [p[len("old_") :] if p.startswith("old_") else p for p in params]
    raise ValueError("No 'update' function found in source")


def _make_callable(fn_name: str, params: list[str], expr: str):
    """Compile an expression string into a callable.

    Registers the generated source in linecache so that inspect.getsource
    (used internally by convert_method) can retrieve it.
    """
    source = f"def {fn_name}({', '.join(params)}):\n    return {expr}\n"
    filename = f"<{fn_name}_inferred>"
    lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, lines, filename)
    code = compile(source, filename, "exec")
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102
    return namespace[fn_name]


def _exprs_to_terms(cert_data: CertificateData, module, source: str) -> CertificateData:
    """Convert inv/ranking expression strings in cert_data to lists of Terms."""

    var_names = _state_var_names(source)
    # cert_body evaluates inv/ranking/prp against ctrl_next (the [1] wire of each
    # ctrl pair).  convert_method reads from wires[name][0], so we point both
    # slots at the next-state wire so the generated terms read from ctrl_next.
    wires = {name: (pair[1], pair[1]) for name, pair in zip(var_names, module.ctrl)}

    if isinstance(cert_data.prp, str):
        cert_data.prp = convert_method(cert_data.prp, wires, [Wire(Bool(1))])

    if isinstance(cert_data.inv, str):
        cert_data.inv = convert_method(cert_data.inv, wires, [Wire(Bool(1))])

    if isinstance(cert_data.ranking, str):
        cert_data.ranking = convert_method(cert_data.ranking, wires, [Wire(Int(1))])

    return cert_data


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
        help="Path to a Python file defining the module (must contain a callable that returns a Module).",
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
        help="Property expression to verify (e.g. 'x == 0'). Required when using --infer.",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Use AI (TA2MagicAI) to infer the invariant and ranking function for --property.",
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

    args = parser.parse_args()

    if args.infer and not args.property:
        parser.error("--infer requires --property")

    cert_data: CertificateData | None = None
    if args.property:
        cert_data = CertificateData(prp=args.property)

    source: str | None = None
    module = load_module_from_file(args.module_file, module_def=args.module_def)
    print(module)

    print(".. Generating lean code")
    project_dir = create_project(
        output_dir=Path(args.output_dir),
        module=module,
        project_name=args.project_name,
        executable=args.executable,
        cert_data=cert_data,
    )

    # FIXME: this is brittle, we rely on hard-coded staff in `create_project`
    lean_code = project_dir / f"{args.project_name}.lean"

    print(".. Doing TA2Magic")
    if args.infer:
        from .magic_ai import TA2MagicAI

        magic = TA2MagicAI(
            lean_code.read_text(), model=args.model, base_url=args.base_url
        )
        cert_data = magic.infer(cert_data)

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

    print(f"\nProject ready at: {project_dir}")


if __name__ == "__main__":
    main()
