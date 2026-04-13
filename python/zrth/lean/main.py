"""CLI entry point for generating a Lean4 project from a reactive module."""

import argparse
import ast
import linecache
from pathlib import Path

from .cert import CertificateData
from .project import create_project, load_module_from_file


def _state_var_names(source: str) -> list[str]:
    """Extract state variable names from the update function's parameters.

    Strips the 'old_' prefix convention (e.g. old_x -> x).
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "update":
            params = [arg.arg for arg in node.args.args if arg.arg != "self"]
            return [p[len("old_"):] if p.startswith("old_") else p for p in params]
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
    from zrth import Wire, Bool, Int
    from zrth.analyzer import convert_method

    var_names = _state_var_names(source)
    wires = {name: pair for name, pair in zip(var_names, module.ctrl)}

    if isinstance(cert_data.prp, str):
        fn = _make_callable("prp", var_names, cert_data.prp)
        cert_data.prp = convert_method(fn, wires, [Wire(Bool(1))])

    if isinstance(cert_data.inv, str):
        fn = _make_callable("inv", var_names, cert_data.inv)
        cert_data.inv = convert_method(fn, wires, [Wire(Bool(1))])

    if isinstance(cert_data.ranking, str):
        fn = _make_callable("ranking", var_names, cert_data.ranking)
        cert_data.ranking = convert_method(fn, wires, [Wire(Int(1))])

    return cert_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Lean4 project from a Python reactive module definition."
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
    if args.infer:
        from .magic_ai import TA2MagicAI

        source = Path(args.module_file).read_text()
        magic = TA2MagicAI(source, model=args.model, base_url=args.base_url)
        cert_data = magic.infer(cert_data)

    module = load_module_from_file(args.module_file, module_def=args.module_def)

    if args.infer and cert_data is not None and source is not None:
        cert_data = _exprs_to_terms(cert_data, module, source)
    project_dir = create_project(
        output_dir=Path(args.output_dir),
        module=module,
        project_name=args.project_name,
        executable=args.executable,
        cert_data=cert_data,
    )
    print(f"\nProject ready at: {project_dir}")


if __name__ == "__main__":
    main()
