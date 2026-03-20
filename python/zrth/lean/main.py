"""CLI entry point for generating a Lean4 project from a reactive module."""

import argparse
from pathlib import Path

from .project import create_project, load_module_from_file


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

    args = parser.parse_args()

    module = load_module_from_file(args.module_file, module_def=args.module_def)
    project_dir = create_project(
        output_dir=Path(args.output_dir),
        module=module,
        project_name=args.project_name,
        executable=args.executable,
    )
    print(f"\nProject ready at: {project_dir}")


if __name__ == "__main__":
    main()
