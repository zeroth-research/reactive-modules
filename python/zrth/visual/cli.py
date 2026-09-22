"""`vis`: render a module named by a FILE:FUNCTION pair to a standalone page.

Usage:
    uv run vis path-to-module.py:module_fun --open
"""
import argparse
import importlib.util
import inspect
import sys
import webbrowser
from pathlib import Path

from ..zrth import Module
from .server import to_html


_DEFAULT_OUTPUT = "module.html"


def _fail(msg):
    raise SystemExit(f"vis: error: {msg}")


def _parse_spec(spec):
    """Split `path/to/file.py:function` into its two halves.

    Split from the right so Windows drive letters survive.
    """
    path, sep, func = spec.rpartition(":")
    if not sep or not path or not func:
        _fail(f"expected a FILE:FUNCTION spec, got {spec!r}")
    return Path(path), func


def _load_function(path, func_name):
    """Import `path` as a throwaway module and pull `func_name` out of it."""
    if not path.is_file():
        _fail(f"no such file: {path}")

    spec = importlib.util.spec_from_file_location(f"_vis_{path.stem}", path)
    if spec is None or spec.loader is None:
        _fail(f"cannot import {path} as a Python module")

    mod = importlib.util.module_from_spec(spec)
    # The file may import its siblings, so let its own directory win on sys.path,
    # and register it before exec so dataclasses/pickle can look the module up.
    sys.path.insert(0, str(path.parent.resolve()))
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        _fail(f"failed to import {path}: {type(e).__name__}: {e}")
    finally:
        sys.path.pop(0)
        sys.modules.pop(spec.name, None)

    try:
        func = getattr(mod, func_name)
    except AttributeError:
        _fail(f"{path} defines no {func_name!r}")
    if not callable(func):
        _fail(f"{path}:{func_name} is a {type(func).__name__}, not a callable")
    return func


def _call(func, spec):
    """Call the entry point with no arguments and insist on a Module back."""
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        params = ()
    required = [p.name for p in params
                if p.default is p.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    if required:
        _fail(f"{spec} takes required argument(s) {', '.join(required)}; "
              "vis calls it with none")

    try:
        module = func()
    except Exception as e:
        _fail(f"{spec} raised {type(e).__name__}: {e}")

    if not isinstance(module, Module):
        _fail(f"{spec} returned {type(module).__name__}, not a zrth Module")
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="vis",
        description="Visualize a reactive module as a standalone HTML page.",
        epilog="example: uv run vis path-to-module.py:module_fun --open",
    )
    parser.add_argument(
        "spec", metavar="FILE:FUNCTION",
        help="a Python file and a no-argument function in it returning a Module",
    )
    parser.add_argument(
        "--output", metavar="PATH", default=_DEFAULT_OUTPUT,
        help=f"where to write the page (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--open", action="store_true", dest="open_browser",
        help="open the page in a browser once written",
    )
    args = parser.parse_args(argv)

    path, func_name = _parse_spec(args.spec)
    module = _call(_load_function(path, func_name), args.spec)

    out = Path(args.output)
    try:
        out.write_text(to_html(module))
    except OSError as e:
        _fail(f"cannot write {out}: {e}")
    print(f"Module visualized in `{out}`")

    if args.open_browser:
        print("Opening in browser ...")
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
