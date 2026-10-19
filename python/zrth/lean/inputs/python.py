"""Python input: a file defining a function that returns the module."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..options import Opt
from .api import InputAdapter, Problem, SourceText


def load_module(filepath: str | Path, module_def: str = "module"):
    """
    Load modules from an external Python file.

    The file must define either a function `module_def` (which defaults into `module`):

        def module() -> Module:
            ...

    returning the Module, or it may define a class that inherits from `Module` and
    has constructor without arguments that create `Module`.
    """
    import importlib.util

    from zrth import Env, Module

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


def _load(source: SourceText, opts: Mapping[str, Any]) -> Problem:
    return Problem(load_module(source.path, opts["module_def"]), source)


ADAPTER = InputAdapter(
    name="python",
    summary="a Python file whose `module()` (or `-d NAME`) returns the module",
    suffixes=(".py",),
    load=_load,
    options=(
        Opt(("-d", "--module-def"), {
            "default": "module",
            "metavar": "NAME",
            "help": (
                "Name of the function or class in the Python file that "
                "produces the Module (default: module)."
            ),
        }),
    ),
)
