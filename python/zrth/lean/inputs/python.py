"""Python input: a file defining a function that returns the module."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..options import Opt
from .api import InputAdapter, Problem, SourceText
from .names import Names


@dataclass(frozen=True)
class Named:
    """A module, with the names a claim about it may call its variables
    by. What a Python input's `module()` returns to be spoken of in its own
    terms rather than by position::

        def module():
            x, u = Var(Int([1, 1])), Var(Int([1, 1]))
            ...
            return Named(Module.sequential([x, u], init, update),
                         {"x": x, "u": u})

    after which `--safety "(<= (X x) 9)"` means `(<= (X @intf.0) 9)`. A
    variable the mapping leaves out is named by position only.
    """

    module: object
    names: Mapping[str, object]


def load_module(filepath: str | Path, module_def: str = "module"):
    """The module the Python file at `filepath` defines, without its names.
    See `_call`."""
    return _as_module(_call(filepath, module_def))[0]


def _as_module(result) -> "tuple[Any, Mapping[str, Any] | None]":
    from zrth import Env, Module

    if isinstance(result, Named):
        return result.module, result.names
    if isinstance(result, Module):
        return result, None
    # main merged the old `Wrapper` into `Env`; wrap non-Module results (e.g.
    # a gym environment) into a reactive Module via `Env`.
    return Env(result), None


def _call(filepath: str | Path, module_def: str = "module"):
    """
    Load modules from an external Python file.

    The file must define either a function `module_def` (which defaults into `module`):

        def module() -> Module:
            ...

    returning the Module (or a `Named` one), or it may define a class that
    inherits from `Module` and has constructor without arguments that create
    `Module`.
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


def _load(source: SourceText, opts: Mapping[str, Any]) -> Problem:
    module, names = _as_module(_call(source.path, opts["module_def"]))
    return Problem(
        module, source,
        Names.of(module, names) if names else Names.positional(module),
    )


ADAPTER = InputAdapter(
    name="python",
    summary=(
        "a Python file whose `module()` (or `-d NAME`) returns the module, "
        "or a `Named` one for claims in its own names"
    ),
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
