"""Every input format verith reads, as one row each.

A row is an `InputAdapter`: the format's name, the file suffixes that
select it, and `load`, which makes a `Problem` -- the module the run is
about, the text it was read from, and the names claims may use (`names`)
-- out of the input. Everything after
loading reads the `Problem`, not the file, so a new format is a new row
rather than a branch in the pipeline.

Lazy imports, as in `infer`: a row module imports only `api` at module
level, and its `load` imports the parser when it is called.

Adding a format: write `inputs/<name>.py` exporting `ADAPTER`, and add it
to the tuple below.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import python, smv
from .api import InputAdapter, InputError, Problem, SourceText, StatedClaim
from .names import ClaimError, Names, to_internal, to_surface
from .python import Named

__all__ = [
    "ADAPTERS", "ClaimError", "InputAdapter", "InputError", "Named", "Names",
    "Problem", "StatedClaim",
    "SourceText", "adapter_by_name", "adapter_for", "load", "to_internal",
    "to_surface",
]

# The order is the order `--format` lists them in.
ADAPTERS: tuple[InputAdapter, ...] = (python.ADAPTER, smv.ADAPTER)

_BY_NAME = {a.name: a for a in ADAPTERS}
assert len(_BY_NAME) == len(ADAPTERS), "two input adapters share a name"


def adapter_by_name(name: str) -> InputAdapter:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"no input format named {name!r} (have: {', '.join(_BY_NAME)})"
        ) from None


def adapter_for(path: str | Path) -> InputAdapter | None:
    """The format `path`'s suffix names, if any does."""
    suffix = Path(path).suffix
    return next((a for a in ADAPTERS if suffix in a.suffixes), None)


def load(path: str | Path, opts: Mapping[str, Any] | None = None,
         *, format: str | None = None) -> Problem:
    """The `Problem` the file at `path` states, read as `format` (by
    default the one its suffix names); `opts` are the format's options by
    `dest`, defaulted as its row says."""
    adapter = adapter_by_name(format) if format else adapter_for(path)
    if adapter is None:
        raise KeyError(
            f"cannot tell the format of {path} from its suffix (have: "
            f"{', '.join(_BY_NAME)})"
        )
    opts = {**{o.dest: o.default for o in adapter.options}, **(opts or {})}
    path = Path(path)
    if not path.exists():
        raise InputError(f"Diagram file not found: {path.resolve()}")
    try:
        text = path.read_text()
    except OSError as e:
        raise InputError(f"cannot read {path.resolve()}: {e}") from e
    source = SourceText(adapter.name, path, text, adapter.suffixes[0])
    return adapter.load(source, opts)
