"""Every input format verith reads, as one row each.

A row is an `InputAdapter`: the format's name, the file suffixes that
select it, and `load`, which makes a `Problem` -- the module the run is
about, and the text it was read from -- out of the input. Everything after
loading reads the `Problem`, not the file, so a new format is a new row
rather than a branch in the pipeline.

Lazy imports, as in `infer_route`: a row module imports only `api` at
module level, and its `load` imports the parser when it is called.

Adding a format: write `inputs/<name>.py` exporting `ADAPTER`, and add it
to the tuple below.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import python
from .api import InputAdapter, Problem, SourceText

__all__ = [
    "ADAPTERS", "InputAdapter", "Problem", "SourceText", "adapter_by_name",
    "load",
]

# The order is the order `--format` lists them in.
ADAPTERS: tuple[InputAdapter, ...] = (python.ADAPTER,)

_BY_NAME = {a.name: a for a in ADAPTERS}
assert len(_BY_NAME) == len(ADAPTERS), "two input adapters share a name"


def adapter_by_name(name: str) -> InputAdapter:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"no input format named {name!r} (have: {', '.join(_BY_NAME)})"
        ) from None


def load(path: str | Path, opts: Mapping[str, Any] | None = None,
         *, format: str = "python") -> Problem:
    """The `Problem` the file at `path` states, read as `format`."""
    adapter = adapter_by_name(format)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Diagram file not found: {path.resolve()}")
    text = path.read_text()
    return adapter.load(SourceText(adapter.name, path, text), opts or {})
