"""The types an input adapter is written against."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zrth import Module


@dataclass(frozen=True)
class SourceText:
    """The input file as read: its format (an adapter's name), where it
    is, and what it says. `suffix` is the format's own (the adapter's
    first), which the project's copy of the input is named with whatever
    the file was called."""

    format: str
    path: Path
    text: str
    suffix: str


@dataclass(frozen=True)
class Problem:
    """What an adapter makes of its input: the module the run is about,
    and the text it was read from."""

    module: Module
    source: SourceText


@dataclass(frozen=True)
class InputAdapter:
    """One input format verith reads.

    `load` turns the input into a `Problem`; it imports its parser when it
    is called, so `verith --help`, which reads every row, imports none.
    """

    name: str                       # the `--format` value
    summary: str                    # one line, for `--format`'s help
    suffixes: tuple[str, ...]       # file suffixes that select it
    load: Callable[[SourceText, Mapping[str, Any]], Problem]
