"""A command-line option that belongs to one row of a registry -- an
inference route (`infer`) or an input format (`inputs`) -- rather
than to verith. A leaf module, so both registries can import it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Opt:
    """One command-line option that belongs to a row, not to verith.

    Given without its row selected it is an error rather than a no-op --
    which is what `--ic3ia`, `--fbk-simplify` and `--fbk-equiv` each said by
    hand, in a block that grew a line per option per route.
    """

    flags: tuple[str, ...]
    kwargs: dict                # forwarded to `add_argument`
    required: bool = False      # the row cannot run without it

    @property
    def dest(self) -> str:
        d = self.kwargs.get("dest")
        return d or self.flags[-1].lstrip("-").replace("-", "_")
