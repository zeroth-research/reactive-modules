"""Options declared by more than one route row.

A row declares every option it accepts, because the row is what `--help`
groups by and what the "only meaningful with" check reads.  An option two rows
spell identically is defined once, here, and named in both.
"""

from __future__ import annotations

from ...core.options import Opt

# The Vampire binary: `--infer houdini --houdini-solver vampire` puts candidates
# to it and `--infer vampire` asks it to derive one.
VAMPIRE = Opt(
    ("--vampire",),
    dict(
        metavar="PATH",
        help=(
            "The Vampire binary, for `--infer vampire` and for "
            "`--infer houdini --houdini-solver vampire`: the "
            "executable, or a directory holding one (the release "
            "zip unpacks to one). When omitted, $VAMPIRE, then "
            "`vampire` on PATH."
        ),
    ),
)
