"""The project's lakefile: the dependencies every project requires
(Mathlib, Cslib, the proof library) at the revisions it pins, and what an
inference route adds to them (`LakefileParts`).
"""

from __future__ import annotations

from dataclasses import dataclass


from .templates import render


MATHLIB_URL = "https://github.com/leanprover-community/mathlib4"
CSLIB_URL = "https://github.com/leanprover-community/cslib"
PROOFS_URL = "https://github.com/zeroth/proof-prototyping"

MATHLIB_REV = "v4.28.0"
CSLIB_REV = MATHLIB_REV
# Lean toolchain version — should match Mathlib's requirement
LEAN_TOOLCHAIN = f"leanprover/lean4:{MATHLIB_REV}"


@dataclass(frozen=True)
class LeanLib:
    """A `[[lean_lib]]` a route adds: `name` is the module name its files are
    imported under, `src_dir` the project-relative directory they are in."""

    name: str
    src_dir: str
    comment: str = ""


@dataclass(frozen=True)
class Require:
    """A `[[require]]` a route adds, from a local checkout at `path`."""

    name: str
    path: str
    comment: str = ""


@dataclass(frozen=True)
class LakefileParts:
    """What an inference route adds to the lakefile: libraries after
    `System`, requires after `smt`, each preceded by its `comment`."""

    libs: tuple[LeanLib, ...] = ()
    requires: tuple[Require, ...] = ()


def generate_lakefile(
    project_name: str,
    executable: bool = False,
    parts: LakefileParts = LakefileParts(),
) -> str:
    """Render the project's lakefile, with `parts` added to the fixed ones.

    A route cannot amend the lakefile after the fact -- it is written while
    the project is created, before any route runs -- so what the route needs
    in it is given here, from its row's `lakefile` hook.
    """
    return render(
        "project/lakefile.toml.j2",
        project_name=project_name,
        executable=executable,
        cslib_rev=CSLIB_REV,
        parts=parts,
    )
