"""The error every stage of the pipeline refuses a run with."""

from __future__ import annotations


class Refused(ValueError):
    """This run cannot be carried out, and the message says why.

    A refusal is a decision the generator makes about its input -- an op with
    no Lean form, a state element the executable cannot print, a name the
    property does not have. It is not a defect, so `main` prints it as
    `error: ...` and exits; everything else keeps its traceback, which is
    what tells the two apart.

    `ValueError` is the base because that is what these sites raised before
    the type existed, and callers that catch it still do.

    `searched` is whether anything was *looked for* before giving up, and it
    is the difference between two refusals that read alike and mean opposite
    things: a route that declines a module's shape has learned nothing about
    it, while one whose budget ran out has learned that this shape is not
    cheap. The default is `False` because most refusals are gates -- a sort
    the route cannot weigh, an operator with no encoding, a flag that does
    not apply -- and every one of those is reached before a solver starts.
    A route that gives up *after* searching says so, and `main` records the
    two under different statuses.

    `setup` is whether the refusal is about how the run was asked for -- a
    package not installed, a key not set -- rather than about the module.
    Filed in `artifacts/` it would read as a fact about a run that never
    started, so `main` leaves it out.
    """

    def __init__(self, *args, searched: bool = False, setup: bool = False):
        super().__init__(*args)
        self.searched = searched
        self.setup = setup


class PredicateRefused(Refused):
    """A refusal about one of the claim's predicates, named by its `role`:
    "property" (of `kind` "safety" or "buchi"), "pre", "invariant" or
    "ranking" -- or any other label, for a term a route made up itself.
    The layers that raise it know predicates by role; a claim's predicate
    is shown as the command-line flag it came from, which is where a user
    can act on it, and this is the one place that spelling lives."""

    def __init__(self, role: str, message: str, *, kind: str = "buchi",
                 searched: bool = False):
        super().__init__(message, searched=searched)
        self.role, self.kind, self.message = role, kind, message

    @property
    def flag(self) -> str:
        if self.role == "property":
            return f"--{self.kind}"
        if self.role in ("pre", "invariant", "ranking"):
            return f"--{self.role}"
        return self.role

    def __str__(self) -> str:
        return f"{self.flag}: {self.message}"


def require(package: str, who: str, install: str | None = None):
    """`package`, imported -- or a refusal saying `who` needs it and how to
    install it: `install`, or the package itself.

    A missing package is about how the run was asked for, not about the
    module, and the `ImportError` from wherever it happened to be imported
    first says neither which flag wanted it nor what to do.
    """
    import importlib

    try:
        return importlib.import_module(package)
    except ImportError as e:
        raise Refused(
            f"{who} needs the `{package}` package, which is not importable. "
            f"Install with: uv pip install {install or package}",
            setup=True,
        ) from e

