"""`--houdini-solver`: which solver judges the search's candidates, resolved
at parse time.

The solvers themselves are `smt.solvers`; this is the choice between them,
and the flags that belong to the one chosen.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.errors import Refused
from ...smt.solvers import DEFAULT_CORES, Cvc5Solver, Solver, VampireSolver
from ...smt.vampire import resolve_vampire

SOLVERS = ("cvc5", "vampire")
DEFAULT_SOLVER = "cvc5"


@dataclass(frozen=True)
class SolverSpec:
    """What `--houdini-solver` resolved to, at parse time.

    A route's `resolve` runs before any project is generated, which is where
    a missing Vampire binary has to be met. cvc5 resolves to nothing to
    look for: it is the package that encoded the module in the first place.
    """

    kind: str = DEFAULT_SOLVER
    exe: "str | None" = None
    cores: int = DEFAULT_CORES

    def build(self, tm, *, seconds: float, log=print) -> Solver:
        if self.kind == "vampire":
            return VampireSolver(self.exe, seconds=seconds,
                                 cores=self.cores, log=log)
        return Cvc5Solver(tm, seconds=seconds, log=log)


def resolve_solver(kind: str, vampire: "str | None" = None,
                   cores: "int | None" = None) -> SolverSpec:
    """`--houdini-solver` and the flags that belong to the one it names."""
    if kind not in SOLVERS:
        raise Refused(
            f"--houdini-solver takes {' or '.join(SOLVERS)}, not {kind!r}"
        )
    if kind != "vampire":
        return SolverSpec(kind)
    return SolverSpec(kind, resolve_vampire(vampire),
                      DEFAULT_CORES if cores is None else int(cores))
