"""`--infer houdini`: Houdini over facts read off simulated runs, a solver deciding each."""

from __future__ import annotations

from ...core.options import Opt
# Constants only -- neither module imports cvc5, so naming the defaults in
# the row costs a bare run nothing.
from ...smt.solvers import DEFAULT_CORES, DEFAULT_TIMEOUT
from .solvers import DEFAULT_SOLVER, SOLVERS
from ..common.options import VAMPIRE
from ..io import InferInput, InferResult
from ..route import InferRoute


def _resolve(opts: dict):
    from .solvers import resolve_solver

    return resolve_solver(opts["houdini_solver"], opts["vampire"],
                          opts["vampire_cores"])


def _check_options(settings, opts) -> "str | None":
    """The flags that belong to a solver this run did not choose.

    Not a cell of the table: `--vampire` is an option of this route
    whichever solver it names, and only the *value* of another option says
    whether it means anything. Refused rather than ignored, for the reason
    `Opt` exists at all -- a flag that quietly does nothing is worse than
    one that is an error.
    """
    if opts["houdini_solver"] == "vampire":
        return None
    spare = [f for f, v in (("--vampire", opts["vampire"]),
                            ("--vampire-cores", opts["vampire_cores"]))
             if v is not None]
    if not spare:
        return None
    return (
        f"{', '.join(spare)} configures the Vampire prover, and this run is "
        f"--houdini-solver {opts['houdini_solver']}, which starts no binary. "
        f"Pass --houdini-solver vampire, or drop the flag."
    )


def _run(inp: InferInput) -> InferResult:
    from .engine import TA2MagicHoudini

    magic = TA2MagicHoudini(
        inp.module,
        solver=inp.config,
        timeout=inp.opts["houdini_timeout"],
        artifacts=inp.project.artifacts,
        log=inp.log,
        encoding=inp.smt,
    )
    return InferResult(magic.infer(inp.claim, inp.given))


ROUTE = InferRoute(
    name="houdini",
    summary=(
        "no LLM: Houdini over facts read off simulated runs of the module "
        "and a ranking function from a fixed list of affine and "
        "lexicographic shapes, each kept only once --houdini-solver "
        "proves its obligation -- cvc5 (default), which decides them and "
        "so answers a failed candidate with a counterexample, or the "
        "Vampire theorem prover, which refutes or times out -- scalar "
        "Int, Bool and Real state, the last ranked through to_int after "
        "scaling by the denominators the program writes"
    ),
    kinds=frozenset({"safety", "buchi"}),
    kinds_refusal="",
    seeds=frozenset({"pre"}),
    seeds_refusal=(
        "the route proposes the invariant and proves which of its facts "
        "stay, and a ranking function is what it searches for, so a "
        "supplied predicate would be one it is not allowed to decide"
    ),
    returns="smt",
    resolve=_resolve,
    extra_check=_check_options,
    run=_run,
    options=(
        Opt(
            ("--houdini-solver",),
            dict(
                choices=list(SOLVERS),
                default=DEFAULT_SOLVER,
                help=(
                    f"Which solver decides the obligations (default: "
                    f"{DEFAULT_SOLVER}). `cvc5` needs no binary -- it is "
                    f"the package that encoded the module -- and it "
                    f"*decides* them: a candidate it cannot prove comes "
                    f"back refuted, with a counterexample that says which "
                    f"facts to drop and that holds at every time limit. "
                    f"`vampire` is a first-order prover on printed "
                    f"SMT-LIB: it refutes the negation or runs out of "
                    f"time, and the two are indistinguishable."
                ),
            ),
        ),
        Opt(
            ("--houdini-timeout",),
            dict(
                type=float,
                default=DEFAULT_TIMEOUT,
                metavar="SECONDS",
                help=(
                    f"Wall-clock budget for all solver calls together "
                    f"(default: {DEFAULT_TIMEOUT}). The search runs whole "
                    f"at a per-call limit of 2 s, then whatever is left; "
                    f"under `--houdini-solver vampire` the rungs are 2 s, "
                    f"10 s, 60 s and the rest, because a short limit "
                    f"makes its portfolio a different schedule rather "
                    f"than a truncated one. Either way only what came "
                    f"back unanswered is asked again."
                ),
            ),
        ),
        VAMPIRE,
        Opt(
            ("--vampire-cores",),
            dict(
                type=int,
                default=None,
                metavar="N",
                help=(
                    f"--houdini-solver vampire: processes each call "
                    f"spreads its portfolio over (default: "
                    f"{DEFAULT_CORES}). Calls also run side by side, as "
                    f"many as the machine's cores allow at this width, up "
                    f"to four."
                ),
            ),
        ),
    ),
)
