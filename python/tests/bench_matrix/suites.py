"""Every (benchmark, property) the route matrix is measured over, and the routes.

Four suites, each already the source of truth for its own harness; nothing is
restated here that one of them already says.

    limits   `tests/limits/cases.py` -- 77 cases over 38 module fixtures, each
             a `--buchi` reachability target. The cases vary the *supplied*
             `inv`/`rank`, which no `--infer` route reads, so they collapse
             onto their (module, property, precondition) key exactly as
             `run_limits.py` collapses them; `shared` keeps the names that
             fell together.
    fbk      `tests/lean/fbk/probes.py` -- 39 `--safety` properties over the
             same fixtures.
    tests    `tests/fixtures/svcomp_*.py` and `counter.py` -- the CLI's own
             fixtures, whose `--buchi` property is stated in their docstrings
             ("Property: x == 0 holds infinitely often") and transcribed here.
    svcomp   `benchmarks/svcomp/dsl/` -- 57 SV-COMP termination benchmarks,
             two properties each, both *derived* rather than written down:
             see `svcomp_rows`.

A row is one question; a (row, route) pair is one `uv run verith`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

SP = Path(__file__).resolve().parent        # python/tests/bench_matrix
PY = SP.parent.parent                       # python/
MODS = PY / "tests" / "limits" / "mods"
FIXTURES = PY / "tests" / "fixtures"
SVCOMP_ADAPTER = SP / "svcomp_mod.py"


# ══════════════════════════════════════════════════════════════════════════
# Rows
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Row:
    """One property of one benchmark: the question a route is asked."""

    suite: str
    bench: str                      # the benchmark, as its suite names it
    kind: str                       # "safety" | "buchi"
    prop: str                       # SMT-LIB 2, over s0..sN-1
    module: Path                    # the file `verith` loads
    prop_label: str = ""            # how the suite names this property
    env: dict = field(default_factory=dict)   # extra env the module needs
    note: str = ""                  # why this property, when it was derived
    shared: tuple = ()              # other case names asking the same thing

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.bench}/{self.prop_label or self.kind}"


def _module_path(mod: str) -> Path:
    """A `limits`/`fbk` module reference as a path. `TESTS/x` is a fixture."""
    if mod.startswith("TESTS/"):
        return FIXTURES / f"{mod.split('/', 1)[1]}.py"
    return MODS / f"{mod}.py"


def limits_rows() -> list[Row]:
    """The limit matrix, collapsed onto the question an inferring route sees."""
    sys.path.insert(0, str(PY / "tests" / "limits"))
    from cases import CASES                                  # noqa: PLC0415

    problems: dict = {}
    for c in CASES:
        if not c.get("P"):
            continue                    # no property: nothing to infer against
        problems.setdefault((c["mod"], c["P"], c.get("pre") or ""), []).append(c)
    out = []
    for (mod, prop, pre), group in problems.items():
        out.append(Row(
            suite="limits", bench=mod, kind="buchi", prop=prop,
            module=_module_path(mod),
            prop_label=group[0]["name"],
            note=("the matrix's own reachability target"
                  + (f". Its precondition {pre} is dropped: no --infer route "
                     "reads a supplied one" if pre else "")),
            shared=tuple(c["name"] for c in group[1:]),
        ))
    return out


def fbk_rows() -> list[Row]:
    """The `--fbk-proveit` sweep's 39 safety properties."""
    sys.path.insert(0, str(PY / "tests" / "lean" / "fbk"))
    from probes import PROBES                                # noqa: PLC0415

    return [Row(suite="fbk", bench=mod, kind="safety", prop=prop,
                module=_module_path(mod), prop_label=name,
                note=f"probe expects `{expect}` of the ic3ia route")
            for name, mod, prop, expect in PROBES]


# The CLI fixtures state their property in their docstring; this is that line,
# as SMT-LIB over `s0..sN-1`. `counter` and the `twobit*` pair are already in
# the limit matrix (as `NNInv`, `BVState`, `BoolState`), so only the
# `svcomp_*` five and `counter`'s own documented property are new here.
TESTS_PROPS = (
    ("counter",               "(= s0 0)",                 "x == 0 infinitely often"),
    ("svcomp_countdown",      "(= s0 0)",                 "x == 0 infinitely often"),
    ("svcomp_collatz_bounded", "(= s0 1)",                "x == 1 infinitely often"),
    ("svcomp_gcd",            "(= s0 s1)",                "a == b (GCD reached)"),
    ("svcomp_nested",         "(and (= s0 0) (= s1 0))",  "i == 0 and j == 0"),
    ("svcomp_twovars",        "(= s0 s1)",                "x == y infinitely often"),
)


def tests_rows() -> list[Row]:
    return [Row(suite="tests", bench=mod, kind="buchi", prop=prop,
                module=FIXTURES / f"{mod}.py", prop_label=label,
                note="the property the fixture's own docstring states")
            for mod, prop, label in TESTS_PROPS]


def svcomp_rows() -> list[Row]:
    """The 57 termination benchmarks, two derived properties each.

    Neither property is written down in the corpus -- these are termination
    benchmarks, and the Farkas pipeline states termination as a claim over
    wires rather than as a one-state predicate. `verith` wants a one-state
    predicate, so both are read off the module:

    **`terminates`** (`--buchi`). `_termination.terminates()`'s domain is
    "some column moves", and `resolve_domain` simplifies it to exactly the
    loop guard, because the encoding writes every update as
    `ite(guard, body, self)`. Leaving that domain is reaching a fixed point,
    which a run never leaves, so `G F not guard` is termination -- reached
    once, it holds forever.

    **`houdini-inv`** (`--safety`). The conjunction of the invariants
    `_invariants.infer_invariants` finds. Inductive by construction, so like
    the fbk suite's `inv-` probes this measures the plumbing rather than the
    search; it is kept unpruned (redundant conjuncts included) because it is
    the invariant the corpus's own inference produced, not a tidied one.
    """
    import z3                                               # noqa: PLC0415

    from benchmarks.svcomp import discover                   # noqa: PLC0415
    from benchmarks.svcomp._farkas import resolve_domain     # noqa: PLC0415
    from benchmarks.svcomp._invariants import infer_invariants  # noqa: PLC0415
    from benchmarks.svcomp._termination import system_of, terminates  # noqa: PLC0415

    def smt(e, state) -> str:
        """`e` as SMT-LIB over `s0..sN-1`.

        A column's z3 symbol is its name; `verith` indexes state by ctrl
        declaration order, which `Bench.state` is, so the substitution is
        positional."""
        sub = [(z3.Int(n), z3.Int(f"s{i}")) for i, n in enumerate(state)]
        return " ".join(z3.substitute(z3.simplify(e), *sub).sexpr().split())

    out = []
    for bench in discover():
        system = system_of(bench)
        env = {"SVCOMP_BENCH": bench.name}
        guard = resolve_domain(system, terminates().domain)
        out.append(Row(
            suite="svcomp", bench=bench.name, kind="buchi",
            prop=smt(z3.Not(guard), bench.state), module=SVCOMP_ADAPTER,
            prop_label="terminates", env=env,
            note=("`G F not guard`: the loop guard, read off the module's "
                  "`ite(guard, body, self)`, negated. Reaching it is the "
                  "fixed point the run never leaves, so it is termination."),
        ))
        facts = infer_invariants(system)
        smap = {n: system.s_map[n] for n in system.names}
        inv = z3.And(*[f(smap) for _, f in facts]) if facts else z3.BoolVal(True)
        out.append(Row(
            suite="svcomp", bench=bench.name, kind="safety",
            prop=smt(inv, bench.state), module=SVCOMP_ADAPTER,
            prop_label="houdini-inv", env=env,
            note=(f"the conjunction of the {len(facts)} invariant"
                  f"{'' if len(facts) == 1 else 's'} Houdini found for this "
                  "module, unpruned. Inductive by construction."),
        ))
    return out


SUITES = {
    "limits": limits_rows,
    "fbk": fbk_rows,
    "tests": tests_rows,
    "svcomp": svcomp_rows,
}


def all_rows(suites=()) -> list[Row]:
    names = list(suites) or list(SUITES)
    return [r for n in names for r in SUITES[n]()]


# ══════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Route:
    """One `--infer` method, as the matrix invokes it."""

    name: str
    kinds: frozenset
    args: tuple = ()
    needs_llm: bool = False
    needs_env: tuple = ()           # env keys that must be set to run it


def routes(*, proveit_dir: str | None = None, ic3ia: str | None = None) -> list[Route]:
    """The route column, with the two paths `fbk-proveit` needs filled in.

    `none` is not an `--infer` route at all: the property alone, with no
    predicates, whose obligations the tactics then cannot close. It is the control
    for the other six -- it says whether `verith` can generate this module
    and whether the project Lean gets is well-formed, which is the question
    underneath "did the certificate discharge"."""
    out = [
        Route("none", frozenset({"safety", "buchi"})),
        Route("ai", frozenset({"buchi"}), ("--infer", "ai"), needs_llm=True),
        Route("ai-cegar", frozenset({"safety", "buchi"}), ("--infer", "ai-cegar"),
              needs_llm=True),
        Route("nuterm", frozenset({"safety", "buchi"}), ("--infer", "nuterm")),
        Route("sygus", frozenset({"safety"}), ("--infer", "sygus")),
        Route("smt-linear", frozenset({"safety", "buchi"}), ("--infer", "smt-linear")),
    ]
    if proveit_dir:
        args = ["--infer", "fbk-proveit", "--proveit-dir", proveit_dir]
        if ic3ia:
            args += ["--ic3ia", ic3ia]
        out.append(Route("fbk-proveit", frozenset({"safety"}), tuple(args),
                         needs_env=("PYTHONPATH",)))
    return out


def pairs(rows, route_list):
    """Every (row, route) the kinds allow, in a stable order."""
    return [(r, rt) for r in rows for rt in route_list if r.kind in rt.kinds]


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    rows = all_rows(sys.argv[1:])
    rts = routes(proveit_dir="/x", ic3ia="/y")
    print(f"{len(rows)} rows, {len(pairs(rows, rts))} (row, route) pairs\n")
    for suite in SUITES:
        sub = [r for r in rows if r.suite == suite]
        if not sub:
            continue
        n = len(pairs(sub, rts))
        print(f"  {suite:<8} {len(sub):>4} rows  {len(set(r.bench for r in sub)):>3} "
              f"benchmarks  {n:>4} runs")
