#!/usr/bin/env python3
"""Compare the certificates a row's routes found, logically rather than as text.

    uv run python tests/bench_matrix/compare_certs.py
    uv run python tests/bench_matrix/compare_certs.py --only philo --verbose
    uv run python tests/bench_matrix/compare_certs.py --routes houdini houdini-vampire

A `VERIFIED` cell says a route certified the row. It does not say *what* it
certified, and two routes that both certify one row need not have found the
same thing -- which is the question the `houdini`/`houdini-vampire` column
pair exists to ask, and the one a verdict-only table cannot answer.

**Why this needs a solver.** The obvious comparison is to diff the strings,
and it is wrong on the first row it meets: on `fbk/m_countdown/InvBase` all
seven routes derive `0 <= s0 <= 100`, and they write it four ways --

    (and (>= s0 0) (<= s0 100))
    (and (<= 0 s0) (<= s0 100))
    (and (>= s0 0) (>= (+ 100 (- s0)) 0))
    (and (<= (+ (- 100) (* (- 101) s0)) 0) (<= (+ (- 100) (* 1 s0)) 0))

-- so a textual diff sorts them into four answers where there is one. The
invariants are parsed back through the module's own cvc5 encoding instead
(`SynthContext`, the same front end the routes state their obligations in)
and compared as formulas.

Measured over the recorded passes, of **148 rows where more than one route
found a certificate: 79 are one invariant, 64 are a chain, and 5 are
mixed** -- and not one row is wholly `incomparable`. So the routes mostly
differ in *how much* they prove rather than in what they prove, and where
they do diverge it is a strand off a chain rather than two unrelated
arguments. The five that mix are worth reading one at a time: on
`fbk/m_lex/LexLinComb` `smt-linear` keeps `s0 >= -2`, which nothing else
says and which orders against nothing else there.

On the `houdini` column pair specifically -- the one that exists to ask
this -- 11 of its 16 rows are one invariant, 5 are ordered, and the
implication goes **both** ways: cvc5 is the stronger answer on two rows and
Vampire on three. That is the "which half of the route the difference is
in" that the pair exists to show, and its verdicts, identical on all but
one row, cannot.

What it reports
===============
Per row, the routes' invariants are grouped by *logical equivalence*, and
the groups are then ordered by implication, which is the interesting
relation and not a tie-breaker: an invariant that implies another is the
**stronger** claim, and a route that reached it looked harder. The four
answers are therefore

    same          one formula, however it is spelled
    stronger      every pair of groups is ordered by implication, so the
                  answers form a chain from the tightest invariant down --
                  the usual shape of a Houdini disagreement, where one
                  prover's minimisation dropped facts the other kept
    mixed         some pairs ordered and some not
    incomparable  no pair is ordered: genuinely different arguments for one
                  property

With two groups there is one pair to relate, so `mixed` cannot arise and
the answer is `stronger` or `incomparable`. With three or more it can, and
that is the reason `verdict` counts pairs rather than asking whether *any*
pair is ordered: `a` may imply `b` while `c` is comparable with neither,
and calling that row "stronger" would hide the pair that is not. 19 of the
recorded rows have three groups or more.

Ranking functions are reported beside the invariants and compared only for
equality. Two ranking functions that are not equal are not thereby ordered
-- `s0` and `2*s0` are both valid and neither is stronger -- so implication
is not asked of them.

An equivalence query that cvc5 leaves `unknown` is reported as undecided
rather than folded into either answer. These are quantifier-free linear
arithmetic in practice, so it does not happen; saying so is cheaper than
finding out from a wrong count.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SP = Path(__file__).resolve().parent
PY = SP.parent.parent
sys.path.insert(0, str(PY))
sys.path.insert(0, str(SP))

from suites import all_rows, column_of                          # noqa: E402

from zrth.lean.cert import CertificateData                      # noqa: E402
from zrth.lean.project import load_module_from_file             # noqa: E402
from zrth.lean.smt_synth import SynthContext                    # noqa: E402

WORK = Path(os.environ.get("VERITH_BENCH_WORK", "/tmp/verith-bench.noindex"))

# Every sort any route reads. This module weighs nobody's obligations, it
# only parses predicates back, so it takes whatever the widest route takes.
ALL_KINDS = ("int", "bool", "bv", "real", "tuple")

# One comparison is a quantifier-free arithmetic query over a handful of
# columns; a second is already generous.
QUERY_MS = 2000


# ══════════════════════════════════════════════════════════════════════════
# The module, as the routes encoded it
# ══════════════════════════════════════════════════════════════════════════

def context_for(row):
    """`row`'s module and property in cvc5, in the routes' own namespace.

    The environment a row asks for is set around the load and put back after
    it -- `petri_mod.py` reads `$PETRI_NET`, so one adapter is twelve
    different modules and loading it under the wrong one silently compares
    the wrong things.
    """
    saved = {k: os.environ.get(k) for k in row.env}
    os.environ.update(row.env)
    try:
        module = load_module_from_file(str(row.module))
    finally:
        for k, v in saved.items():
            os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    cd = CertificateData(prp=row.prop, kind=row.kind,
                         init_pre=row.pre or None, update_pre=row.pre or None)
    return SynthContext.build(module, cd, route="houdini", takes=ALL_KINDS)


# ══════════════════════════════════════════════════════════════════════════
# Asking cvc5
# ══════════════════════════════════════════════════════════════════════════

class Judge:
    """Equivalence and implication between predicates of one module.

    Answers are memoised on the pair, because grouping asks about the same
    representative once per later route.
    """

    def __init__(self, ctx):
        import cvc5                                   # noqa: PLC0415
        from cvc5 import Kind                         # noqa: PLC0415

        self.ctx, self.cvc5, self.kind = ctx, cvc5, Kind
        self._asked: dict = {}

    def parse(self, src: str):
        """`src` as a term of the module's encoding, or `None`.

        `None` is "this route's answer cannot be read here" -- the `ai`
        routes write Lean rather than SMT-LIB, and a row whose certificate
        does not parse is reported as unread rather than as different.
        """
        try:
            term = self.ctx.env.parse_expr(src)
        except Exception:                              # noqa: BLE001
            return None
        return None if term.isNull() else term

    def _unsat(self, formula) -> "bool | None":
        """Whether `formula` has no model; `None` when cvc5 did not settle."""
        solver = self.cvc5.Solver(self.ctx.tm)
        solver.setLogic("ALL")
        solver.setOption("tlimit", str(QUERY_MS))
        solver.setOption("tlimit-per", str(QUERY_MS))
        try:
            solver.assertFormula(formula)
            result = solver.checkSat()
        except RuntimeError:                           # pragma: no cover
            return None
        return True if result.isUnsat() else (False if result.isSat() else None)

    def implies(self, a, b) -> "bool | None":
        """Whether every state satisfying `a` satisfies `b`."""
        key = ("=>", str(a), str(b))
        if key not in self._asked:
            tm, Kind = self.ctx.tm, self.kind
            self._asked[key] = self._unsat(
                tm.mkTerm(Kind.AND, a, tm.mkTerm(Kind.NOT, b)))
        return self._asked[key]

    def same(self, a, b) -> "bool | None":
        """Whether `a` and `b` hold of exactly the same states.

        Asked as one query rather than as two implications: `a != b` is
        unsatisfiable exactly when they agree everywhere, and it is the
        query that decides a `ranking` too, where implication means nothing.
        """
        key = ("=", str(a), str(b))
        if key not in self._asked:
            tm, Kind = self.ctx.tm, self.kind
            self._asked[key] = self._unsat(
                tm.mkTerm(Kind.NOT, tm.mkTerm(Kind.EQUAL, a, b)))
        return self._asked[key]


# ══════════════════════════════════════════════════════════════════════════
# One row
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Group:
    """Routes whose invariants are one formula, and that formula."""

    routes: list = field(default_factory=list)
    src: str = ""
    term: object = None


@dataclass
class Comparison:
    """What the routes on one row found, and how the answers relate."""

    key: str
    groups: list = field(default_factory=list)       # of Group
    unread: dict = field(default_factory=dict)       # route -> its source
    ranks: dict = field(default_factory=dict)        # route -> ranking source
    rank_groups: list = field(default_factory=list)  # of Group, over ranks
    # (i, j) for each pair of groups where group i's invariant implies group
    # j's -- so `i` is the stronger claim. Undecided pairs are absent from
    # this and present in `undecided`.
    stronger: list = field(default_factory=list)
    undecided: list = field(default_factory=list)
    error: str = ""

    @property
    def verdict(self) -> str:
        """The row's headline, in the vocabulary the module docstring sets.

        Told apart by how much of the implication order is *there*, which
        with three groups or more is a real question and not a formality:
        `a` may imply `b` while `c` is comparable with neither, and calling
        that row "stronger" because one pair is ordered would hide the pair
        that is not. Counting the pairs is what keeps the four answers
        honest as the group count grows -- 19 of the recorded rows have
        three groups or more, and 5 of those are `mixed`.
        """
        if self.error:
            return "unread"
        if len(self.groups) <= 1:
            return "same"
        if self.undecided:
            return "undecided"
        # A pair is ordered when either direction holds; `stronger` records
        # the direction, so one entry is enough to order the pair.
        ordered = {frozenset(p) for p in self.stronger}
        total = len(self.groups) * (len(self.groups) - 1) // 2
        if not ordered:
            return "incomparable"
        return "stronger" if len(ordered) == total else "mixed"


def compare_row(row, certs: dict) -> Comparison:
    """Group `certs` -- route -> {inv, ranking} -- by what they mean.

    Nothing here needs the row to have been verified: a route that found a
    certificate Lean then rejected still found something, and what it found
    is as comparable as any other. The caller decides which cells to pass.
    """
    out = Comparison(key=row.key)
    try:
        judge = Judge(context_for(row))
    except Exception as e:                             # noqa: BLE001
        out.error = f"{type(e).__name__}: {e}"
        return out

    for route, found in sorted(certs.items()):
        src = found.get("inv")
        if not src:
            continue
        term = judge.parse(src)
        if term is None:
            out.unread[route] = src
            continue
        for group in out.groups:
            if judge.same(group.term, term):
                group.routes.append(route)
                break
        else:
            out.groups.append(Group([route], src, term))
        if found.get("ranking"):
            out.ranks[route] = found["ranking"]

    for i, a in enumerate(out.groups):
        for j, b in enumerate(out.groups):
            if i == j:
                continue
            answer = judge.implies(a.term, b.term)
            if answer is None:
                out.undecided.append((i, j))
            elif answer:
                out.stronger.append((i, j))

    for route, src in sorted(out.ranks.items()):
        term = judge.parse(src)
        if term is None:
            continue
        for group in out.rank_groups:
            if judge.same(group.term, term):
                group.routes.append(route)
                break
        else:
            out.rank_groups.append(Group([route], src, term))
    return out


# ══════════════════════════════════════════════════════════════════════════
# A whole pass
# ══════════════════════════════════════════════════════════════════════════

def certificates(runs: dict, keep=()) -> dict:
    """Every certificate a pass recorded, as `row key -> route -> found`.

    Keyed by the *column*, which is what a cell is looked up by and what
    tells `houdini` from `houdini-vampire` -- the `route` field beside it
    names the route, and is one string for both.
    """
    out: dict = {}
    for pair, run in runs.items():
        key, column = pair.split("::", 1) if "::" in pair else (pair, "")
        if keep and column not in keep:
            continue
        found = (run.get("gen") or {}).get("inferred") or {}
        if found.get("inv"):
            out.setdefault(key, {})[column] = found
    return out


def report(cmp_: Comparison, verbose: bool) -> list[str]:
    lines = []
    if cmp_.error:
        return [f"  could not read the module: {cmp_.error}"]
    for i, g in enumerate(cmp_.groups):
        mark = ""
        if any(a == i for a, _b in cmp_.stronger):
            mark = "  [stronger]"
        lines.append(f"  {', '.join(g.routes)}{mark}")
        if verbose or len(cmp_.groups) > 1:
            lines.append(f"      {g.src}")
    for route, src in sorted(cmp_.unread.items()):
        lines.append(f"  {route}: not readable as SMT-LIB here")
        if verbose:
            lines.append(f"      {src}")
    if len(cmp_.rank_groups) > 1:
        lines.append(f"  ranking: {len(cmp_.rank_groups)} different")
        for g in cmp_.rank_groups:
            lines.append(f"      {', '.join(g.routes)}: {g.src}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(WORK / "results.json"),
                    nargs="*", metavar="PATH",
                    help="results file(s) to read certificates from")
    ap.add_argument("--routes", nargs="*", default=[], metavar="NAME",
                    help="compare only these columns (default: all of them)")
    ap.add_argument("--only", nargs="*", default=[], metavar="TEXT",
                    help="substring match on the row key")
    ap.add_argument("--verbose", action="store_true",
                    help="print every certificate, not only the differing ones")
    ap.add_argument("--agreed", action="store_true",
                    help="print the rows every route agreed on too")
    args = ap.parse_args()

    paths = [args.results] if isinstance(args.results, str) else args.results
    runs: dict = {}
    for path in paths:
        runs.update(json.loads(Path(path).read_text())["runs"])
    rows = {r.key: r for r in all_rows()}

    certs = certificates(runs, keep=tuple(args.routes))
    wanted = {k: v for k, v in certs.items()
              if len(v) > 1 and (not args.only
                                 or any(t in k for t in args.only))}
    print(f"{len(wanted)} rows where more than one route found a certificate"
          + (f", of {len(certs)} with any" if len(certs) != len(wanted) else ""))

    tally: Counter = Counter()
    for key in sorted(wanted):
        cmp_ = compare_row(rows[key], wanted[key])
        tally[cmp_.verdict] += 1
        if cmp_.verdict == "same" and not (args.agreed or args.verbose):
            continue
        print(f"\n{key}  [{cmp_.verdict}]")
        for line in report(cmp_, args.verbose):
            print(line)

    print("\n" + "  ".join(f"{v} {k}" for k, v in sorted(tally.items())))
    if tally["same"] and not (args.agreed or args.verbose):
        print(f"({tally['same']} rows every route agreed on are not listed; "
              f"pass --agreed to see them)")


if __name__ == "__main__":
    main()
