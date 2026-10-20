"""The invariant facts a search starts from, read off the module and
off its simulated runs (`invariant_candidates`), and the value-set
facts a floored rank needs kept (`pinned_values`).
"""

from __future__ import annotations

from fractions import Fraction

from ...smt.claim import SynthContext
from ...smt.script import smt_int, smt_real
from ..common.synth import Column, columns, moduli
from .simulate import Runs

# A Real component whose runs stay within this many values is offered the
# disjunction of them as one fact. It bounds how long a certificate can get:
# the disjunction is one `step_inv` case per value. The other half of that
# bound, the scale a ranking function is floored after, is
# `infer.common.synth._MAX_SCALE`, since every route that reads a Real rank
# wants it.
_MAX_VALUES = 8


def reading(col: Column) -> str:
    """`col` as an integer: a Bool one is `0`/`1`, anything else itself."""
    return f"(ite {col.src} 1 0)" if col.sort.isBoolean() else col.src


def _bounds(values, constants) -> tuple["int | None", "int | None"]:
    """The tightest program constants below and above every value seen.

    A constant rather than the extreme value seen: the bound a run happens to
    reach is a fact about the run, and the one the program's guard names is
    usually the fact about the program -- `NNInv` needs `x <= 9`, and `9` is
    only in the guard.
    """
    lo, hi = min(values), max(values)
    below = [c for c in constants if c <= lo]
    above = [c for c in constants if c >= hi]
    return (max(below) if below else None, min(above) if above else None)


def pinned_values(ctx: SynthContext, runs: Runs) -> dict[str, str]:
    """For each Real column the runs keep to a few values, those values.

    Over the reals an interval is hardly ever inductive -- `0 <= x <= 5`
    admits `x = 1/2`, and `x' = x - 1` steps it out -- while the values a
    run takes are finitely many and closed under the round. It is the fact
    the LRA certificates `tests/limits` carries by hand are written with,
    and the one a floored ranking function needs: see `Search.pinning`.

    Keyed by the column's own source, so an element of a matrix-shaped
    component is pinned on its own rather than with the component around it.
    """
    out: dict[str, str] = {}
    for col in columns(ctx):
        if not (col.sort.isReal() and runs.states):
            continue
        values = sorted({col.at(s) for s in runs.states})
        if 1 < len(values) <= _MAX_VALUES:
            out[col.src] = ("(or " + " ".join(f"(= {col.src} {smt_real(v)})"
                                              for v in values) + ")")
    return out


def invariant_candidates(ctx: SynthContext, runs: Runs,
                         constants: tuple[int, ...],
                         conjuncts: list[str],
                         rationals: tuple = ()) -> list[str]:
    """Facts every simulated state satisfies, as SMT-LIB over `s0..`.

    Stated over :func:`columns` rather than over components: a fact about a
    matrix-shaped component is a fact about one of its elements, since the
    tuple itself has nothing to bound.
    """
    consts = tuple(sorted(set(constants) | {0}))
    rconsts = tuple(sorted({Fraction(c) for c in consts}
                           | {Fraction(q) for q in rationals}))
    states = runs.states
    cols = columns(ctx)
    ints = [c for c in cols if c.sort.isInteger()]
    bools = [c for c in cols if c.sort.isBoolean()]
    reals = [c for c in cols if c.sort.isReal()]
    out: list[str] = list(conjuncts)
    if not states:
        return out

    def bounded(form: str, values, cs=consts, lit=smt_int) -> list[str]:
        lo, hi = _bounds(values, cs)
        facts = []
        if len(set(values)) == 1 and len(states) > 1:
            facts.append(f"(= {form} {lit(values[0])})")
        if lo is not None:
            facts.append(f"(<= {lit(lo)} {form})")
        if hi is not None:
            facts.append(f"(<= {form} {lit(hi)})")
        return facts

    def near_zero(cs):
        return tuple(c for c in cs if abs(c) <= 1)

    for col in ints:
        values = [col.at(s) for s in states]
        out += bounded(col.src, values)
        distinct = set(values)
        for k in moduli(consts):
            residues = {v % k for v in values}
            if len(residues) == 1 and len(distinct) > 2:
                out.append(f"(= (mod {col.src} {k}) {residues.pop()})")
    pins = pinned_values(ctx, runs)
    for col in reals:
        values = [col.at(s) for s in states]
        # The values themselves, before any bound -- listed first so that
        # the minimiser, which drops from the end, tries the bounds before
        # the fact that carries the module.
        if col.src in pins:
            out.append(pins[col.src])
        out += bounded(col.src, values, rconsts, smt_real)
    for col in bools:
        values = {col.at(s) for s in states}
        if len(values) == 1:
            out.append(col.src if values.pop() else f"(not {col.src})")
            continue
        for flag, lit in ((True, col.src), (False, f"(not {col.src})")):
            side = [s for s in states if col.at(s) == flag]
            for other in ints:
                for fact in bounded(other.src, [other.at(s) for s in side]):
                    if fact not in out:
                        out.append(f"(=> {lit} {fact})")
            for other in reals:
                for fact in bounded(other.src, [other.at(s) for s in side],
                                    rconsts, smt_real):
                    if fact not in out:
                        out.append(f"(=> {lit} {fact})")
    # Relations between two columns of the same sort -- mixing an Int one
    # with a Real one would need a `to_real` on the sum, and the certificate
    # is easier to read one sort at a time -- against the constants near zero
    # only: a sum bounded by some large literal is rarely what holds a run in.
    for group, cs, lit in ((ints, near_zero(consts), smt_int),
                           (reals, near_zero(rconsts), smt_real)):
        for a, x in enumerate(group):
            for y in group[a + 1:]:
                for form, values in (
                    (f"(- {x.src} {y.src})",
                     [x.at(s) - y.at(s) for s in states]),
                    (f"(+ {x.src} {y.src})",
                     [x.at(s) + y.at(s) for s in states]),
                ):
                    out += bounded(form, values, cs, lit)
    return list(dict.fromkeys(out))
