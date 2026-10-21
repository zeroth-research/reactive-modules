"""The certificate with its coefficients left open: a `Template` of
bounded `Row`s over the columns a row may bound (`weighable`), and the
templates a run asks about in turn (`templates`).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from ...smt.claim import SynthContext
from ..common.synth import columns, program_constants

from cvc5 import Kind


# Differences are quadratic in the width; past this the second template is
# more holes than any answer-literal search has been measured to close.
_MAX_WIDTH_FOR_DIFFERENCES = 4


# How far outside the module's own numbers a coefficient is looked for. The
# bound that makes an invariant true is almost always a constant the program
# already contains -- what `program_constants` is seeded from and what
# `tests/limits/README.md` records about Houdini -- so the window only has
# to be wide enough to hold a little arithmetic around them.
_WINDOW_MARGIN = 10
_MIN_WINDOW = 100


def window_for(ctx: SynthContext) -> int:
    """How far a template coefficient may reach from zero."""
    biggest = max((abs(c) for c in program_constants(ctx)), default=0)
    return max(_MIN_WINDOW, biggest * _WINDOW_MARGIN)


@dataclass(frozen=True)
class Row:
    """One `lo <= body <= hi` of a template, with `lo` and `hi` still holes.

    `at` is which *columns* the body reads -- the scalars the state is made
    of, so an element of a matrix-shaped component and not the component
    around it. One index is the column itself, two is their difference. It
    is kept as indices rather than as text because the row is needed twice
    over -- as a cvc5 term in the question, and as SMT-LIB in the
    certificate -- and a body that is printed one way and parsed back the
    other is a way for those two to disagree without anything saying so.

    `reads` is how those same columns are written, carried alongside for
    the certificate: `s0` for a scalar, `((_ tuple.select k) s0)` for an
    element.
    """

    lo: str
    hi: str
    at: tuple[int, ...]
    reads: tuple[str, ...]

    def term(self, tm, state: list):
        """The body over `state`, which is one term per column."""
        if len(self.at) == 1:
            return state[self.at[0]]
        i, j = self.at
        return tm.mkTerm(Kind.SUB, state[i], state[j])

    @property
    def smt(self) -> str:
        """The body as the certificate carries it."""
        if len(self.reads) == 1:
            return self.reads[0]
        return f"(- {self.reads[0]} {self.reads[1]})"


@dataclass(frozen=True)
class Template:
    """The certificate with its coefficients left open.

    `holes` are the variables of the conjecture's existential, in the order
    Vampire reports them, which is how an answer tuple is read back.
    """

    name: str
    holes: tuple[str, ...]
    rows: tuple[Row, ...]
    rank: tuple[str, ...] = ()      # rank coefficient holes, then the constant
    # Which columns this template says anything about, in column order, and
    # so what `rank`'s coefficients are against. Not `range(width)` once a
    # module may carry a column no row can bound.
    at: tuple[int, ...] = ()

    @property
    def all_holes(self) -> tuple[str, ...]:
        return self.holes + self.rank


def weighable(ctx: SynthContext) -> list[int]:
    """Which columns a template row can bound: the arithmetic ones.

    `lo <= x <= hi` needs an order on `x` and two integer endpoints, which
    an `Int` and a `Real` column have and a `Bool` or a bitvector one does
    not. Such a column is **left out of the template** rather than made to
    refuse the module: a certificate that says nothing about it is a weaker
    certificate, not an unsound one, and the obligations are still stated
    over the module's own transition, which reads the column whether the
    invariant mentions it or not.

    That is worth the reach it costs, and the sizing says by how much. The
    26 cells this admits are the hybrid and petri modules with a mode flag
    beside a Real plant, and **25 of them are verified by no route in the
    matrix at all** -- including `--infer houdini`, which reads a Bool, and
    reads it far more richly than any template here would (the bounds a
    component keeps on each side of a flag). So the flag is not what stops
    them. The 26th, `petri/p_timed/deadline`, is certified by three routes
    and what they found is `0 <= s0 <= 5` -- a Real interval with the flag
    unconstrained, which is exactly what this states.

    The three-case Bool row that this file once wanted -- always true,
    always false, or free -- would therefore buy one extra shape on 26
    cells where 25 need something else entirely. Left undone deliberately,
    and this is the measurement that says so.
    """
    return [i for i, c in enumerate(columns(ctx))
            if c.sort.isInteger() or c.sort.isReal()]


def template(ctx: SynthContext, which: str, *, ranked: bool) -> Template:
    """The `which` template over the columns it can bound.

    Columns, not components: a matrix-shaped component is an interval per
    element, because the tuple itself has no order for `lo <= _ <= hi` to
    bound. Holes are named after the *column*, so `A3` bounds column 3
    whether or not columns 0 to 2 are ones this template can say anything
    about -- which keeps a hole name meaning one thing across a module
    whose columns are of mixed sorts.
    """
    cols = columns(ctx)
    at = weighable(ctx)
    rows, holes = [], []
    for i in at:
        lo, hi = f"A{i}", f"B{i}"
        rows.append(Row(lo, hi, (i,), (cols[i].src,)))
        holes += [lo, hi]
    if which == "differences":
        for i, j in combinations(at, 2):
            lo, hi = f"C{i}_{j}", f"D{i}_{j}"
            rows.append(Row(lo, hi, (i, j), (cols[i].src, cols[j].src)))
            holes += [lo, hi]
    # A coefficient per column the rank can weigh, which is the same set:
    # `hrank` lands in `Nat`, and a Bool has no integer reading here that is
    # not an `ite` -- the one term this route refuses by measurement.
    rank = tuple([f"R{i}" for i in at] + ["Rc"]) if ranked else ()
    return Template(which, tuple(holes), tuple(rows), rank, tuple(at))


def templates(ctx: SynthContext, *, ranked: bool) -> list[Template]:
    """The templates this module is worth asking about, smallest first.

    The differences template is quadratic in the columns it relates, and
    the width it is measured against is how many of those there are -- a
    32-element vector is 32 columns whatever it is declared as, and a Bool
    column is none, since no row reads one.
    """
    width = len(weighable(ctx))
    out = [template(ctx, "intervals", ranked=ranked)]
    if 1 < width <= _MAX_WIDTH_FOR_DIFFERENCES:
        out.append(template(ctx, "differences", ranked=ranked))
    return out
