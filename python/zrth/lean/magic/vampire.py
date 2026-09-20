"""TA2Magic by synthesis: the certificate Vampire's answer literals derive.

``--infer vampire``.  Nothing here proposes a candidate.  The route states
the certificate's obligations with the certificate *itself* left open --
a template whose coefficients are existentially quantified -- and Vampire
answers with the coefficients, or does not answer.  Where ``--infer
houdini`` asks "is this fact inductive?" once per fact, this asks "what
numbers make the whole thing inductive?" once.

That is Vampire's *answer literal* mechanism (``--question_answering
plain``).  Handed a conjecture ``?[A,B]: phi(A,B)`` it refutes the negation
and reports the substitutions the refutation used::

    % SZS answers Tuple [([1,100]|[0,100]|[0,100])|_] for cert

which carries `0 <= s0 <= 100` for `m_countdown`, derived rather than
checked.  The alternatives are a *disjunctive* answer -- one of them is a
witness, not each of them, because a split refutation closes each branch
under its own hypothesis -- so all of them are tried below, and the one the
obligations accept is the certificate.

What the obligations have to look like
======================================
Three restrictions, and the first is the one that decides the shape of this
whole module.

**No ``ite``, anywhere.**  Vampire's answer-literal search is superposition
plus theory reasoning, and a conditional in the goal defeats it -- measured:
`m_countdown`'s own step obligation, stated with the transition as one
``ite``, does not come back inside 40 s, and the same obligation split into
its two guarded branches is answered in under one.  So the transition is
*branch-split* before it is printed: every ``ite`` condition the update and
the init terms mention becomes a case, cvc5's rewriter folds the
conditionals away under each assignment of them, and each case is one more
implication with its guard as a hypothesis.  A term that still holds an
``ite`` after that -- a condition over something the split did not reach --
is refused by name rather than printed into a question Vampire reads
perfectly well and never answers.

**No uninterpreted function for the round.**  Same measurement: defining
``nxt(S)`` by an axiom and asking about ``nxt`` is a time limit where
inlining the same arithmetic is an answer.  So the successor state is
substituted into the obligation, once per branch.

**Something arithmetic to bound.**  A template row bounds an `Int` or a
`Real` column, or an element of a matrix of either.  A **Bool** column it
cannot bound -- there is no order and no integer endpoint -- and that is
not a reason to refuse the module: Vampire reads a Bool perfectly well and
the obligations are stated over the module's own transition whether or not
the invariant mentions every column of it, so such a column is carried
*unbounded* and the certificate is weaker rather than absent.  A
**bitvector** is the different limit and is still refused by name: Vampire's
front end has none, so one anywhere makes the script unreadable rather than
the certificate weaker.  `weighable` holds both rules and sizes what the
first costs.

The coefficients stay integers whatever the column -- Vampire reports an
answer as a literal and this route reads an integer one -- so a Real column
gets an interval with *integer* endpoints, which is a restriction on reach
and not on soundness: `0 <= s0 <= 9` is a true and useful thing to say
about a tank level, and every answer is put back to cvc5 against the
module's own encoding before it is emitted.

The rank is the one place a Real cannot stay one, because ``hrank`` lands
in ``Nat``.  There a Real column is read as ``to_int`` of itself scaled by
the module's least common denominator -- the reading `--infer smt-linear`
ranks one by, for the reason it has: a quantity that falls by less than one
need not floor to anything smaller.  So the two halves of a Buchi
certificate read the same column differently, the invariant keeping its
rationals and the rank flooring them, which is what
`smt_synth.component_readings`'s ``floor_reals`` is the general form of.

cvc5 writes the question
========================
The obligations are cvc5 terms -- the module's own encoding, the same terms
`--infer houdini` proves -- so the conjecture is *built* as one cvc5 term,
holes and quantifiers and all, and cvc5 prints it as SMT-LIB.  There is no
printer in this module on purpose: one was here, in TPTP, and every kind it
did not know was a way to state something other than what was encoded (a
bitvector reached it once and crashed it).  What is left is the two
refusals a printer gave for free -- an ``ite`` that survived the split, and
a constant no quantifier binds -- said by name, and `assert-not`, which is
how SMT-LIB marks the formula that ``--question_answering`` attaches answer
literals to.

Naming is asked both ways
=========================
Vampire introduces a Tseitin definition for each subformula it judges worth
naming, and on a conjecture with holes that is decisive in *both*
directions.  Measured over `tests/limits`, 30 s each: naming answers
`m_max` and `m_toward5` where ``--naming 0`` does not; ``--naming 0``
answers `m_relu` where naming does not; they agree on `m_countdown`,
`m_deep` and `m_min`.  Neither setting is the right one, so both are asked,
each on a share of what is left, and the union is what this route reaches.

``--time_limit`` is a *scheduling* input before it is a cap: Vampire slices
it between strategies, so a small one runs a different search rather than
the same one cut short.  `m_max`'s question reaches the limit at
``--time_limit 20`` and answers after 2.5 s at 25.  So every call declares
the run's whole budget and the wall clock is what bounds it.

Two holes at a time
===================
Two holes is what this mechanism closes; three is a time limit, measured
both ways and against five option settings that moved neither ceiling.  So
the questions are kept to two where they can be:

* ``--safety`` is one question.  ``inv -> prp`` is what pins the interval
  down, and with it the coefficients are two.
* ``--buchi`` is two questions: an inductive invariant, then a rank that
  drops on *that* invariant, with its coefficients already numbers.  The
  joint question is four holes for a one-component module and answered
  nothing at all over `tests/limits`; the split derives `m_countdown` and
  `m_deep`, invariant in about a second and rank in about fifteen.

Splitting a search can cost what a joint one would have found -- an
inductive invariant need not admit a rank, and the second question cannot
go back and ask for another.  So every alternative of the first answer is
carried into the second.

An invariant asked for on its own is *looser* than one asked for with the
property, and loose is harder: with nothing pinning the interval the search
is over the whole of `Int`, and `m_countdown`'s invariant-only question does
not come back inside 30 s.  Bounded to a window around the module's own
constants it answers ``[0,100]`` in three.  The same window costs the safety
questions `m_max` and `m_toward5`, whose obligations pin the interval
already -- so it is asked for where it is needed and nowhere else.

The templates, smallest first
=============================
A template is a formula with holes, and the holes are what Vampire is asked
for.  They are tried in order, because every extra hole is a wider search
and the narrow one is usually enough:

* **intervals** -- ``A_i <= s_i <= B_i`` per component.  Two holes each.
* **intervals and differences** -- the above, plus ``C_ij <= s_i - s_j <=
  D_ij`` for each pair.  What a module whose components move together needs,
  and quadratic in the width, which is why it is second.

Under ``--buchi`` the ranking function is ``R_0*s_0 + ... + R_n-1*s_n-1 +
R_c``, asked for in the second question above.  Its obligation is stated as
``rank(s) > 0`` where the property fails and ``rank(s') < rank(s)``, which
is ite-free and implies the clamped ``Int.toNat`` form `rule_buchi` asks
for: where ``r > 0`` and ``r' < r``, ``max(r',0) < max(r,0)`` whatever the
sign of ``r'``.

What comes back is checked
==========================
An answer literal is a substitution *a* refutation used, and a refutation of
a mis-stated question proves nothing about the module -- nor does one branch
of a disjunctive answer.  So the coefficients Vampire returns are put back
into the template and the obligations are re-asked of cvc5 -- through the
same :mod:`zrth.lean.houdini_solver` seam `--infer houdini` uses, in
milliseconds.  A certificate that does not survive that is reported as not
found, with what Vampire said.  This is not Houdini: nothing is filtered and
nothing is proposed, it is the one check that the derivation is honest.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

from ..cert import CertificateData
from ..common import Refused
from ..houdini_solver import (
    DEFAULT_TIMEOUT,
    Cvc5Solver,
    _kill_group,
    decimals,
)
from . import TA2Magic
from .houdini import Candidate, Obligations, columns
from ..smt_synth import (
    SynthContext,
    denominator_scale,
    floor_real,
    floor_real_src,
    program_constants,
    program_rationals,
    smt_int,
)

try:
    import cvc5                                      # type: ignore
    from cvc5 import Kind
except ImportError:                                  # pragma: no cover
    cvc5 = None
    Kind = None

# The templates, in the order they are tried.
TEMPLATES = ("intervals", "differences")
DEFAULT_TEMPLATE = TEMPLATES[0]

# A branch is one assignment of the transition's `ite` conditions, so the
# case count doubles with each. Past this the question is too long to be
# worth asking, and the route says so rather than printing it.
_MAX_CONDITIONS = 5
_MAX_BRANCHES = 2 ** _MAX_CONDITIONS
# Differences are quadratic in the width; past this the second template is
# more holes than any answer-literal search has been measured to close.
_MAX_WIDTH_FOR_DIFFERENCES = 4


# ══════════════════════════════════════════════════════════════════════════
# Branches: the transition without its conditionals
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Branch:
    """One case of a round: its guard, and the state terms under that guard.

    `guard` is the conjunction of condition literals that selects this case
    and `state` is the terms with every `ite` folded away by cvc5's rewriter
    under that assignment. The guard is a hypothesis of the obligation
    rather than a condition inside it, which is the whole point: Vampire
    answers an implication and does not answer a conditional.
    """

    guard: tuple                # cvc5 Bool terms, all of them true here
    state: tuple                # the folded terms, ite-free


def ite_conditions(terms) -> list:
    """Every condition an `ite` in `terms` branches on, outermost first.

    Deduplicated by printed form: a transition that tests `s0 = 0` in three
    components is one case split, not three.
    """
    seen: dict = {}

    def walk(t):
        if t.getKind() == Kind.ITE:
            seen.setdefault(str(t[0]), t[0])
        for child in t:
            walk(child)

    for t in terms:
        walk(t)
    return list(seen.values())


def split(ctx: SynthContext, terms, what: str) -> list[Branch]:
    """`terms`, split into the ite-free cases of their own conditions.

    The round and the initial state are split *apart*, because they are not
    quantified over the same things: the initial state is a function of the
    inputs alone, so a guard taken from the round -- which tests the latched
    state -- would leave a variable in the entry obligation that nothing
    binds. Every assignment is emitted, the unsatisfiable ones included: an
    unsatisfiable guard makes its implication vacuously true, which costs
    Vampire a clause and costs this module the logic that would have worked
    out which ones those are.
    """
    tm = ctx.tm
    rewriter = cvc5.Solver(tm)
    terms = list(terms)
    conds = ite_conditions(terms)
    if len(conds) > _MAX_CONDITIONS:
        raise Refused(
            f"--infer vampire splits {what} on the {len(conds)} conditions "
            f"its `ite`s test, which is {2 ** len(conds)} cases and past the "
            f"{_MAX_BRANCHES} this route will print. `--infer houdini` "
            f"states the round once, conditionals and all."
        )
    out = []
    for bits in product((True, False), repeat=len(conds)):
        subs = [tm.mkBoolean(b) for b in bits]
        folded = tuple(rewriter.simplify(t.substitute(conds, subs))
                       if conds else t for t in terms)
        guard = tuple(c if b else tm.mkTerm(Kind.NOT, c)
                      for c, b in zip(conds, bits))
        out.append(Branch(guard, folded))
    return out


# ══════════════════════════════════════════════════════════════════════════
# SMT-LIB: cvc5 prints the question it built
# ══════════════════════════════════════════════════════════════════════════

_PRELUDE = "(set-logic ALL)\n"
# Not `(assert (not ...))`: `assert-not` is what marks a formula as *the*
# conjecture, which is what `--question_answering` attaches answer literals
# to. An ordinary negated assertion is refuted without an answer.
_CONJECTURE = "(assert-not {})\n(check-sat)\n"


def script_for(conjecture) -> str:
    """One cvc5 term as a whole SMT-LIB problem for Vampire.

    Through `decimals`, which is the one rewrite cvc5's printing needs
    before Vampire reads it: a rational prints as `(/ 1 2)`, whose
    arguments are Int, and Vampire's SMT-LIB front end sorts literals
    strictly enough to call that "invalid sort $int for interpretation /".
    `houdini_solver` has said so since it began passing its own scripts
    through the same function; this one did not, and nothing noticed while
    the sorts gate refused every module with a rational in it.

    The term *is* the question, so cvc5 prints it and this adds the two
    lines around it. There is no printer here on purpose: the obligations
    are cvc5 terms the module's own encoding produced, and every printer
    between them and the prover is a chance to state something other than
    what was encoded.
    """
    return _PRELUDE + decimals(_CONJECTURE.format(conjecture))


def refuse_ites(term) -> None:
    """An `ite` the split did not reach is refused rather than printed.

    It would print perfectly well -- SMT-LIB has `ite` and Vampire reads it
    -- and then never be answered, which reaches the user as a timeout
    rather than as the unsupported shape it is. Asked of the whole assembled
    question, so a conditional in the property or in a precondition is met
    here too and not only one the branch split left behind.
    """
    if ite_conditions([term]):
        raise Refused(
            "--infer vampire cannot state an obligation that still holds "
            "an `ite` after the transition was branch-split: Vampire's "
            "answer-literal search does not see through a conditional. "
            "`--infer houdini` proves the same obligation with one."
        )


def free_constants(term) -> list:
    """Every uninterpreted constant `term` still mentions, outermost first.

    What the old TPTP printer refused for free by having no variable for a
    symbol: cvc5 prints a stray constant perfectly happily, and Vampire
    reads an undeclared one as an error three steps later.
    """
    seen: dict = {}

    def walk(t):
        if t.getKind() == Kind.CONSTANT:
            seen.setdefault(str(t), t)
        for child in t:
            walk(child)

    walk(term)
    return list(seen.values())


# ══════════════════════════════════════════════════════════════════════════
# The template
# ══════════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════════
# The question
# ══════════════════════════════════════════════════════════════════════════


class Question:
    """One module's obligations as one cvc5 term with holes in it.

    The obligations are built over the constants `Obligations` already made
    -- `v_s0`, `v_el0`, `v_en0` -- and turned into bound variables only on
    the way out, by `_forall`. That way every term here is a term of the
    module's own encoding right up to the point where it is quantified, and
    the question Vampire reads is cvc5's printing of it.
    """

    def __init__(self, ctx: SynthContext, ob: Obligations,
                 entry: list[Branch], step: list[Branch]):
        self.ctx, self.ob, self.tm = ctx, ob, ctx.tm
        self.entry, self.step = entry, step
        # One bound variable per constant, and the constants are one per
        # *element* -- a matrix-shaped component is not a name Vampire can
        # be given. The names reach Vampire through cvc5's printer, so they
        # have to be distinct: two `mkVar`s of one name print alike and
        # mean different things. `v_s0_1` is bound as `s0_1`, which is the
        # constant's own name without the prefix that marks it a constant.
        flat_s = ob.flatten(ob.s)
        flat_e = ob.flatten(list(ob.el) + list(ob.en))
        self.consts = flat_s + flat_e
        self.state = [ctx.tm.mkVar(c.getSort(), str(c)[2:]) for c in flat_s]
        # The latched state as one term per column, which is what a row and
        # a rank coefficient are indexed by.
        self.here = flat_s
        self.inputs = [ctx.tm.mkVar(c.getSort(), f"e{i}")
                       for i, c in enumerate(flat_e)]
        self.vars = self.state + self.inputs
        self.holes: dict = {}
        self.window = window_for(ctx)
        # What a Real column is multiplied by before the rank floors it.
        # 1 for a module with no rationals in it, which is every module this
        # route took before it read them.
        self.scale = denominator_scale(program_rationals(ctx))

    # --- the template, as a term over a state ----------------------------

    def _hole(self, name: str):
        """The variable standing for one coefficient, made once per name.

        Always an integer, whatever the column it bounds. Vampire reports an
        answer as a literal and `_number` reads an integer one, so a Real
        hole would come back as a rational this route cannot carry -- and an
        integer endpoint on a rational column is still a true statement about
        it. `_fit` is what makes the sorts agree.
        """
        if name not in self.holes:
            self.holes[name] = self.tm.mkVar(self.tm.getIntegerSort(), name)
        return self.holes[name]

    def _fit(self, hole, body):
        """`hole` in `body`'s arithmetic, so the printed question is sorted.

        cvc5 coerces an `Int` against a `Real` silently and prints `(<= A
        x)`, which is well sorted only if the reader coerces too. Vampire's
        SMT-LIB front end is the reader, and the one thing this module has
        learned about it is not to find out: `to_real` is written down.
        """
        return (self.tm.mkTerm(Kind.TO_REAL, hole)
                if body.getSort().isReal() else hole)

    def _rows(self, tpl: Template, at: list):
        """`tpl` read at `at`, which is one term per component."""
        out = []
        for row in tpl.rows:
            body = row.term(self.tm, at)
            out += [self.tm.mkTerm(Kind.LEQ, self._fit(self._hole(row.lo), body),
                                   body),
                    self.tm.mkTerm(Kind.LEQ, body,
                                   self._fit(self._hole(row.hi), body))]
        return self._and(out)

    def _rank(self, tpl: Template, at: list):
        """`Rc + R0*s0 + ... `, the rank with its coefficients left open.

        Over the *floored* columns: `hrank` lands in `Nat`, so a Real column
        is read as `to_int` of itself scaled by the module's least common
        denominator -- the same reading `--infer smt-linear` ranks one by,
        and for the same reason, that a quantity falling by less than one
        need not floor to anything smaller. The invariant rows above keep
        their rationals; only the rank has to be an integer.

        `tpl.at` says which columns have a coefficient, so a module with a
        Bool column among its Reals ranks over the Reals and leaves the flag
        out -- the flag has no integer reading here that is not an `ite`,
        which is the one term this route refuses by measurement.
        """
        out = self._hole(tpl.rank[-1])
        for c, v in zip(tpl.rank, self.ranked_cols(tpl, at)):
            out = self.tm.mkTerm(Kind.ADD, out,
                                 self.tm.mkTerm(Kind.MULT, self._hole(c), v))
        return out

    def ranked_cols(self, tpl: Template, at: list) -> list:
        """The columns `tpl` ranks over, each Real one floored."""
        return [floor_real(self.tm, at[i], self.scale) for i in tpl.at]

    def _and(self, parts):
        kept = [p for p in parts
                if not (p.getKind() == Kind.CONST_BOOLEAN
                        and p.getBooleanValue())]
        if not kept:
            return self.tm.mkBoolean(True)
        return kept[0] if len(kept) == 1 else self.tm.mkTerm(Kind.AND, *kept)

    # --- the obligations --------------------------------------------------

    def conjecture(self, tpl: Template, prp) -> str:
        """`--safety`: an inductive invariant that implies the property."""
        obs = self._obligations(tpl, prp, inductive=True, implies=True,
                                ranked=False)
        return script_for(self._ask_for(tpl.holes, obs))

    def invariant_conjecture(self, tpl: Template) -> str:
        """`init_inv` and `step_inv` alone -- an inductive invariant, with
        no claim about the property. Stage one of the Buchi split."""
        obs = self._obligations(tpl, None, inductive=True, implies=False,
                                ranked=False)
        return script_for(self._ask_for(tpl.holes, obs, bounded=True))

    def rank_conjecture(self, tpl: Template, prp, fixed: dict) -> str:
        """`hrank` over an invariant already derived. Stage two.

        Only the rank obligations: the invariant is in them already, as the
        hypothesis `here`, and with its coefficients fixed to numbers it is
        a statement about the module rather than a hole to fill. `init_inv`
        and `step_inv` are settled and asking them again is clauses for
        nothing.
        """
        obs = self._obligations(tpl, prp, inductive=False, implies=False,
                                ranked=True)
        return script_for(self._ask_for(tpl.rank, obs, fixed=fixed))

    def _obligations(self, tpl: Template, prp, *, inductive: bool,
                     implies: bool, ranked: bool) -> list:
        tm, ob = self.tm, self.ob
        obligations = []
        # init_inv. The initial state is a function of the inputs alone, so
        # this is quantified over the inputs and split on its own
        # conditions -- a guard from the round would leave a latched-state
        # variable here that nothing binds.
        if inductive:
            for br in self.entry:
                hyp = self._and([ob.init_pre, *br.guard])
                obligations.append(self._forall(
                    self.inputs,
                    self._implies(hyp, self._rows(tpl, self._cols(br)))))
        for br in self.step:
            here = self._rows(tpl, self.here)
            if inductive:
                hyp = self._and([here, ob.update_pre, *br.guard])
                obligations.append(self._forall(
                    self.vars,
                    self._implies(hyp, self._rows(tpl, self._cols(br)))))
            if ranked:
                # hrank, ite-free: positive where the property fails, and
                # smaller after the round. Implies the `Int.toNat` form.
                now = self._rank(tpl, self.here)
                drops = self._and([
                    tm.mkTerm(Kind.GT, now, tm.mkInteger(0)),
                    tm.mkTerm(Kind.LT, self._rank(tpl, self._cols(br)), now),
                ])
                fails = tm.mkTerm(Kind.NOT, self._at(prp, ob.s))
                obligations.append(self._forall(
                    self.vars,
                    self._implies(
                        self._and([here, fails, ob.update_pre, *br.guard]),
                        drops)))
        if implies:
            obligations.append(self._forall(
                self.state,
                self._implies(self._rows(tpl, self.here),
                              self._at(prp, ob.s))))
        return obligations

    def _ask_for(self, holes: tuple[str, ...], obligations: list,
                 fixed: "dict | None" = None, bounded: bool = False):
        """`obligations`, with `fixed` filled in and `holes` left open.

        `bounded` confines each open hole to the window, and it is asked for
        only where the obligations leave the coefficients loose. Measured
        both ways over `tests/limits`: bounding costs the safety questions
        `m_max` and `m_toward5`, whose `inv -> prp` obligation pins the
        interval already so the window is clauses for nothing; and it is
        what makes the invariant-only question answerable at all --
        `m_countdown`'s does not come back inside 30 s free and answers
        `[0,100]` in three seconds bounded. A refutation only has to produce
        *a* witness, and where nothing else pins one down a window is what
        makes producing one a finite matter rather than a guess.
        """
        tm = self.tm
        body = self._and(obligations)
        if fixed:
            body = body.substitute(
                [self._hole(h) for h in fixed],
                [tm.mkInteger(v) for v in fixed.values()])
        parts = []
        if bounded:
            window = tm.mkInteger(self.window)
            parts = [b for h in holes for b in (
                tm.mkTerm(Kind.LEQ, tm.mkTerm(Kind.NEG, window),
                          self._hole(h)),
                tm.mkTerm(Kind.LEQ, self._hole(h), window))]
        out = tm.mkTerm(Kind.EXISTS,
                        tm.mkTerm(Kind.VARIABLE_LIST,
                                  *[self._hole(h) for h in holes]),
                        self._and(parts + [body]))
        refuse_ites(out)
        return out

    def _implies(self, hyp, goal):
        if hyp.getKind() == Kind.CONST_BOOLEAN and hyp.getBooleanValue():
            return goal
        return self.tm.mkTerm(Kind.IMPLIES, hyp, goal)

    def _at(self, term, vs):
        # Folded: substituting a matrix-shaped holder leaves a select of a
        # constructor, and a tuple is the one thing Vampire cannot be shown.
        return self.ob._readable(term.substitute(self.ctx.state, vs))

    def _cols(self, br: Branch) -> list:
        """A branch's successor state, one term per column."""
        return self.ob.flatten(self.ob.s, list(br.state))

    def _forall(self, vs: list, body):
        """`body` over the constants, quantified over the variables for them.

        The substitution is what turns `v_s0` into the bound `s0`, and only
        the constants `vs` stands for are substituted: a constant left over
        is one this obligation is not quantified over, and it is refused by
        name rather than printed as a symbol nothing declares. That is the
        entry obligation's own failure mode -- a guard from the round
        mentions the latched state, which nothing binds there.
        """
        names = {str(v) for v in vs}
        pairs = [(c, v) for c, v in zip(self.consts, self.vars)
                 if str(v) in names]
        if pairs:
            body = body.substitute([c for c, _ in pairs],
                                   [v for _, v in pairs])
        stray = free_constants(body)
        if stray:
            raise Refused(
                f"--infer vampire states an obligation over "
                f"{', '.join(sorted(str(v) for v in vs)) or 'nothing'}, and "
                f"it mentions {', '.join(str(c) for c in stray)}, which "
                f"nothing there binds"
            )
        if not vs:
            return body
        return self.tm.mkTerm(Kind.FORALL,
                              self.tm.mkTerm(Kind.VARIABLE_LIST, *vs), body)


# ══════════════════════════════════════════════════════════════════════════
# Asking Vampire for an answer rather than a yes
# ══════════════════════════════════════════════════════════════════════════

# `% SZS answers Tuple [([1,100]|[0,100]|[0,100])|_] for cert`, and the
# single form `% SZS answers Tuple [[0,100]|_]`.
#
# The alternatives after `|` are a *disjunctive* answer: the refutation
# established that one of them is a witness, not that each is. A split
# refutation -- AVATAR's, which is what closes these questions -- reports one
# tuple per branch it closed, and a branch closed under a hypothesis that
# does not hold contributes a tuple that is not a witness at all. Measured on
# `m_countdown`: twelve alternatives, ten of them the `[0,100]` that is an
# invariant and the first two `[1,100]`, which is not preserved (`s0 = 1`
# steps to `0`). So every alternative is a candidate and they are tried in
# turn against the obligations; taking the first is how this route spent a
# while reporting no certificate for a module it had been handed one for.
_ANSWER = "SZS answers Tuple "


def answer_tuples(line: str) -> "list[list[str]] | None":
    """Every alternative in a `SZS answers` line, one string per hole.

    Scanned with a bracket depth rather than matched with a regex: a hole
    the refutation never pinned down prints as `∀X0.[X0]`, brackets and all,
    and a pattern that stops at the first `]` reads that as the end of the
    tuple and comes back one value short.
    """
    at = line.find(_ANSWER)
    if at < 0:
        return None
    rest = line[at + len(_ANSWER):].lstrip()
    if not rest.startswith("["):
        return None
    rest = rest[1:].lstrip()
    if rest.startswith("("):                         # the per-disjunct form
        rest = rest[1:].lstrip()
    out = []
    while rest.startswith("["):
        one, rest = _one_tuple(rest)
        if one is None:
            break
        out.append(one)
        rest = rest.lstrip()
        if not rest.startswith("|"):
            break
        rest = rest[1:].lstrip()
    return out or None


def _one_tuple(rest: str) -> "tuple[list[str] | None, str]":
    """One `[a,b,...]`, and what is left of the line after it."""
    depth, out, cur = 0, [], ""
    for i, ch in enumerate(rest[1:], start=2):
        if ch in "([":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "]":
            if depth == 0:
                out.append(cur)
                return out, rest[i:]
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(cur)
            cur = ""
            continue
        cur += ch
    return None, ""


@dataclass(frozen=True)
class Derived:
    """What Vampire answered: the alternatives, each one value per hole.

    Any one of them may be the witness, and which one is settled by putting
    it back into the template and re-asking the obligations.
    """

    answers: tuple[tuple[int, ...], ...]
    secs: float


class Answers:
    """Vampire in question-answering mode, on this run's deadline."""

    # Both ways of naming subformulas, because on a conjecture with holes
    # neither is the right one -- see the module docstring for what each
    # answers that the other does not. Named rather than a bare tuple of
    # flags so the log can say which one came back.
    STRATEGIES = (("naming", ()), ("no naming", ("--naming", "0")))
    # The short rung's wall clock. Measured over `tests/limits` with the
    # declared limit left alone: every answer but `m_max`'s and `m_relu`'s
    # arrives inside eight seconds, so one pass at this over every strategy
    # is what keeps the questions that do answer from waiting behind the
    # ones that do not.
    SHORT = 8.0

    def __init__(self, exe: str, *, seconds: float, log=print):
        self.exe = exe
        self.deadline = time.monotonic() + seconds
        # What Vampire is *told* it has, which is not what it is given.
        # `--time_limit` is a scheduling input before it is a cap: Vampire
        # slices it between strategies, so a small one runs a different
        # search rather than the same one cut short. Measured on `m_max`'s
        # question -- `--time_limit 20` reaches the limit, `--time_limit 25`
        # answers after 2.5 s. So every call declares the run's whole
        # budget, and what actually bounds a call is the wall clock below.
        self.declared = max(1, int(seconds))
        self.log = log
        self.calls = 0
        self.spent = 0.0

    def left(self) -> float:
        return self.deadline - time.monotonic()

    def ask(self, script: str, holes: int) -> "Derived | None":
        """The first answer any strategy gives, short rung first.

        The rungs are wall clock, and the declared limit does not move with
        them -- which is the whole reason a short rung is safe here. A short
        `--time_limit` would have Vampire run a *different* search; a short
        wall runs the same one and stops it early, and most answers measured
        over `tests/limits` arrive in under four seconds.
        """
        for short in (True, False):
            for n, (name, extra) in enumerate(self.STRATEGIES):
                left = self.left()
                # On the long rung what is left is shared out between the
                # strategies still to come, so the last is still asked.
                budget = (min(self.SHORT, left) if short
                          else left / (len(self.STRATEGIES) - n))
                answer = self._call(script, holes, budget, extra)
                if answer is not None:
                    self.log(f"[vampire] answered with {name}")
                    return answer
        return None

    def _call(self, script: str, holes: int, budget: float,
              extra: tuple) -> "Derived | None":
        if budget < 1:
            return None
        with tempfile.NamedTemporaryFile(
            "w", suffix=".smt2", prefix="verith-vampire-qa-", delete=False
        ) as f:
            f.write(script)
            path = f.name
        # Default mode, not `portfolio`: measured on this route's own
        # question for `m_countdown`, `--mode portfolio --schedule casc`
        # reaches the time limit where the default mode answers `[0,100]`
        # at once. A portfolio strategy is tuned to find a refutation, and
        # what is wanted here is the *substitution* a refutation carries.
        cmd = [
            self.exe, "--input_syntax", "smtlib2",
            "--question_answering", "plain",
            *extra,
            "--time_limit", f"{self.declared}",
            path,
        ]
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        try:
            # The real bound: `--time_limit` told Vampire what schedule to
            # run, and this is the share of the deadline it actually gets.
            out, _ = proc.communicate(timeout=budget)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            out = ""
        finally:
            _kill_group(proc)
            Path(path).unlink(missing_ok=True)
        dt = time.perf_counter() - t0
        self.calls += 1
        self.spent += dt
        error = next((ln for ln in out.splitlines()
                      if "User error" in ln or "Parsing Error" in ln), None)
        if error is not None:
            raise Refused(
                f"--infer vampire wrote a question Vampire cannot read: "
                f"{error.strip()[:300]}"
            )
        return self._read(out, holes, dt)

    def _read(self, out: str, holes: int, dt: float) -> "Derived | None":
        alts = next(
            (t for t in (answer_tuples(ln) for ln in out.splitlines())
             if t is not None), None)
        if alts is None:
            return None
        seen, kept = set(), []
        for parts in alts:
            values = []
            for part in parts:
                v = _integer(part.strip())
                if v is None:
                    # A hole the refutation never had to pin down: any value
                    # serves, and zero is the one the certificate reads
                    # best. It is checked with the rest before anything is
                    # emitted.
                    v = 0
                values.append(v)
            if len(values) != holes:
                self.log(f"[vampire] an answer has {len(values)} values for "
                         f"{holes} holes; ignoring it")
                continue
            if tuple(values) not in seen:
                seen.add(tuple(values))
                kept.append(tuple(values))
        return Derived(tuple(kept), dt) if kept else None


def _integer(text: str) -> "int | None":
    text = text.strip()
    m = re.fullmatch(r"\$uminus\((\d+)\)", text)
    if m:
        return -int(m.group(1))
    return int(text) if re.fullmatch(r"-?\d+", text) else None


# ══════════════════════════════════════════════════════════════════════════
# The route
# ══════════════════════════════════════════════════════════════════════════


class TA2MagicVampire(TA2Magic):
    """The certificate Vampire derives, checked before it is handed on."""

    def __init__(self, module, *, vampire: str, timeout: float = DEFAULT_TIMEOUT,
                 artifacts=None, log=print):
        super().__init__("")
        if cvc5 is None:                             # pragma: no cover
            raise Refused(
                "--infer vampire needs the `cvc5` package to encode the "
                "module, which is not importable. Install with: "
                "uv pip install cvc5"
            )
        self.module = module
        self.exe = vampire
        self.timeout = float(timeout)
        self.artifacts = artifacts
        self.log = log

    # --- driver -----------------------------------------------------------

    def infer(self, cd: CertificateData) -> CertificateData:
        # `SynthContext.build` refuses a component this route cannot weigh
        # as an integer -- Real, bitvector, matrix-shaped -- before anything
        # here is printed, which is the same gate `--infer smt-linear` uses.
        ctx = SynthContext.build(self.module, cd, route="vampire",
                                 takes=("int", "bool", "bv", "real", "tuple"))
        self._check_sorts(ctx)
        self.ctx = ctx
        self.width = len(columns(ctx))
        self.ob = ob = Obligations(ctx)
        entry = split(ctx, ob.init, "the initial state")
        step = split(ctx, ob.next, "the round")
        q = Question(ctx, ob, entry, step)
        # The question owns the scale; `_read_back` prints the same floor it
        # put in the conjecture, so there is one of them and not two.
        self.scale = q.scale
        bounded = len(weighable(ctx))
        carried = ("" if bounded == self.width else
                   f" ({self.width - bounded} carried unbounded)")
        self.log(f"[vampire] {len(step)} branch(es) of the round, "
                 f"{len(entry)} of the initial state, "
                 f"{bounded} of {self.width} column(s) bounded{carried}")
        if self.scale != 1:
            self.log(f"[vampire] Real column(s): the rank reads them "
                     f"through a floor after scaling by {self.scale}")

        asker = Answers(self.exe, seconds=self.timeout, log=self.log)
        derive = self._safety if cd.is_safety else self._ranked
        tried = []
        for tpl in templates(ctx, ranked=not cd.is_safety):
            if asker.left() < 1:
                break
            tried.append(tpl.name)
            out = derive(q, tpl, asker, cd)
            if out is not None:
                return out
            self.log(f"[vampire] nothing from {tpl.name} "
                     f"({asker.spent:.1f} s spent)")
        self.log(f"[vampire] {asker.calls} Vampire call(s), "
                 f"{asker.spent:.1f} s")
        return self._none(cd, tried, asker)

    def _safety(self, q: Question, tpl: Template, asker: Answers,
                cd: CertificateData) -> "CertificateData | None":
        """One question: the invariant and the property it has to imply.

        `inv -> prp` is what pins the coefficients down, which is why this
        is one question where `--buchi` is two.
        """
        script = q.conjecture(tpl, self.ctx.prp)
        self.log(f"[vampire] template {tpl.name}: {len(tpl.holes)} holes, "
                 f"{len(script)} chars")
        answer = asker.ask(script, len(tpl.holes))
        if answer is None:
            return None
        for named in self._alternatives(answer, tpl.holes, asker):
            inv, _ = self._read_back(tpl, named, cd)
            if self._checks_out(inv, None, cd):
                return self._emit(cd, inv, None, tpl)
        return None

    def _ranked(self, q: Question, tpl: Template, asker: Answers,
                cd: CertificateData) -> "CertificateData | None":
        """Two questions: an inductive invariant, then a rank that drops on
        it.

        Asked apart rather than as one conjecture with every hole in it,
        because two holes is what this mechanism reaches: the joint question
        for a one-component module is four, and it answered nothing at all
        over `tests/limits`, where the split derives `m_countdown` and
        `m_deep` -- invariant in about a second, rank in about fifteen.

        Splitting a search can cost what a joint one would have found: an
        invariant that is inductive need not admit a rank, and this one
        cannot go back and ask for another. So every alternative of the
        first answer is carried into the second, which is the disjunctive
        answer earning its keep twice.
        """
        script = q.invariant_conjecture(tpl)
        self.log(f"[vampire] template {tpl.name}: an invariant in "
                 f"{len(tpl.holes)} holes, {len(script)} chars")
        answer = asker.ask(script, len(tpl.holes))
        if answer is None:
            return None
        for inv_named in self._alternatives(answer, tpl.holes, asker):
            ranks = asker.ask(q.rank_conjecture(tpl, self.ctx.prp, inv_named),
                              len(tpl.rank))
            if ranks is None:
                self.log("[vampire] no ranking over that invariant")
                continue
            for named in self._alternatives(ranks, tpl.rank, asker):
                inv, rank = self._read_back(tpl, {**inv_named, **named}, cd)
                if self._checks_out(inv, rank, cd):
                    return self._emit(cd, inv, rank, tpl)
        return None

    def _alternatives(self, answer: Derived, holes: tuple[str, ...],
                      asker: Answers):
        """Each alternative of a disjunctive answer, named, while time lasts.

        One of them is a witness, not each of them, so each is a candidate
        until the obligations say otherwise. Checking one is cvc5 on a few
        small queries, but there can be a dozen, so the run's own deadline
        bounds the loop -- after the first, which is always worth it.
        """
        for n, values in enumerate(answer.answers):
            if n and asker.left() < 0:
                self.log(f"[vampire] out of time with "
                         f"{len(answer.answers) - n} alternative(s) of the "
                         f"answer unchecked")
                return
            named = dict(zip(holes, values))
            self.log("[vampire] Vampire answered "
                     + ", ".join(f"{h}={v}" for h, v in named.items()))
            yield named

    # --- what the answer means --------------------------------------------

    def _read_back(self, tpl: Template, named: dict,
                   cd: CertificateData) -> tuple:
        """The template with Vampire's numbers in it, as SMT-LIB."""
        facts = []
        for row in tpl.rows:
            facts.append(f"(<= {smt_int(named[row.lo])} {row.smt})")
            facts.append(f"(<= {row.smt} {smt_int(named[row.hi])})")
        inv = ("true" if not facts else facts[0] if len(facts) == 1
               else "(and " + " ".join(facts) + ")")
        if cd.is_safety:
            return inv, None
        # A coefficient of one is the column itself: `(* 1 s0)` is what the
        # template says and `s0` is what the certificate should read. The
        # reads are taken from `tpl.at` rather than from the row order, so
        # an element says `((_ tuple.select k) s0)` and a Real one says
        # `(to_int ...)` -- the same floor `_rank` put in the question,
        # because `hrank` lands in `Nat` -- and a module carrying a column
        # no row bounds still pairs each coefficient with its own column.
        cols = columns(self.ctx)
        reads = [floor_real_src(cols[i].src, cols[i].sort, self.scale)
                 for i in tpl.at]
        terms = [reads[i] if named[c] == 1
                 else f"(* {smt_int(named[c])} {reads[i]})"
                 for i, c in enumerate(tpl.rank[:-1]) if named[c]]
        const = named[tpl.rank[-1]]
        parts = ([smt_int(const)] if const or not terms else []) + terms
        rank = parts[0] if len(parts) == 1 else "(+ " + " ".join(parts) + ")"
        return inv, rank

    def _checks_out(self, inv_src: str, rank_src: "str | None",
                    cd: CertificateData) -> bool:
        """The four obligations, restated and put to cvc5.

        An answer literal is the substitution *a* refutation used, and this
        route wrote the question it refuted -- so the certificate is checked
        against the module's own encoding before anything is emitted, the
        same way `--pre-check cvc5` would check one supplied by hand.
        """
        ob, ctx = self.ob, self.ctx
        try:
            facts = [Candidate(inv_src, ctx.env.parse_expr(inv_src))]
            rank = (None if rank_src is None
                    else Candidate(rank_src, ctx.env.parse_expr(rank_src)))
        except Exception as e:                       # noqa: BLE001
            self.log(f"[vampire] the answer does not parse back: {e}")
            return False
        solver = Cvc5Solver(ctx.tm, seconds=max(5.0, min(30.0, self.timeout)),
                            log=self.log)
        queries = [ob.holds_at_entry(facts), ob.preserved(facts, facts)]
        if cd.is_safety:
            queries.append(ob.implies(facts, ctx.prp))
        else:
            queries.append(ob.drops(facts, rank))
        for name, query in zip(("init_inv", "step_inv", "the last"), queries):
            answer = solver.prove(query, 10)
            if not answer.proved:
                self.log(f"[vampire] {name} does not hold of the answer "
                         f"({answer.verdict.value})")
                return False
        return True

    # --- reading the module -----------------------------------------------

    def _check_sorts(self, ctx: SynthContext) -> None:
        """What this route cannot *state*, told apart from what it cannot bound.

        These were one refusal and are two limits, which is why 26 cells
        were turned away for the wrong reason.

        **A bitvector it cannot state.** Vampire's SMT-LIB front end has
        integer and real arithmetic and datatypes and no bitvectors, so a
        bitvector anywhere -- a state column or an input -- makes the whole
        script unreadable rather than the certificate weaker. That is a
        refusal, and it names the wire.

        **A Bool it cannot bound**, which is a different thing and not a
        reason to turn the module away. `lo <= x <= hi` wants an order and
        two integer endpoints and a Bool has neither, but Vampire reads a
        Bool perfectly well, and the obligations are stated over the
        module's own transition whether or not the invariant mentions every
        column of it. So a Bool column is left out of the template --
        `weighable` says so, and sizes what it costs -- and a Bool *input*
        was never bounded by anything here to begin with.

        A **Real** is arithmetic and is bounded. An interval with integer
        endpoints is a perfectly good statement about a rational column --
        `0 <= s0 <= 9` says something true and useful about a tank level --
        and the rank, which does have to land in `Nat`, reads such a column
        through a floor. What the endpoints cannot be is fractional, and
        that is a restriction on what this route reaches rather than a
        soundness question: every answer is put back to cvc5 against the
        module's own encoding by `_checks_out` before anything is emitted.

        What is left is the module with nothing to say anything *about* --
        no column a row can bound -- which would search for the invariant
        `true` and is refused up front instead.
        """
        def bitvectors(sort) -> bool:
            """Whether `sort`, or an element of it, is one."""
            if sort.isTuple():
                return any(e.isBitVector() for e in sort.getTupleSorts())
            return sort.isBitVector()

        bad = [f"s{i} is {s}" for i, s in enumerate(ctx.env.state_sorts)
               if bitvectors(s)]
        bad += [f"{c} is {c.getSort()}"
                for c in list(ctx.extl_next) + list(ctx.extl_latched)
                if bitvectors(c.getSort())]
        if bad:
            raise Refused(
                f"--infer vampire states its obligations as SMT-LIB for "
                f"Vampire, whose front end has no bitvectors, and this "
                f"module has {', '.join(bad)}. Nothing here can state the "
                f"question, so this is not a matter of which certificate is "
                f"asked for. `--infer smt-linear` weighs a bitvector as its "
                f"unsigned value; `--infer houdini` on cvc5 reads one too."
            )
        if not weighable(ctx):
            sorts = ", ".join(f"s{i} is {c.sort}"
                              for i, c in enumerate(columns(ctx)))
            raise Refused(
                f"--infer vampire bounds each state column between two "
                f"integers, and this module has no column it can bound: "
                f"{sorts}. A Bool column is carried unbounded where there "
                f"is something else to bound, but here the invariant would "
                f"be `true`, which is inductive and proves nothing. "
                f"`--infer houdini` states the bounds a component keeps on "
                f"each side of a Bool flag, which is what this shape wants."
            )

    # --- out ---------------------------------------------------------------

    def _emit(self, cd: CertificateData, inv_src: str,
              rank_src: "str | None", tpl: Template) -> CertificateData:
        self.log(f"[vampire] inv: {inv_src}")
        cd.inv = cd.inv_smt = inv_src
        if rank_src is not None:
            self.log(f"[vampire] ranking: {rank_src}")
            cd.ranking = cd.ranking_smt = rank_src
        self._record(inv_src, rank_src, tpl)
        return cd

    def _record(self, inv_src: str, rank_src: "str | None",
                tpl: Template) -> None:
        if self.artifacts is None:
            return
        what = (f"Derived by Vampire's answer literals from a `{tpl.name}` "
                f"template with {len(tpl.all_holes)} holes -- the "
                f"obligations were stated with the certificate open and "
                f"Vampire returned the coefficients -- then checked against "
                f"the module's encoding with cvc5.")
        self.artifacts.put("inv", inv_src, status="proved", language="smt",
                           what=f"The invariant this run found. {what}")
        if rank_src is not None:
            self.artifacts.put(
                "ranking", rank_src, status="proved", language="smt",
                what=f"The ranking function this run found. {what}")

    def _none(self, cd, tried, asker) -> CertificateData:
        what = ("inductive invariant implying the property" if cd.is_safety
                else "invariant and ranking function")
        detail = (
            f"Vampire was asked for one as the coefficients of "
            f"{' and '.join(tried) or 'no'} template(s), over "
            f"{self.width} column(s), in {asker.calls} call(s) taking "
            f"{asker.spent:.1f} s. An answer literal comes back only when a "
            f"refutation pins the coefficients down, so this is not a proof "
            f"that no certificate of this shape exists -- a longer "
            f"--vampire-timeout, or a shape outside the templates, may be "
            f"what is missing. `--infer houdini` searches the same module by "
            f"proposing facts and proving them, which is a different reach."
        )
        if self.artifacts is not None:
            self.artifacts.note(
                f"--infer vampire derived no {what}. {detail}\n",
                status="unknown",
                what=f"A derivation of {what} that did not succeed.",
                why="Vampire answers or times out; nothing was refuted.",
            )
        raise Refused(f"--infer vampire derived no {what}. {detail}",
                      searched=True)
