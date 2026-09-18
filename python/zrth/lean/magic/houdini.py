"""TA2Magic by proposing and proving: Houdini, and a ranking search.

``--infer houdini``.  Nothing here searches for a certificate.  The route
*proposes* -- candidate invariant facts read off the module and off
simulated runs of it, ranking functions from a fixed list of shapes -- and a
solver decides what stays:

* **the invariant** is Houdini's.  Candidate facts are the bound a component
  stays within, stated with a constant the program mentions; a congruence
  the runs keep; a relation between two components; the bounds a component
  keeps on each side of a Bool flag; and under ``--safety`` the property's
  own conjuncts.  Each is put to the solver (it holds at entry; a round
  preserves it given the others) and dropped when not proved, until a pass
  drops nothing.
* **the ranking function** (``--buchi``) is one of a fixed list of shapes --
  an affine form of one or two components, or ``K*x + y`` for a
  lexicographic pair, shifted to stay positive where the property fails and
  optionally zeroed where it holds -- tried smallest first against
  ``rule_buchi``'s own ``hrank``.
* **the certificate is then cut down to what the proofs used.**  Both
  solvers report an unsat core, which names the invariant facts a refutation
  needed, so the invariant handed on is the closure of those under the
  preservation proofs rather than everything Houdini kept.  Every fact is
  one more `step_inv` implication for Lean to close.

Which solver
============
``--houdini-solver`` picks it and :mod:`zrth.lean.houdini_solver` is the
seam.  The engine here never asks which one it is holding; it asks each
query for an unsat core and for a counter-model and uses whichever came
back.  The difference the choice makes is one thing, and it runs through
everything below:

**cvc5 decides these obligations; Vampire only refutes them.**  The queries
are quantifier-free -- the inputs are free constants, so what an obligation
quantifies over is its free variables -- and what is left is linear integer
and real arithmetic with ``to_int``, ``div``, ``mod`` and ``ite``.  cvc5
answers `sat` or `unsat`; Vampire, handed the negation, either refutes it or
runs out of time, and those two failures look the same from outside.

So with cvc5 a failed proof carries a **counterexample**, and the engine
spends it twice.  A Houdini pass that fails learns from one model which
facts to drop -- the model is a state every kept fact holds at whose
successor breaks some of them, which is Houdini's step exactly -- where
Vampire has to ask again once per fact.  And a refuted candidate is gone for
the rest of the run rather than retried at the next rung, because `sat`
holds at every time limit there will ever be.

Two things follow from Vampire's one-sidedness, both measured, and both are
why the shape below is what it is even though cvc5 no longer needs them.

**A candidate that is false costs a whole time limit**, because nothing
comes back early to say so.  That is why the module is *simulated* first:
a fact some reachable state violates is gone before any solver is asked,
which is free, and what is left is mostly true.  The runs quantify the
inputs the way the obligations do -- each round's inputs drawn afresh,
subject to ``--pre`` -- so a state they reach is one every certificate has
to cover, and under ``--safety`` a reached state where the property fails
ends the route there, with the state.  It stays on both solvers: it is
cheaper than a call either way, and it is what makes the *candidates* good
rather than only the answers.

**The time limit climbs: 2 s, 10 s, 60 s, then what is left of
``--houdini-timeout``.**  Vampire's portfolio divides its time limit among
its strategies, so a short limit is a different schedule rather than a
truncated long one, and a proof one limit finds another may not.  Measured
on the 399 obligations of certificates the benchmark matrix verified, the
`smtcomp` portfolio refuted 369 within 2 s, nearly all in 10-20 ms; the rest
were encodings it cannot read (bitvectors, tuples, reals) or obligations
that are false.  So a whole search runs at one limit, and only a search that
found nothing is repeated at the next -- proofs already found are kept, and
only what failed is asked again.  cvc5 climbs a shorter ladder for a
different reason: its search at a longer limit contains the shorter one, so
the rungs are there only to stop one query eating a whole budget.

What the route needs of a module
===============================
Scalar ``Int``, ``Bool`` and ``Real`` state and inputs, whichever solver is
chosen.  Vampire's SMT-LIB front end has integer and real arithmetic and
datatypes but no bitvectors; cvc5 has bitvectors, but the candidate shapes
below do not -- a bound stated with a program constant, a congruence, an
affine relation -- and a matrix-shaped component would be a column per
element.  Both are refused by name before a solver is started.

A Real component changes two things, and only two.

**Its invariant is a set of values, not an interval.**  Over the reals a
bound is almost never inductive on its own -- ``m_lra_lin`` steps
``x' = x - 1`` while ``x > 0``, so ``0 <= x <= 5`` admits ``x = 1/2`` and
steps it to ``-1/2`` -- while the values a run actually takes are finitely
many and closed under the round.  So a Real component whose runs stay
within a few values is offered that disjunction as one fact, which is the
shape `tests/limits` records the hand-written LRA certificates needing.

**Its ranking function is floored, and scaled first.**  `rule_buchi` ranks
by a `Nat`, so a real-valued rank is read through ``to_int``; and flooring
a quantity that falls by less than one need not fall at all --
``m_lra_half`` steps by ``1/2``, where ``to_int x`` repeats.  The scale is
the least common denominator of the literals the program mentions, so the
rank is ``(to_int (* 2.0 s0))`` there and ``(to_int s0)`` where the program
is integral.  Either solver proves the floored obligation directly:
``to_int (x - 1.0) < to_int x`` is one of their unsat answers, not something
this module reasons about.

Literals are printed as decimals throughout, because Vampire's front end
sorts them strictly: it reads neither cvc5's ``(/ 1 2)`` for one half nor a
bare ``3`` where a Real is expected.  cvc5 reads a decimal as happily, so
the certificate carries one spelling whoever proved it.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from ..cert import CertificateData
from ..common import Refused
from ..houdini_solver import (
    DEFAULT_SOLVER,
    DEFAULT_TIMEOUT,
    Query,
    Solver,
    SolverSpec,
    decimals,
    smt_lit,
    smt_real,
)
from . import TA2Magic
from ..smt_synth import (
    SynthContext,
    affine_smt,
    component_slots,
    denominator_scale,
    element,
    moduli,
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

# How much of the module the simulation may evaluate, and for how long. The
# runs only *filter*: a fact they keep is still proved or dropped by the
# solver.
_MAX_ROUNDS = 4000
_SIM_SECONDS = 3.0
_TRIES = 64                     # input draws per round before a run is ended
_OVERFLOW = 10**9               # a run past this is diverging, not informing

# Sampled rounds: how long one batch may take, and how many batches a
# Houdini pass draws before asking the solver.
_SAMPLE_SECONDS = 1.0
_WIDE = 10**6
_SAMPLE_PASSES = 4
# The most a certificate's minimisation may take, of what is left.
_MINIMISE_SECONDS = 15.0

# A ranking search asks about at most this many shapes, smallest first; fits
# them to at most this many rounds; branches on at most this many
# conditions, with at most this many forms on each side of one.
_MAX_RANKS = 48
_MAX_OUTSIDE = 1000
_MAX_SPLITS = 24
_MAX_BRANCH_FORMS = 6
# Up to this many columns, every {-1, 0, 1} combination is a form; past it,
# one or two columns at a time. There is no third rung, and a cap on the
# pairs was measured and dropped: they are quadratic in the columns, but on
# the widest module here -- a 32-element vector, so 2048 forms against 64 --
# dropping them saves 2.3 s of a 5.5 s search and changes no answer on any
# of the seven matrix-shaped modules. A guard that costs reach and buys
# nothing measurable is not worth the rung.
_MAX_DENSE = 5
# Constant-derived shifts offered above a fitted one, per form.
_MAX_SHIFTS = 2

# A Real component whose runs stay within this many values is offered the
# disjunction of them as one fact. It bounds how long a certificate can get:
# the disjunction is one `step_inv` case per value. The other half of that
# bound, the scale a ranking function is floored after, is
# `smt_synth._MAX_SCALE`, since every route that reads a Real rank wants it.
_MAX_VALUES = 8


# ══════════════════════════════════════════════════════════════════════════
# The obligations
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Candidate:
    """One invariant fact or ranking function: its source and its term."""

    src: str                # SMT-LIB over `s0..`, what the certificate carries
    term: object            # the same, parsed, over `ctx.state`


class Obligations:
    """One module's obligations, as questions rather than as files.

    The round is stated once per query as `defines` -- the successor state
    `v_sp*` over the latched state `v_s*` and the inputs -- and every fact
    is read at it, so a query about twenty facts carries the transition once
    rather than twenty times.

    A query that asks about several facts at once names each of them as a
    *probe*, so a solver that returns a counter-model says which of them it
    breaks. That is the whole of what the engine needs to take Houdini's
    step from one answer instead of one answer per fact; a solver that
    returns no model ignores them and the engine asks fact by fact.
    """

    def __init__(self, ctx: SynthContext):
        tm = ctx.tm
        self.ctx = ctx

        def consts(vs, prefix):
            return [tm.mkConst(v.getSort(), f"{prefix}{i}")
                    for i, v in enumerate(vs)]

        self.s = consts(ctx.state, "v_s")
        self.sp = consts(ctx.state, "v_sp")
        self.si = consts(ctx.state, "v_si")
        self.el = consts(ctx.extl_latched, "v_el")
        self.en = consts(ctx.extl_next, "v_en")
        rewriter = cvc5.Solver(tm)

        def readable(t):
            """`t`, with the tuples a matrix-shaped intermediate leaves folded.

            A module with scalar state can still compute through a matrix --
            `m_max` takes the max of a 2-vector -- and cvc5 encodes that as
            `tuple`/`tuple.select`, which Vampire's SMT-LIB front end does not
            have. The rewriter folds a select of a constructor away, so the
            round Vampire is shown is the same function without them. It is
            only called where a tuple is printed: a term that has none is
            handed over exactly as encoded.
            """
            return rewriter.simplify(t) if "tuple" in str(t) else t

        self.next = [readable(t) for t in ctx.msmt.update_state(self.s, self.el, self.en)]
        self.init = [readable(t) for t in ctx.msmt.init_state(self.en)]
        self.update_pre = readable(ctx.with_inputs(ctx.update_pre, self.el, self.en))
        self.init_pre = readable(ctx.with_inputs(ctx.init_pre, self.el, self.en))

    # --- the queries ------------------------------------------------------

    def holds_at_entry(self, facts: list[Candidate]) -> Query:
        """`init_pre e -> fact (init e)`, for all of `facts` at once."""
        at = [self._at(f.term, self.si) for f in facts]
        return self._query([("pre", self.init_pre)], self._and(at),
                           defines=zip(self.si, self.init), probes=at)

    def preserved(self, hyps: list[Candidate], goals: list[Candidate]) -> Query:
        """`hyps s /\\ update_pre e -> goals (update s e)`."""
        named = [(f"h{i}", self._at(h.term, self.s)) for i, h in enumerate(hyps)]
        at = [self._at(g.term, self.sp) for g in goals]
        return self._query(named + [("pre", self.update_pre)], self._and(at),
                           defines=zip(self.sp, self.next), probes=at)

    def implies(self, hyps: list[Candidate], goal) -> Query:
        """`hyps s -> goal s`: the invariant is a proof of the property."""
        named = [(f"h{i}", self._at(h.term, self.s)) for i, h in enumerate(hyps)]
        return self._query(named, self._at(goal, self.s))

    def drops(self, hyps: list[Candidate], rank: Candidate) -> Query:
        """`hrank`: `inv s /\\ ~P s /\\ update_pre e -> V (update s e) < V s`.

        With `V = Int.toNat rank`, exactly as `rule_buchi` states it and
        `smt_query.check_hrank` restates it: a clamp on both sides, so the
        rank may be anything where the property holds.
        """
        tm = self.ctx.tm
        named = [(f"h{i}", self._at(h.term, self.s)) for i, h in enumerate(hyps)]
        named += [
            ("notP", tm.mkTerm(Kind.NOT, self._at(self.ctx.prp, self.s))),
            ("pre", self.update_pre),
        ]
        goal = tm.mkTerm(
            Kind.LT,
            self._clamp(self._at(rank.term, self.sp)),
            self._clamp(self._at(rank.term, self.s)),
        )
        return self._query(named, goal, defines=zip(self.sp, self.next))

    # --- plumbing ---------------------------------------------------------

    def _at(self, term, vs):
        return term.substitute(self.ctx.state, vs)

    def _and(self, terms):
        tm = self.ctx.tm
        if not terms:
            return tm.mkBoolean(True)
        return terms[0] if len(terms) == 1 else tm.mkTerm(Kind.AND, *terms)

    def _clamp(self, t):
        tm = self.ctx.tm
        zero = tm.mkInteger(0)
        return tm.mkTerm(Kind.ITE, tm.mkTerm(Kind.GEQ, t, zero), t, zero)

    @staticmethod
    def _query(hyps, goal, *, defines=(), probes=()) -> Query:
        """Named hypotheses, a goal, and the conjuncts of it worth naming.

        A probe is named `g<i>` against the position of the fact it came
        from, which is how `_survivors` reads a counter-model back: the
        names are positions in the list the caller asked about. A goal that
        is one thing gets no probes -- there would be nothing to learn from
        naming it that `REFUTED` does not already say.
        """
        probes = list(probes)
        return Query(
            hyps=tuple(hyps),
            goal=goal,
            defines=tuple(defines),
            probes=tuple((f"g{i}", t) for i, t in enumerate(probes))
            if len(probes) > 1 else (),
        )


# ══════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════


def _term(tm, sort, value):
    """`value` as a cvc5 literal of `sort`.

    A matrix-shaped value is a Python tuple, element by element, which is
    how the simulation carries one and the only shape `tuple.select` reads
    back.
    """
    if sort.isTuple():
        return tm.mkTuple([_term(tm, s, v)
                           for s, v in zip(sort.getTupleSorts(), value)])
    if sort.isBoolean():
        return tm.mkBoolean(bool(value))
    if sort.isReal():
        q = Fraction(value)
        return tm.mkReal(q.numerator, q.denominator)
    return tm.mkInteger(int(value))


class _Unevaluable(Exception):
    """A term with no value here: a division by zero, or an op not covered."""


# The value of a division by zero. SMT-LIB leaves it unspecified, so nothing
# is claimed about a state that needs one: it propagates, and a result that
# depends on it is `_Unevaluable`.
_UNDEF = object()


def _strict(fn):
    def apply(vals):
        return _UNDEF if any(v is _UNDEF for v in vals) else fn(vals)
    return apply


def _ite(vals):
    c = vals[0]
    return _UNDEF if c is _UNDEF else vals[1] if c else vals[2]


def _and(vals):
    if any(v is False for v in vals):
        return False
    return _UNDEF if any(v is _UNDEF for v in vals) else True


def _or(vals):
    if any(v is True for v in vals):
        return True
    return _UNDEF if any(v is _UNDEF for v in vals) else False


def _product(vals):
    out = 1
    for v in vals:
        out *= v
    return out


def _chain(op):
    return _strict(lambda vs: all(op(a, b) for a, b in zip(vs, vs[1:])))


def _division(vals):
    a, b = vals
    if a is _UNDEF or b is _UNDEF or b == 0:
        return _UNDEF
    return (a - a % abs(b)) // b


def _modulus(vals):
    a, b = vals
    if a is _UNDEF or b is _UNDEF or b == 0:
        return _UNDEF
    # SMT-LIB's is Euclidean: the remainder is never negative.
    return a % abs(b)


def _quotient(vals):
    """Real `/`. Exact, on `Fraction`: a float would round a guard's corner."""
    a, b = vals
    if a is _UNDEF or b is _UNDEF or b == 0:
        return _UNDEF
    return Fraction(a) / Fraction(b)


def _tuple(vals):
    """A matrix-shaped value: its elements, in the tuple's own order."""
    return _UNDEF if any(v is _UNDEF for v in vals) else tuple(vals)


def _selector(j: int):
    """Element `j` of one, which is what `((_ tuple.select j) x)` means."""
    def pick(vals):
        return _UNDEF if vals[0] is _UNDEF else vals[0][j]
    return pick


def _selected(t) -> int:
    """Which element `t`, an `APPLY_SELECTOR`, reads.

    By identity against the datatype's own selectors rather than by reading
    the index out of the printed name: the name is cvc5's to spell.
    """
    dt = t[1].getSort().getDatatype()[0]
    for j in range(dt.getNumSelectors()):
        if t[0] == dt[j].getTerm():
            return j
    raise _Unevaluable(f"no element for {t}")      # pragma: no cover


_OPS = {} if Kind is None else {
    Kind.ITE: _ite,
    Kind.AND: _and,
    Kind.OR: _or,
    Kind.NOT: _strict(lambda vs: not vs[0]),
    Kind.IMPLIES: lambda vs: _or([_strict(lambda v: not v[0])(vs[:1]), vs[1]]),
    Kind.XOR: _strict(lambda vs: vs[0] != vs[1]),
    Kind.ADD: _strict(sum),
    Kind.SUB: _strict(lambda vs: vs[0] - sum(vs[1:])),
    Kind.NEG: _strict(lambda vs: -vs[0]),
    Kind.MULT: _strict(_product),
    Kind.ABS: _strict(lambda vs: abs(vs[0])),
    Kind.EQUAL: _strict(lambda vs: all(v == vs[0] for v in vs[1:])),
    Kind.DISTINCT: _strict(lambda vs: len(set(vs)) == len(vs)),
    Kind.LT: _chain(lambda a, b: a < b),
    Kind.LEQ: _chain(lambda a, b: a <= b),
    Kind.GT: _chain(lambda a, b: a > b),
    Kind.GEQ: _chain(lambda a, b: a >= b),
    Kind.INTS_DIVISION: _division,
    Kind.INTS_MODULUS: _modulus,
    Kind.DIVISION: _quotient,
    # The Real/Int boundary. `to_real` is the identity on a value this
    # evaluator already keeps exactly; `to_int` is the floor `Int.toNat`
    # sees, and the reason a ranking function over a Real component has to
    # be scaled before it is read.
    Kind.TO_REAL: _strict(lambda vs: Fraction(vs[0])),
    Kind.TO_INTEGER: _strict(lambda vs: math.floor(vs[0])),
    Kind.IS_INTEGER: _strict(lambda vs: Fraction(vs[0]).denominator == 1),
}


class Evaluator:
    """cvc5 terms at concrete values, in Python.

    A term is compiled once into a flat list of operations, one per distinct
    subterm, so a transition that shares its subterms -- the `let`s a deep
    network prints with -- is evaluated in time linear in its DAG rather than
    its tree. The kinds a module's integer transition is made of are Python
    arithmetic; anything else is handed to cvc5's rewriter with the values
    substituted in, which evaluates a ground term without deciding anything.
    """

    def __init__(self, tm):
        self.tm = tm
        self._solver = None
        self._programs: dict = {}

    def __call__(self, term, env: dict):
        return self.many([term], env)[0]

    def many(self, terms, env: dict) -> list:
        """Several terms at one assignment, sharing what they share."""
        key = tuple(t.getId() for t in terms)
        program = self._programs.get(key)
        if program is None:
            program = self._programs[key] = self._compile(terms)
        vals: list = []
        for kind, arg in program[0]:
            if kind == 0:
                vals.append(arg)
            elif kind == 1:
                try:
                    vals.append(env[arg])
                except KeyError as e:
                    raise _Unevaluable(f"no value for {arg}") from e
            elif kind == 2:
                vals.append(self._rewrite(arg, env))
            else:
                vals.append(kind([vals[j] for j in arg]))
        out = [vals[i] for i in program[1]]
        if any(v is _UNDEF for v in out):
            raise _Unevaluable("division by zero")
        return out

    def _compile(self, terms):
        slot: dict = {}
        ops: list = []
        stack = [(t, False) for t in reversed(list(terms))]
        while stack:
            t, expanded = stack.pop()
            tid = t.getId()
            if tid in slot:
                continue
            k = t.getKind()
            if k == Kind.CONST_INTEGER:
                op = (0, t.getIntegerValue())
            elif k == Kind.CONST_RATIONAL:
                op = (0, t.getRealValue())
            elif k == Kind.CONST_BOOLEAN:
                op = (0, t.getBooleanValue())
            elif k == Kind.CONSTANT:
                op = (1, t.getSymbol())
            elif k in (Kind.APPLY_SELECTOR, Kind.APPLY_CONSTRUCTOR):
                # The tuples a matrix-shaped component and the wires around
                # it are encoded as. Child 0 is the selector or constructor
                # symbol, which stands for the operation rather than for a
                # value, so it is the one child not compiled.
                kids = list(t)[1:]
                if not expanded:
                    stack.append((t, True))
                    stack.extend((c, False) for c in reversed(kids))
                    continue
                fn = (_selector(_selected(t)) if k == Kind.APPLY_SELECTOR
                      else _tuple)
                op = (fn, tuple(slot[c.getId()] for c in kids))
            elif k not in _OPS:
                op = (2, t)
            elif not expanded:
                stack.append((t, True))
                stack.extend((c, False) for c in reversed(list(t)))
                continue
            else:
                op = (_OPS[k], tuple(slot[c.getId()] for c in t))
            slot[tid] = len(ops)
            ops.append(op)
        return ops, [slot[t.getId()] for t in terms]

    def _rewrite(self, t, env):
        from ..smt_query import _constants

        tm = self.tm
        if self._solver is None:
            self._solver = cvc5.Solver(tm)
        consts = list(_constants(t, {}).values())
        vals = []
        for c in consts:
            v = env.get(c.getSymbol())
            if v is None:
                raise _Unevaluable(f"no value for {c.getSymbol()}")
            vals.append(_term(tm, c.getSort(), v))
        r = self._solver.simplify(t.substitute(consts, vals) if consts else t)
        if r.getKind() == Kind.CONST_INTEGER:
            return r.getIntegerValue()
        if r.getKind() == Kind.CONST_RATIONAL:
            return r.getRealValue()
        if r.getKind() == Kind.CONST_BOOLEAN:
            return r.getBooleanValue()
        raise _Unevaluable(f"{t.getKind()} does not evaluate to a literal")


# ══════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class Runs:
    """What the simulation reached: states, and the rounds between them.

    A state is a tuple of component values, in `ctx.state` order.
    """

    states: list
    rounds: list
    # Under `--safety`, a reached state where the property fails, and how
    # many rounds it took.
    violation: "tuple | None" = None
    depth: int = 0


class Draws:
    """Input values, and states near the ones a run reached.

    Integers come from the program's own constants half the time -- the
    value a guard compares against is the one worth hitting -- and from a
    range a little wider than those constants otherwise.

    A real is drawn on the lattice the program lives on -- multiples of
    `1/scale`, plus the rationals it mentions -- most of the time, and off
    it the rest: a state with a denominator the program never writes is what
    shows an interval is not inductive over the reals, and a state on the
    lattice is what a fact about the values a run takes has to be tested at.
    """

    def __init__(self, ctx: SynthContext, constants, seed: int,
                 wide: bool = False, rationals=(), scale: int = 1):
        self.rng = random.Random(seed)
        self.pool = sorted(set(constants) | {-1, 0, 1})
        self.reach = min(1000, 2 * max(abs(c) for c in self.pool) + 10)
        # `wide` draws integers far past anything the program mentions. A
        # shape fitted to states within the program's own range can hold
        # only *because* they were small -- `11 - x` is positive on every
        # state with `x <= 10` -- and states from a wider range are what tell
        # that apart from a shape the invariant actually bounds.
        if wide:
            self.reach = _WIDE
        self.fresh = 0.5 if wide else 0.25
        self.sorts = [v.getSort() for v in ctx.state]
        # Where a state can be moved, and in what sort: one place per
        # *element*, so a matrix-shaped component is as many as it has.
        self.slots = [(i, slot, srt)
                      for i in range(len(self.sorts))
                      for slot, srt in component_slots(ctx, i)]
        self.rpool = sorted(set(rationals) | {Fraction(c) for c in self.pool})
        self.step = Fraction(1, scale)

    def value(self, sort):
        rng = self.rng
        if sort.isTuple():
            return tuple(self.value(s) for s in sort.getTupleSorts())
        if sort.isBoolean():
            return rng.random() < 0.5
        if sort.isReal():
            roll = rng.random()
            if roll < 0.4:
                return rng.choice(self.rpool)
            if roll < 0.8:
                return self.step * rng.randint(-self.reach, self.reach)
            return Fraction(rng.randint(-self.reach, self.reach),
                            rng.choice((3, 5, 7, 8)))
        if rng.random() < 0.5:
            return rng.choice(self.pool)
        return rng.randint(-self.reach, self.reach)

    def near(self, seeds: list) -> tuple:
        """A state: a reached one with some elements moved, or a fresh one.

        Elements rather than components: a 3-vector moved as a unit is three
        coordinates that always change together, and the state that tells a
        fact about `v[0]` apart from one about the whole vector is the state
        where only `v[0]` moved.
        """
        rng = self.rng
        if not seeds or rng.random() < self.fresh:
            return tuple(self.value(srt) for srt in self.sorts)
        s = list(rng.choice(seeds))
        for i, slot, srt in rng.sample(self.slots,
                                       k=rng.randint(1, len(self.slots))):
            old = s[i] if slot is None else s[i][slot]
            if srt.isBoolean():
                new = not old
            elif rng.random() < 0.6:
                new = old + (self.step if srt.isReal() else 1) * rng.choice(
                    (-2, -1, 1, 2))
            else:
                new = self.value(srt)
            if slot is None:
                s[i] = new
            else:
                s[i] = s[i][:slot] + (new,) + s[i][slot + 1:]
        return tuple(s)


def simulate(ctx: SynthContext, ob: Obligations, ev: Evaluator,
             constants: tuple[int, ...], *, safety: bool, seed: int = 0,
             rationals=(), scale: int = 1) -> Runs:
    """Run the module from its entry states, drawing inputs as `--pre` allows.

    A run is a run of the module: each round's latched inputs are the values
    the round before awaited, and only the awaited ones are drawn. So every
    state reached is reachable -- a fact it violates is not an invariant, and
    a `--safety` property it violates does not hold.
    """
    draws = Draws(ctx, constants, seed, rationals=rationals, scale=scale)
    names = ctx.names
    latched = [str(c) for c in ob.el]
    awaited = [(str(c), c.getSort()) for c in ob.en]

    def inputs(pre, base, before):
        """Awaited inputs `pre` allows. At entry nothing is latched yet, so
        the latched ones are drawn too -- `init_inv` quantifies them."""
        for _ in range(_TRIES):
            env = dict(base)
            if before is None:
                env.update({n: draws.value(srt)
                            for n, (_, srt) in zip(latched, awaited)})
            else:
                env.update(zip(latched, before))
            env.update({n: draws.value(srt) for n, srt in awaited})
            try:
                if ev(pre, env):
                    return env
            except _Unevaluable:
                return None
        return None

    def prop_fails(state) -> bool:
        try:
            return not ev(ctx.prp, dict(zip(names, state)))
        except _Unevaluable:
            return False

    states: dict = {}
    rounds: dict = {}
    deterministic = not awaited
    trajectories = 1 if deterministic else 32
    length = _MAX_ROUNDS if deterministic else 128
    t_end = time.monotonic() + _SIM_SECONDS
    for _ in range(trajectories):
        if len(rounds) >= _MAX_ROUNDS or time.monotonic() > t_end:
            break
        env = inputs(ob.init_pre, {}, None)
        if env is None:
            continue
        try:
            s = tuple(ev.many(ob.init, env))
        except _Unevaluable:
            continue
        before = [env[n] for n, _ in awaited]
        seen_here = set()
        for depth in range(length + 1):
            states.setdefault(s, None)
            if safety and prop_fails(s):
                return Runs(list(states), list(rounds), violation=s, depth=depth)
            if deterministic and s in seen_here:
                break
            seen_here.add(s)
            if (len(rounds) >= _MAX_ROUNDS or time.monotonic() > t_end
                    or any(not isinstance(v, bool) and abs(v) > _OVERFLOW
                           for v in _scalars(s))):
                break
            base = {str(c): v for c, v in zip(ob.s, s)}
            env = inputs(ob.update_pre, base, before)
            if env is None:
                break
            try:
                sp = tuple(ev.many(ob.next, env))
            except _Unevaluable:
                break
            before = [env[n] for n, _ in awaited]
            rounds.setdefault((s, sp), None)
            s = sp
    return Runs(list(states), list(rounds))


def sample_rounds(ctx: SynthContext, ob: Obligations, ev: Evaluator,
                  draws: Draws, seeds: list, keep, *, want: int = 300,
                  seconds: float = _SAMPLE_SECONDS) -> list:
    """Rounds from states `keep` accepts, not necessarily reachable ones.

    What an obligation quantifies over is every state the invariant admits,
    with any inputs `--pre` allows -- latched ones included -- so a round
    from such a state that breaks a fact, or that a ranking function does not
    drop on, is a counterexample to that obligation as a solver would be
    asked it. Finding one here is free, where a call costs milliseconds at
    best and, on a solver that cannot refute, a whole time limit to learn
    nothing.
    """
    ins = [(str(c), c.getSort()) for c in ob.el + ob.en]
    names = [str(c) for c in ob.s]
    out: list = []
    t_end = time.monotonic() + seconds
    for _ in range(40 * want):
        if len(out) >= want or time.monotonic() > t_end:
            break
        s = draws.near(seeds)
        try:
            if not keep(s):
                continue
            base = dict(zip(names, s))
            for _ in range(8):
                env = dict(base)
                env.update({n: draws.value(srt) for n, srt in ins})
                if ev(ob.update_pre, env):
                    out.append((s, tuple(ev.many(ob.next, env))))
                    break
        except _Unevaluable:
            continue
    return out


# ══════════════════════════════════════════════════════════════════════════
# Candidates
# ══════════════════════════════════════════════════════════════════════════


def _scalars(state):
    """Every scalar in a simulated state, matrix-shaped ones flattened."""
    for v in state:
        if isinstance(v, tuple):
            yield from v
        else:
            yield v


@dataclass(frozen=True)
class Column:
    """One scalar a candidate or a ranking function may be stated over.

    A scalar component is its own column. A matrix-shaped one is a column
    per element -- `((_ tuple.select k) s0)`, the spelling `smt_to_lean`
    renders and a property already uses -- which is what lets a fact be
    said about `v[0]` when the component it lives in is a 3-vector. Without
    it the only sayable facts are about the whole tuple, and there are
    none: a tuple has no order to bound and no arithmetic to relate.

    `sort` is the *element's*, never the component's, because everything
    downstream branches on it -- an integer column is bounded by program
    constants, a Real one by rationals and pinned to the values a run
    takes, a Bool one splits a fact in two.
    """

    src: str                        # SMT-LIB over `s0..`
    sort: object                    # the element's cvc5 sort
    index: int                      # which component it reads
    slot: "int | None"              # which element of it, or the whole thing

    def at(self, state):
        """This column's value in a simulated state."""
        v = state[self.index]
        return v if self.slot is None else v[self.slot]


def columns(ctx: SynthContext) -> list[Column]:
    """Every scalar this module's state is made of, in certificate order."""
    return [Column(element(ctx, i, slot)[0], srt, i, slot)
            for i in range(len(ctx.state))
            for slot, srt in component_slots(ctx, i)]


def _reading(col: Column) -> str:
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
    and the one a floored ranking function needs: see `_pins`.

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


def split_conditions(ctx: SynthContext, ob: Obligations) -> list[str]:
    """What a piecewise ranking function may branch on, as SMT-LIB over `s0..`.

    First the comparisons the property and the transition branch on -- where
    a run changes what it does, and so where the quantity that falls changes
    too -- then each Bool column, the sign of each integer one, and the
    order of each pair: `max(y, z) - x` falls on `while (x < y) {x++; y = z}`
    and is `(ite (<= y z) (- z x) (- y x))`, a branch the program never
    spells.
    """
    from ..smt_query import _constants

    out: list[str] = []
    state = set(ctx.names)
    roots = [ctx.prp] + [t.substitute(ob.s, ctx.state) for t in ob.next]
    seen: set = set()
    stack = list(roots)
    while stack:
        t = stack.pop()
        if t.getId() in seen:
            continue
        seen.add(t.getId())
        k = t.getKind()
        kids = list(t)
        if k in (Kind.LT, Kind.LEQ, Kind.GT, Kind.GEQ) or (
            k == Kind.EQUAL and kids
            and (kids[0].getSort().isInteger() or kids[0].getSort().isReal())
        ):
            if set(_constants(t, {})) <= state:
                out.append(decimals(str(t)))
        stack.extend(kids)
    cols = columns(ctx)
    ints = [c for c in cols if c.sort.isInteger()]
    reals = [c for c in cols if c.sort.isReal()]
    for col in cols:
        if col.sort.isBoolean():
            out.append(col.src)
        else:
            out.append(f"(<= {smt_lit(0, col.sort)} {col.src})")
    for group in (ints, reals):
        out += [f"(<= {x.src} {y.src})"
                for a, x in enumerate(group) for y in group[a + 1:]]
    return list(dict.fromkeys(out))[:_MAX_SPLITS]


class Ranks:
    """Ranking functions fitted to rounds, in a few fixed shapes.

    Each shape is an affine form of the state's integer readings -- one
    column, a sum or difference of two, or `K*x + y` for a lexicographic
    pair -- and the *shift* that makes it at least one on every round that
    has to drop is read off those rounds rather than enumerated. A piecewise
    rank `(ite c f g)` fits a form to each side of `c` the same way, and the
    offset between the sides from the rounds that cross from one to the
    other. Every shape is also offered zeroed where the property holds,
    since `hrank` never constrains those states.
    """

    def __init__(self, ctx: SynthContext, ev: Evaluator, prp_src: str,
                 splits: list[str], constants: tuple[int, ...] = (),
                 rationals: tuple = (), scale: int = 1):
        self.ctx, self.ev, self.prp_src = ctx, ev, prp_src
        # What a real-valued form is multiplied by before it is floored, and
        # whether it has to be: a state of `Int` and `Bool` is read as it is.
        self.scale = max(1, int(scale))
        self.cols = columns(ctx)
        self.real = any(c.sort.isReal() for c in self.cols)
        # Shifts a guard's constant suggests: `c` and one past it. A shift
        # fitted to sampled rounds is only as low as the lowest one sampled,
        # and the round that needs the largest shift is usually the corner
        # the guard names -- `y <= 100 && z <= x` needs `101 + x - y - z`,
        # where samples that never hit `y = 100, z = x` exactly fit `95`.
        # A real literal suggests one at the scale the rank is floored at.
        shifts = set(constants) | {math.floor(self.scale * Fraction(q))
                                   for q in rationals}
        self.shift_pool = sorted({c + d for c in shifts for d in (0, 1)})
        # The env the evaluator wants is keyed by *component*, which is what
        # a state is a tuple of; the forms below are over columns, which is
        # what a matrix-shaped component has several of.
        self.names = ctx.names
        n = len(self.cols)
        self.readings = [self._reading(c) for c in self.cols]
        if n <= _MAX_DENSE:
            # Every form with coefficients in {-1, 0, 1}: `100 - y + x - z`
            # ranks `ColonSipma-TACAS2001-Fig1`, and no pair of its three
            # columns does.
            forms = [v for v in product((1, -1, 0), repeat=n) if any(v)]
        else:
            forms = []
            for i in range(n):
                for sign in (1, -1):
                    forms.append(tuple(sign if k == i else 0 for k in range(n)))
            for i in range(n):
                for j in range(i + 1, n):
                    for si, sj in ((1, -1), (-1, 1), (1, 1), (-1, -1)):
                        forms.append(tuple(si if k == i else sj if k == j else 0
                                           for k in range(n)))
        forms.sort(key=lambda v: sum(map(abs, v)))
        self.forms = forms
        self.splits = []
        for src in splits:
            try:
                term = ctx.env.parse_expr(src)
            except Exception:                        # noqa: BLE001
                continue
            if not term.isNull():
                self.splits.append((src, term))
        self._truth: dict = {}

    def _reading(self, col: Column) -> str:
        """`col` in the arithmetic this rank's forms are summed in.

        Integer, unless some column is Real -- then every reading is
        lifted to Real, because a sum has one sort and `to_real` is the only
        way an `Int` column joins it.
        """
        if not self.real:
            return _reading(col)
        if col.sort.isBoolean():
            return f"(ite {col.src} 1.0 0.0)"
        return col.src if col.sort.isReal() else f"(to_real {col.src})"

    def _src(self, shift: int, form) -> str:
        """The shape as the certificate carries it: an `Int`-sorted term.

        Over a real form that is `(to_int (scale*form + shift))`. Flooring
        after the shift is the same number as flooring before it -- the
        shift is a whole number -- which is what lets everything downstream
        fit shifts and offsets in integers.
        """
        if not self.real:
            return affine_smt(shift, form, self.readings)
        coeffs = [self.scale * c for c in form]
        if not any(coeffs):
            return smt_int(shift)
        parts = [] if not shift else [smt_real(shift)]
        for c, name in zip(coeffs, self.readings):
            if c == 1:
                parts.append(name)
            elif c == -1:
                parts.append(f"(- {name})")
            elif c:
                parts.append(f"(* {smt_real(c)} {name})")
        body = parts[0] if len(parts) == 1 else "(+ " + " ".join(parts) + ")"
        return f"(to_int {body})"

    def holds(self, state, term=None) -> bool:
        key = (state, None if term is None else term.getId())
        if key not in self._truth:
            try:
                self._truth[key] = bool(self.ev(
                    self.ctx.prp if term is None else term,
                    dict(zip(self.names, state))))
            except _Unevaluable:
                self._truth[key] = term is None
        return self._truth[key]

    def fit(self, rounds: list) -> list[str]:
        """Every shape that drops on all of `rounds`, smallest first."""
        outside = [(s, sp) for s, sp in rounds if not self.holds(s)]
        if len(outside) > _MAX_OUTSIDE:
            outside = random.Random(0).sample(outside, _MAX_OUTSIDE)
        if not outside:
            # Nothing to rank on. A constant is then the first thing worth
            # asking: it proves the property holds wherever the invariant does.
            return ["0"]
        forms = list(self.forms) + self._lex(outside)
        out: list[str] = []
        for form in dict.fromkeys(forms):
            shift = 1 - min(self._at(form, s) for s, _ in outside)
            above = [c for c in self.shift_pool if c > shift][:_MAX_SHIFTS]
            for sh in dict.fromkeys((0, shift, *above)):
                src = self._src(sh, form)
                out += self._accept(src, lambda st, f=form, sh=sh: self._at(f, st) + sh,
                                    outside)
        for c_src, c_term in self.splits:
            out += self._piecewise(c_src, c_term, outside)
        out = list(dict.fromkeys(out))
        out.sort(key=len)
        return out

    # --- shapes -----------------------------------------------------------

    def _lex(self, outside) -> list:
        """`K*x + y`, with `K` wider than the range `y` takes on these rounds."""
        n = len(self.cols)
        out = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                unit = tuple(1 if k == j else 0 for k in range(n))
                ys = [self._at(unit, s) for s, _ in outside]
                width = max(ys) - min(ys) + 1
                if width > 1000:
                    continue
                K = max(2, width + 1)
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    out.append(tuple(si * K if k == i else sj if k == j else 0
                                     for k in range(n)))
        return out

    def _piecewise(self, c_src, c_term, outside) -> list[str]:
        side = {b: [(s, sp) for s, sp in outside
                    if self.holds(s, c_term) == b] for b in (True, False)}
        if not side[True] or not side[False]:
            return []
        zero = tuple(0 for _ in self.cols)
        fits = {}
        for b in (True, False):
            stays = [(s, sp) for s, sp in side[b] if self.holds(sp, c_term) == b]
            fits[b] = [f for f in [zero] + self.forms
                       if all(self._at(f, sp) < self._at(f, s) for s, sp in stays)
                       ][:_MAX_BRANCH_FORMS]
        out = []
        for f in fits[True]:
            for g in fits[False]:
                sf = 1 - min(self._at(f, s) for s, _ in side[True])
                sg = 1 - min(self._at(g, s) for s, _ in side[False])
                offsets = self._offsets(c_term, f, g, sf, sg, side)
                if offsets is None:
                    continue
                sf, sg = sf + offsets[0], sg + offsets[1]
                src = (f"(ite {c_src} {self._src(sf, f)} "
                       f"{self._src(sg, g)})")

                def rank(st, f=f, g=g, sf=sf, sg=sg):
                    if self.holds(st, c_term):
                        return self._at(f, st) + sf
                    return self._at(g, st) + sg

                out += self._accept(src, rank, outside)
        return out

    def _offsets(self, c_term, f, g, sf, sg, side):
        """How far to lift each side so that crossing to the other drops."""
        of = og = 0
        for _ in range(6):
            moved = False
            for s, sp in side[True]:
                if not self.holds(sp, c_term):
                    need = max(self._at(g, sp) + sg + og, 0) + 1 - (self._at(f, s) + sf)
                    if need > of:
                        of, moved = need, True
            for s, sp in side[False]:
                if self.holds(sp, c_term):
                    need = max(self._at(f, sp) + sf + of, 0) + 1 - (self._at(g, s) + sg)
                    if need > og:
                        og, moved = need, True
            if not moved:
                return of, og
        return None                                  # each side lifts the other

    def _accept(self, src, rank, outside) -> list[str]:
        """`src`, and its zeroed form, for whichever drops on every round."""
        out = []
        before = [(rank(s), sp) for s, sp in outside]
        if all(_drops(r, rank(sp)) for r, sp in before):
            out.append(src)
        if all(_drops(r, 0 if self.holds(sp) else rank(sp)) for r, sp in before):
            out.append(f"(ite {self.prp_src} 0 {src})")
        return out

    def _at(self, form, state) -> int:
        """What `_src(0, form)` evaluates to at `state` -- floored, if real."""
        total = 0
        for c, col in zip(form, self.cols):
            if c:
                v = col.at(state)
                total += c * (int(v) if isinstance(v, bool) else v)
        return math.floor(self.scale * total) if self.real else int(total)


def _drops(before: int, after: int) -> bool:
    """`Int.toNat after < Int.toNat before`."""
    return max(after, 0) < max(before, 0)


# ══════════════════════════════════════════════════════════════════════════
# The route
# ══════════════════════════════════════════════════════════════════════════


class TA2MagicHoudini(TA2Magic):
    """Houdini and a ranking search, put to whichever solver was chosen.

    `solver` is a `SolverSpec` -- what `--houdini-solver` and the flags
    belonging to the one it names resolved to at parse time -- or the bare
    name of one, which is what a caller with no binary to point at wants.
    The solver itself is built in `infer`, because it needs the cvc5
    `TermManager` the module is encoded into and because its deadline
    starts when the search does, not when the object is made.
    """

    def __init__(self, module, *, solver=DEFAULT_SOLVER,
                 timeout: float = DEFAULT_TIMEOUT, artifacts=None, log=print):
        super().__init__("")
        if cvc5 is None:                             # pragma: no cover
            raise Refused(
                "--infer houdini needs the `cvc5` package to encode the "
                "module, which is not importable. Install with: "
                "uv pip install cvc5"
            )
        self.spec = solver if isinstance(solver, SolverSpec) else SolverSpec(solver)
        self.module = module
        self.timeout = float(timeout)
        self.artifacts = artifacts
        self.log = log

    # --- driver ---------------------------------------------------------

    def infer(self, cd: CertificateData) -> CertificateData:
        ctx = SynthContext.build(self.module, cd, route="houdini",
                                 takes=("int", "bool", "bv", "real", "tuple"))
        self._check_sorts(ctx)
        self.ctx = ctx
        self.cols = columns(ctx)
        self.ob = ob = Obligations(ctx)
        self.ev = ev = Evaluator(ctx.tm)
        constants = program_constants(ctx)
        rationals = program_rationals(ctx)
        scale = denominator_scale(rationals)
        self.log(f"[houdini] solver: {self.spec.kind}")
        self.log("[houdini] columns: "
                 + ", ".join(_reading(c) for c in self.cols))
        if any(srt.isReal() for srt in ctx.env.state_sorts):
            self.log(f"[houdini] Real state: a ranking function is floored "
                     f"after scaling by {scale}")

        runs = simulate(ctx, ob, ev, constants, safety=cd.is_safety,
                        rationals=rationals, scale=scale)
        self.log(f"[houdini] simulated {len(runs.rounds)} rounds, "
                 f"{len(runs.states)} distinct states")
        if runs.violation is not None:
            state = ", ".join(f"s{i} = {v}" for i, v in enumerate(runs.violation))
            raise Refused(
                f"--infer houdini: the property does not hold -- a run of "
                f"{runs.depth} round(s), with inputs drawn as `--pre` allows, "
                f"reaches {state}. No invariant implies a property the module "
                f"violates, so there is nothing to put to a solver."
            )
        self.runs = runs
        self.draws = Draws(ctx, constants, seed=1,
                           rationals=rationals, scale=scale)
        self.wide = Draws(ctx, constants, seed=2, wide=True,
                          rationals=rationals, scale=scale)

        self.pins = pinned_values(ctx, runs)
        conjuncts = self._conjuncts(ctx, cd) if cd.is_safety else []
        facts = self._parse(invariant_candidates(ctx, runs, constants,
                                                 conjuncts, rationals))
        self.log(f"[houdini] {len(facts)} candidate facts hold on the runs")
        ranks = None if cd.is_safety else Ranks(ctx, ev, cd.prp,
                                                split_conditions(ctx, ob),
                                                constants, rationals, scale)

        prover = self.spec.build(ctx.tm, seconds=self.timeout, log=self.log)
        found, reached, tried = None, 0.0, 0
        for limit in prover.ladder(self.timeout):
            if prover.left() < 0.5:
                break
            reached = limit
            self.log(f"[houdini] time limit {limit:g} s "
                     f"({prover.left():.0f} s of the budget left)")
            inv = self._houdini(prover, facts, limit)
            self.log(f"[houdini] Houdini kept {len(inv)} of {len(facts)}: "
                     + " ".join(f.src for f in inv))
            if cd.is_safety:
                need = self._property_rests_on(prover, inv, conjuncts, limit)
                if need is not None:
                    inv = self._shrink(prover, inv, need, limit)
                    found = (self._minimise(prover, inv, conjuncts, None), None)
                    break
                continue
            candidates = self._ranks(ranks, inv)
            tried = max(tried, len(candidates))
            if not candidates:
                continue
            at = prover.first([ob.drops(inv, r) for r in candidates], limit)
            if at is not None:
                rank = candidates[at]
                self.log(f"[houdini] ranking function proved: {rank.src}")
                core = prover.prove(ob.drops(inv, rank), limit, core=True).core
                pins = self._pins(inv, rank) if ranks.real else []
                need = self._named(inv, core) + pins
                inv = self._shrink(prover, inv, need, limit)
                found = (self._minimise(prover, inv, [f.src for f in pins],
                                        (ranks, rank)), rank)
                break
        refuted = f", {prover.refuted} refuted" if prover.refutes else ""
        self.log(f"[houdini] {prover.calls} {prover.name} calls{refuted}, "
                 f"{prover.spent:.1f} s")
        if found is None:
            return self._none(cd, len(facts), tried, reached, prover)
        inv, rank = found
        return self._emit(cd, inv, rank)

    # --- Houdini ----------------------------------------------------------

    def _holds(self, fact: Candidate, state) -> bool:
        try:
            return bool(self.ev(fact.term, dict(zip(self.ctx.names, state))))
        except _Unevaluable:
            return True

    def _houdini(self, prover: Solver, facts: list[Candidate],
                 limit: float) -> list[Candidate]:
        """The largest subset of `facts` the solver proves inductive at `limit`.

        Each pass first drops what a sampled round refutes -- a state every
        kept fact holds at, whose successor breaks one -- which is Houdini's
        own step taken without a solver, and then asks about the rest.

        One question is asked per pass, not one per fact, and what happens
        when it comes back false is where the solvers part. A counter-model
        *is* the pass's next step -- a state the kept facts admit whose
        successor breaks some of them, named by `_survivors` and dropped
        together -- so cvc5 converges in one call per pass. Vampire has no
        model to read, so the pass falls back to asking about each fact on
        its own, which is where the old cost per pass came from.
        """
        ob = self.ob
        if not facts:
            return []
        kept = self._at_entry(prover, list(facts), limit)
        while kept and prover.left() >= 0.5:
            kept = self._refuted_by_sampling(kept)
            if not kept:
                break
            step = prover.prove(ob.preserved(kept, kept), limit, model=True)
            if step.proved:
                return kept
            survivors = self._survivors(
                prover, step, kept,
                lambda fs: [ob.preserved(fs, [f]) for f in fs], limit)
            if len(survivors) == len(kept):
                # Nothing to drop and the conjunction was not proved: the
                # solver ran out of time on it, not out of facts.
                return kept
            kept = survivors
        # Out of budget mid-pass: what is left was not shown inductive.
        return [] if kept and prover.left() < 0.5 else kept

    def _at_entry(self, prover: Solver, kept: list[Candidate],
                  limit: float) -> list[Candidate]:
        """`kept` less the facts the initial state does not satisfy.

        Asked again until it is proved, because one counter-model names the
        facts *that* initial state breaks and another may break others --
        narrowing from models is a fixpoint where asking fact by fact
        answers in a single round. Both have to reach the same set: a fact
        that fails at entry and is kept anyway is an `init_inv` Lean cannot
        close, and nothing downstream asks this question again.
        """
        ob = self.ob
        while kept:
            entry = prover.prove(ob.holds_at_entry(kept), limit, model=True)
            if entry.proved:
                break
            survivors = self._survivors(
                prover, entry, kept,
                lambda fs: [ob.holds_at_entry([f]) for f in fs], limit)
            if len(survivors) == len(kept):
                break               # nothing to drop: unanswered, not false
            kept = survivors
        return kept

    def _survivors(self, prover: Solver, answer, kept: list[Candidate],
                   per_fact, limit: float) -> list[Candidate]:
        """`kept` less what `answer` says is broken.

        From the counter-model where there is one -- the probes are the
        facts in the order they were asked about, so a name is a position --
        and otherwise by asking about each fact on its own, which is what a
        solver that cannot produce a model leaves as the only way to find
        out. An empty `broken` on a refuted answer means the model could not
        be read back rather than that nothing is wrong, so that falls back
        too.
        """
        if answer.broken:
            out = [f for i, f in enumerate(kept) if f"g{i}" not in answer.broken]
            self.log(f"[houdini] a counterexample drops "
                     f"{len(kept) - len(out)} of {len(kept)} facts")
            return out
        answers = prover.prove_all(per_fact(kept), limit)
        return [f for f, a in zip(kept, answers) if a.proved]

    def _refuted_by_sampling(self, kept: list[Candidate]) -> list[Candidate]:
        for n in range(_SAMPLE_PASSES):
            rounds = sample_rounds(
                self.ctx, self.ob, self.ev,
                self.wide if n % 2 else self.draws, self.runs.states,
                lambda s: all(self._holds(f, s) for f in kept))
            broken = {i for i, f in enumerate(kept)
                      if any(not self._holds(f, sp) for _, sp in rounds)}
            if not broken:
                break
            kept = [f for i, f in enumerate(kept) if i not in broken]
        return kept

    def _ranks(self, ranks: Ranks, inv: list[Candidate]) -> list[Candidate]:
        """Shapes fitted to the rounds `inv` admits, smallest first.

        Fitted to the reached rounds and to sampled ones from states `inv`
        admits where the property fails -- `hrank` quantifies over all of
        those, so a shape that does not drop on a sampled round is one the
        solver would only be asked about to be told no.
        """
        sampled = sample_rounds(
            self.ctx, self.ob, self.ev, self.draws, self.runs.states,
            lambda s: not ranks.holds(s) and all(self._holds(f, s) for f in inv))
        fitted = self._parse(ranks.fit(list(self.runs.rounds) + sampled))
        wide = sample_rounds(
            self.ctx, self.ob, self.ev, self.wide, self.runs.states,
            lambda s: not ranks.holds(s) and all(self._holds(f, s) for f in inv))
        kept = [r for r in fitted if self._drops_on(r, wide)]
        self.log(f"[houdini] {len(fitted)} ranking functions drop on "
                 f"{len(self.runs.rounds)} reached and {len(sampled)} sampled "
                 f"rounds; {len(kept)} also on {len(wide)} rounds from a wider "
                 f"range")
        return kept[:_MAX_RANKS]

    def _drops_on(self, rank: Candidate, rounds: list) -> bool:
        names = self.ctx.names
        for s, sp in rounds:
            try:
                if not _drops(self.ev(rank.term, dict(zip(names, s))),
                              self.ev(rank.term, dict(zip(names, sp)))):
                    return False
            except _Unevaluable:
                continue
        return True

    def _pins(self, inv: list[Candidate], rank: Candidate) -> list[Candidate]:
        """The value-set facts about the components a floored rank reads.

        Kept whether or not a proof used them, which is the one place this
        route carries a fact the proofs can do without. A real ranking function
        reaches Lean as `Int.toNat ⌊·⌋`, and `hrank` asks for `0 < ⌊x⌋`
        wherever the property fails; the tactics get there by *computing*
        the floor, which needs `x` pinned to a value rather than bounded or
        related to something else. Measured on `m_lra_conv`: the cores leave
        `x - y = -2` and `y in {2..5}`, from which a solver proves everything
        and Lean proves nothing, while adding `x in {0..3}` back -- a fact
        the other two imply -- builds in 2.6 s.
        """
        from ..smt_query import _constants

        read = set(_constants(rank.term, {}))
        # A pin is keyed by its column, and what the rank's term mentions is
        # the *component* -- an element and the tuple it lives in are one
        # constant to a solver -- so the column is mapped back to it.
        holder = {c.src: f"s{c.index}" for c in self.cols}
        wanted = {src for key, src in self.pins.items()
                  if holder.get(key) in read}
        return [f for f in inv if f.src in wanted]

    def _property_rests_on(self, prover, inv, conjuncts,
                           limit) -> "list[Candidate] | None":
        """The facts of `inv` that imply the property, or `None` if it does not.

        Houdini keeps a seeded conjunct exactly when it is inductive with the
        rest, so the usual answer is the conjuncts themselves. When one was
        dropped the rest may still imply it, and then the proof's core says
        which facts that took.
        """
        mine = [f for f in inv if f.src in conjuncts]
        if len(mine) == len(conjuncts):
            return mine
        if not inv:
            return None
        answer = prover.prove(self.ob.implies(inv, self.ctx.prp), limit, core=True)
        return self._named(inv, answer.core) if answer.proved else None

    # --- cutting the certificate down ---------------------------------------

    def _shrink(self, prover: Solver, inv: list[Candidate],
                need: list[Candidate], limit: float) -> list[Candidate]:
        """The facts `need` rests on, closed under the preservation proofs.

        Each proof of "a round preserves `need`" is asked from all of `inv`,
        and its core names the facts it used; those join `need` until a proof
        uses nothing new. The set that comes out is inductive by those
        proofs, and it is checked once more on its own before it replaces
        `inv` -- a core over-trimmed would otherwise surface as a Lean
        failure, and neither solver promises a core is one its goal needs. Anything that does not come back proved keeps `inv` whole.
        """
        ob = self.ob
        need = list(dict.fromkeys(need))
        while True:
            if not need:
                return need
            answer = prover.prove(ob.preserved(inv, need), limit, core=True)
            if not answer.proved or answer.core is None:
                return inv
            more = [f for f in self._named(inv, answer.core) if f not in need]
            if not more:
                break
            need += more
        if len(need) == len(inv):
            return inv
        if prover.prove(ob.preserved(need, need), limit).proved:
            self.log(f"[houdini] invariant cut to the {len(need)} of "
                     f"{len(inv)} facts its proofs use")
            return need
        return inv

    def _minimise(self, prover: Solver, inv: list[Candidate],
                  keep: list[str], ranked) -> list[Candidate]:
        """`inv` with each fact the rest do without taken out, one at a time.

        An unsat core is whatever the refutation happened to touch, not the
        least it could have, so what `_shrink` returns can still carry facts
        nothing needs. Dropping one is tried against sampled rounds first --
        a state the rest admit whose successor breaks one of them, or that
        the ranking function does not drop on -- and only a drop no sample
        refutes is put to the solver, at a short limit and on a small share
        of the budget: a smaller certificate is worth having, and not worth
        the search that found it.
        """
        ob = self.ob
        deadline = time.monotonic() + min(_MINIMISE_SECONDS, max(prover.left(), 0) / 4)
        kept = list(inv)
        for fact in reversed(list(inv)):
            if fact.src in keep or time.monotonic() > deadline:
                continue
            rest = [f for f in kept if f is not fact]
            if self._sampled_counterexample(rest, ranked):
                continue
            scripts = [ob.preserved(rest, rest)]
            if ranked is not None:
                scripts.append(ob.drops(rest, ranked[1]))
            limit = min(prover.short, deadline - time.monotonic())
            if limit > 0.5 and all(a.proved for a in prover.prove_all(scripts, limit)):
                kept = rest
        if len(kept) < len(inv):
            self.log(f"[houdini] invariant minimised to {len(kept)} of {len(inv)} facts")
        return kept

    def _sampled_counterexample(self, facts: list[Candidate], ranked) -> bool:
        ranks, rank = ranked if ranked is not None else (None, None)

        def admitted(s):
            if ranks is not None and ranks.holds(s):
                return False
            return all(self._holds(f, s) for f in facts)

        rounds = sample_rounds(self.ctx, self.ob, self.ev, self.draws,
                               self.runs.states, admitted, want=100,
                               seconds=_SAMPLE_SECONDS / 4)
        for s, sp in rounds:
            if any(not self._holds(f, sp) for f in facts):
                return True
            if rank is not None:
                try:
                    before = self.ev(rank.term, dict(zip(self.ctx.names, s)))
                    after = self.ev(rank.term, dict(zip(self.ctx.names, sp)))
                except _Unevaluable:
                    continue
                if not _drops(before, after):
                    return True
        return False

    @staticmethod
    def _named(inv: list[Candidate], core) -> list[Candidate]:
        if core is None:
            return list(inv)
        return [f for i, f in enumerate(inv) if f"h{i}" in core]

    # --- reading the module and the property ------------------------------

    def _check_sorts(self, ctx: SynthContext) -> None:
        def scalar(s) -> bool:
            """A sort this route has candidate shapes for.

            A matrix-shaped component passes when its *elements* do: it is a
            column per element here, and a column of a sort with no shapes
            is no better inside a tuple than outside one.
            """
            if s.isTuple():
                return all(scalar(e) for e in s.getTupleSorts())
            return s.isInteger() or s.isBoolean() or s.isReal()

        bad = [f"s{i} is {s}" for i, s in enumerate(ctx.env.state_sorts)
               if not scalar(s)]
        bad += [f"{c} is {c.getSort()}"
                for c in list(ctx.extl_next) + list(ctx.extl_latched)
                if not scalar(c.getSort())]
        if bad:
            why = (
                "Vampire's SMT-LIB front end has no bitvector theory"
                if self.spec.kind == "vampire" else
                "cvc5 has a bitvector theory, but the candidate shapes here "
                "do not -- a bound stated with a program constant, a "
                "congruence, an affine relation -- so there would be nothing "
                "to propose about one"
            )
            raise Refused(
                f"--infer houdini reads Int, Bool and Real state and inputs, "
                f"and this module has {', '.join(bad)}. {why}. `--infer "
                f"smt-linear` weighs a bitvector as its unsigned value."
            )

    def _conjuncts(self, ctx: SynthContext, cd: CertificateData) -> list[str]:
        """The property's top-level conjuncts, seeded as facts of their own.

        Split because Houdini keeps facts, not formulas: `(and a b)` is kept
        whole or dropped whole, where `a` alone may be inductive and `b`
        inductive given it.
        """
        p = ctx.prp
        parts = list(p) if p.getKind() == Kind.AND else [p]
        return [str(t) if len(parts) > 1 else cd.prp.strip() for t in parts]

    def _parse(self, srcs: list[str]) -> list[Candidate]:
        out = []
        for src in dict.fromkeys(srcs):
            try:
                term = self.ctx.env.parse_expr(src)
            except Exception:                        # noqa: BLE001
                continue
            if term.isNull():
                continue
            out.append(Candidate(src, term))
        return out

    # --- out --------------------------------------------------------------

    def _emit(self, cd: CertificateData, inv: list[Candidate],
              rank: "Candidate | None") -> CertificateData:
        srcs = [f.src for f in inv]
        inv_src = ("true" if not srcs else srcs[0] if len(srcs) == 1
                   else "(and " + " ".join(srcs) + ")")
        self.log(f"[houdini] inv: {inv_src}")
        cd.inv = cd.inv_smt = inv_src
        if rank is not None:
            self.log(f"[houdini] ranking: {rank.src}")
            cd.ranking = cd.ranking_smt = rank.src
        self._record(inv_src, rank)
        return cd

    def _record(self, inv_src: str, rank: "Candidate | None") -> None:
        if self.artifacts is None:
            return
        by = self.spec.kind
        what = (f"Houdini over facts read off simulated runs of the module, "
                f"each proved by {by} -- it holds at entry and every round "
                f"preserves it -- and cut down to the facts the proofs used.")
        self.artifacts.put("inv", inv_src, status="proved", language="smt",
                           what=f"The invariant this run found. {what}")
        if rank is not None:
            self.artifacts.put(
                "ranking", rank.src, status="proved", language="smt",
                what=f"The ranking function this run found: {by} proved "
                     f"`hrank` for it over the invariant beside it.",
            )

    def _none(self, cd, n_facts, n_ranks, reached, prover) -> CertificateData:
        if cd.is_safety:
            what, article = "inductive invariant implying the property", "an"
            started = f"Houdini started from {n_facts} facts the simulated runs keep"
        else:
            what, article = "ranking function", "a"
            started = (f"Houdini started from {n_facts} facts the simulated "
                       f"runs keep, and at most {n_ranks} ranking functions "
                       f"dropped on every reached and sampled round")
        refuted = (f", {prover.refuted} of them refuted"
                   if prover.refutes else "")
        detail = (
            f"{started}; the time limit reached {reached:g} s over "
            f"{prover.calls} {prover.name} calls{refuted}. "
            f"{prover.inconclusive}"
        )
        if self.artifacts is not None:
            self.artifacts.note(
                f"--infer houdini found no {what}. {detail}\n",
                status="unknown",
                what=f"A search for {article} {what} that did not succeed.",
                why=(f"{prover.name} decided every candidate it was asked "
                     f"about; the shapes proposed were the limit."
                     if prover.refutes else
                     f"{prover.name} proves or times out; nothing here was "
                     f"refuted."),
            )
        raise Refused(f"--infer houdini found no {what}. {detail}")
