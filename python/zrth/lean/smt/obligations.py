"""Obligations as questions: named hypotheses, a goal, and the round they
read.

A `Query` is one obligation. What `rule_buchi` and `rule_globally` ask --
`init_inv`, `step_inv`, `inv_imp_P`, `hrank`, and `hrank` split into
`rank_decrease` and `rank_positive` -- is stated once, here, over a *frame*:
the constants an obligation is stated over and the round from them.

* `Frame` is inline: fresh constants `{prefix}s0..`, `{prefix}el0..` and
  `{prefix}en0..`, and the successor is the encoded round substituted in.
* `Obligations` is defined: one frame `v_*` whose successor is constants
  `v_sp*` each query defines, so a query about twenty facts carries the
  round once.

A precondition is taken as given: `None` is no hypothesis, and a literal
`true` is one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .claim import SynthContext


@dataclass(frozen=True)
class Query:
    """One obligation: named hypotheses, a goal, and the round they read.

    `defines` states the round once -- the successor state over the latched
    state and the inputs -- so a query about twenty facts carries the
    transition once rather than twenty times.

    `probes` are the goal's conjuncts under their own names.  A solver that
    returns a counter-model reports which of them that model falsifies; one
    that does not, ignores them.  They are deliberately outside `key`: the
    same obligation asked with and without probes is the same question, and
    a cached answer that already carries what the caller wants is reused.
    """

    hyps: tuple[tuple[str, object], ...]
    goal: object
    defines: tuple[tuple[object, object], ...] = ()
    probes: tuple[tuple[str, object], ...] = ()
    # Terms whose values a counter-model is read at -- the state and its
    # successor, for an obligation whose refutation is a round to learn
    # from. Outside `key` for the reason `probes` are.
    witness: tuple = ()

    @property
    def key(self):
        return (self.hyps, self.goal, self.defines)

    def negation(self, tm):
        """The hypotheses and the goal's negation, conjoined: unsat exactly
        when the obligation holds. Inline only -- a defined successor would
        be left free."""
        from cvc5 import Kind

        assert not self.defines, "a defined round has no negation form"
        return conjunction(tm, [t for _, t in self.hyps]
                           + [tm.mkTerm(Kind.NOT, self.goal)])

    def implication(self, tm):
        """`hyps -> goal`, every hypothesis kept as given. Inline only."""
        from cvc5 import Kind

        assert not self.defines, "a defined round has no implication form"
        if not self.hyps:
            return self.goal
        return tm.mkTerm(Kind.IMPLIES,
                         conjunction(tm, [t for _, t in self.hyps]), self.goal)

    def labelled(self, tm) -> dict:
        """`negation` as named parts, the goal's negation as `goal`. A
        literal `true` or `false` is left out: an unsat core never needs it."""
        from cvc5 import Kind

        parts = dict(self.hyps)
        parts["goal"] = tm.mkTerm(Kind.NOT, self.goal)
        return {k: v for k, v in parts.items()
                if v.getKind() != Kind.CONST_BOOLEAN}


@dataclass(frozen=True)
class Candidate:
    """One invariant fact or ranking function: its source and its term."""

    src: str                # SMT-LIB over `s0..`, what the certificate carries
    term: object            # the same, parsed, over `ctx.state`


# ══════════════════════════════════════════════════════════════════════════
# The obligations, over a frame
#
# A frame has `tm`; the constants `s`; `succ`, the state after one round from
# them, and `entry`, the state after tick 0; the preconditions `init_pre` and
# `update_pre` over its inputs; `at(pred, vs)`, a predicate read at the state
# `vs`; and `define_next()` / `define_entry()`, what a query must define for
# `succ` / `entry` to mean the round. A predicate is a term over the parser's
# state, or a function of a state (a template with its coefficients open).
# `given` are hypotheses about nothing a frame reads, such as the guard of
# one case of a round; they follow the precondition. Each statement builds
# its terms in the order written, which is the order cvc5 numbers them in.
# ══════════════════════════════════════════════════════════════════════════


def conjunction(tm, terms):
    """`terms` conjoined: `true` for none, the term itself for one."""
    from cvc5 import Kind

    if not terms:
        return tm.mkBoolean(True)
    return terms[0] if len(terms) == 1 else tm.mkTerm(Kind.AND, *terms)


def clamp(tm, t):
    """`t` as Lean's `Int.toNat` reads an Int: itself when non-negative,
    else 0."""
    from cvc5 import Kind

    zero = tm.mkInteger(0)
    return tm.mkTerm(Kind.ITE, tm.mkTerm(Kind.GEQ, t, zero), t, zero)


def _named(f, hyps) -> list:
    return [(name, f.at(term, f.s)) for name, term in hyps]


def _given(given) -> list:
    return [(f"case{i}", g) for i, g in enumerate(given)]


def _outside(f, hyps, prp) -> list:
    """What a ranking obligation assumes: `hyps` and `~P` at `s`, and the
    update precondition."""
    from cvc5 import Kind

    return _named(f, hyps) + [
        ("notP", f.tm.mkTerm(Kind.NOT, f.at(prp, f.s))),
        ("pre", f.update_pre),
    ]


def _query(hyps, goal, *, defines=(), probes=()) -> Query:
    """Named hypotheses, a goal, and the conjuncts of it worth naming.

    A probe is named `g<i>` against the position of the fact it came from,
    which is how a counter-model is read back: the names are positions in
    the list the caller asked about. A goal that is one thing gets no probes
    -- there would be nothing to learn from naming it that `REFUTED` does
    not already say. An absent precondition is no hypothesis.
    """
    probes = list(probes)
    return Query(
        hyps=tuple((n, h) for n, h in hyps if h is not None),
        goal=goal,
        defines=tuple(defines),
        probes=tuple((f"g{i}", t) for i, t in enumerate(probes))
        if len(probes) > 1 else (),
    )


def init_inv(f, goals: list, *, given=()) -> Query:
    """`init_pre e -> goals (init e)`, for all of `goals` at once."""
    entry = f.entry
    pre = f.init_pre
    at = [f.at(g, entry) for g in goals]
    goal = conjunction(f.tm, at)
    return _query([("pre", pre)] + _given(given), goal,
                  defines=f.define_entry(), probes=at)


def step_inv(f, hyps: list, goals: list, *, given=()) -> Query:
    """`hyps s /\\ update_pre e -> goals (update s e)`; `hyps` are named."""
    succ = f.succ
    named = _named(f, hyps)
    pre = f.update_pre
    at = [f.at(g, succ) for g in goals]
    goal = conjunction(f.tm, at)
    return _query(named + [("pre", pre)] + _given(given), goal,
                  defines=f.define_next(), probes=at)


def inv_imp_P(f, hyps: list, goal) -> Query:
    """`hyps s -> goal s`: the invariant is a proof of the property."""
    named = _named(f, hyps)
    return _query(named, f.at(goal, f.s))


def hrank(f, hyps: list, prp, rank) -> Query:
    """`hyps s /\\ ~P s /\\ update_pre e -> V (update s e) < V s`.

    With `V = Int.toNat rank`, exactly as `rule_buchi` states it: a clamp
    on both sides, so the rank may be anything where the property holds.
    """
    from cvc5 import Kind

    succ = f.succ
    named = _outside(f, hyps, prp)
    goal = f.tm.mkTerm(Kind.LT, clamp(f.tm, f.at(rank, succ)),
                       clamp(f.tm, f.at(rank, f.s)))
    return _query(named, goal, defines=f.define_next())


def ranked(f, hyps: list, prp, goal, *, given=()) -> Query:
    """`hyps s /\\ ~P s /\\ update_pre e -> goal`, for a route that states
    `hrank`'s conclusion in a shape of its own (read at `f.succ`)."""
    named = _outside(f, hyps, prp)
    return _query(named + _given(given), goal, defines=f.define_next())


def rank_decrease(f, hyps: list, prp, rank) -> Query:
    """`hrank`'s first half: `rank (update s e) < rank s`, unclamped.

    With `rank_positive`, exactly `hrank` for an Int rank: `toNat r' <
    toNat r` holds iff `r >= 1` and `r' < r`.
    """
    from cvc5 import Kind

    succ = f.succ
    named = _outside(f, hyps, prp)
    goal = f.tm.mkTerm(Kind.LT, f.at(rank, succ), f.at(rank, f.s))
    return _query(named, goal, defines=f.define_next())


def rank_positive(f, hyps: list, prp, rank) -> Query:
    """`hrank`'s second half: `rank s >= 1` wherever it has to fall."""
    from cvc5 import Kind

    named = _outside(f, hyps, prp)
    goal = f.tm.mkTerm(Kind.GEQ, f.at(rank, f.s), f.tm.mkInteger(1))
    return _query(named, goal)


# ══════════════════════════════════════════════════════════════════════════
# The round as relations
#
# `--infer sygus` states `G P` as SyGuS-IF does, over `pre(s)` and
# `trans(s, s')`: predicates on states alone, so the inputs that make a
# state initial, or take one state to the next, are closed existentially.
# Existential rather than fresh constants, because the relation is about the
# state alone; and existential rather than universal, because `init_inv`
# asks the invariant to hold at whatever the inputs made, not at every value
# some other input would have made. A precondition is over the parser's
# inputs, and `None` is none.
# ══════════════════════════════════════════════════════════════════════════


def initial(enc, s: list, init_pre=None, *, prefix: str = "pre_"):
    """`s` is a state tick 0 can produce: `exists el en. init_pre /\\ s =
    init en`."""
    el, en = enc.bound_inputs(prefix)
    return _closing(enc, el + en, enc.with_inputs(init_pre, el, en),
                    lambda: _same(enc.tm, s, enc.init_state(en)))


def transition(enc, s: list, s2: list, update_pre=None, *,
               prefix: str = "tr_"):
    """`s -> s2`: some input, allowed by `update_pre`, takes one to the
    other -- `exists el en. update_pre /\\ s2 = update s el en`."""
    el, en = enc.bound_inputs(prefix)
    return _closing(enc, el + en, enc.with_inputs(update_pre, el, en),
                    lambda: _same(enc.tm, s2, enc.next_state(s, el, en)))


def _closing(enc, vs: list, pre, same):
    """`exists vs. pre /\\ same()`. `same` is built after `pre`, which is
    the order cvc5 numbers the terms in, and so the order the relation
    has always been built in."""
    from cvc5 import Kind

    tm = enc.tm
    body = same() if pre is None else tm.mkTerm(Kind.AND, pre, same())
    if not vs:
        return body
    return tm.mkTerm(Kind.EXISTS, tm.mkTerm(Kind.VARIABLE_LIST, *vs), body)


def _same(tm, lhs: list, rhs: list):
    from cvc5 import Kind

    return conjunction(tm, [tm.mkTerm(Kind.EQUAL, a, b)
                            for a, b in zip(lhs, rhs)])


class Frame:
    """An inline frame: fresh constants, and the encoded round from them.

    `enc` is the run's `smt.encoding.ModuleEncoding`; `init_pre` and
    `update_pre` are over the parser's inputs, as it reads them. `succ`,
    `entry` and the preconditions are built on first use, so a statement
    that never reads one never encodes it.
    """

    def __init__(self, enc, prefix: str, *, init_pre=None, update_pre=None,
                 consts=None):
        self.enc = enc
        self.tm = enc.tm
        self.s, self.el, self.en = consts or enc.fresh(prefix)
        self._pres = (init_pre, update_pre)

    @classmethod
    def over(cls, enc, s, el, en, *, init_pre=None, update_pre=None) -> "Frame":
        """A frame over terms the caller made: bound variables, say."""
        return cls(enc, "", init_pre=init_pre, update_pre=update_pre,
                   consts=(list(s), list(el), list(en)))

    @cached_property
    def succ(self) -> list:
        return self.enc.next_state(self.s, self.el, self.en)

    @cached_property
    def entry(self) -> list:
        return self.enc.init_state(self.en)

    @cached_property
    def init_pre(self):
        return self.enc.with_inputs(self._pres[0], self.el, self.en)

    @cached_property
    def update_pre(self):
        return self.enc.with_inputs(self._pres[1], self.el, self.en)

    def at(self, pred, vs):
        return pred(vs) if callable(pred) else self.enc.at(pred, vs)

    def define_next(self) -> tuple:
        return ()

    def define_entry(self) -> tuple:
        return ()


class Obligations:
    """One module's obligations, as questions rather than as files: a
    defined frame (see the module docstring).

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

    def __init__(self, ctx: "SynthContext"):
        tm = ctx.tm
        self.ctx = ctx

        def consts(vs, prefix):
            """One holder per component, over one *constant* per element.

            A scalar component is a constant, as it always was. A
            matrix-shaped one is the constructor over a constant per
            element -- `(tuple v_s0_0 v_s0_1 v_s0_2)` -- so that a
            component is still one thing to substitute while every name
            that reaches a solver is a scalar. `_readable` folds the
            selects that leaves behind, and a prover with no tuple theory
            then never sees a tuple: not in a `declare-fun`, not in a
            `define-fun`, and not in an obligation.
            """
            out = []
            for i, v in enumerate(vs):
                sort = v.getSort()
                if not sort.isTuple():
                    out.append(tm.mkConst(sort, f"{prefix}{i}"))
                    continue
                out.append(tm.mkTuple([tm.mkConst(e, f"{prefix}{i}_{k}")
                                       for k, e in
                                       enumerate(sort.getTupleSorts())]))
            return out

        self.s = consts(ctx.state, "v_s")
        self.sp = consts(ctx.state, "v_sp")
        self.si = consts(ctx.state, "v_si")
        self.el = consts(ctx.extl_latched, "v_el")
        self.en = consts(ctx.extl_next, "v_en")
        readable = self._readable

        self.next = [readable(t) for t in ctx.enc.next_state(self.s, self.el, self.en)]
        self.init = [readable(t) for t in ctx.enc.init_state(self.en)]
        self.update_pre = readable(ctx.enc.with_inputs(ctx.update_pre, self.el, self.en))
        self.init_pre = readable(ctx.enc.with_inputs(ctx.init_pre, self.el, self.en))

    def _readable(self, t):
        """`t`, with every tuple the encoding leaves folded away.

        A module with scalar state can still compute through a matrix --
        `m_max` takes the max of a 2-vector -- and cvc5 encodes that as
        `tuple`/`tuple.select`, which Vampire's SMT-LIB front end does not
        have. The rewriter folds a select of a constructor away, so the
        round a prover is shown is the same function without them. Since
        the state holders are constructors too (`consts`), that covers a
        matrix-shaped *component* and not only an intermediate. It is only
        called where a tuple is printed: a term that has none is handed
        over exactly as encoded.
        """
        return self.ctx.enc.simplify(t) if "tuple" in str(t) else t

    def flatten(self, holders, terms=None) -> list:
        """One term per element: `terms` shaped by `holders`, or the holders.

        `holders` says what has elements and how many -- a constructor is a
        matrix-shaped component, anything else is a scalar -- and `terms`
        are the terms standing in those positions, which for the round is
        the successor and for the holders themselves is the constants. The
        two are separate because a caller that wants the *names* and one
        that wants the round both need the same shape read off the same
        place.
        """
        from cvc5 import Kind

        out = []
        for h, t in zip(holders, holders if terms is None else terms):
            if h.getKind() != Kind.APPLY_CONSTRUCTOR:
                out.append(t)
            else:
                out += [self._select(t, k)
                        for k in range(h.getNumChildren() - 1)]
        return out

    def slots(self, holders) -> list:
        """`(name, sort)` for every constant in `holders` -- one per element.

        What a caller drawing input values needs: the names are the ones a
        term mentions, so a matrix-shaped input is drawn element by element
        rather than as a tuple under a name nothing uses.
        """
        return [(str(c), c.getSort()) for c in self.flatten(holders)]

    def env_of(self, state, holders=None) -> dict:
        """A simulated state as values for the constants `holders` names.

        One entry per element, because that is what the constants are: a
        matrix-shaped component contributes its elements under the names
        `consts` gave them, not a tuple under the component's name.
        """
        from cvc5 import Kind

        out: dict = {}
        for h, v in zip(holders if holders is not None else self.s, state):
            if h.getKind() == Kind.APPLY_CONSTRUCTOR:
                for c, x in zip(list(h)[1:], v):
                    out[c.getSymbol()] = x
            else:
                out[str(h)] = v
        return out

    def _select(self, t, k):
        """Element `k` of `t`, pushed down to where the tuple is built.

        `simplify` folds a select of a constructor, but not one of an `ite`
        *between* constructors -- and that is exactly the shape of a round
        that branches, as `m_mixed` does when it resets its vector instead
        of decrementing it. Pushing the select into the branches is what
        leaves a scalar per element rather than a tuple under a selector.
        """
        from cvc5 import Kind

        tm = self.tm
        if t.getKind() == Kind.ITE:
            return tm.mkTerm(Kind.ITE, t[0],
                             self._select(t[1], k), self._select(t[2], k))
        if t.getKind() == Kind.APPLY_CONSTRUCTOR:
            return list(t)[1 + k]
        ctor = t.getSort().getDatatype()[0]
        return self._readable(
            tm.mkTerm(Kind.APPLY_SELECTOR, ctor[k].getTerm(), t))

    def _defines(self, holders, values) -> list:
        """The round as `define-fun`s: one per constant, so one per element.

        There is no constant of tuple sort to define -- by construction, see
        `consts` -- so a matrix-shaped component is defined element by
        element, the `k`-th from the `k`-th element of the round.
        """
        from cvc5 import Kind

        out = []
        for h, v in zip(holders, values):
            if h.getKind() != Kind.APPLY_CONSTRUCTOR:
                out.append((h, v))
                continue
            for k, c in enumerate(list(h)[1:]):
                out.append((c, self._select(v, k)))
        return out

    # --- the frame ------------------------------------------------------

    @property
    def tm(self):
        return self.ctx.tm

    @property
    def succ(self) -> list:
        return self.sp

    @property
    def entry(self) -> list:
        return self.si

    def at(self, pred, vs):
        """`pred` read at `vs` -- a term folded, because substituting a
        matrix-shaped holder leaves a select of a constructor where the
        component was."""
        if callable(pred):
            return pred(vs)
        return self._readable(pred.substitute(self.ctx.state, vs))

    def case(self, state) -> "_Case":
        """This frame with `state` for both the successor and the initial
        state, which nothing defines: one case of a round, its `ite`s
        folded away."""
        return _Case(self, state)

    def define_next(self) -> list:
        return self._defines(self.sp, self.next)

    def define_entry(self) -> list:
        return self._defines(self.si, self.init)

    # --- the queries ------------------------------------------------------

    def holds_at_entry(self, facts: list[Candidate]) -> Query:
        """`init_pre e -> fact (init e)`, for all of `facts` at once."""
        return init_inv(self, [f.term for f in facts])

    def preserved(self, hyps: list[Candidate], goals: list[Candidate]) -> Query:
        """`hyps s /\\ update_pre e -> goals (update s e)`."""
        return step_inv(self, _facts(hyps), [g.term for g in goals])

    def implies(self, hyps: list[Candidate], goal) -> Query:
        """`hyps s -> goal s`: the invariant is a proof of the property."""
        return inv_imp_P(self, _facts(hyps), goal)

    def drops(self, hyps: list[Candidate], rank: Candidate) -> Query:
        """`hrank` over the facts `hyps`."""
        from cvc5 import Kind

        q = hrank(self, _facts(hyps), self.ctx.prp, rank.term)
        # A refutation is a round the rank does not drop on, and the round
        # is what the fit learns from -- read back only for a state of
        # scalars, which is the shape a simulated round has.
        if all(h.getKind() != Kind.APPLY_CONSTRUCTOR for h in self.s):
            q = replace(q, witness=tuple(self.s) + tuple(self.sp))
        return q


class _Case:
    """`Obligations.case`."""

    def __init__(self, ob: Obligations, state):
        self.tm, self.s, self.el, self.en = ob.tm, ob.s, ob.el, ob.en
        self.init_pre, self.update_pre = ob.init_pre, ob.update_pre
        self.succ = self.entry = list(state)
        self.at = ob.at

    def define_next(self) -> tuple:
        return ()

    def define_entry(self) -> tuple:
        return ()


def _facts(hyps: list[Candidate]) -> list:
    """`hyps` named `h<i>`, by position: how an unsat core is read back."""
    return [(f"h{i}", h.term) for i, h in enumerate(hyps)]
