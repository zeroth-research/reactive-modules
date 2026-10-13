"""Obligations as questions: named hypotheses, a goal, and the round they
read.

`Obligations` states a module's obligations over one set of constants, and
`Query` is what it hands a solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..smt_synth import SynthContext


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

    def __init__(self, ctx: "SynthContext"):
        import cvc5

        tm = ctx.tm
        self.ctx = ctx
        self._rewriter = cvc5.Solver(tm)

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
        self.update_pre = readable(ctx.with_inputs(ctx.update_pre, self.el, self.en))
        self.init_pre = readable(ctx.with_inputs(ctx.init_pre, self.el, self.en))

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
        return self._rewriter.simplify(t) if "tuple" in str(t) else t

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

        tm = self.ctx.tm
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

    # --- the queries ------------------------------------------------------

    def holds_at_entry(self, facts: list[Candidate]) -> Query:
        """`init_pre e -> fact (init e)`, for all of `facts` at once."""
        at = [self._at(f.term, self.si) for f in facts]
        return self._query([("pre", self.init_pre)], self._and(at),
                           defines=self._defines(self.si, self.init),
                           probes=at)

    def preserved(self, hyps: list[Candidate], goals: list[Candidate]) -> Query:
        """`hyps s /\\ update_pre e -> goals (update s e)`."""
        named = [(f"h{i}", self._at(h.term, self.s)) for i, h in enumerate(hyps)]
        at = [self._at(g.term, self.sp) for g in goals]
        return self._query(named + [("pre", self.update_pre)], self._and(at),
                           defines=self._defines(self.sp, self.next),
                           probes=at)

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
        from cvc5 import Kind

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
        q = self._query(named, goal, defines=self._defines(self.sp, self.next))
        # A refutation is a round the rank does not drop on, and the round
        # is what the fit learns from -- read back only for a state of
        # scalars, which is the shape a simulated round has.
        if all(h.getKind() != Kind.APPLY_CONSTRUCTOR for h in self.s):
            q = replace(q, witness=tuple(self.s) + tuple(self.sp))
        return q

    # --- plumbing ---------------------------------------------------------

    def _at(self, term, vs):
        # Folded, because substituting a matrix-shaped holder leaves a
        # select of a constructor where the component was.
        return self._readable(term.substitute(self.ctx.state, vs))

    def _and(self, terms):
        from cvc5 import Kind

        tm = self.ctx.tm
        if not terms:
            return tm.mkBoolean(True)
        return terms[0] if len(terms) == 1 else tm.mkTerm(Kind.AND, *terms)

    def _clamp(self, t):
        from cvc5 import Kind

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
