"""Inference by synthesis: the certificate Vampire's answer literals derive.

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
`infer.common.synth.component_readings`'s ``floor_reals`` is the general
form of.

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
same :mod:`zrth.lean.smt.solvers` seam `--infer houdini` uses, in
milliseconds.  A certificate that does not survive that is reported as not
found, with what Vampire said.  This is not Houdini: nothing is filtered and
nothing is proposed, it is the one check that the derivation is honest.
"""

from __future__ import annotations


from ...core.certificate import Certificate, Claim
from ...core.errors import PredicateRefused, Refused
from ...smt.solvers import DEFAULT_TIMEOUT, Cvc5Solver
from ..engine import Engine
from ...smt.obligations import Candidate, Obligations
from ...artifacts.record import found, note
from ...smt.script import affine_smt, conjunction_smt, smt_int
from ...smt.claim import SynthContext, sorts_where
from ..common.synth import columns, floor_real_src
from .answers import Answers, Derived
from .branches import split
from .question import Question
from .template import Template, templates, weighable


class VampireEngine(Engine):
    """The certificate Vampire derives, checked before it is handed on."""

    def __init__(self, module, *, vampire: str, timeout: float = DEFAULT_TIMEOUT,
                 artifacts=None, log=print, encoding=None):
        self.module = module
        # The run's `smt.encoding.lazy` encoding; a library caller's module gets
        # its own.
        self.encoding = encoding
        self.exe = vampire
        self.timeout = float(timeout)
        self.artifacts = artifacts
        self.log = log

    # --- driver -----------------------------------------------------------

    def infer(self, claim: Claim,
              given: Certificate = Certificate()) -> Certificate:
        """The certificate of `claim`, as SMT-LIB. `given` is not read: the
        row seeds from nothing but `--pre`, which is in the claim."""
        # `SynthContext.build` takes every reading kind here; what this route
        # cannot state (a bitvector) or bound (no column at all) is
        # `_check_sorts`'s to refuse, before anything is printed.
        ctx = SynthContext.build(self.encoding or self.module, claim,
                                 route="vampire",
                                 takes=("int", "bool", "bv", "real", "tuple"))
        self._check_sorts(ctx)
        width = len(columns(ctx))
        ob = Obligations(ctx)
        entry = split(ctx, ob.init, "the initial state")
        step = split(ctx, ob.next, "the round")
        # Everything a step below reads of the module is the question's:
        # its context, its obligations and its scale -- `_read_back` prints
        # the same floor the question put in the conjecture, so there is
        # one of them and not two.
        q = Question(ctx, ob, entry, step)
        bounded = len(weighable(ctx))
        carried = ("" if bounded == width else
                   f" ({width - bounded} carried unbounded)")
        self.log(f"[vampire] {len(step)} branch(es) of the round, "
                 f"{len(entry)} of the initial state, "
                 f"{bounded} of {width} column(s) bounded{carried}")
        if q.scale != 1:
            self.log(f"[vampire] Real column(s): the rank reads them "
                     f"through a floor after scaling by {q.scale}")

        asker = Answers(self.exe, seconds=self.timeout, log=self.log)
        derive = self._safety if claim.is_safety else self._ranked
        tried = []
        for tpl in templates(ctx, ranked=not claim.is_safety):
            if asker.left() < 1:
                break
            tried.append(tpl.name)
            out = derive(q, tpl, asker, claim)
            if out is not None:
                return out
            self.log(f"[vampire] nothing from {tpl.name} "
                     f"({asker.spent:.1f} s spent)")
        self.log(f"[vampire] {asker.calls} Vampire call(s), "
                 f"{asker.spent:.1f} s")
        return self._none(claim, tried, asker, width)

    def _safety(self, q: Question, tpl: Template, asker: Answers,
                claim: Claim) -> "Certificate | None":
        """One question: the invariant and the property it has to imply.

        `inv -> prp` is what pins the coefficients down, which is why this
        is one question where `--buchi` is two.
        """
        script = q.conjecture(tpl, q.ctx.prp)
        self.log(f"[vampire] template {tpl.name}: {len(tpl.holes)} holes, "
                 f"{len(script)} chars")
        answer = asker.ask(script, len(tpl.holes))
        if answer is None:
            return None
        for named in self._alternatives(answer, tpl.holes, asker):
            inv, _ = self._read_back(q, tpl, named, claim)
            if self._checks_out(q, inv, None, claim):
                return self._emit(inv, None, tpl)
        return None

    def _ranked(self, q: Question, tpl: Template, asker: Answers,
                claim: Claim) -> "Certificate | None":
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
            ranks = asker.ask(q.rank_conjecture(tpl, q.ctx.prp, inv_named),
                              len(tpl.rank))
            if ranks is None:
                self.log("[vampire] no ranking over that invariant")
                continue
            for named in self._alternatives(ranks, tpl.rank, asker):
                inv, rank = self._read_back(q, tpl, {**inv_named, **named},
                                            claim)
                if self._checks_out(q, inv, rank, claim):
                    return self._emit(inv, rank, tpl)
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

    def _read_back(self, q: Question, tpl: Template, named: dict,
                   claim: Claim) -> tuple:
        """The template with Vampire's numbers in it, as SMT-LIB."""
        facts = []
        for row in tpl.rows:
            facts.append(f"(<= {smt_int(named[row.lo])} {row.smt})")
            facts.append(f"(<= {row.smt} {smt_int(named[row.hi])})")
        inv = conjunction_smt(facts)
        if claim.is_safety:
            return inv, None
        # A coefficient of one is the column itself: `(* 1 s0)` is what the
        # template says and `s0` is what the certificate should read. The
        # reads are taken from `tpl.at` rather than from the row order, so
        # an element says `((_ tuple.select k) s0)` and a Real one says
        # `(to_int ...)` -- the same floor `_rank` put in the question,
        # because `hrank` lands in `Nat` -- and a module carrying a column
        # no row bounds still pairs each coefficient with its own column.
        cols = columns(q.ctx)
        reads = [floor_real_src(cols[i].src, cols[i].sort, q.scale)
                 for i in tpl.at]
        rank = affine_smt(named[tpl.rank[-1]],
                          [named[c] for c in tpl.rank[:-1]], reads)
        return inv, rank

    def _checks_out(self, q: Question, inv_src: str, rank_src: "str | None",
                    claim: Claim) -> bool:
        """The four obligations, restated and put to cvc5.

        An answer literal is the substitution *a* refutation used, and this
        route wrote the question it refuted -- so the certificate is checked
        against the module's own encoding before anything is emitted, the
        same way `--pre-check cvc5` would check one supplied by hand.
        """
        ob, ctx = q.ob, q.ctx
        try:
            facts = [Candidate(inv_src, ctx.enc.parse(inv_src, "invariant"))]
            rank = (None if rank_src is None
                    else Candidate(rank_src, ctx.enc.parse(
                        rank_src, "ranking", want="Int")))
        except PredicateRefused as e:
            self.log(f"[vampire] the answer does not parse back: {e}")
            return False
        solver = Cvc5Solver(ctx.tm, seconds=max(5.0, min(30.0, self.timeout)),
                            log=self.log)
        queries = [ob.holds_at_entry(facts), ob.preserved(facts, facts)]
        if claim.is_safety:
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
        bad = sorts_where(ctx, lambda s: s.isBitVector())
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

    def _emit(self, inv_src: str, rank_src: "str | None",
              tpl: Template) -> Certificate:
        self.log(f"[vampire] inv: {inv_src}")
        if rank_src is not None:
            self.log(f"[vampire] ranking: {rank_src}")
        self._record(inv_src, rank_src, tpl)
        return Certificate(inv_src, rank_src)

    def _record(self, inv_src: str, rank_src: "str | None",
                tpl: Template) -> None:
        what = (f"Derived by Vampire's answer literals from a `{tpl.name}` "
                f"template with {len(tpl.all_holes)} holes -- the "
                f"obligations were stated with the certificate open and "
                f"Vampire returned the coefficients -- then checked against "
                f"the module's encoding with cvc5.")
        found(self.artifacts, "inv", inv_src,
              what=f"The invariant this run found. {what}")
        if rank_src is not None:
            found(self.artifacts, "ranking", rank_src,
                  what=f"The ranking function this run found. {what}")

    def _none(self, claim: Claim, tried, asker, width: int):
        what = ("inductive invariant implying the property" if claim.is_safety
                else "invariant and ranking function")
        detail = (
            f"Vampire was asked for one as the coefficients of "
            f"{' and '.join(tried) or 'no'} template(s), over "
            f"{width} column(s), in {asker.calls} call(s) taking "
            f"{asker.spent:.1f} s. An answer literal comes back only when a "
            f"refutation pins the coefficients down, so this is not a proof "
            f"that no certificate of this shape exists -- a longer "
            f"--vampire-timeout, or a shape outside the templates, may be "
            f"what is missing. `--infer houdini` searches the same module by "
            f"proposing facts and proving them, which is a different reach."
        )
        note(
            self.artifacts,
            f"--infer vampire derived no {what}. {detail}\n",
            status="unknown",
            what=f"A derivation of {what} that did not succeed.",
            why="Vampire answers or times out; nothing was refuted.",
        )
        raise Refused(f"--infer vampire derived no {what}. {detail}",
                      searched=True)
