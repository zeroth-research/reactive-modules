"""Inference by proposing and proving: Houdini, and a ranking search.

``--infer houdini``.  Nothing here searches for a certificate.  The route
*proposes* -- candidate invariant facts read off the module and off
simulated runs of it, ranking functions from a fixed list of shapes -- and a
solver decides what stays:

* **the invariant** is Houdini's.  Candidate facts are the bound a component
  stays within, stated with a constant the program mentions; a congruence
  the runs keep; a relation between two components; the bounds a component
  keeps on each side of a Bool flag; the bounds each mode of the Bool
  columns keeps, computed rather than read (:mod:`.modes`); and under
  ``--safety`` the property's own conjuncts.  Each is put to the solver (it holds at entry; a round
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
``--houdini-solver`` picks it and :mod:`zrth.lean.smt.solvers` is the
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
import time

from ...core.certificate import Certificate, Claim
from ...core.errors import PredicateRefused, Refused
from ...smt.claim import SynthContext
from .solvers import DEFAULT_SOLVER, SolverSpec
from ...smt.solvers import DEFAULT_TIMEOUT, Solver
from ..engine import Engine
from .modes import mode_bounds
from .candidates import invariant_candidates, pinned_values, reading
from .evaluate import Evaluator, Unevaluable
from .ranks import Ranks, drops
from .simulate import SAMPLE_SECONDS, Draws, sample_rounds, simulate
from ...smt.obligations import Candidate, Obligations
from ...artifacts.record import found, note
from ...smt.script import conjunction_smt
from ...smt.terms import free_constants
from ..common.synth import (
    columns,
    denominator_scale,
    program_constants,
    program_rationals,
    split_conditions,
)

try:
    import cvc5                                      # type: ignore
    from cvc5 import Kind
except ImportError:                                  # pragma: no cover
    cvc5 = None
    Kind = None


# Batches of sampled rounds a Houdini pass draws before asking the solver.
_SAMPLE_PASSES = 4

# The most a certificate's minimisation may take, of what is left.
_MINIMISE_SECONDS = 15.0

# A ranking search asks about at most this many shapes, smallest first.
_MAX_RANKS = 48

# Times the shapes are fitted again to the wide rounds every one of them
# failed on, before the search gives up on them.
_REFITS = 3

# Ranking functions a refit reads the solver's counter-models of.
_WITNESSED = 4

# How long a rung's fits may run, refits included: a refit that would end
# past it is skipped. The first fit always runs.
_REFIT_SECONDS = 20.0


class HoudiniEngine(Engine):
    """Houdini and a ranking search, put to whichever solver was chosen.

    `solver` is a `SolverSpec` -- what `--houdini-solver` and the flags
    belonging to the one it names resolved to at parse time -- or the bare
    name of one, which is what a caller with no binary to point at wants.
    The solver itself is built in `infer`, because it needs the cvc5
    `TermManager` the module is encoded into and because its deadline
    starts when the search does, not when the object is made.
    """

    def __init__(self, module, *, solver=DEFAULT_SOLVER,
                 timeout: float = DEFAULT_TIMEOUT, artifacts=None, log=print,
                 encoding=None):
        self.spec = solver if isinstance(solver, SolverSpec) else SolverSpec(solver)
        self.module = module
        # The run's `smt.encoding.lazy` encoding; a library caller's module gets
        # its own.
        self.encoding = encoding
        self.timeout = float(timeout)
        self.artifacts = artifacts
        self.log = log

    # --- driver ---------------------------------------------------------

    def infer(self, claim: Claim,
              given: Certificate = Certificate()) -> Certificate:
        """The certificate of `claim`, as SMT-LIB. `given` is not read: the
        row seeds from nothing but `--pre`, which is in the claim."""
        ctx = SynthContext.build(self.encoding or self.module, claim,
                                 route="houdini",
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
                 + ", ".join(reading(c) for c in self.cols))
        if any(srt.isReal() for srt in ctx.env.state_sorts):
            self.log(f"[houdini] Real state: a ranking function is floored "
                     f"after scaling by {scale}")

        runs = simulate(ctx, ob, ev, constants, safety=claim.is_safety,
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
        conjuncts = self._conjuncts(ctx, claim) if claim.is_safety else []
        # The computed bounds go last: the minimiser drops from the end, so
        # a certificate the read ones carry keeps the program's constants.
        facts = self._parse(invariant_candidates(ctx, runs, constants,
                                                 conjuncts, rationals)
                            + mode_bounds(ctx, ob, log=self.log))
        self.log(f"[houdini] {len(facts)} candidate facts hold on the runs")
        ranks = None if claim.is_safety else Ranks(ctx, ev, claim.prp,
                                                   split_conditions(ctx, ob),
                                                   constants, rationals, scale)

        prover = self.spec.build(ctx.tm, seconds=self.timeout, log=self.log)
        found, reached, tried = None, 0.0, 0
        # Rounds a refutation taught the fit. Kept across rungs: a round
        # cvc5 refuted a shape with is a counterexample at every limit.
        learnt: list = []
        for limit in prover.ladder(self.timeout):
            if prover.left() < 0.5:
                break
            reached = limit
            self.log(f"[houdini] time limit {limit:g} s "
                     f"({prover.left():.0f} s of the budget left)")
            inv = self._houdini(prover, facts, limit)
            self.log(f"[houdini] Houdini kept {len(inv)} of {len(facts)}: "
                     + " ".join(f.src for f in inv))
            if claim.is_safety:
                need = self._property_rests_on(prover, inv, conjuncts, limit)
                if need is not None:
                    inv = self._shrink(prover, inv, need, limit)
                    found = (self._minimise(prover, inv, conjuncts, None), None)
                    break
                continue
            at, candidates = None, []
            fit_until = time.monotonic() + _REFIT_SECONDS
            for _ in range(_REFITS + 1):
                started = time.monotonic()
                candidates = self._ranks(ranks, inv, learnt, fit_until)
                tried = max(tried, len(candidates))
                if not candidates:
                    break
                at = prover.first([ob.drops(inv, r) for r in candidates], limit)
                if at is not None or not prover.refutes:
                    break
                if time.monotonic() + (time.monotonic() - started) > fit_until:
                    break
                fresh = [r for r in self._refuting(prover, inv, candidates, limit)
                         if r not in learnt]
                if not fresh:
                    break
                self.log(f"[houdini] refitting to {len(fresh)} round(s) "
                         f"{prover.name} refuted the ranking functions with")
                learnt += fresh
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
            return self._none(claim, len(facts), tried, reached, prover)
        inv, rank = found
        return self._emit(inv, rank)

    # --- Houdini ----------------------------------------------------------

    def _holds(self, fact: Candidate, state) -> bool:
        try:
            return bool(self.ev(fact.term, dict(zip(self.ctx.names, state))))
        except Unevaluable:
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

    def _ranks(self, ranks: Ranks, inv: list[Candidate],
               seen=(), fit_until: float = math.inf) -> list[Candidate]:
        """Shapes fitted to the rounds `inv` admits, smallest first.

        Fitted to the reached rounds and to sampled ones from states `inv`
        admits where the property fails -- `hrank` quantifies over all of
        those, so a shape that does not drop on a sampled round is one the
        solver would only be asked about to be told no -- and to `seen`,
        rounds the solver refuted earlier shapes with, as far as `inv` still
        admits them. A refit is skipped when another fit would run past
        `fit_until`: on a wide module one fit is tens of seconds of Python.
        """
        sampled = sample_rounds(
            self.ctx, self.ob, self.ev, self.draws, self.runs.states,
            lambda s: not ranks.holds(s) and all(self._holds(f, s) for f in inv))
        wide = sample_rounds(
            self.ctx, self.ob, self.ev, self.wide, self.runs.states,
            lambda s: not ranks.holds(s) and all(self._holds(f, s) for f in inv))
        # A wide round no fitted shape drops on is fitted to in turn: the
        # shift of a shape is read off the rounds it was fitted to, and the
        # round that decides it -- one crossing between two modes from
        # exactly a bound -- is one a draw finds by luck.
        learnt: list = [r for r in seen
                        if all(self._holds(f, r[0]) for f in inv)]
        for _ in range(_REFITS + 1):
            started = time.monotonic()
            fitted = self._parse(ranks.fit(list(self.runs.rounds) + sampled,
                                           keep=learnt),
                                 "ranking", want="Int")
            broken = [self._broken_on(r, wide) for r in fitted]
            kept = [r for r, b in zip(fitted, broken) if b is None]
            fresh = [b for b in dict.fromkeys(b for b in broken if b is not None)
                     if b not in learnt]
            if kept or not fresh:
                break
            if time.monotonic() + (time.monotonic() - started) > fit_until:
                break
            learnt += fresh
        self.log(f"[houdini] {len(fitted)} ranking functions drop on "
                 f"{len(self.runs.rounds)} reached and "
                 f"{len(sampled) + len(learnt)} sampled rounds; {len(kept)} "
                 f"also on {len(wide)} rounds from a wider range")
        return kept[:_MAX_RANKS]

    def _refuting(self, prover: Solver, inv: list[Candidate],
                  ranks: list[Candidate], limit: float) -> list:
        """The rounds the solver refutes the first few of `ranks` with.

        `hrank` quantifies over real states the lattice the draws live on
        never reaches -- a tank at 11.9, just short of full, where a rank
        floored after scaling is already 0 -- and a counter-model is exactly
        such a state, with its successor.
        """
        n = len(self.ctx.state)
        out = []
        for r in ranks[:_WITNESSED]:
            a = prover.prove(self.ob.drops(inv, r), limit, model=True)
            if a.refuted and a.values is not None:
                out.append((a.values[:n], a.values[n:]))
        return list(dict.fromkeys(out))

    def _broken_on(self, rank: Candidate, rounds: list):
        """The first of `rounds` `rank` does not drop on, or None."""
        names = self.ctx.names
        for s, sp in rounds:
            try:
                if not drops(self.ev(rank.term, dict(zip(names, s))),
                             self.ev(rank.term, dict(zip(names, sp)))):
                    return (s, sp)
            except Unevaluable:
                continue
        return None

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
        read = set(free_constants([rank.term]))
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

        The closure is `_closure`, and it is taken **again over what it
        returned** until a round cuts nothing. One round is not a fixed
        point, because a core names what the refutation happened to touch
        and a pool of 87 facts offers more to touch than the 35 that come
        back: asking the same question of the smaller pool is a different
        refutation, and on measurement a much tighter one. The rounds
        shrink strictly or stop, so there are at most as many as there are
        facts, and each costs a handful of calls against `_minimise`'s one
        per fact -- which is what makes this the half that carries a
        refutation-only prover, whose every unproved call costs a whole time
        limit.

        The set that comes out is inductive by those proofs, and it is
        checked once more on its own before it replaces `inv` -- a core
        over-trimmed would otherwise surface as a Lean failure, and neither
        solver promises a core is one its goal needs. Anything that does not
        come back proved keeps `inv` whole.
        """
        pool = inv
        while True:
            cut = self._closure(prover, pool, need, limit)
            # `None` is a round that could not answer, and it leaves the
            # round before it standing: what that one cut was closed under
            # its own proofs and is not made wrong by a later time limit.
            if cut is None or len(cut) >= len(pool):
                break
            pool = cut
            if not pool:
                return pool
        if len(pool) == len(inv):
            return inv
        if prover.prove(self.ob.preserved(pool, pool), limit).proved:
            self.log(f"[houdini] invariant cut to the {len(pool)} of "
                     f"{len(inv)} facts its proofs use")
            return pool
        return inv

    def _closure(self, prover: Solver, pool: list[Candidate],
                 need: list[Candidate], limit: float) -> "list[Candidate] | None":
        """`need` closed under "a round preserves it", drawing from `pool`.

        Each proof is asked from all of `pool` and its core names the facts
        it used; those join `need` until a proof uses nothing new. `None` is
        "no answer" -- a proof that did not come back, or one with no core
        -- and leaves the caller with what it had.
        """
        need = list(dict.fromkeys(need))
        while need:
            answer = prover.prove(self.ob.preserved(pool, need), limit,
                                  core=True)
            if not answer.proved or answer.core is None:
                return None
            more = [f for f in self._named(pool, answer.core) if f not in need]
            if not more:
                break
            need += more
        return need

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
                               seconds=SAMPLE_SECONDS / 4)
        for s, sp in rounds:
            if any(not self._holds(f, sp) for f in facts):
                return True
            if rank is not None:
                try:
                    before = self.ev(rank.term, dict(zip(self.ctx.names, s)))
                    after = self.ev(rank.term, dict(zip(self.ctx.names, sp)))
                except Unevaluable:
                    continue
                if not drops(before, after):
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

    def _conjuncts(self, ctx: SynthContext, claim: Claim) -> list[str]:
        """The property's top-level conjuncts, seeded as facts of their own.

        Split because Houdini keeps facts, not formulas: `(and a b)` is kept
        whole or dropped whole, where `a` alone may be inductive and `b`
        inductive given it.
        """
        p = ctx.prp
        parts = list(p) if p.getKind() == Kind.AND else [p]
        return [str(t) if len(parts) > 1 else claim.prp.strip() for t in parts]

    def _parse(self, srcs: list[str], role: str = "invariant",
               want: str = "Bool") -> list[Candidate]:
        """The candidates that read as a `want` over the module, through the
        run's encoding. One that does not is no candidate, not an error:
        the generators propose shapes, and a shape may not fit."""
        out = []
        for src in dict.fromkeys(srcs):
            try:
                out.append(Candidate(src, self.ctx.enc.parse(src, role, want=want)))
            except PredicateRefused:
                continue
        return out

    # --- out --------------------------------------------------------------

    def _emit(self, inv: list[Candidate],
              rank: "Candidate | None") -> Certificate:
        inv_src = conjunction_smt(f.src for f in inv)
        self.log(f"[houdini] inv: {inv_src}")
        if rank is not None:
            self.log(f"[houdini] ranking: {rank.src}")
        self._record(inv_src, rank)
        return Certificate(inv_src, None if rank is None else rank.src)

    def _record(self, inv_src: str, rank: "Candidate | None") -> None:
        by = self.spec.kind
        what = (f"Houdini over facts read off simulated runs of the module, "
                f"each proved by {by} -- it holds at entry and every round "
                f"preserves it -- and cut down to the facts the proofs used.")
        found(self.artifacts, "inv", inv_src,
              what=f"The invariant this run found. {what}")
        if rank is not None:
            found(self.artifacts, "ranking", rank.src,
                  what=f"The ranking function this run found: {by} proved "
                       f"`hrank` for it over the invariant beside it.")

    def _none(self, claim: Claim, n_facts, n_ranks, reached, prover):
        if claim.is_safety:
            what, article = "inductive invariant implying the property", "an"
            started = f"Houdini started from {n_facts} facts the simulated runs keep"
        else:
            what, article = "ranking function", "a"
            started = (f"Houdini started from {n_facts} facts the simulated "
                       f"runs keep, and at most {n_ranks} ranking functions "
                       f"dropped on every reached and sampled round")
        refuted = (f", {prover.refuted} of them refuted"
                   if prover.refutes else "")
        if not claim.is_safety and not n_ranks:
            started += f", so no ranking function was put to {prover.name}"
        # What was spent, and the per-call limit the last call ran under --
        # which used to be printed as "the time limit reached 120 s" on runs
        # that took under a second, because with no candidate to ask about
        # every rung of the ladder is passed at once.
        detail = (
            f"{started}; {prover.calls} {prover.name} calls took "
            f"{prover.spent:.1f} s, the last under a {reached:g} s limit"
            f"{refuted}. {prover.inconclusive}"
        )
        note(
            self.artifacts,
            f"--infer houdini found no {what}. {detail}\n",
            status="unknown",
            what=f"A search for {article} {what} that did not succeed.",
            why=(f"{prover.name} decided every candidate it was asked "
                 f"about; the shapes proposed were the limit."
                 if prover.refutes else
                 f"{prover.name} proves or times out; nothing here was "
                 f"refuted."),
        )
        raise Refused(f"--infer houdini found no {what}. {detail}",
                      searched=True)
