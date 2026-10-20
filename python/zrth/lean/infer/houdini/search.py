"""One run's search, over what the simulation found (`Search`): Houdini's
fixpoint over the candidate facts, the ranking functions fitted to the
rounds and put to the prover, and the certificate cut down to what the
proofs used.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from ...core.errors import PredicateRefused
from ...smt.claim import SynthContext
from ...smt.obligations import Candidate, Obligations
from ...smt.solvers import Solver
from ...smt.terms import free_constants
from ..common.synth import Column
from .evaluate import Evaluator, Unevaluable
from .ranks import Ranks, drops
from .simulate import SAMPLE_SECONDS, Draws, Runs, sample_rounds


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


@dataclass(frozen=True)
class Search:
    """Everything a step of the search reads, fixed when the search starts:
    the context and the obligations stated over it, the evaluator, the
    columns, the simulated runs, the two input draws and the pinned
    values a floored rank needs kept. Each step is a method, handed the
    prover and the time limit of the rung it runs at.

    The draws are the one part that moves: their generators advance with
    every round they sample.
    """

    ctx: SynthContext
    ob: Obligations
    ev: Evaluator
    cols: list[Column]
    runs: Runs
    draws: Draws
    wide: Draws
    pins: dict[str, str]
    log: Callable = print

    def holds(self, fact: Candidate, state) -> bool:
        """Whether `fact` holds at `state`. A fact with no value there is
        not refuted by it."""
        try:
            return bool(self.ev(fact.term, dict(zip(self.ctx.names, state))))
        except Unevaluable:
            return True

    def houdini(self, prover: Solver, facts: list[Candidate],
                limit: float) -> list[Candidate]:
        """The largest subset of `facts` the solver proves inductive at `limit`.

        Each pass first drops what a sampled round refutes -- a state every
        kept fact holds at, whose successor breaks one -- which is Houdini's
        own step taken without a solver, and then asks about the rest.

        One question is asked per pass, not one per fact, and what happens
        when it comes back false is where the solvers part. A counter-model
        *is* the pass's next step -- a state the kept facts admit whose
        successor breaks some of them, named by `survivors` and dropped
        together -- so cvc5 converges in one call per pass. Vampire has no
        model to read, so the pass falls back to asking about each fact on
        its own, which is where the old cost per pass came from.
        """
        ob = self.ob
        if not facts:
            return []
        kept = self.at_entry(prover, list(facts), limit)
        while kept and prover.left() >= 0.5:
            kept = self._refuted_by_sampling(kept)
            if not kept:
                break
            step = prover.prove(ob.preserved(kept, kept), limit, model=True)
            if step.proved:
                return kept
            survivors = self.survivors(
                prover, step, kept,
                lambda fs: [ob.preserved(fs, [f]) for f in fs], limit)
            if len(survivors) == len(kept):
                # Nothing to drop and the conjunction was not proved: the
                # solver ran out of time on it, not out of facts.
                return kept
            kept = survivors
        # Out of budget mid-pass: what is left was not shown inductive.
        return [] if kept and prover.left() < 0.5 else kept

    def at_entry(self, prover: Solver, kept: list[Candidate],
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
            survivors = self.survivors(
                prover, entry, kept,
                lambda fs: [ob.holds_at_entry([f]) for f in fs], limit)
            if len(survivors) == len(kept):
                break               # nothing to drop: unanswered, not false
            kept = survivors
        return kept

    def survivors(self, prover: Solver, answer, kept: list[Candidate],
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
                lambda s: all(self.holds(f, s) for f in kept))
            broken = {i for i, f in enumerate(kept)
                      if any(not self.holds(f, sp) for _, sp in rounds)}
            if not broken:
                break
            kept = [f for i, f in enumerate(kept) if i not in broken]
        return kept

    def _fitted(self, ranks: Ranks, inv: list[Candidate],
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
            lambda s: not ranks.holds(s) and all(self.holds(f, s) for f in inv))
        wide = sample_rounds(
            self.ctx, self.ob, self.ev, self.wide, self.runs.states,
            lambda s: not ranks.holds(s) and all(self.holds(f, s) for f in inv))
        # A wide round no fitted shape drops on is fitted to in turn: the
        # shift of a shape is read off the rounds it was fitted to, and the
        # round that decides it -- one crossing between two modes from
        # exactly a bound -- is one a draw finds by luck.
        learnt: list = [r for r in seen
                        if all(self.holds(f, r[0]) for f in inv)]
        for _ in range(_REFITS + 1):
            started = time.monotonic()
            fitted = self.parse(ranks.fit(list(self.runs.rounds) + sampled,
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

    def ranking(self, prover: Solver, ranks: Ranks, inv: list[Candidate],
                learnt: list, limit: float) -> "tuple[Candidate | None, int]":
        """The first fitted ranking function `prover` proves `hrank` for
        over `inv` at `limit`, or `None`; and how many were asked about.

        A refuted shape teaches the fit: the rounds the prover refuted the
        first few with are fitted to again, up to `_REFITS` times inside
        `_REFIT_SECONDS`. `learnt` is those rounds, which the caller keeps
        across rungs -- a round cvc5 refuted a shape with is a
        counterexample at every limit -- and this adds to.
        """
        at, candidates, tried = None, [], 0
        fit_until = time.monotonic() + _REFIT_SECONDS
        for _ in range(_REFITS + 1):
            started = time.monotonic()
            candidates = self._fitted(ranks, inv, learnt, fit_until)
            tried = max(tried, len(candidates))
            if not candidates:
                break
            at = prover.first([self.ob.drops(inv, r) for r in candidates],
                              limit)
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
            learnt.extend(fresh)
        return (None if at is None else candidates[at]), tried

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

    def pinning(self, inv: list[Candidate], rank: Candidate) -> list[Candidate]:
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

    def property_rests_on(self, prover, inv, conjuncts,
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
        return named(inv, answer.core) if answer.proved else None

    # --- cutting the certificate down ---------------------------------------

    def shrink(self, prover: Solver, inv: list[Candidate],
               need: list[Candidate], limit: float) -> list[Candidate]:
        """The facts `need` rests on, closed under the preservation proofs.

        The closure is `_closure`, and it is taken **again over what it
        returned** until a round cuts nothing. One round is not a fixed
        point, because a core names what the refutation happened to touch
        and a pool of 87 facts offers more to touch than the 35 that come
        back: asking the same question of the smaller pool is a different
        refutation, and on measurement a much tighter one. The rounds
        shrink strictly or stop, so there are at most as many as there are
        facts, and each costs a handful of calls against `minimise`'s one
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
            more = [f for f in named(pool, answer.core) if f not in need]
            if not more:
                break
            need += more
        return need

    def minimise(self, prover: Solver, inv: list[Candidate],
                 keep: list[str], ranked) -> list[Candidate]:
        """`inv` with each fact the rest do without taken out, one at a time.

        An unsat core is whatever the refutation happened to touch, not the
        least it could have, so what `shrink` returns can still carry facts
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
            return all(self.holds(f, s) for f in facts)

        rounds = sample_rounds(self.ctx, self.ob, self.ev, self.draws,
                               self.runs.states, admitted, want=100,
                               seconds=SAMPLE_SECONDS / 4)
        for s, sp in rounds:
            if any(not self.holds(f, sp) for f in facts):
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

    def parse(self, srcs: list[str], role: str = "invariant",
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


def named(facts: list[Candidate], core) -> list[Candidate]:
    """The facts an unsat core names: `h<i>` is the `i`th hypothesis, the
    order they were asked in. No core is no answer, and keeps them all."""
    if core is None:
        return list(facts)
    return [f for i, f in enumerate(facts) if f"h{i}" in core]
