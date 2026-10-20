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

from ...artifacts.record import found, note
from ...core.certificate import Certificate, Claim
from ...core.errors import Refused
from ...smt.claim import SynthContext
from ...smt.obligations import Candidate, Obligations
from ...smt.script import conjunction_smt
from ...smt.solvers import DEFAULT_TIMEOUT
from ..common.synth import (
    columns,
    denominator_scale,
    program_constants,
    program_rationals,
    split_conditions,
)
from ..engine import Engine
from .candidates import invariant_candidates, pinned_values, reading
from .evaluate import Evaluator
from .modes import mode_bounds
from .ranks import Ranks
from .search import Search, named
from .simulate import Draws, simulate
from .solvers import DEFAULT_SOLVER, SolverSpec

try:
    import cvc5                                      # type: ignore
    from cvc5 import Kind
except ImportError:                                  # pragma: no cover
    cvc5 = None
    Kind = None


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
        cols = columns(ctx)
        ob = Obligations(ctx)
        ev = Evaluator(ctx.tm)
        constants = program_constants(ctx)
        rationals = program_rationals(ctx)
        scale = denominator_scale(rationals)
        self.log(f"[houdini] solver: {self.spec.kind}")
        self.log("[houdini] columns: " + ", ".join(reading(c) for c in cols))
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
        search = Search(
            ctx, ob, ev, cols, runs,
            draws=Draws(ctx, constants, seed=1, rationals=rationals,
                        scale=scale),
            wide=Draws(ctx, constants, seed=2, wide=True, rationals=rationals,
                       scale=scale),
            pins=pinned_values(ctx, runs),
            log=self.log,
        )

        conjuncts = self._conjuncts(ctx, claim) if claim.is_safety else []
        # The computed bounds go last: the minimiser drops from the end, so
        # a certificate the read ones carry keeps the program's constants.
        facts = search.parse(invariant_candidates(ctx, runs, constants,
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
            inv = search.houdini(prover, facts, limit)
            self.log(f"[houdini] Houdini kept {len(inv)} of {len(facts)}: "
                     + " ".join(f.src for f in inv))
            if claim.is_safety:
                need = search.property_rests_on(prover, inv, conjuncts, limit)
                if need is not None:
                    inv = search.shrink(prover, inv, need, limit)
                    found = (search.minimise(prover, inv, conjuncts, None), None)
                    break
                continue
            rank, asked = search.ranking(prover, ranks, inv, learnt, limit)
            tried = max(tried, asked)
            if rank is not None:
                self.log(f"[houdini] ranking function proved: {rank.src}")
                core = prover.prove(ob.drops(inv, rank), limit, core=True).core
                pins = search.pinning(inv, rank) if ranks.real else []
                need = named(inv, core) + pins
                inv = search.shrink(prover, inv, need, limit)
                found = (search.minimise(prover, inv, [f.src for f in pins],
                                         (ranks, rank)), rank)
                break
        refuted = f", {prover.refuted} refuted" if prover.refutes else ""
        self.log(f"[houdini] {prover.calls} {prover.name} calls{refuted}, "
                 f"{prover.spent:.1f} s")
        if found is None:
            return self._none(claim, len(facts), tried, reached, prover)
        inv, rank = found
        return self._emit(inv, rank)

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
