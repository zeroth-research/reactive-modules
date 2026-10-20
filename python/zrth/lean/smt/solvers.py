"""The judge a Houdini search puts its candidates to.

``--infer houdini`` proposes and a solver decides.  *What* it proposes --
the invariant facts read off simulated runs of the module, the ranking
shapes tried smallest first -- is :mod:`zrth.lean.infer.houdini`
(``candidates``, ``ranks``), and is the same whichever solver answers.
This module is the seam between the two: an obligation as a *question*
rather than as a file, and the solvers that answer one.

**A query names more than it asks.**  Besides its hypotheses and its goal a
:class:`Query` names two things, so that a solver able to report them can:

* the hypotheses are *named*, so that a refutation's unsat core says which
  of them the proof used and the certificate can be cut down to those.
  Both solvers do this, and it is why the invariant that reaches Lean is
  the small one rather than everything Houdini kept.
* the *probes* are the goal's conjuncts, named one by one, so that a
  counter-model says which of them it falsifies.  Only cvc5 does this, and
  it is the difference between one call and one call per fact.

A solver fills in what it has.  Nothing in the engine asks which solver it
is holding: it asks for a core and for a model, and reads whichever came
back.

**An answer is three-valued, and that is the whole reason to have cvc5
here.**  Vampire is a refutation prover.  Handed the negation of an
obligation it either refutes it -- the obligation holds, and the refutation
is the warrant -- or it runs out of time, and those two failures are
indistinguishable from outside.  cvc5 *decides*: `sat` is a counterexample,
which says the candidate is false at every time limit there will ever be.
:class:`Verdict` is that distinction, and it is what the cache is keyed on:
a `PROVED` and a `REFUTED` are kept forever, an `UNKNOWN` only against the
limit it was reached at, so a later rung re-asks exactly what the last one
could not answer.

**Both solvers climb a ladder of time limits, for different reasons.**
Vampire's portfolio divides its limit among its strategies, so a short limit
is a different schedule rather than a truncated long one and a proof one
limit finds another may not.  cvc5's search at a longer limit simply
contains the shorter one -- but a single query must still not be able to eat
a whole budget, so its rungs are per-call caps: a cheap pass that answers
essentially everything, then one with the rest of the budget for whatever
came back `UNKNOWN`.  Repeating a pass is nearly free either way, because
the cache re-asks only what is unsettled.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from ..core.errors import Refused, require
from .obligations import Query
from .limits import bounded_solver
from .script import decimals, script
from .vampire import run_vampire

DEFAULT_TIMEOUT = 120
DEFAULT_CORES = 4


# ══════════════════════════════════════════════════════════════════════════
# The question
# ══════════════════════════════════════════════════════════════════════════


class Verdict(Enum):
    """What a solver settled about one obligation.

    `REFUTED` is the one a first-order prover cannot reach, and the one
    worth the most: it holds at every time limit, so the candidate behind it
    is gone for the rest of the run rather than retried at the next rung.
    """

    PROVED = "proved"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Answer:
    """What came back, and how much of what was asked for it carries."""

    verdict: Verdict
    secs: float = 0.0
    # The named hypotheses the refutation used, when a core was asked for
    # and the solver gave one.
    core: "frozenset[str] | None" = None
    # The named probes the counter-model falsifies, when a model was asked
    # for and the solver gave one. Empty is not the same as `None`: `None`
    # is "no model", and the caller falls back to asking fact by fact.
    broken: "frozenset[str] | None" = None
    # The query's `witness` terms at the counter-model, when a model was
    # asked for and the solver gave one.
    values: "tuple | None" = None

    @property
    def proved(self) -> bool:
        return self.verdict is Verdict.PROVED

    @property
    def refuted(self) -> bool:
        return self.verdict is Verdict.REFUTED

    def serves(self, *, core: bool, model: bool) -> bool:
        """Whether this settles the question *and* carries what was asked.

        An `UNKNOWN` never serves -- it is the absence of an answer, and the
        ledger of limits is what keeps it from being asked again too soon.
        """
        if self.verdict is Verdict.PROVED:
            return self.core is not None or not core
        if self.verdict is Verdict.REFUTED:
            return (self.broken is not None or self.values is not None
                    or not model)
        return False


# ══════════════════════════════════════════════════════════════════════════
# The solver
# ══════════════════════════════════════════════════════════════════════════


class Solver:
    """A judge on this run's deadline, with what it has already answered.

    The scheduling is here and the answering is in the subclass: the
    deadline, the ladder of limits, the cache, and running several queries
    side by side are the same question whoever answers it.
    """

    name = "?"
    # Whether a failed proof can come back as `REFUTED` -- that is, whether
    # this solver decides or only refutes. The engine reads it to know
    # whether an exhausted search is evidence of anything.
    refutes = False
    # What a search that found nothing does and does not mean, for the
    # refusal that reports one.
    inconclusive = ""
    # The rungs below the final one, in seconds; the final rung is whatever
    # is left of the budget. The whole of the difference between the two
    # solvers' scheduling is this tuple.
    LADDER: tuple[float, ...] = ()

    def __init__(self, *, seconds: float, log=print):
        self.deadline = time.monotonic() + seconds
        self.log = log
        self.jobs = 1
        self.calls = 0
        self.refuted = 0
        self.spent = 0.0
        self._answered: dict = {}
        self._gave_up: dict = {}

    # --- what the engine drives -------------------------------------------

    def ladder(self, total: float) -> tuple[float, ...]:
        """The per-call time limits a search runs at, ending at `total`."""
        return tuple(t for t in self.LADDER if t < total) + (float(total),)

    @property
    def short(self) -> float:
        """The cheapest rung: what a query worth asking only if it is cheap
        gets. Minimising a certificate is the one that asks on those terms."""
        return float(self.LADDER[0]) if self.LADDER else 2.0

    def left(self) -> float:
        return self.deadline - time.monotonic()

    def prove(self, q: Query, limit: float, *,
              core: bool = False, model: bool = False) -> Answer:
        hit = self._answered.get(q.key)
        if hit is not None and hit.serves(core=core, model=model):
            return hit
        if hit is None and self._gave_up.get(q.key, 0.0) >= limit:
            return Answer(Verdict.UNKNOWN)
        budget = min(limit, self.left())
        if budget < 0.5:
            return Answer(Verdict.UNKNOWN)
        answer = self._run(q, budget, core, model)
        if answer.verdict is Verdict.UNKNOWN:
            self._gave_up[q.key] = max(self._gave_up.get(q.key, 0.0), limit)
        else:
            self._answered[q.key] = answer
            self.refuted += answer.refuted
        return answer

    def prove_all(self, queries: list[Query], limit: float) -> list[Answer]:
        """Several queries, `jobs` at a time.

        No `model` here on purpose: the callers that batch are asking about
        one fact per query, where the goal is a single thing and `REFUTED`
        already says everything a model would.
        """
        if len(queries) <= 1 or self.jobs == 1:
            return [self.prove(q, limit) for q in queries]
        with ThreadPoolExecutor(self.jobs) as pool:
            return list(pool.map(lambda q: self.prove(q, limit), queries))

    def first(self, queries: list[Query], limit: float) -> "int | None":
        """The index of the first query proved, trying `jobs` at a time."""
        for at in range(0, len(queries), self.jobs):
            if self.left() < 0.5:
                return None
            batch = queries[at:at + self.jobs]
            for i, answer in enumerate(self.prove_all(batch, limit)):
                if answer.proved:
                    return at + i
        return None

    # --- what a subclass answers ------------------------------------------

    def _run(self, q: Query, budget: float, core: bool, model: bool) -> Answer:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════
# cvc5: in this process, and it decides
# ══════════════════════════════════════════════════════════════════════════


class Cvc5Solver(Solver):
    """cvc5 on the terms themselves -- nothing is printed and nothing forked.

    The obligations are quantifier-free: the inputs are free constants, so
    the universal quantification an obligation means is the free variables,
    and what is left is linear integer and real arithmetic with `to_int`,
    `div`, `mod` and `ite`. cvc5 decides that, which is why `sat` here is a
    counterexample rather than a shrug.

    One query is one fresh `cvc5.Solver`. They are cheap -- measured in
    milliseconds against the tens of seconds a `lake build` costs -- and a
    fresh one is what makes `tlimit` mean this call's limit rather than a
    running total.
    """

    name = "cvc5"
    refutes = True
    inconclusive = (
        "cvc5 decides these obligations, so a candidate it answered was "
        "either proved or refuted: what is missing is a fact or a shape "
        "outside the candidates -- a disjunction, a relation of three "
        "components, a constant the program never mentions -- rather than "
        "more time, unless the log shows obligations it left unknown."
    )

    # A cheap pass answers essentially everything; the final rung is for
    # whatever came back unknown, which in this theory is rare. Not a
    # portfolio reshuffle, as Vampire's rungs are -- only a cap that stops
    # one query eating a whole budget.
    LADDER = (2,)

    def __init__(self, tm, *, seconds: float = DEFAULT_TIMEOUT, log=print):
        super().__init__(seconds=seconds, log=log)
        # Lazily, so that the route table above can name this class without
        # a bare `verith` run paying for the import.
        self.kind = require("cvc5", "--houdini-solver cvc5").Kind
        self.tm = tm

    def _run(self, q: Query, budget: float, core: bool, model: bool) -> Answer:
        tm, Kind = self.tm, self.kind
        t0 = time.perf_counter()
        options = {}
        if core:
            options["produce_unsat_cores"] = "true"
        if model and (q.probes or q.witness):
            options["produce_models"] = "true"
        solver = bounded_solver(tm, max(1, int(budget * 1000)), logic="ALL",
                                **options)
        try:
            # The round, as equalities rather than `define-fun`: the same
            # constraint, and it leaves `v_sp*` as constants the model can
            # be read at.
            for v, body in q.defines:
                solver.assertFormula(tm.mkTerm(Kind.EQUAL, v, body))
            for _n, t in q.hyps:
                solver.assertFormula(t)
            solver.assertFormula(tm.mkTerm(Kind.NOT, q.goal))
            result = solver.checkSat()
            if result.isUnsat():
                answer = Answer(Verdict.PROVED, 0.0,
                                self._core(solver, q) if core else None)
            elif result.isSat():
                answer = Answer(Verdict.REFUTED, 0.0, None,
                                self._broken(solver, q) if model else None,
                                self._values(solver, q) if model else None)
            else:
                answer = Answer(Verdict.UNKNOWN)
        except RuntimeError as e:
            # Total, the way `smt.precheck` is total: the terms came out of
            # cvc5's own encoder, so anything raised here is a resource the
            # run does not have rather than a question it asked wrongly.
            self.log(f"[houdini] cvc5 could not answer a query: {e}")
            answer = Answer(Verdict.UNKNOWN)
        dt = time.perf_counter() - t0
        self.calls += 1
        self.spent += dt
        return Answer(answer.verdict, dt, answer.core, answer.broken,
                      answer.values)

    @staticmethod
    def _core(solver, q: Query) -> "frozenset[str] | None":
        """The named hypotheses in the refutation, matched back by term."""
        try:
            used = list(solver.getUnsatCore())
        except RuntimeError:                         # pragma: no cover
            return None
        return frozenset(n for n, t in q.hyps if any(t == c for c in used))

    @staticmethod
    def _values(solver, q: Query) -> "tuple | None":
        """The `witness` terms at this counter-model, as Python values: a
        Fraction for a Real, an int, a bool. None if any cannot be read."""
        if not q.witness:
            return None
        out = []
        try:
            for t in q.witness:
                v = solver.getValue(t)
                sort = t.getSort()
                if sort.isBoolean():
                    out.append(v.getBooleanValue())
                elif sort.isInteger():
                    out.append(v.getIntegerValue())
                elif sort.isReal():
                    out.append(Fraction(v.getRealValue()))
                else:
                    return None
        except (RuntimeError, AttributeError):       # pragma: no cover
            return None
        return tuple(out)

    def _broken(self, solver, q: Query) -> "frozenset[str] | None":
        """The named probes this counter-model falsifies.

        A probe that cannot be read back is left *unbroken* -- the caller
        drops what is named, so the safe direction is to name less. That
        makes an empty answer to a refuted query the signal it should be:
        nothing to drop, so ask fact by fact instead.
        """
        if not q.probes:
            return None
        out = set()
        for name, term in q.probes:
            try:
                if not solver.getValue(term).getBooleanValue():
                    out.add(name)
            except (RuntimeError, AttributeError):   # pragma: no cover
                continue
        return frozenset(out)


# ══════════════════════════════════════════════════════════════════════════
# Vampire: a printed script, and one-sided answers
# ══════════════════════════════════════════════════════════════════════════


class VampireSolver(Solver):
    """Vampire on a printed script, in its own process session.

    Everything here is downstream of the one-sidedness. A candidate that is
    false costs a whole time limit, because nothing comes back early to say
    so; `sat` is never reported and a model never exists, so a query's
    probes are ignored and the engine falls back to asking fact by fact.
    """

    name = "vampire"
    refutes = False
    inconclusive = (
        "Vampire cannot refute a candidate, so this is not a proof that "
        "none exists: a fact or a shape outside the candidates -- a "
        "disjunction, a relation of three components, a constant the "
        "program never mentions -- or a longer time limit may be what is "
        "missing. `--houdini-solver cvc5` decides these obligations, and "
        "would say which."
    )

    LADDER = (2, 10, 60)

    def __init__(self, exe: str, *, seconds: float = DEFAULT_TIMEOUT,
                 cores: int = DEFAULT_CORES, log=print):
        super().__init__(seconds=seconds, log=log)
        self.exe = exe
        self.cores = max(1, int(cores))
        # Portfolio calls side by side. Each spreads its strategies over
        # `cores` processes; more calls than the machine has cores for would
        # only make every limit mean less time.
        self.jobs = max(1, min(4, (os.cpu_count() or 4) // self.cores))

    def _run(self, q: Query, budget: float, core: bool, model: bool) -> Answer:
        run = run_vampire(self.exe, self._script(q), [
            "--mode", "portfolio", "--schedule", "smtcomp",
            "--cores", str(self.cores),
            "--output_mode", "ucore" if core else "smtcomp",
            # Deciseconds: a limit below a second is a real limit.
            "--time_limit", f"{max(1, int(budget * 10))}d",
        ], timeout=budget + 10)
        dt = run.secs
        self.calls += 1
        self.spent += dt
        lines = run.lines
        if "unsat" in lines:
            names = None
            if core:
                at = lines.index("unsat") + 1
                reported = frozenset(
                    ln for ln in lines[at:] if ln and ln not in "()"
                    and not ln.startswith("%")
                )
                names = frozenset(n for n, _t in q.hyps if n in reported)
            return Answer(Verdict.PROVED, dt, names)
        error = run.error
        if error is not None:
            detail = run.out[run.out.find(error):].strip().splitlines()
            raise Refused(
                f"Vampire cannot read this module's encoding: "
                f"{' '.join(detail[:2])[:300]}. `--houdini-solver cvc5` reads "
                f"what this module writes, because it is what wrote it."
            )
        # Not `REFUTED`: out of time and false are the same output here.
        return Answer(Verdict.UNKNOWN, dt)

    @staticmethod
    def _script(q: Query) -> str:
        """The query as a script whose `unsat` is a proof.

        The goal is asserted negated. Hypotheses are named so that an unsat
        core can say which of them the proof used; the goal is named too, so
        a core is never empty for a reason other than the goal being false.
        """
        defines = [(str(v), (), v.getSort(), body) for v, body in q.defines]
        asserts = [f"(! {t} :named {n})" for n, t in q.hyps]
        asserts.append(f"(! (not {q.goal}) :named goal)")
        mentions = ([body for _v, body in q.defines]
                    + [t for _n, t in q.hyps] + [q.goal])
        return decimals(script(asserts, defines=defines, mentions=mentions,
                               spaced=False))
