"""A template's coefficients solved for (`decide`): what a `Template`
states, answered `sat` with the values, `unsat` -- no coefficients do -- or
`unknown`.

Two ways of answering the same question, and the second exists
because of a measurement: for a BitVec column cvc5 answers the
quantified form `unknown (INCOMPLETE)` in *one millisecond* -- not a
timeout, a refusal to state it. So a module whose state is a
bitvector cannot be reached by the direct query at all, and the same
question has to be asked in a form that stays quantifier-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.errors import Refused
from ...smt.claim import SynthContext
from ...smt.limits import bounded_solver, call_ms
from ..common.synth import program_constants

from cvc5 import Kind


# How many counterexamples the fallback loop may collect before it gives up.
# It is not the real leash -- the run's budget is, and it is checked every
# iteration -- but a loop that cannot terminate needs a number. Measured: an
# 8-bit counter's invariant takes 55 samples, so the cap is well clear of
# what a small module needs and a big one runs out of budget long before it.
DEFAULT_SAMPLES = 256

HOW_QUANTIFIED = (
    "the query is `exists c. forall s. obligations(c)` over the "
    "coefficients, and cvc5 came back unsat"
)
_HOW_CEGIS = (
    "cvc5 cannot state that query for this module, so the same question ran "
    "as a counterexample-guided loop -- propose coefficients that fit the "
    "states seen so far, verify against the whole transition, add the "
    "counterexample -- and the proposal became unsatisfiable after "
    "{samples} counterexamples, which is the same proof one sample set at "
    "a time"
)


@dataclass(frozen=True)
class Template:
    """A shape with its coefficients left open, and what they must satisfy.

    `build` states the obligations for *given* state and input terms, which
    is what lets one template serve both engines: the quantified query
    passes bound variables and wraps the result in a `forall`, and the
    counterexample-guided loop passes concrete values and fresh constants.
    """

    unknowns: list                  # the cvc5 Int constants to solve for
    build: object                   # (state, extl_l, extl_n) -> Bool term


@dataclass(frozen=True)
class Outcome:
    """What a template was answered, and what the answer is a claim about."""

    answer: str                     # "sat" | "unsat" | "unknown"
    values: "list[int] | None" = None
    how: str = HOW_QUANTIFIED
    caveat: str = ""                # what narrows an `unsat`, if anything


def decide(ctx: SynthContext, tpl: Template, *, budget,
           log=print) -> Outcome:
    """Solve `tpl`, by whichever engine can state it.

    The quantified query first: it is one call, it is decided over
    *every* integer coefficient, and for an integer state it answers in
    milliseconds. Only when cvc5 will not state it does the loop run,
    and what that loop proves is narrower -- which is why the verdict
    carries the caveat rather than the caller having to remember.
    """
    verdict = _quantified(ctx, tpl, budget)
    if verdict.answer != "unknown":
        return verdict
    bound = _coefficient_bound(ctx)
    log(
        f"[smt-linear] cvc5 will not state the quantified query for this "
        f"module; searching coefficients in [-{bound}, {bound}] by "
        f"counterexample instead"
    )
    return _cegis(ctx, tpl, bound, budget)


def _quantified(ctx: SynthContext, tpl: Template, budget) -> Outcome:
    """`exists c. forall s, inputs. obligations(c)`, in one call."""
    tm = ctx.tm
    solver = bounded_solver(tm, call_ms(budget), produce_models="true")
    solver.setLogic("ALL")
    st, el, en = ctx.enc.bound("q_")
    body = tpl.build(st, el, en)
    quantified = st + el + en
    solver.assertFormula(
        tm.mkTerm(Kind.FORALL,
                  tm.mkTerm(Kind.VARIABLE_LIST, *quantified), body)
        if quantified else body
    )
    try:
        result = solver.checkSat()
    except RuntimeError as e:
        raise Refused(
            f"cvc5 could not state this module's transition against a "
            f"linear template: {e}"
        ) from e
    if result.isSat():
        return Outcome("sat", [_value(solver, x)
                               for x in tpl.unknowns])
    return Outcome("unsat" if result.isUnsat() else "unknown")


def _cegis(ctx: SynthContext, tpl: Template, bound: int,
           budget) -> Outcome:
    """The same question as a loop of quantifier-free queries.

    *Propose* the coefficients against the states seen so far -- with
    the state concrete every product `c * value` is a literal multiple,
    so this is linear arithmetic whatever the module's own sorts are --
    then *verify* the proposal against the whole transition, which is
    quantifier-free once the coefficients are concrete. A
    counterexample becomes the next sample.

    Both answers are still proofs. `unsat` at the propose step means no
    coefficients fit even finitely many states, so none fit everywhere;
    `unsat` at the verify step is the obligation holding for every
    state there is. What is *not* the same is the scope of the first
    one: the loop needs a finite candidate space to terminate, so the
    coefficients are bounded, and `caveat` says so wherever the answer
    is written down.
    """
    tm = ctx.tm
    samples: list = []
    for step in range(DEFAULT_SAMPLES):
        if budget is not None and budget.exhausted:
            return Outcome("unknown")
        propose = bounded_solver(tm, call_ms(budget),
                                 produce_models="true")
        propose.setLogic("ALL")
        for x in tpl.unknowns:
            propose.assertFormula(
                tm.mkTerm(Kind.LEQ, x, tm.mkInteger(bound)))
            propose.assertFormula(
                tm.mkTerm(Kind.GEQ, x, tm.mkInteger(-bound)))
        for sample in samples:
            propose.assertFormula(tpl.build(*sample))
        answer = propose.checkSat()
        if answer.isUnsat():
            return Outcome(
                "unsat",
                how=_HOW_CEGIS.format(samples=step),
                caveat=_caveat(bound),
            )
        if not answer.isSat():
            return Outcome("unknown")
        values = [propose.getValue(x) for x in tpl.unknowns]

        verify = bounded_solver(tm, call_ms(budget),
                                produce_models="true")
        verify.setLogic("ALL")
        st, el, en = ctx.enc.fresh(f"cex{step}_")
        verify.assertFormula(
            tm.mkTerm(Kind.NOT,
                      tpl.build(st, el, en).substitute(tpl.unknowns, values))
        )
        refuted = verify.checkSat()
        if refuted.isUnsat():
            return Outcome(
                "sat",
                [int(v.getIntegerValue()) for v in values],
                how=_HOW_CEGIS.format(samples=step),
                caveat=_caveat(bound),
            )
        if not refuted.isSat():
            return Outcome("unknown")
        samples.append(
            tuple([verify.getValue(x) for x in group]
                  for group in (st, el, en))
        )
    return Outcome("unknown")


def _coefficient_bound(ctx: SynthContext) -> int:
    """How far the loop's coefficients may range.

    Read off the program, like the grammar route's constants and for the
    same reason: the coefficient that makes an invariant true is a bound
    the module already mentions. Room to spare above it, because a
    coefficient is not always a literal -- and not much room, because
    the bound *is* the thing that makes the loop converge. Measured on
    an 8-bit counter: 55 samples inside a tight bound, and no answer in
    200 with a loose one.
    """
    consts = program_constants(ctx)
    return max(16, 2 * max((abs(c) for c in consts), default=0) + 2)


def _caveat(bound: int) -> str:
    return (
        f" The coefficients were searched in [-{bound}, {bound}]: the "
        f"counterexample loop needs a finite candidate space, so this is "
        f"a proof about that space rather than about every integer."
    )


def _value(solver, term) -> int:
    return int(solver.getValue(term).getIntegerValue())
