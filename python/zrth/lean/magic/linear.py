"""TA2Magic by template: the certificate cvc5 can find in one quantified query.

``--infer smt-linear``.  No LLM, no learner, no grammar -- the *shape* is
fixed and its coefficients are ordinary constants, so the whole search is one
`checkSat`::

    exists c. forall s, inputs. obligations(c . s)

``--buchi`` asks it for a ranking function ``c0 + c1*s0 + ... + cn*sn`` over
a fixed invariant.  ``--safety`` asks it for the invariant itself, as a
conjunction of ``a0 + a1*s0 + ... >= 0`` rows.

**Every scalar component is a column, whatever its sort.**  An ``Int`` is
itself, a ``Bool`` is ``0``/``1`` (``(ite s0 1 0)``, which is how a row says
``b`` or ``¬b`` and how a rank falls when a flag flips), and a bitvector is
its unsigned value (``(ubv_to_int s0)``).  One integer template then covers
a mixed state instead of a template per sort -- and the lifting is only the
search space: the obligations are stated against the module's own transition
in its own sorts, wraparound and all, so what survives is right about the
bitvector rather than about a story told over it.

**The refutation is the point.**  A `sat` answer is a certificate, and there
is a cheaper route to most of those; an `unsat` answer is a *proof that no
certificate of that shape exists*, which nothing else in this package can
produce.  Measured: `m_toward5`, `m_twovars` and `m_lex` have no linear
ranking function and cvc5 says so in 2-30 ms, where an LLM asked for one
spends an API call per attempt discovering the same thing.  That answer is
written to `artifacts/` as a note, and `--infer ai-cegis` reads it into its
prompt on the next run -- which is the whole reason this route is worth
running before the expensive one.

Two things follow from how the cost behaves, both measured:

* **The rows are tried one at a time.**  A one-row invariant for
  `m_countdown |- G (s0 <= 100)` is 11 ms; asking for two rows at once does
  not finish in 30 s, because each row multiplies the nonlinear search. So
  `--linear-rows` is a ceiling, the queries run `1, 2, ... n`, and the first
  that answers wins.
* **"Empty" means the widest shape actually decided.**  If one row came back
  unsat and two rows timed out, what was proved is about one row, and the
  note says one row. A note that overstated this would be worse than no
  note: the next run's prompt would be told something false.

And one thing follows from a sort: **a bitvector column takes the query out
of what cvc5 will state.**  Measured, the `exists c. forall s.` form comes
back ``unknown (INCOMPLETE)`` in *one millisecond* for any width -- not a
timeout, a refusal.  So there is a second engine, and it asks the same
question with both halves quantifier-free (:meth:`TA2MagicLinear._cegis`).
It is a fallback rather than the engine, because what the direct query
proves is stronger: its ``unsat`` is about every integer coefficient, where
the loop needs a bounded candidate space to terminate and its ``unsat`` is
about that box -- which is why the note carries the bound when the loop is
what answered.
"""

from __future__ import annotations

from ..cert import CertificateData
from ..common import Refused
from . import TA2Magic
from dataclasses import dataclass

from ..smt_synth import (
    Search,
    SynthContext,
    affine_smt,
    bounded_solver,
    conjunction_smt,
    int_readings,
    program_constants,
    record,
)

try:
    import cvc5                                      # type: ignore
    from cvc5 import Kind
except ImportError:                                  # pragma: no cover
    cvc5 = None
    Kind = None

DEFAULT_ROWS = 2

# How many counterexamples the fallback loop may collect before it gives up.
# It is not the real leash -- the run's budget is, and it is checked every
# iteration -- but a loop that cannot terminate needs a number. Measured: an
# 8-bit counter's invariant takes 55 samples, so the cap is well clear of
# what a small module needs and a big one runs out of budget long before it.
DEFAULT_SAMPLES = 256

_HOW_QUANTIFIED = (
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
class _Template:
    """A shape with its coefficients left open, and what they must satisfy.

    `build` states the obligations for *given* state and input terms, which
    is what lets one template serve both engines: the quantified query
    passes bound variables and wraps the result in a `forall`, and the
    counterexample-guided loop passes concrete values and fresh constants.
    """

    unknowns: list                  # the cvc5 Int constants to solve for
    build: object                   # (state, extl_l, extl_n) -> Bool term


@dataclass(frozen=True)
class _Verdict:
    """What an engine answered, and what the answer is a claim about."""

    answer: str                     # "sat" | "unsat" | "unknown"
    values: "list[int] | None" = None
    how: str = _HOW_QUANTIFIED
    caveat: str = ""                # what narrows an `unsat`, if anything


def _shape(names, prefix: str = "c") -> str:
    """The template as a person would write it: `c0 + c1*s0 + c2*s1`.

    Elided in the middle only when there is a middle to elide -- a
    one-column module printing `c0 + c1*s0 + ... + c1*s0` would name the
    same coefficient twice and read as a wider shape than was searched.
    """
    terms = [f"{prefix}{i + 1}*{n}" for i, n in enumerate(names)]
    if len(terms) > 3:
        terms = terms[:2] + ["..."] + terms[-1:]
    return " + ".join([f"{prefix}0"] + terms)


def _normalise(values: list[int]) -> list[int]:
    """`values` divided by their gcd, which is the same predicate.

    `(>= (+ 200 (* (- 2) s0)) 0)` and `(>= (+ 100 (- s0)) 0)` hold of exactly
    the same states -- dividing through by a positive common factor is an
    equivalence over the integers for both `>= 0` and a strict decrease --
    and the second is the one the certificate should carry: every digit of
    it is restated in Lean and re-elaborated on every obligation.
    """
    from math import gcd

    g = 0
    for v in values:
        g = gcd(g, abs(int(v)))
    return list(values) if g in (0, 1) else [int(v) // g for v in values]


class TA2MagicLinear(TA2Magic):
    """Infer a certificate of fixed linear shape, or prove there is none."""

    def __init__(
        self,
        module,
        *,
        rows: int = DEFAULT_ROWS,
        budget=None,
        artifacts=None,
        log=print,
    ):
        super().__init__("")
        if cvc5 is None:                             # pragma: no cover
            raise Refused(
                "--infer smt-linear needs the `cvc5` package, which is not "
                "importable. Install with: uv pip install cvc5"
            )
        self.module = module
        self.rows = max(1, int(rows))
        self.budget = budget
        self.artifacts = artifacts
        self.log = log

    # --- driver ---------------------------------------------------------

    def infer(self, cd: CertificateData) -> CertificateData:
        ctx = SynthContext.build(self.module, cd, route="smt-linear")
        cols = int_readings(ctx, allow=("int", "bool", "bv"), route="smt-linear")
        self.log(f"[smt-linear] columns: {', '.join(c.name for c in cols)}")
        if cd.is_safety:
            search = self._safety(ctx, cols, cd)
            if search.found is not None:
                self.log(f"[smt-linear] inv: {search.found}")
            record(self.artifacts, search, log=self.log)
            if search.found is None:
                raise Refused(f"--infer smt-linear found no invariant. {search.note}")
            cd.inv = cd.inv_smt = search.found
            return cd

        given = cd.inv if isinstance(cd.inv, str) and cd.inv.strip() else "true"
        if given == "true":
            # Said rather than assumed: a rank that has to hold in *every*
            # state is a much harder ask than one over a real invariant, and
            # a note saying "no linear rank" would be read as a fact about
            # the module when it is a fact about `true`.
            self.log(
                "[smt-linear] no invariant to rank over (none supplied and "
                "none resumed): searching under `true`, which is the weakest "
                "invariant there is"
            )
        search = self._buchi(ctx, cols, given)
        if search.found is not None:
            self.log(f"[smt-linear] inv: {given}")
            self.log(f"[smt-linear] ranking: {search.found}")
        record(self.artifacts, search, log=self.log)
        if search.found is None:
            raise Refused(
                f"--infer smt-linear found no ranking function. {search.note}"
            )
        cd.inv = cd.inv_smt = given
        cd.ranking = cd.ranking_smt = search.found
        return cd

    # --- the ranking function, over a fixed invariant -------------------

    def _buchi(self, ctx: SynthContext, cols: list, inv_src: str) -> Search:
        """`inv /\\ ~P -> rank >= 1 /\\ rank(next) < rank`.

        `rule_buchi`'s one ranking obligation, `hrank`, over the Lean ranking
        `Int.toNat rank`: `toNat r' < toNat r` holds exactly when `r >= 1` and
        `r' < r`. Only the states where `P` fails are constrained -- where it
        holds, the rank may be anything, negative included.

        This used to ask `inv -> rank >= 0` over *every* state as well, which
        `rule_buchi` never states. On an unbounded integer state that rules out
        every linear rank, since the property's own states run to minus
        infinity, and the route reported the empty space as a proof: `while (y
        >= 0) y := y - 1` was "proved" to have no linear rank, and Lean accepts
        `(1 + y).toNat` for it under the invariant `true`.
        """
        from ..smt_prompt import parse_predicate

        tm, n = ctx.tm, len(cols)
        inv_term = parse_predicate(ctx.env, inv_src)
        Int = tm.getIntegerSort()
        c = [tm.mkConst(Int, f"c{i}") for i in range(n + 1)]

        def rank(vs):
            """`c0 + Σ ci·(column i read at `vs`)`.

            The columns are terms over `ctx.state`, so reading them at the
            successor is the same substitution the obligations already make
            -- and a Bool or bitvector column is read there too, which is
            the whole of what those sorts cost here.
            """
            t = c[0]
            for k, col in zip(c[1:], cols):
                t = tm.mkTerm(
                    Kind.ADD, t,
                    tm.mkTerm(Kind.MULT, k, col.term.substitute(ctx.state, vs)),
                )
            return t

        def build(st, el, en):
            inv_s = inv_term.substitute(ctx.state, st)
            nxt = ctx.msmt.update_state(st, el, en)
            return tm.mkTerm(
                Kind.IMPLIES,
                tm.mkTerm(Kind.AND, inv_s,
                          tm.mkTerm(Kind.NOT, ctx.at(ctx.prp, st)),
                          ctx.with_inputs(ctx.update_pre, el, en)),
                tm.mkTerm(Kind.AND,
                          tm.mkTerm(Kind.GEQ, rank(st), tm.mkInteger(1)),
                          tm.mkTerm(Kind.LT, rank(nxt), rank(st))),
            )

        shape = f"`{_shape([col.name for col in cols])}`"
        verdict = self._decide(ctx, _Template(unknowns=c, build=build))
        if verdict.answer == "sat":
            coeffs = _normalise(verdict.values)
            return Search(
                "ranking",
                f"It is linear in the state -- {shape}, no branching -- and "
                f"ranks under the invariant `{inv_src}`.",
                found=affine_smt(coeffs[0], coeffs[1:],
                                 [col.name for col in cols]),
            )
        return Search(
            "ranking",
            f"No ranking function linear in the state -- {shape}, integer "
            f"coefficients, no branching -- drops by at least one and stays "
            f"positive wherever the property fails, under the invariant "
            f"`{inv_src}`.{verdict.caveat}",
            how=verdict.how,
            exhausted=verdict.answer == "unsat",
            detail=(
                "A ranking function for this module needs something outside "
                "that shape: a branch (`ite`), which is what a program whose "
                "run wraps around needs, or a stronger invariant to rank "
                "over. `--infer ai-cegis` and `--infer nuterm` both search "
                "shapes that have one."
                if verdict.answer == "unsat" else
                "Raise `--smt-timeout`, or try `--infer nuterm` / "
                "`--infer ai-cegis`."
            ),
        )

    # --- the invariant, one row at a time -------------------------------

    def _safety(self, ctx: SynthContext, cols: list,
                cd: CertificateData) -> Search:
        """`init -> inv`, `inv -> inv'`, `inv -> P`, over conjunctions of rows.

        A supplied invariant is not replaced but *strengthened*: it is the
        first conjunct and the rows are what is added to it, so
        `--invariant` here means "this much is known, find the rest" -- the
        same thing a resumed `inv` artifact means to `--infer ai-cegis`.

        The widths are separate queries, smallest first, because the cost is
        in the width: one row is 11 ms on `m_countdown` and two do not finish
        in 30 s. Each `unsat` on the way is a fact worth keeping, and the
        widest one reached is what the note is allowed to claim.
        """
        given = cd.inv if isinstance(cd.inv, str) and cd.inv.strip() else None
        if given:
            self.log(f"[smt-linear] strengthening the given invariant: {given}")
        decided, how, caveat = 0, _HOW_QUANTIFIED, ""
        for k in range(0 if given else 1, self.rows + 1):
            found, verdict = self._safety_at(ctx, cols, given, k)
            if found is not None:
                return Search("inv", self._found_rows(cols, given, k),
                              found=found)
            if verdict.answer != "unsat":
                return self._empty_rows(cols, given, decided, undecided_at=k,
                                        how=how, caveat=caveat)
            decided, how, caveat = k, verdict.how, verdict.caveat
            self.log(f"[smt-linear] no invariant with {k} row(s); widening")
        return self._empty_rows(cols, given, decided, how=how, caveat=caveat)

    def _safety_at(
        self, ctx: SynthContext, cols: list, given: "str | None", k: int
    ) -> tuple["str | None", "_Verdict"]:
        """The invariant with `k` added rows, or why there is none."""
        from ..smt_prompt import parse_predicate

        tm, n = ctx.tm, len(cols)
        Int = tm.getIntegerSort()
        rows = [[tm.mkConst(Int, f"a{j}_{i}") for i in range(n + 1)]
                for j in range(k)]
        given_term = parse_predicate(ctx.env, given) if given else None

        def inv(vs):
            parts = []
            if given_term is not None:
                parts.append(given_term.substitute(ctx.state, vs))
            for row in rows:
                t = row[0]
                for a, col in zip(row[1:], cols):
                    t = tm.mkTerm(
                        Kind.ADD, t,
                        tm.mkTerm(Kind.MULT, a,
                                  col.term.substitute(ctx.state, vs)),
                    )
                parts.append(tm.mkTerm(Kind.GEQ, t, tm.mkInteger(0)))
            if not parts:                            # pragma: no cover
                return tm.mkBoolean(True)
            return parts[0] if len(parts) == 1 else tm.mkTerm(Kind.AND, *parts)

        def build(st, el, en):
            return tm.mkTerm(
                Kind.AND,
                tm.mkTerm(Kind.IMPLIES,
                          ctx.with_inputs(ctx.init_pre, el, en),
                          inv(ctx.msmt.init_state(en))),
                tm.mkTerm(
                    Kind.IMPLIES,
                    tm.mkTerm(Kind.AND, inv(st),
                              ctx.with_inputs(ctx.update_pre, el, en)),
                    inv(ctx.msmt.update_state(st, el, en)),
                ),
                tm.mkTerm(Kind.IMPLIES, inv(st), ctx.at(ctx.prp, st)),
            )

        flat = [x for row in rows for x in row]
        verdict = self._decide(ctx, _Template(unknowns=flat, build=build))
        if verdict.answer != "sat":
            return None, verdict
        values = list(verdict.values)
        parts = [given] if given else []
        for j in range(k):
            row = _normalise(values[j * (n + 1):(j + 1) * (n + 1)])
            if not any(row[1:]) and row[0] >= 0:
                # `2 >= 0`: a row the solver was free to leave vacuous, and a
                # conjunct that says nothing is one more thing for every
                # obligation downstream to carry.
                continue
            parts.append(
                f"(>= {affine_smt(row[0], row[1:], [c.name for c in cols])} 0)"
            )
        return conjunction_smt(parts), verdict

    # --- what the widths add up to, as prose for the artifact -----------

    def _found_rows(self, cols: list, given: "str | None", k: int) -> str:
        over = f", strengthening the given `{given}`" if given else ""
        if k == 0:
            return f"The given invariant `{given}` needed no strengthening."
        return (
            f"It is a conjunction of {k} linear "
            f"{'inequality' if k == 1 else 'inequalities'} "
            f"`{_shape([c.name for c in cols], prefix='a')} >= 0`{over}."
        )

    def _empty_rows(self, cols: list, given: "str | None", decided: int,
                    undecided_at: int = 0, how: str = _HOW_QUANTIFIED,
                    caveat: str = "") -> Search:
        over = f" beyond the given `{given}`" if given else ""
        exhausted = decided > 0
        if not exhausted:
            space = (
                f"Nothing was proved about invariants that are conjunctions "
                f"of linear inequalities{over} for this module and property."
            )
        else:
            space = (
                f"No inductive invariant implying the property is a "
                f"conjunction of {decided} linear "
                f"{'inequality' if decided == 1 else 'inequalities'} "
                f"`{_shape([c.name for c in cols], prefix='a')} >= 0`{over}."
            )
        detail = (
            f"Wider conjunctions were not tried: `--linear-rows` stopped at "
            f"{self.rows}. An invariant may exist with more rows, or outside "
            f"linear arithmetic altogether -- `--infer sygus` adds "
            f"congruences (`x` even), which no conjunction of inequalities "
            f"can state."
            if not undecided_at else
            f"The query at {undecided_at} row(s) did not finish inside the "
            f"budget, so nothing is known at that width or beyond it. Raise "
            f"`--smt-timeout`, or try `--infer sygus`."
        )
        return Search("inv", space + (caveat if exhausted else ""),
                      exhausted=exhausted, detail=detail, how=how)

    # ══════════════════════════════════════════════════════════════════
    # The engine
    #
    # Two ways of answering the same question, and the second exists
    # because of a measurement: for a BitVec column cvc5 answers the
    # quantified form `unknown (INCOMPLETE)` in *one millisecond* -- not a
    # timeout, a refusal to state it. So a module whose state is a
    # bitvector cannot be reached by the direct query at all, and the same
    # question has to be asked in a form that stays quantifier-free.
    # ══════════════════════════════════════════════════════════════════

    def _decide(self, ctx: SynthContext, tpl: _Template) -> _Verdict:
        """Solve `tpl`, by whichever engine can state it.

        The quantified query first: it is one call, it is decided over
        *every* integer coefficient, and for an integer state it answers in
        milliseconds. Only when cvc5 will not state it does the loop run,
        and what that loop proves is narrower -- which is why the verdict
        carries the caveat rather than the caller having to remember.
        """
        verdict = self._quantified(ctx, tpl)
        if verdict.answer != "unknown":
            return verdict
        bound = self._coefficient_bound(ctx)
        self.log(
            f"[smt-linear] cvc5 will not state the quantified query for this "
            f"module; searching coefficients in [-{bound}, {bound}] by "
            f"counterexample instead"
        )
        return self._cegis(ctx, tpl, bound)

    def _quantified(self, ctx: SynthContext, tpl: _Template) -> _Verdict:
        """`exists c. forall s, inputs. obligations(c)`, in one call."""
        tm = ctx.tm
        solver = bounded_solver(tm, self.budget, produce_models="true")
        solver.setLogic("ALL")
        st, el, en = self._bound_vars(ctx)
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
            return _Verdict("sat", [self._value(solver, x)
                                    for x in tpl.unknowns])
        return _Verdict("unsat" if result.isUnsat() else "unknown")

    def _cegis(self, ctx: SynthContext, tpl: _Template, bound: int) -> _Verdict:
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
            if self.budget is not None and self.budget.exhausted:
                return _Verdict("unknown")
            propose = bounded_solver(tm, self.budget, produce_models="true")
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
                return _Verdict(
                    "unsat",
                    how=_HOW_CEGIS.format(samples=step),
                    caveat=self._caveat(bound),
                )
            if not answer.isSat():
                return _Verdict("unknown")
            values = [propose.getValue(x) for x in tpl.unknowns]

            verify = bounded_solver(tm, self.budget, produce_models="true")
            verify.setLogic("ALL")
            st, el, en = self._fresh_consts(ctx, f"cex{step}_")
            verify.assertFormula(
                tm.mkTerm(Kind.NOT,
                          tpl.build(st, el, en).substitute(tpl.unknowns, values))
            )
            refuted = verify.checkSat()
            if refuted.isUnsat():
                return _Verdict(
                    "sat",
                    [int(v.getIntegerValue()) for v in values],
                    how=_HOW_CEGIS.format(samples=step),
                    caveat=self._caveat(bound),
                )
            if not refuted.isSat():
                return _Verdict("unknown")
            samples.append(
                tuple([verify.getValue(x) for x in group]
                      for group in (st, el, en))
            )
        return _Verdict("unknown")

    def _coefficient_bound(self, ctx: SynthContext) -> int:
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

    def _caveat(self, bound: int) -> str:
        return (
            f" The coefficients were searched in [-{bound}, {bound}]: the "
            f"counterexample loop needs a finite candidate space, so this is "
            f"a proof about that space rather than about every integer."
        )

    def _bound_vars(self, ctx: SynthContext):
        """Fresh bound variables for the state and the inputs."""
        tm = ctx.tm
        st = [tm.mkVar(v.getSort(), f"q{i}") for i, v in enumerate(ctx.state)]
        el = [tm.mkVar(v.getSort(), f"bel{i}")
              for i, v in enumerate(ctx.extl_latched)]
        en = [tm.mkVar(v.getSort(), f"ben{i}")
              for i, v in enumerate(ctx.extl_next)]
        return st, el, en

    def _fresh_consts(self, ctx: SynthContext, prefix: str):
        """Fresh *constants* for the state and the inputs, for a model."""
        tm = ctx.tm
        st = [tm.mkConst(v.getSort(), f"{prefix}q{i}")
              for i, v in enumerate(ctx.state)]
        el = [tm.mkConst(v.getSort(), f"{prefix}l{i}")
              for i, v in enumerate(ctx.extl_latched)]
        en = [tm.mkConst(v.getSort(), f"{prefix}e{i}")
              for i, v in enumerate(ctx.extl_next)]
        return st, el, en

    def _value(self, solver, term) -> int:
        return int(solver.getValue(term).getIntegerValue())
