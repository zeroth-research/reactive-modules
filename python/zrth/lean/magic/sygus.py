"""TA2Magic by synthesis: the invariant cvc5's SyGuS track finds.

``--infer sygus``.  A safety certificate *is* an invariant -- `rule_globally`
takes no ranking function -- and an invariant is exactly what SyGuS-IF's
invariant track is for: ``addSygusInvConstraint(inv, pre, trans, post)`` is
the three obligations ``G P`` is made of, stated once, with `inv` the
function to be found.

Why this route exists beside `--infer nuterm`, which also needs no LLM: the
one thing a Houdini lattice of signs and pairwise relations cannot say is a
**congruence**.  `m_step2` steps by two and resets, so `x` is even in every
reachable state and no candidate Houdini has expresses that; the limit
matrix records it as the single Buchi case nuterm misses on the certificate
rather than on reach.  A grammar can carry `(= (mod lin 2) 0)`, and with it
cvc5 answers that invariant in 11 ms.

What the route needs of a module
================================
Scalar integer state, and a transition in linear integer arithmetic -- the
fragment the grammar lives in.  A Real, Bool, bitvector or matrix-shaped
state is refused by name (:meth:`SynthContext.build`), and a transition
outside `LIA` is refused when cvc5 rejects the constraint rather than
silently searching a space that cannot express it.

External inputs are allowed.  SyGuS-IF's `pre` and `trans` are predicates on
the state alone, so an input is existentially quantified inside them: `s` is
initial when *some* input makes `init` produce it, and `s -> s'` is a
transition when *some* input does -- which is the same relation the
certificate's `init_inv` and `step_inv` obligations quantify over, and a
`--pre` precondition restricts it exactly as it restricts them.

How wide an invariant it looks for
==================================
One width at a time, `1, 2, ... --sygus-conjuncts`, the first that answers
winning -- not one query at the ceiling.  The widths nest, so this costs no
reach and the *narrowest* answer is the one that comes back; and because the
search is not monotone in the width, asking only at the ceiling was losing
both halves of what this route is for.  :meth:`TA2MagicSygus._synthesise`
has the measurement.

What it leaves behind
=====================
The invariant, in `artifacts/`, as a resumable `inv`: `--infer ai-cegis` in
a later run takes it as given and infers only what is left.  That is the
route's second purpose and the reason it is worth running even when the
project it generates is not the one wanted -- the certificate is the answer,
and the artifact is the answer being *reusable*.
"""

from __future__ import annotations

from ..cert import CertificateData
from ..common import Refused
from . import TA2Magic
from ..smt_synth import (
    Search,
    SynthContext,
    body_of,
    bounded_solver,
    readings,
    moduli,
    program_constants,
    record,
)

try:
    import cvc5                                      # type: ignore
    from cvc5 import Kind
except ImportError:                                  # pragma: no cover
    cvc5 = None
    Kind = None

# Grammars, by what their atoms can say.  `congruence` is the default because
# it is the only one that says anything the rest of the package cannot: the
# linear half of it is what `--infer smt-linear` decides in milliseconds and
# what Houdini's lattice already covers.
GRAMMARS = ("congruence", "linear")

# The *ceiling* on how many atoms the invariant may be a conjunction of: the
# widths are asked one at a time, `1, 2, ... n`, and the first that answers
# wins. A bound rather than a free `B -> A | (and B B)` for two reasons, and
# the second is the one that matters: a bounded conjunction is a *finite*
# space, and a finite space can be decided empty -- `hasNoSolution` rather
# than `unknown`, which is the difference between leaving a proof behind and
# leaving a shrug. (The first reason is the ordinary one: every conjunct is
# restated in Lean and is one more implication in the obligation, which
# `magic.learn._prune` exists for.)
# Measured on `m_step2` with the congruence atoms dropped, which has no
# answer in this space: decided empty in 10 / 25 / 166 ms at 1 / 2 / 3.
DEFAULT_CONJUNCTS = 3


class TA2MagicSygus(TA2Magic):
    """Infer a safety invariant with cvc5's SyGuS invariant track.

    ``grammar`` chooses what an atom may be: `linear` gives comparisons of
    affine combinations of the state, `congruence` adds `(= (mod lin k) 0)`
    for the `k` the program mentions.  ``conjuncts`` is a *ceiling*: the
    widths are asked one at a time and the first that answers wins.
    """

    def __init__(
        self,
        module,
        *,
        grammar: str = "congruence",
        conjuncts: int = DEFAULT_CONJUNCTS,
        budget=None,
        artifacts=None,
        log=print,
    ):
        super().__init__("")
        if cvc5 is None:                             # pragma: no cover
            raise Refused(
                "--infer sygus needs the `cvc5` package, which is not "
                "importable. Install with: uv pip install cvc5"
            )
        if grammar not in GRAMMARS:
            raise Refused(
                f"--sygus-grammar takes one of {', '.join(GRAMMARS)}; "
                f"{grammar!r} is not one"
            )
        self.module = module
        self.grammar = grammar
        self.conjuncts = max(1, int(conjuncts))
        self.budget = budget
        self.artifacts = artifacts
        self.log = log

    # --- driver ---------------------------------------------------------

    def infer(self, cd: CertificateData) -> CertificateData:
        if not cd.is_safety:
            # The row refuses `--buchi` at parse time; this is the same rule
            # where the search is, so the class is not usable another way.
            raise Refused(
                "--infer sygus certifies `G P`: the invariant is the whole "
                "certificate and there is no ranking function to synthesise"
            )
        ctx = SynthContext.build(self.module, cd, route="sygus")
        # Integers only: the invariant is a `synthFun` over integer
        # arguments, so a Bool or bitvector column has nowhere to go. The
        # template route weighs those, and the refusal says so.
        readings(ctx, allow=("int",), route="sygus")
        self.log(f"[sygus] columns: {', '.join(ctx.names)}")
        consts = program_constants(ctx)
        self.log(f"[sygus] grammar: {self.grammar}, constants {list(consts)}")

        search = self._synthesise(ctx, consts)
        record(self.artifacts, search, log=self.log)
        if search.found is None:
            raise Refused(
                f"--infer sygus found no invariant. {search.note}",
                searched=True,
            )
        self.log(f"[sygus] inv: {search.found}")
        return self._emit(cd, search.found)

    # --- the search -----------------------------------------------------

    def _synthesise(self, ctx: SynthContext, consts) -> Search:
        """`pre -> inv`, `inv /\\ trans -> inv'`, `inv -> post`, width by width.

        One query per width, narrowest first, as `magic.linear` runs its
        rows -- and here for a stronger reason than cost, because **the
        search is not monotone in the width**.  The widths nest: the start
        rule at width `w` is `A | A /\\ A | ... ` up to `w`, so a wider
        grammar's language contains every narrower one and an answer found
        at 2 is inside the space searched at 3.  cvc5 still does not find it
        there.  Measured over the 59 matrix cells that reach the grammar:
        `wise` finds a two-atom invariant at width 3 and times out at 2;
        `T5Exact` finds one at 2 and times out at 3; `NiLexNe4` and
        `RelTvImplies` find *one-atom* invariants in under a second and time
        out at 3.  So the ceiling is the worst single width to ask at, and
        asking only there is what this route used to do: 28 found, 1 space
        decided empty, 30 cells that learned nothing.  Every width, first
        answer wins: 31 found, 22 decided, 6 that learned nothing.

        Nesting is also what lets the note claim the *widest* width that
        came back empty rather than a prefix of them -- an empty width 3
        subsumes 1 and 2 whatever they did.
        """
        decided, undecided = 0, []
        for k in range(1, self.conjuncts + 1):
            found, exhausted = self._at_width(ctx, consts, k)
            if found is not None:
                self.log(f"[sygus] found at {k} atom(s)")
                return Search("inv", self._found_space(consts, k), found=found)
            if exhausted:
                decided = k
                self.log(f"[sygus] no invariant of {k} atom(s); widening")
            else:
                undecided.append(k)
                self.log(f"[sygus] {k} atom(s) did not finish; widening")
            if self.budget is not None and self.budget.exhausted:
                break
        # Only a width *above* the decided one is still open: the widths
        # nest, so one that ran out of budget under a width that came back
        # empty was decided after all, by the wider query that contains it.
        return self._empty(
            consts, decided,
            undecided_at=min((k for k in undecided if k > decided), default=0),
        )

    def _at_width(self, ctx: SynthContext, consts, width: int):
        """One SyGuS query, over a grammar of at most `width` atoms.

        Returns the invariant if there is one, else whether the space was
        *decided* empty rather than merely not searched.
        """
        tm, n = ctx.tm, len(ctx.state)
        solver = bounded_solver(tm, self.budget, sygus="true")
        try:
            solver.setLogic("LIA")
        except Exception as e:                       # pragma: no cover
            raise Refused(f"cvc5 refused the logic for this module: {e}") from e

        Int, Bool = tm.getIntegerSort(), tm.getBooleanSort()
        args = [tm.mkVar(Int, f"x{i}") for i in range(n)]
        grammar = self._grammar(solver, ctx, args, consts, width)
        inv = solver.synthFun("inv", args, Bool, grammar)

        state = [tm.mkVar(Int, f"s{i}_") for i in range(n)]
        nxt = [tm.mkVar(Int, f"sp{i}_") for i in range(n)]
        try:
            solver.addSygusInvConstraint(
                inv,
                solver.defineFun("pre", state, Bool, self._pre(ctx, state)),
                solver.defineFun(
                    "trans", state + nxt, Bool, self._trans(ctx, state, nxt)
                ),
                solver.defineFun(
                    "post", state, Bool, ctx.at(ctx.prp, state)
                ),
            )
            result = solver.checkSynth()
        except RuntimeError as e:
            # A transition outside `LIA` -- a product of two state components,
            # say -- is met here, as cvc5 refusing the constraint. It is a
            # fact about the module, not a bug, so it is named as one.
            raise Refused(
                f"cvc5 cannot state this module's transition in the linear "
                f"integer fragment this route's grammar lives in: {e}"
            ) from e

        if result.hasSolution():
            body, bound = body_of(solver.getSynthSolution(inv))
            # cvc5 `let`-binds a repeated subterm when it prints, and that is
            # fine: the term is parsed back before it is rendered, and
            # `smt_to_lean` gives what is shared a Lean `let` of its own.
            src = str(body.substitute(bound, ctx.state) if bound else body)
            return src, False
        return None, result.hasNoSolution()

    # --- the three formulas `G P` is made of ----------------------------

    def _pre(self, ctx: SynthContext, state: list):
        """`s` is an initial state: some input makes `init` produce it.

        Existential rather than a fresh constant, because SyGuS-IF's `pre` is
        a predicate on the state alone -- and existential rather than
        universal because `init_inv` asks the invariant to hold at whatever
        the inputs made, not at every value some other input would have made.
        """
        tm = ctx.tm
        el = [tm.mkVar(v.getSort(), f"pre_el{i}")
              for i, v in enumerate(ctx.extl_latched)]
        en = [tm.mkVar(v.getSort(), f"pre_en{i}")
              for i, v in enumerate(ctx.extl_next)]
        body = tm.mkTerm(
            Kind.AND,
            ctx.with_inputs(ctx.init_pre, el, en),
            self._same(ctx, state, ctx.msmt.init_state(en)),
        )
        return self._exists(ctx, el + en, body)

    def _trans(self, ctx: SynthContext, state: list, nxt: list):
        """`s -> s'`: some input, allowed by `--pre`, takes one to the other."""
        tm = ctx.tm
        el = [tm.mkVar(v.getSort(), f"tr_el{i}")
              for i, v in enumerate(ctx.extl_latched)]
        en = [tm.mkVar(v.getSort(), f"tr_en{i}")
              for i, v in enumerate(ctx.extl_next)]
        body = tm.mkTerm(
            Kind.AND,
            ctx.with_inputs(ctx.update_pre, el, en),
            self._same(ctx, nxt, ctx.msmt.update_state(state, el, en)),
        )
        return self._exists(ctx, el + en, body)

    def _same(self, ctx: SynthContext, lhs: list, rhs: list):
        eqs = [ctx.tm.mkTerm(Kind.EQUAL, a, b) for a, b in zip(lhs, rhs)]
        if not eqs:                                  # pragma: no cover
            return ctx.tm.mkBoolean(True)
        return eqs[0] if len(eqs) == 1 else ctx.tm.mkTerm(Kind.AND, *eqs)

    def _exists(self, ctx: SynthContext, vs: list, body):
        if not vs:
            return body
        return ctx.tm.mkTerm(
            Kind.EXISTS, ctx.tm.mkTerm(Kind.VARIABLE_LIST, *vs), body
        )

    # --- the grammar ----------------------------------------------------

    def _grammar(self, solver, ctx: SynthContext, args: list, consts, width):
        """Conjunctions of comparisons between affine combinations.

        Shaped as a *template* rather than a free arithmetic grammar, because
        the free one is measurably worse: on a ranking function an
        unrestricted `LIA` grammar and a loose "linear combinations plus
        `ite`" grammar both ran *slower* than this shape, and what prunes is
        fixing the affine form and enumerating only its coefficients.
        """
        tm = ctx.tm
        Int, Bool = tm.getIntegerSort(), tm.getBooleanSort()
        start = tm.mkVar(Bool, "B")
        atom = tm.mkVar(Bool, "A")
        lin = tm.mkVar(Int, "L")
        coeff = tm.mkVar(Int, "K")
        grammar = solver.mkGrammar(args, [start, atom, lin, coeff])
        # The finite ladder: one atom, two, ... up to `width`. `B -> (and
        # B B)` would say the same thing about what is *reachable* and make
        # the space infinite, and then "nothing here works" stops being
        # something cvc5 can answer. Keeping the narrower rungs in is what
        # makes the widths nest, which is what lets an empty width subsume
        # the ones below it.
        ladder = [atom]
        wider = atom
        for _ in range(width - 1):
            wider = tm.mkTerm(Kind.AND, wider, atom)
            ladder.append(wider)
        grammar.addRules(start, ladder)
        atoms = [
            tm.mkTerm(Kind.LEQ, lin, tm.mkInteger(0)),
            tm.mkTerm(Kind.EQUAL, lin, tm.mkInteger(0)),
        ]
        if self.grammar == "congruence":
            atoms += [
                tm.mkTerm(
                    Kind.EQUAL,
                    tm.mkTerm(Kind.INTS_MODULUS, lin, tm.mkInteger(k)),
                    tm.mkInteger(0),
                )
                for k in moduli(consts)
            ]
        grammar.addRules(atom, atoms)
        body = coeff
        for a in args:
            body = tm.mkTerm(Kind.ADD, body, tm.mkTerm(Kind.MULT, coeff, a))
        grammar.addRules(lin, [body])
        grammar.addRules(coeff, [tm.mkInteger(c) for c in consts])
        return grammar

    # --- what the widths add up to, as prose for the artifact -----------

    def _atoms(self, consts) -> str:
        atoms = "`c0 + c1*s0 + ... <= 0` and `= 0`"
        if self.grammar == "congruence":
            atoms += (
                f", and `(= (mod c0 + c1*s0 + ... k) 0)` for k in "
                f"{list(moduli(consts))}"
            )
        return atoms

    def _found_space(self, consts, width: int) -> str:
        return (
            f"It is a conjunction of at most {width} "
            f"{'atom' if width == 1 else 'atoms'} of the form "
            f"{self._atoms(consts)}, with coefficients from {list(consts)}."
        )

    def _empty(self, consts, decided: int, undecided_at: int = 0) -> Search:
        """What the widths add up to, and no more than that.

        `decided` is the widest width cvc5 enumerated, and it is all the note
        may claim. A width that ran out of budget is not evidence of anything
        -- the narrower widths it sits above still count, because they are
        inside it, but the ones above it do not.
        """
        shape = (
            f"{self._atoms(consts)}, with coefficients from {list(consts)} "
            f"-- the constants this module and property mention"
        )
        if not decided:
            # The shape is named even here. What was searched is the useful
            # half of "and nothing came of it", and a reader who is choosing
            # the next route needs it whether or not the query finished.
            space = (
                f"Nothing was proved about invariants that are conjunctions "
                f"of atoms of the form {shape}."
            )
        else:
            space = (
                f"No inductive invariant implying the property is a "
                f"conjunction of at most {decided} "
                f"{'atom' if decided == 1 else 'atoms'} of the form {shape}."
            )
        return Search(
            "inv",
            space,
            exhausted=decided > 0,
            how=(
                f"the grammar is finite -- at most {decided} "
                f"{'atom' if decided == 1 else 'atoms'} over a fixed set of "
                f"coefficients -- and cvc5 enumerated it"
            ),
            detail=(
                # The timeout is named last and as what it is -- a knob, not
                # a remedy -- because naming it first sends the reader to
                # spend two minutes arriving back here. The measurement
                # behind it is six benchmark modules re-run at 12 to 24 times
                # the budget, and it is quoted as the six it is rather than
                # as a property of the route.
                # `Search.note` has already said a query did not finish, so
                # what is left to say is *which width* is still open -- the
                # one thing the reader cannot infer from the rest.
                f"It stopped at {undecided_at} "
                f"{'atom' if undecided_at == 1 else 'atoms'}"
                f"{f', above the {decided} proved empty' if decided else ''}. "
                f"`--infer ai-cegis` searches no fixed space and is what to "
                f"try next; `--infer smt-linear` decides a narrower shape "
                f"rather than enumerating this one. `--smt-timeout` raises "
                f"the per-query budget, though tried at 12 to 24 times the "
                f"default on six modules that stopped here it finished none "
                f"of them."
                if undecided_at else
                "An invariant for this module needs something the grammar "
                "does not have -- a disjunction, a product of two "
                "components, a constant the program never mentions, or more "
                "than `--sygus-conjuncts` of them. `--infer ai-cegis` "
                "searches no fixed space and is what to try next."
            ),
        )

    # --- out ------------------------------------------------------------

    def _emit(self, cd: CertificateData, inv_smt: str) -> CertificateData:
        """The certificate, as SMT-LIB. The pipeline renders the Lean.

        Unlike `magic.cegar` and `magic.learn` this route holds no rendering
        of its own -- it never built one -- so handing back the SMT alone is
        what its row's `returns="smt"` means, and `main` prints it once.
        """
        cd.inv, cd.inv_smt = inv_smt, inv_smt
        return cd
