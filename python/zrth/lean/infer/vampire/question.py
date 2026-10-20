"""The question Vampire is asked: the obligations stated over a
template's holes and the transition's branches (`Question`), printed
as the SMT-LIB script cvc5 writes for it (`script_for`).
"""

from __future__ import annotations


from ...core.errors import Refused
from ...smt import obligations as obl
from ...smt.obligations import Obligations
from ...smt.script import decimals, script
from ...smt.terms import free_constants, ite_conditions
from ...smt.claim import SynthContext
from ..common.synth import (
    denominator_scale,
    floor_real,
    program_rationals,
)
from .branches import Branch
from .template import Template, window_for

try:
    import cvc5                                      # type: ignore
    from cvc5 import Kind
except ImportError:                                  # pragma: no cover
    cvc5 = None
    Kind = None


def script_for(conjecture) -> str:
    """One closed cvc5 term as a whole SMT-LIB problem for Vampire.

    The term *is* the question: cvc5 prints it, and `decimals` rewrites the
    rationals Vampire will not read. It goes under `assert-not`, not
    `(assert (not ...))`, because that marks it as *the* conjecture, which
    is what `--question_answering` attaches answer literals to.
    """
    return decimals(script([conjecture], command="assert-not", spaced=False))


def refuse_ites(term) -> None:
    """An `ite` the split did not reach is refused rather than printed.

    It would print perfectly well -- SMT-LIB has `ite` and Vampire reads it
    -- and then never be answered, which reaches the user as a timeout
    rather than as the unsupported shape it is. Asked of the whole assembled
    question, so a conditional in the property or in a precondition is met
    here too and not only one the branch split left behind.
    """
    if ite_conditions([term]):
        raise Refused(
            "--infer vampire cannot state an obligation that still holds "
            "an `ite` after the transition was branch-split: Vampire's "
            "answer-literal search does not see through a conditional. "
            "`--infer houdini` proves the same obligation with one."
        )


class Question:
    """One module's obligations as one cvc5 term with holes in it.

    The obligations are built over the constants `Obligations` already made
    -- `v_s0`, `v_el0`, `v_en0` -- and turned into bound variables only on
    the way out, by `_forall`. That way every term here is a term of the
    module's own encoding right up to the point where it is quantified, and
    the question Vampire reads is cvc5's printing of it.
    """

    def __init__(self, ctx: SynthContext, ob: Obligations,
                 entry: list[Branch], step: list[Branch]):
        self.ctx, self.ob, self.tm = ctx, ob, ctx.tm
        self.entry, self.step = entry, step
        # One bound variable per constant, and the constants are one per
        # *element* -- a matrix-shaped component is not a name Vampire can
        # be given. The names reach Vampire through cvc5's printer, so they
        # have to be distinct: two `mkVar`s of one name print alike and
        # mean different things. `v_s0_1` is bound as `s0_1`, which is the
        # constant's own name without the prefix that marks it a constant.
        flat_s = ob.flatten(ob.s)
        flat_e = ob.flatten(list(ob.el) + list(ob.en))
        self.consts = flat_s + flat_e
        self.state = [ctx.tm.mkVar(c.getSort(), str(c)[2:]) for c in flat_s]
        # The latched state as one term per column, which is what a row and
        # a rank coefficient are indexed by.
        self.here = flat_s
        self.inputs = [ctx.tm.mkVar(c.getSort(), f"e{i}")
                       for i, c in enumerate(flat_e)]
        self.vars = self.state + self.inputs
        self.holes: dict = {}
        self.window = window_for(ctx)
        # What a Real column is multiplied by before the rank floors it.
        # 1 for a module with no rationals in it, which is every module this
        # route took before it read them.
        self.scale = denominator_scale(program_rationals(ctx))

    # --- the template, as a term over a state ----------------------------

    def _hole(self, name: str):
        """The variable standing for one coefficient, made once per name.

        Always an integer, whatever the column it bounds. Vampire reports an
        answer as a literal and `_number` reads an integer one, so a Real
        hole would come back as a rational this route cannot carry -- and an
        integer endpoint on a rational column is still a true statement about
        it. `_fit` is what makes the sorts agree.
        """
        if name not in self.holes:
            self.holes[name] = self.tm.mkVar(self.tm.getIntegerSort(), name)
        return self.holes[name]

    def _fit(self, hole, body):
        """`hole` in `body`'s arithmetic, so the printed question is sorted.

        cvc5 coerces an `Int` against a `Real` silently and prints `(<= A
        x)`, which is well sorted only if the reader coerces too. Vampire's
        SMT-LIB front end is the reader, and the one thing this module has
        learned about it is not to find out: `to_real` is written down.
        """
        return (self.tm.mkTerm(Kind.TO_REAL, hole)
                if body.getSort().isReal() else hole)

    def _rows(self, tpl: Template, at: list):
        """`tpl` read at `at`, which is one term per component."""
        out = []
        for row in tpl.rows:
            body = row.term(self.tm, at)
            out += [self.tm.mkTerm(Kind.LEQ, self._fit(self._hole(row.lo), body),
                                   body),
                    self.tm.mkTerm(Kind.LEQ, body,
                                   self._fit(self._hole(row.hi), body))]
        return self._and(out)

    def _rank(self, tpl: Template, at: list):
        """`Rc + R0*s0 + ... `, the rank with its coefficients left open.

        Over the *floored* columns: `hrank` lands in `Nat`, so a Real column
        is read as `to_int` of itself scaled by the module's least common
        denominator -- the same reading `--infer smt-linear` ranks one by,
        and for the same reason, that a quantity falling by less than one
        need not floor to anything smaller. The invariant rows above keep
        their rationals; only the rank has to be an integer.

        `tpl.at` says which columns have a coefficient, so a module with a
        Bool column among its Reals ranks over the Reals and leaves the flag
        out -- the flag has no integer reading here that is not an `ite`,
        which is the one term this route refuses by measurement.
        """
        out = self._hole(tpl.rank[-1])
        for c, v in zip(tpl.rank, self.ranked_cols(tpl, at)):
            out = self.tm.mkTerm(Kind.ADD, out,
                                 self.tm.mkTerm(Kind.MULT, self._hole(c), v))
        return out

    def ranked_cols(self, tpl: Template, at: list) -> list:
        """The columns `tpl` ranks over, each Real one floored."""
        return [floor_real(self.tm, at[i], self.scale) for i in tpl.at]

    def _and(self, parts):
        """`parts` conjoined, less any literal `true`."""
        return obl.conjunction(self.tm, [
            p for p in parts
            if not (p.getKind() == Kind.CONST_BOOLEAN and p.getBooleanValue())
        ])

    # --- the obligations --------------------------------------------------

    def conjecture(self, tpl: Template, prp) -> str:
        """`--safety`: an inductive invariant that implies the property."""
        obs = self._obligations(tpl, prp, inductive=True, implies=True,
                                ranked=False)
        return script_for(self._ask_for(tpl.holes, obs))

    def invariant_conjecture(self, tpl: Template) -> str:
        """`init_inv` and `step_inv` alone -- an inductive invariant, with
        no claim about the property. Stage one of the Buchi split."""
        obs = self._obligations(tpl, None, inductive=True, implies=False,
                                ranked=False)
        return script_for(self._ask_for(tpl.holes, obs, bounded=True))

    def rank_conjecture(self, tpl: Template, prp, fixed: dict) -> str:
        """`hrank` over an invariant already derived. Stage two.

        Only the rank obligations: the invariant is in them already, as the
        hypothesis `here`, and with its coefficients fixed to numbers it is
        a statement about the module rather than a hole to fill. `init_inv`
        and `step_inv` are settled and asking them again is clauses for
        nothing.
        """
        obs = self._obligations(tpl, prp, inductive=False, implies=False,
                                ranked=True)
        return script_for(self._ask_for(tpl.rank, obs, fixed=fixed))

    def _obligations(self, tpl: Template, prp, *, inductive: bool,
                     implies: bool, ranked: bool) -> list:
        """The obligations, one per case of the round: each case's guard is
        a hypothesis of its own, and its folded state the successor."""
        tm, ob = self.tm, self.ob

        def rows(vs):
            return self._rows(tpl, ob.flatten(ob.s, vs))

        inv = [("inv", rows)]
        obligations = []
        # init_inv. The initial state is a function of the inputs alone, so
        # this is quantified over the inputs and split on its own
        # conditions -- a guard from the round would leave a latched-state
        # variable here that nothing binds.
        if inductive:
            for br in self.entry:
                q = obl.init_inv(ob.case(br.state), [rows], given=br.guard)
                obligations.append(self._forall(self.inputs, self._stated(q)))
        for br in self.step:
            case = ob.case(br.state)
            if inductive:
                q = obl.step_inv(case, inv, [rows], given=br.guard)
                obligations.append(self._forall(self.vars, self._stated(q)))
            if ranked:
                # hrank, ite-free: positive where the property fails, and
                # smaller after the round. Implies the `Int.toNat` form.
                now = self._rank(tpl, self.here)
                drops = self._and([
                    tm.mkTerm(Kind.GT, now, tm.mkInteger(0)),
                    tm.mkTerm(Kind.LT, self._rank(tpl, self._cols(br)), now),
                ])
                q = obl.ranked(case, inv, prp, drops, given=br.guard)
                obligations.append(self._forall(self.vars, self._stated(q)))
        if implies:
            q = obl.inv_imp_P(ob, inv, prp)
            obligations.append(self._forall(self.state, self._stated(q)))
        return obligations

    def _stated(self, q):
        """`q` as the implication Vampire reads, a literal `true` left out."""
        return self._implies(self._and([h for _, h in q.hyps]), q.goal)

    def _ask_for(self, holes: tuple[str, ...], obligations: list,
                 fixed: "dict | None" = None, bounded: bool = False):
        """`obligations`, with `fixed` filled in and `holes` left open.

        `bounded` confines each open hole to the window, and it is asked for
        only where the obligations leave the coefficients loose. Measured
        both ways over `tests/limits`: bounding costs the safety questions
        `m_max` and `m_toward5`, whose `inv -> prp` obligation pins the
        interval already so the window is clauses for nothing; and it is
        what makes the invariant-only question answerable at all --
        `m_countdown`'s does not come back inside 30 s free and answers
        `[0,100]` in three seconds bounded. A refutation only has to produce
        *a* witness, and where nothing else pins one down a window is what
        makes producing one a finite matter rather than a guess.
        """
        tm = self.tm
        body = self._and(obligations)
        if fixed:
            body = body.substitute(
                [self._hole(h) for h in fixed],
                [tm.mkInteger(v) for v in fixed.values()])
        parts = []
        if bounded:
            window = tm.mkInteger(self.window)
            parts = [b for h in holes for b in (
                tm.mkTerm(Kind.LEQ, tm.mkTerm(Kind.NEG, window),
                          self._hole(h)),
                tm.mkTerm(Kind.LEQ, self._hole(h), window))]
        out = tm.mkTerm(Kind.EXISTS,
                        tm.mkTerm(Kind.VARIABLE_LIST,
                                  *[self._hole(h) for h in holes]),
                        self._and(parts + [body]))
        refuse_ites(out)
        return out

    def _implies(self, hyp, goal):
        if hyp.getKind() == Kind.CONST_BOOLEAN and hyp.getBooleanValue():
            return goal
        return self.tm.mkTerm(Kind.IMPLIES, hyp, goal)

    def _cols(self, br: Branch) -> list:
        """A branch's successor state, one term per column."""
        return self.ob.flatten(self.ob.s, list(br.state))

    def _forall(self, vs: list, body):
        """`body` over the constants, quantified over the variables for them.

        The substitution is what turns `v_s0` into the bound `s0`, and only
        the constants `vs` stands for are substituted: a constant left over
        is one this obligation is not quantified over, and it is refused by
        name rather than printed as a symbol nothing declares. That is the
        entry obligation's own failure mode -- a guard from the round
        mentions the latched state, which nothing binds there.
        """
        names = {str(v) for v in vs}
        pairs = [(c, v) for c, v in zip(self.consts, self.vars)
                 if str(v) in names]
        if pairs:
            body = body.substitute([c for c, _ in pairs],
                                   [v for _, v in pairs])
        # What the old TPTP printer refused for free by having no variable
        # for a symbol: cvc5 prints a stray constant perfectly happily, and
        # Vampire reads an undeclared one as an error three steps later.
        stray = list(free_constants([body]).values())
        if stray:
            raise Refused(
                f"--infer vampire states an obligation over "
                f"{', '.join(sorted(str(v) for v in vs)) or 'nothing'}, and "
                f"it mentions {', '.join(str(c) for c in stray)}, which "
                f"nothing there binds"
            )
        if not vs:
            return body
        return self.tm.mkTerm(Kind.FORALL,
                              self.tm.mkTerm(Kind.VARIABLE_LIST, *vs), body)
