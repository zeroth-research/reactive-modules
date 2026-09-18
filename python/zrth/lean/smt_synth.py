"""Certificates cvc5 searches for itself: one fixed shape, or one grammar.

Two routes stand on this module, and they are the same idea at two
strengths.  ``--infer smt-linear`` fixes the *shape* of the answer -- a
ranking function linear in the state, an invariant that is a conjunction of
linear inequalities -- and asks one quantified query per shape:
``exists c. forall s. obligations(c . s)``.  ``--infer sygus`` fixes a
*grammar* instead and hands cvc5's SyGuS invariant track the three formulas
``G P`` is made of.

Neither is a search that gives up.  A refuted template query is a **proof
that the shape is empty**, and that is worth keeping: it is exactly what an
LLM route should be told before it proposes the same shape again.  So a
route here does not only return a certificate, it returns a
:class:`Search` -- what was looked for, and what became of the looking --
and ``record`` leaves that in ``artifacts/``, a found predicate as a
resumable ``inv``/``ranking`` and an empty space as a note the next run's
prompt can carry.

Measured on the `tests/limits` fixtures with cvc5 1.3.4, which is what the
shapes below are chosen from:

* a linear rank is decided in **2-30 ms** either way -- `m_countdown` sat
  (`rank = s0`), `m_toward5`, `m_twovars` and `m_lex` all unsat, which is the
  useful half: those three need a branch, and now nothing has to discover
  that by proposing one.
* a linear invariant costs what its *conjunct count* costs, not what the
  module costs: `m_countdown |- G (s0 <= 100)` is 11 ms at one row and does
  not finish in 30 s at two.  So the rows are tried in order and each one is
  a separate query, rather than asking for the widest conjunction outright.
* the SyGuS invariant track finds `m_step2`'s `x` even in **11 ms** through
  `addSygusInvConstraint`, where the same four constraints hand-rolled
  around a rank as well did not return in 300 s.  The dedicated track is not
  a convenience here, it is the difference.

**Everything is on cvc5's own leash, and `tlimit-per` is the option that
holds it.**  Measured: `tlimit` alone does *not* stop `checkSynth` (a 1000 ms
limit ran 13.9 s and returned a solution) and does not stop a quantified
`checkSat` either (a 20 s limit ran past 400 s); `tlimit-per` stops both
(1001 ms and 5007 ms, `unknown`).  `smt_query._solver` already sets the
pair, and :func:`bounded_solver` is the same pair for the searches here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .common import Refused

try:
    import cvc5                                      # type: ignore
    from cvc5 import Kind
except ImportError:                                  # pragma: no cover
    cvc5 = None
    Kind = None


# ══════════════════════════════════════════════════════════════════════════
# The module, as terms
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class SynthContext:
    """One module and its property in cvc5, for a search over shapes.

    The same front end the prompt routes use (`ModuleSMT` under
    `CegarPromptEnv`), for the same reason: the property arrives as SMT-LIB
    over ``s0..sN-1`` and that is the namespace every predicate this package
    reads or writes is in, so a certificate found here is in it too.
    """

    module: object
    tm: object
    msmt: object
    env: object
    prp: object
    init_pre: object
    update_pre: object

    @property
    def state(self) -> list:
        return self.env.state_vars

    @property
    def extl_latched(self) -> list:
        return self.env.extl_latched_vars

    @property
    def extl_next(self) -> list:
        return self.env.extl_next_vars

    @property
    def names(self) -> list[str]:
        return [f"s{i}" for i in range(len(self.state))]

    @classmethod
    def build(cls, module, cd, *, route: str,
              reals: bool = False) -> "SynthContext":
        """The context, or the reason this module is not one these routes read.

        What is refused here is what *neither* search can weigh: a Real
        component (the templates are integer arithmetic and a ranking
        function has to land in `Nat`) and a matrix-shaped one (its elements
        would each be a column, which is a wider change than a sort check).
        Which of the remaining sorts a particular route reads is that
        route's question, asked through :func:`int_readings` -- the grammar
        one takes integers only, the template one weighs Bools and
        bitvectors as well.

        ``reals`` is a route saying it has an answer to the Real question:
        `--infer houdini` states its candidates in real arithmetic and reads
        a ranking function through `to_int`, so it asks for the component
        rather than an integer reading of it.
        """
        if cvc5 is None:                             # pragma: no cover
            raise Refused(
                f"--infer {route} needs the `cvc5` package, which is not "
                f"importable. Install with: uv pip install cvc5"
            )
        from .smt_module import ModuleSMT, keep_alive
        from .smt_prompt import CegarPromptEnv, parse_predicate

        if not isinstance(cd.prp, str):
            raise Refused(
                f"--infer {route} needs a property: pass --safety or --buchi"
            )
        tm = cvc5.TermManager()
        msmt = ModuleSMT(tm=tm, module=module)
        env = CegarPromptEnv(msmt)
        keep_alive(tm, msmt, env)

        def weighable(sort) -> bool:
            if sort.isInteger() or sort.isBoolean() or sort.isBitVector():
                return True
            return reals and sort.isReal()

        bad = [
            f"s{i} is {sort}"
            for i, sort in enumerate(env.state_sorts)
            if not weighable(sort)
        ]
        if bad:
            takes = ("reads a scalar state" if reals else
                     "weighs each state component as an integer")
            why = "" if reals else (
                "A Real component has no integer reading and a ranking "
                "function has to land in `Nat`. "
            )
            raise Refused(
                f"--infer {route} {takes}, and this module has one it "
                f"cannot: {', '.join(bad)}. {why}A matrix-shaped component "
                f"would be a column per element. Use --infer ai-cegis."
            )
        true = tm.mkBoolean(True)
        return cls(
            module=module,
            tm=tm,
            msmt=msmt,
            env=env,
            prp=parse_predicate(env, cd.prp),
            init_pre=(parse_predicate(env, cd.init_pre)
                      if isinstance(cd.init_pre, str) else true),
            update_pre=(parse_predicate(env, cd.update_pre)
                        if isinstance(cd.update_pre, str) else true),
        )

    # --- terms the searches build on ------------------------------------

    def at(self, term, vs: list):
        """`term` with the state variables replaced by `vs`."""
        return term.substitute(self.state, vs)

    def with_inputs(self, term, el: list, en: list):
        """`term` with the env's input constants replaced by `el` / `en`.

        A precondition is written over `e0..`/`el0..`, which are constants of
        the parsing env; an obligation quantifies its own, so the two have to
        be brought together exactly as `magic.cegar._subst_inputs` does it.
        """
        old = list(self.extl_next) + list(self.extl_latched)
        new = list(en) + list(el)
        return term.substitute(old, new) if old else term


@dataclass(frozen=True)
class Reading:
    """One state component as the integer a template weighs it by.

    An `Int` component is itself. A `Bool` one is `0`/`1` -- which is what
    makes `b` and `¬b` expressible as rows (`(ite s0 1 0) - 1 >= 0`), and
    what lets a rank fall when a flag flips. A bitvector is its unsigned
    value, `ubv_to_int`, so a counter in `BitVec 8` is ranked by the number
    it holds rather than by its bits.

    The lifting is only the *search space*. Every obligation is still stated
    against the module's own transition, in its own sorts, so a candidate
    that survives is correct about the bitvector and not about a story told
    over it -- wraparound included, since the query quantifies over every
    state the real transition can reach.
    """

    name: str                       # SMT-LIB over `s0..`, for printing
    term: object                    # the same, over `ctx.state`
    sort: str                       # "int" | "bool" | "bv"


def int_readings(ctx: "SynthContext", *, allow: tuple[str, ...],
                 route: str) -> list[Reading]:
    """The state components `route` can weigh, or why one of them is not.

    `allow` is the route's own contract rather than the context's: the
    grammar route's `synthFun` takes integer arguments and has nowhere to
    put a Bool column, while a template weighs whatever has an integer
    reading. Refused by name either way -- naming the component and the
    route that does read it is the difference between "try something else"
    and "try everything else".
    """
    tm = ctx.tm
    out: list[Reading] = []
    refused: list[str] = []
    for i, (var, sort) in enumerate(zip(ctx.state, ctx.env.state_sorts)):
        if sort.isInteger():
            kind = "int"
            name, term = f"s{i}", var
        elif sort.isBoolean():
            kind = "bool"
            name = f"(ite s{i} 1 0)"
            term = tm.mkTerm(Kind.ITE, var, tm.mkInteger(1), tm.mkInteger(0))
        else:
            kind = "bv"
            name = f"(ubv_to_int s{i})"
            term = tm.mkTerm(Kind.BITVECTOR_UBV_TO_INT, var)
        if kind not in allow:
            refused.append(f"s{i} is {sort}")
            continue
        out.append(Reading(name=name, term=term, sort=kind))
    if refused:
        raise Refused(
            f"--infer {route} reads a state of scalar integers, and this "
            f"module's is not one: {', '.join(refused)}. `--infer smt-linear` "
            f"weighs a Bool component as 0/1 and a bitvector as its unsigned "
            f"value, and `--infer ai-cegis` reads whatever cvc5 encodes."
        )
    return out


# ══════════════════════════════════════════════════════════════════════════
# Grammar and template material
# ══════════════════════════════════════════════════════════════════════════

# Always available, whatever the program says: the coefficients that make a
# difference, a comparison against zero, and a step of one.
_BASE_CONSTANTS = (-2, -1, 0, 1, 2)

# A literal wider than this is a constant matrix entry or an address, not a
# bound an invariant is stated against, and every one of them multiplies the
# grammar.
_CONSTANT_LIMIT = 10_000
_CONSTANT_COUNT = 16


def program_constants(ctx: SynthContext) -> tuple[int, ...]:
    """The integer literals this module and its property mention.

    Seeded rather than guessed, for the reason `tests/limits/README.md`
    records about Houdini: the bound that makes an invariant true is almost
    always a constant the program already contains -- `NNInv` needs
    `0 <= x <= 9`, whose `9` only the guard mentions -- and a grammar over
    invented numbers is both bigger and worse.
    """
    seen: set[int] = set(_BASE_CONSTANTS)
    roots = [ctx.prp, ctx.init_pre, ctx.update_pre]
    roots += list(ctx.msmt.init_state(ctx.extl_next))
    roots += list(ctx.msmt.update_state(ctx.state, ctx.extl_latched, ctx.extl_next))
    stack = list(roots)
    visited: set[int] = set()
    while stack:
        t = stack.pop()
        key = t.getId() if hasattr(t, "getId") else id(t)
        if key in visited:
            continue
        visited.add(key)
        if t.getKind() == Kind.CONST_INTEGER:
            value = int(t.getIntegerValue())
            if abs(value) <= _CONSTANT_LIMIT:
                # A literal is worth its neighbours *and* its negation.
                #
                # The neighbours because the bound is usually the guard's
                # constant and the one step past it -- `x + 1 == 10` bounds
                # `x` at 9 -- which is the normalisation `tests/limits`
                # records Houdini needing before it could seed them.
                #
                # The negation because of how an atom is *written*. The
                # grammar's atoms are `c0 + c1*s0 + ... <= 0`, so the
                # invariant `s0 <= 100` of a module that counts down from
                # 100 is spelled `-100 + s0 <= 0` -- and a set carrying
                # `100` without `-100` cannot say it. Measured: with the
                # negations missing, `m_countdown |- G (s0 <= 100)` came
                # back as a *proof* that the space was empty, which it was,
                # for a space that should not have been that one.
                near = {value, value - 1, value + 1}
                seen.update(near | {-v for v in near})
        stack.extend(list(t))
    out = sorted(seen, key=lambda v: (abs(v), v))[:_CONSTANT_COUNT]
    return tuple(sorted(out))


def moduli(constants: tuple[int, ...]) -> tuple[int, ...]:
    """The `k` worth a `(= (mod lin k) 0)` atom.

    A congruence is the one fact Houdini's lattice cannot state -- `m_step2`
    steps by two, so its invariant is that `x` is even and nothing in a
    lattice of signs and pairwise relations says so. Which `k` to offer is
    read off the program the same way the constants are: a module that steps
    by two mentions two.
    """
    ks = {2} | {abs(c) for c in constants if 2 <= abs(c) <= 16}
    return tuple(sorted(ks))


def bounded_solver(tm, budget, **options):
    """A solver on this run's leash, with `options` set before the logic.

    Both limits, because only one of them binds what these routes run:
    `tlimit` does not stop `checkSynth` or a quantified `checkSat`, and
    `tlimit-per` stops both. `smt_query._solver` sets the same pair for the
    obligation queries.
    """
    solver = cvc5.Solver(tm)
    limit = _limit_ms(budget)
    solver.setOption("tlimit-per", str(limit))
    solver.setOption("tlimit", str(limit))
    for name, value in options.items():
        solver.setOption(name.replace("_", "-"), value)
    return solver


def _limit_ms(budget) -> int:
    """How long the next query may take, from the run's budget."""
    if budget is None:
        from .smt_query import DEFAULT_CALL_MS

        return DEFAULT_CALL_MS
    return max(1, budget.next_call_ms())


# ══════════════════════════════════════════════════════════════════════════
# SMT-LIB out
# ══════════════════════════════════════════════════════════════════════════


def smt_int(value: int) -> str:
    """An integer literal. SMT-LIB has no negative numerals, so one is a negation."""
    value = int(value)
    return str(value) if value >= 0 else f"(- {-value})"


def affine_smt(const: int, coeffs, names) -> str:
    """``k + Σ cᵢ·sᵢ`` as SMT-LIB, with the terms that say nothing dropped.

    Written out here rather than printed from the model's own term: what this
    string has to survive is being parsed back and rendered as the body of a
    Lean definition, and a printer that `let`-binds a repeated subterm
    produces something that cannot be one (`magic.learn._smt_term` refuses
    exactly that).
    """
    parts = [] if not int(const) else [smt_int(const)]
    for c, n in zip(coeffs, names):
        c = int(c)
        if c == 1:
            parts.append(n)
        elif c == -1:
            parts.append(f"(- {n})")
        elif c:
            parts.append(f"(* {smt_int(c)} {n})")
    if not parts:
        return "0"
    return parts[0] if len(parts) == 1 else "(+ " + " ".join(parts) + ")"


def conjunction_smt(parts) -> str:
    """SMT-LIB Bool terms as one. Empty is `true`, one is itself."""
    rendered = list(dict.fromkeys(p for p in parts if p and p != "true"))
    if not rendered:
        return "true"
    if len(rendered) == 1:
        return rendered[0]
    return "(and " + " ".join(rendered) + ")"


def body_of(solution) -> tuple:
    """A synthesised `(lambda ((x0 Int) ...) body)` as `(body, bound vars)`.

    `getSynthSolution` answers with the function, and what the rest of this
    package reads is a predicate over `s0..sN-1`, so the caller substitutes
    its own variables for the bound ones. A constant solution is not a
    lambda, and then there is nothing to substitute.
    """
    if solution.getKind() != Kind.LAMBDA:
        return solution, []
    return solution[1], list(solution[0])


# ══════════════════════════════════════════════════════════════════════════
# What became of the looking
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Search:
    """One search for one field of a certificate, and its outcome.

    Three outcomes, and the middle one is why this class exists:

    * ``found`` -- the predicate, already checked by construction.
    * ``exhausted`` -- the space was *decided empty*. Not a failure: a proof
      that nothing of this shape works, which is a fact about the module
      worth as much as a certificate and costs the next run nothing to know.
    * neither -- the query did not finish inside the budget. Nothing is
      known, and the note says so rather than implying absence.
    """

    field: str                      # "inv" | "ranking" -- the artifact role
    space: str                      # prose: the shape that was searched
    found: str | None = None
    exhausted: bool = False
    detail: str = ""                # what to do instead, or why it stopped
    # How the space was decided, for the note to say. A template query and a
    # finite grammar are both proofs and are not the same proof, and a note
    # that described one as the other would be wrong about the only thing it
    # is for.
    how: str = "cvc5 decided the whole space at once"

    @property
    def decided(self) -> bool:
        return self.found is not None or self.exhausted

    @property
    def note(self) -> str:
        """The note an empty space leaves, written to be read by an LLM.

        It goes into `artifacts/` and out again into the `--infer ai-cegis`
        prompt verbatim, so it is prose about this module and not a log line:
        what was ruled out, that it was *proved* rather than not found, and
        what that leaves.
        """
        if self.exhausted:
            return (
                f"{self.space} This is a proof that the space is empty, not "
                f"a search that ran out of time: {self.how}. "
                f"{self.detail}".strip()
            )
        return (
            f"{self.space} The query did not finish inside the budget, so "
            f"nothing is known either way: the space may or may not contain "
            f"one. {self.detail}".strip()
        )


def record(store, search: Search, *, log=None) -> None:
    """Leave `search` in `artifacts/` for the next run.

    A found predicate is written as its own role, in SMT-LIB and with the
    status that makes it resumable -- `infer_route.FIXABLE` is what a later
    `--infer ai-cegis` will take as given. An empty space is written as a
    note, because what it is worth is what it *says*, and `no_solution` is
    the status that tells a consumer the saying was proved.
    """
    if store is None:
        return
    if search.found is not None:
        store.put(
            search.field,
            search.found,
            status="proved",
            language="smt",
            what=(
                f"The {search.field} this run found. {search.space} It "
                f"satisfies the obligations by construction: cvc5 answered "
                f"the search, so what came back is a witness, and "
                f"`--pre-check cvc5` restates the obligations against it."
            ),
        )
        return
    store.note(
        search.note + "\n",
        status="no_solution" if search.exhausted else "unknown",
        what=(
            f"What this run ruled out while looking for a {search.field}: "
            f"a shape that contains no certificate for this module and "
            f"property."
            if search.exhausted else
            f"A search for a {search.field} that did not finish."
        ),
        why=(
            "The space was decided empty rather than searched unsuccessfully, "
            "so a route that proposes the same shape again is proposing "
            "something that provably does not work."
            if search.exhausted else
            "The budget ran out first; nothing is known about the space."
        ),
    )
