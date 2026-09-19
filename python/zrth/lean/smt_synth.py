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

* a linear rank is decided in **2-30 ms** under a fixed invariant --
  `m_countdown` sat (`rank = s0`), `m_toward5`, `m_twovars` and `m_lex` all
  unsat under `true`, which is the useful half: nothing has to discover that
  by proposing one.  Searching the rank *with* an invariant to rank under
  finds one for `m_toward5` and `m_lex` and costs a second or two; only
  `m_twovars` is empty at every width.
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

import math
from dataclasses import dataclass
from fractions import Fraction

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
              takes: tuple[str, ...] = ("int", "bool", "bv")) -> "SynthContext":
        """The context, or the reason this module is not one `route` reads.

        ``takes`` is the route's contract, as :data:`READING_KINDS`, and it
        is the same tuple :func:`readings` is given -- a route that declines
        a kind here declines it there, so there is one answer to "what can
        this route read" rather than two that have to agree. What a sort
        costs beyond being readable is the route's own business:
        `--infer sygus` takes the context's Bool and then refuses it in
        `readings`, because its `synthFun` has nowhere to put a Bool column
        even though the context can weigh one.

        A kind outside ``takes`` is refused here, before a solver starts,
        and named: `--infer houdini` states its candidates in real
        arithmetic and so asks for `real`; `--infer smt-linear` reads a
        Real through a floor and asks for it too.
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

        bad = [
            f"s{i} is {sort}"
            for i, sort in enumerate(env.state_sorts)
            if reading_kind(sort) not in takes
        ]
        if bad:
            reads = ("reads a scalar state" if "real" in takes else
                     "weighs each state component as an integer")
            why = "" if "real" in takes else (
                "A Real component has no integer reading and a ranking "
                "function has to land in `Nat`. "
            )
            shaped = "" if "tuple" in takes else (
                "A matrix-shaped component would be a column per element. "
            )
            raise Refused(
                f"--infer {route} {reads}, and this module has one it "
                f"cannot: {', '.join(bad)}. {why}{shaped}"
                f"Use --infer ai-cegis."
            )
        # The transition, encoded once here rather than lazily by whichever
        # route first builds an obligation out of it. A module with an
        # operator cvc5 has no term for -- `Transpose`, say -- is a refusal
        # either way, but raised from the middle of a search it is a
        # `ValueError` traceback several steps past the fact, for a user who
        # asked about a module. Named up front instead, which is what
        # `native.lean_gaps` does for the Lean side.
        try:
            msmt.init_state(env.extl_next_vars)
            msmt.update_state(env.state_vars, env.extl_latched_vars,
                              env.extl_next_vars)
        except ValueError as e:
            raise Refused(
                f"--infer {route} encodes this module's transition for cvc5, "
                f"and an operator in it has no encoding. {e}"
            ) from e
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
    """One quantity a template can weigh a state component by.

    An `Int` component is itself. A `Bool` one is `0`/`1` -- which is what
    makes `b` and `¬b` expressible as rows (`(ite s0 1 0) - 1 >= 0`), and
    what lets a rank fall when a flag flips. A bitvector is its unsigned
    value, `ubv_to_int`, so a counter in `BitVec 8` is ranked by the number
    it holds rather than by its bits. A Real is a floor, or itself, and
    which one is the caller's `floor_reals` -- see :func:`component_readings`
    for why that is a question about ranking rather than about sorts.

    The lifting is only the *search space*. Every obligation is still stated
    against the module's own transition, in its own sorts, so a candidate
    that survives is correct about the bitvector and not about a story told
    over it -- wraparound included, since the query quantifies over every
    state the real transition can reach.

    A component need not have exactly one reading: :func:`component_readings`
    returns a list, so a matrix-shaped component is a column per element.
    """

    name: str                       # SMT-LIB over `s0..`, for printing
    term: object                    # the same, over `ctx.state`
    kind: str                       # one of `READING_KINDS`, of the *element*
    index: int                      # which state component it reads
    slot: "int | None" = None       # which element of it, or the whole thing


# Every kind of state component there is a reading for. A sort absent from
# here has none at all, which is a different refusal from a route declining
# a kind that exists: the first is this front end's limit, the second is the
# route's own contract.
#
# `tuple` is the odd one out, and deliberately: it is not a kind anything is
# *read* as, it is the kind that has elements. A route that takes it is
# saying it states itself over elements rather than over components -- see
# :func:`component_slots` -- and what each element is read as is one of the
# other four.
READING_KINDS = ("int", "bool", "bv", "real", "tuple")


def reading_kind(sort) -> "str | None":
    """Which of :data:`READING_KINDS` a cvc5 sort is read as, or `None`."""
    for name, is_it in (("int", sort.isInteger), ("bool", sort.isBoolean),
                        ("bv", sort.isBitVector), ("real", sort.isReal),
                        ("tuple", sort.isTuple)):
        if is_it():
            return name
    return None


def component_slots(ctx: "SynthContext", i: int) -> list[tuple]:
    """The scalars state component `i` is made of: `(slot, sort)` apiece.

    A scalar component is one slot, `None` -- it *is* the scalar, and there
    is no selector to write. A matrix-shaped one is a slot per element, in
    the order `smt_encode.wire_sort` laid the tuple out: flat, `m*n`
    elements of one scalar sort, row by row. Flat is why this is one level
    and not a recursion -- a 2-D component is `(Tuple Int Int ...)`, never a
    tuple of tuples -- and it is the same order `smt_to_lean` renders the
    selectors in and the Lean state is written in, so a certificate stated
    over slot `k` here means element `k` there.

    This is the one place that answers "what is this component made of",
    and it is separate from :func:`component_readings` because the routes
    that need elements do not all want them read the same way: `smt-linear`
    wants an integer per element, `houdini` wants each element in its own
    sort so a Real one keeps its rationals, and `vampire` wants a variable
    per element because its prover has no tuples at all.
    """
    sort = ctx.env.state_sorts[i]
    if not sort.isTuple():
        return [(None, sort)]
    return list(enumerate(sort.getTupleSorts()))


def element(ctx: "SynthContext", i: int, slot: "int | None") -> tuple:
    """One slot of component `i`: how it is written, and the term for it."""
    var = ctx.state[i]
    if slot is None:
        return f"s{i}", var
    ctor = ctx.env.state_sorts[i].getDatatype()[0]
    return (f"((_ tuple.select {slot}) s{i})",
            ctx.tm.mkTerm(Kind.APPLY_SELECTOR, ctor[slot].getTerm(), var))


def component_readings(ctx: "SynthContext", i: int, *, scale: int = 1,
                       floor_reals: bool = True) -> list[Reading]:
    """Every arithmetic reading of state component `i`, in column order.

    One reading for a scalar; one *per element* for a matrix-shaped
    component, which cvc5 encodes as a tuple and which is projected with the
    selectors `smt_to_lean` already renders.

    An element is read by *its own* sort rather than by the component's,
    which is the only reading of a matrix of Bools or of Reals that is
    arithmetic at all: the selector alone is a Bool, and a template that
    multiplies it by a coefficient is a sort error four steps later.

    ``floor_reals`` is what a Real costs, and it is the caller's question
    rather than this function's. A **ranking** function has to land in
    `Nat`, so a route that weighs one reads a Real through `to_int` -- the
    floor `Int.toNat` sees -- after multiplying by ``scale``, because
    flooring a quantity that falls by less than one need not fall at all.
    An **invariant** has no such obligation: it is a predicate, and
    flooring one is not conservative in either direction -- `to_int s0 +
    to_int s1 <= 1` neither implies nor is implied by `s0 + s1 <= 1.0`, as
    `s0 = s1 = 0.6` shows. So a route that only ever states an invariant
    passes `floor_reals=False` and keeps the rationals, and its linear
    combinations live in Real arithmetic instead.
    """
    tm = ctx.tm
    out: list[Reading] = []
    for slot, sort in component_slots(ctx, i):
        name, var = element(ctx, i, slot)
        kind = reading_kind(sort)
        if kind == "int":
            out.append(Reading(name, var, kind, i, slot))
        elif kind == "bool":
            out.append(Reading(f"(ite {name} 1 0)",
                               tm.mkTerm(Kind.ITE, var, tm.mkInteger(1),
                                         tm.mkInteger(0)), kind, i, slot))
        elif kind == "bv":
            out.append(Reading(f"(ubv_to_int {name})",
                               tm.mkTerm(Kind.BITVECTOR_UBV_TO_INT, var),
                               kind, i, slot))
        elif kind == "real":
            if not floor_reals:
                out.append(Reading(name, var, kind, i, slot))
                continue
            inner = var
            if scale != 1:
                inner = tm.mkTerm(Kind.MULT, tm.mkReal(scale, 1), var)
                name = f"(* {scale}.0 {name})"
            out.append(Reading(f"(to_int {name})",
                               tm.mkTerm(Kind.TO_INTEGER, inner), kind, i, slot))
    return out


def readings(ctx: "SynthContext", *, allow: tuple[str, ...], route: str,
             scale: int = 1, floor_reals: bool = True) -> list[Reading]:
    """The columns `route` weighs this module by, or why one component is not.

    `allow` is the route's own contract rather than the context's: a route
    whose obligations are stated in integer arithmetic has nowhere to put a
    Real column, while a template weighs whatever has a reading. Refused by
    name either way -- naming the component and the route that does read it
    is the difference between "try something else" and "try everything
    else".

    ``scale`` and ``floor_reals`` are passed through to
    :func:`component_readings`, which is where what a Real costs is
    written down.
    """
    out: list[Reading] = []
    refused: list[str] = []
    for i, sort in enumerate(ctx.env.state_sorts):
        if reading_kind(sort) not in allow:
            refused.append(f"s{i} is {sort}")
            continue
        out.extend(component_readings(ctx, i, scale=scale,
                                      floor_reals=floor_reals))
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


def program_rationals(ctx: SynthContext) -> tuple:
    """The Real literals this module and its property mention, as fractions.

    :func:`program_constants` reads the integer ones, and an LRA module has
    none of those to read: `m_lra_lin` writes `0.0`, `1.0` and `5.0`, so the
    bounds, the value sets and the lattice a state is drawn on all come from
    here instead.
    """
    roots = [ctx.prp, ctx.init_pre, ctx.update_pre]
    roots += list(ctx.msmt.init_state(ctx.extl_next))
    roots += list(ctx.msmt.update_state(ctx.state, ctx.extl_latched,
                                        ctx.extl_next))
    seen: set = set()
    visited: set = set()
    stack = list(roots)
    while stack:
        t = stack.pop()
        if t.getId() in visited:
            continue
        visited.add(t.getId())
        if t.getKind() == Kind.CONST_RATIONAL:
            q = t.getRealValue()
            if abs(q) <= _CONSTANT_LIMIT:
                # As `program_constants` does: a literal is worth its
                # neighbours, since the bound is usually one step past the
                # guard's own constant.
                seen.update({q, q - 1, q + 1, -q, 1 - q, -1 - q})
        stack.extend(list(t))
    return tuple(sorted(seen)[:32])


# The scale a real-valued ranking function is floored after is at most this:
# the scale becomes a literal in the rank, so it bounds how long the emitted
# certificate can get.
_MAX_SCALE = 64


def denominator_scale(rationals) -> int:
    """What a real-valued ranking function is multiplied by before flooring.

    `Int.toNat` reads a rank through `to_int`, and a quantity that falls by
    less than one need not floor to a smaller number -- `m_lra_half` steps
    by `1/2`, where `to_int x` stalls on every other round. An affine
    transition moves a state by the literals it writes, so the least common
    denominator of those is a scale at which a fall is a whole number.
    """
    scale = 1
    for q in rationals:
        den = Fraction(q).denominator
        scale = scale * den // math.gcd(scale, den)
        if scale > _MAX_SCALE:
            return _MAX_SCALE
    return scale


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
