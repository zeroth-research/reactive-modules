"""Shape-directed tactic generation for the certificate proofs.

The three obligations (`init_inv`, `step_inv`, `hrank`) used to be closed by
one fixed tactic chain, identical for every module. That is wasteful in both
directions. An integer module with a linear ranking wants `omega` and nothing
else, yet paid for `decide` and `bv_decide` on every goal — and `bv_decide`,
which can only ever help a BitVec module, took one BitVec certificate from
10s to 977s without closing anything. And a shape the
chain did not anticipate had no way to ask for the step it needed: a Real
state whose invariant pins exact values needs its equalities *substituted*
before any closer runs, which is a prep step, not one more alternative to try
at the end.

So the plan is read off the obligations instead:

* which theories the state actually uses — `omega` is emitted only where
  integers are in play and `decide` only for a finite state, and `bv_decide`
  not at all (see the note beside `decide` below);
* whether the predicates are nonlinear — `nlinarith` is expensive and useless
  on a linear goal;
* whether they branch, conjoin or disjoin — `split_ifs` and `casesm*` are
  skipped when there is nothing to split;
* whether a Real state carries equalities — then `simp_all` runs in *prep*, so
  every closer afterwards sees numerals rather than an opaque `⌊4 - x⌋`;
* how big the module is — `maxRecDepth` scales with the term count, and the
  heartbeat budget is left low unless something in the plan is known to be
  slow, so a failing proof fails fast;
* whether the state is finite *and* narrow — then each element is enumerated
  over its two values before anything tries to close, which is the only way
  to discharge a branch that is contradictory purely because the state has
  finitely many inhabitants.

`plan_for` returns a `TacticPlan`; `Certificate.lean.j2` renders it as two
macros, `cert_prep` and `cert_close`, which the three proofs then share.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from zrth import Bool, Int, Real, BitVec

from .common import _accessor, dtype_shape, itype_name


# ══════════════════════════════════════════════════════════════════════
# Feature detection
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Features:
    """What the module and its certificate predicates actually contain."""

    theories: set[str] = field(default_factory=set)
    ops: set[str] = field(default_factory=set)
    n_terms: int = 0
    n_slots: int = 0

    # (accessor, lemma) per state element that ranges over exactly two values,
    # empty if any element does not. See `_two_valued_slots`.
    two_valued: list[tuple[str, str]] = field(default_factory=list)

    # `min`/`max`/`ite` occurrences in the predicates. Each is a case: omega
    # splits `min`/`max` internally, `split_ifs` fans an `ite` out, and
    # `hrank` mentions the ranking twice. This, not the module's size, is
    # what makes a neural certificate expensive.
    n_branch: int = 0

    # Distinct branch *conditions*. `split_ifs` splits once per condition
    # and reuses the hypothesis for repeats, so this, not `n_branch`, is
    # what its fan-out costs. Falls back to `n_branch` without cvc5.
    n_conditions: int = 0

    has_ite: bool = False
    has_and: bool = False
    has_or: bool = False
    has_eq: bool = False
    has_floor: bool = False
    nonlinear: bool = False

    @property
    def has_int(self) -> bool:
        # Bool predicates decide into `0`/`1` arithmetic, and every ranking is
        # a `Nat`, so omega is in play whenever nothing forces it out.
        return bool({"Int", "Bool"} & self.theories) or not self.theories

    @property
    def has_real(self) -> bool:
        return "Real" in self.theories

    @property
    def has_bitvec(self) -> bool:
        return "BitVec" in self.theories

    @property
    def finite_state(self) -> bool:
        """A state `decide` could enumerate: Bool or BitVec, no Int or Real."""
        return bool(self.theories) and not ({"Int", "Real"} & self.theories)


_SORT_NAMES = ((Bool, "Bool"), (Int, "Int"), (Real, "Real"), (BitVec, "BitVec"))


def _sort_name(dtype) -> str:
    for cls, name in _SORT_NAMES:
        if isinstance(dtype, cls):
            return name
    return "Int"


# A `*` whose two operands both mention the state parameter. An operand is a
# parenthesised group or an application chain such as `s.1 0 0`, so scanning
# has to stop at an operator or an unmatched paren, not at the first space.
_VAR = re.compile(r"(?:^|[^A-Za-z0-9_])(?:s|e|el)\d*(?:[.\[ )]|$)")
_BOUNDARY = set("+-*/%<>=\u2264\u2265\u2260\u2227\u2228\u00ac,;")


def _skip_group_left(text: str, k: int) -> int:
    depth = 0
    while k >= 0:
        if text[k] == ")":
            depth += 1
        elif text[k] == "(":
            depth -= 1
            if depth == 0:
                return k
        k -= 1
    return 0


def _skip_group_right(text: str, k: int) -> int:
    depth = 0
    while k < len(text):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return len(text) - 1


def _operand_left(text: str, i: int) -> str:
    """The operand ending just before position `i`."""
    j = i - 1
    while j >= 0 and text[j] == " ":
        j -= 1
    if j < 0:
        return ""
    if text[j] == ")":
        return text[_skip_group_left(text, j) : j + 1]
    k = j
    while k >= 0 and text[k] not in _BOUNDARY and text[k] != "(":
        if text[k] == ")":
            k = _skip_group_left(text, k)
        k -= 1
    return text[k + 1 : j + 1]


def _operand_right(text: str, i: int) -> str:
    """The operand starting just after position `i`."""
    j = i + 1
    while j < len(text) and text[j] == " ":
        j += 1
    if j >= len(text):
        return ""
    if text[j] == "(":
        return text[j : _skip_group_right(text, j) + 1]
    k = j
    while k < len(text) and text[k] not in _BOUNDARY and text[k] != ")":
        if text[k] == "(":
            k = _skip_group_right(text, k)
        k += 1
    return text[j:k]


def _mentions_var(operand: str) -> bool:
    return bool(_VAR.search(operand))


def is_nonlinear(text: str) -> bool:
    """True if any `*` in `text` multiplies two state-dependent operands.

    Multiplication by a literal weight — every affine layer of a net — stays
    linear and must not drag `nlinarith` in. Over-reporting is the safe
    direction: an extra `nlinarith` is only ever tried after the cheaper
    closers have failed, whereas a missed one loses the proof.
    """
    for m in re.finditer(r"\*", text):
        i = m.start()
        if text[i : i + 2] == "**":
            continue
        if _mentions_var(_operand_left(text, i)) and _mentions_var(
            _operand_right(text, i)
        ):
            return True
    return False


# A finite state is only worth enumerating if the fan-out stays small: each
# slot doubles the number of branches every later tactic has to run on.
MAX_ENUMERABLE_SLOTS = 4


def _two_valued_slots(ctx) -> list[tuple[str, str]]:
    """One `(accessor, lemma)` per state element that has exactly two values.

    Returns `[]` unless *every* element does, since enumerating some of a
    state and not the rest buys nothing. `accessor` is relative to the state
    variable, e.g. `.2.1 0 0`, so the caller can prefix whichever binder the
    obligation uses.
    """
    wires = list(ctx.ctrl_next)
    n = len(wires)
    out: list[tuple[str, str]] = []
    for i, w in enumerate(wires):
        dt = w.dtype
        if isinstance(dt, Bool):
            lemma = "Bool.eq_false_or_eq_true"
        elif isinstance(dt, BitVec) and dt._0 == 1:
            lemma = "BitVec.eq_zero_or_eq_one"
        else:
            # Wider BitVec has 2^w values and no two-way lemma; Int and Real
            # are not finite at all.
            return []
        acc = _accessor(i, n)
        shape = dtype_shape(dt)
        rows = shape[0] if shape else 1
        cols = shape[1] if len(shape) > 1 else 1
        for r in range(rows):
            for c in range(cols):
                out.append((f"{acc} {r} {c}", lemma))
    return out if len(out) <= MAX_ENUMERABLE_SLOTS else []


def features_for(ctx, pred_text: str, facts=None) -> Features:
    """Read the shape of `ctx`'s module and its certificate predicates.

    `facts` is `smt_query.PredicateFacts` when the predicates came from
    SMT-LIB and cvc5 could parse them: the same questions answered against
    the terms rather than against the printed Lean. It is `None` for
    predicates compiled from Python IR, and then everything below falls back
    to reading the text, exactly as it did before.
    """
    f = Features()

    for w in (*ctx.ctrl_next, *ctx.extl_next):
        f.theories.add(_sort_name(w.dtype))
        shape = dtype_shape(w.dtype)
        n = 1
        for d in shape:
            n *= d
        f.n_slots += n

    terms = list(ctx.atom.init) + list(ctx.atom.update)
    f.n_terms = len(terms)
    for t in terms:
        f.ops.add(itype_name(t.itype))
        for w in (*t.read, *t.write):
            f.theories.add(_sort_name(w.dtype))

    # The transition branches if it has an `Ite`, and a ReLU/Min/Max reduces
    # to a branch in the goal even though the term itself is not an `Ite`.
    f.has_ite = bool({"Ite", "ReLU", "Min", "Max"} & f.ops) or "if " in pred_text
    f.has_and = " ∧ " in pred_text
    f.has_or = " ∨ " in pred_text
    f.has_eq = " = " in pred_text
    f.has_floor = "⌊" in pred_text
    f.nonlinear = is_nonlinear(pred_text)
    text_branch = (
        pred_text.count("(max ") + pred_text.count("(min ") + pred_text.count("if ")
    )
    f.n_branch = text_branch
    if facts is not None:
        # Exact where the text is a guess. `n_branch` takes the larger of the
        # two: the terms cover only the fields that came from SMT-LIB, so a
        # predicate compiled from Python IR alongside them would otherwise go
        # uncounted, and over-counting only ever buys a bigger budget.
        f.nonlinear = facts.nonlinear
        f.has_and = f.has_and or facts.has_and
        f.has_or = f.has_or or facts.has_or
        f.has_eq = f.has_eq or facts.has_eq
        f.has_ite = f.has_ite or facts.has_ite
        f.n_branch = max(text_branch, facts.n_branch)
        f.n_conditions = facts.n_conditions
    else:
        f.n_conditions = text_branch
    if f.finite_state:
        f.two_valued = _two_valued_slots(ctx)
    return f


# ══════════════════════════════════════════════════════════════════════
# Plan
# ══════════════════════════════════════════════════════════════════════


@dataclass
class TacticPlan:
    prep: list[str]
    closers: list[str]
    max_rec_depth: int
    max_heartbeats: int
    features: Features
    # `smt_query.SolverHints` when `--smt-tactics=cvc5` ran and cvc5 had
    # something to add; `None` otherwise, and every property below degrades
    # to the tactic the plan would have emitted without it.
    hints: object | None = None

    @property
    def prep_tactic(self) -> str:
        """The prep steps as one `<;>`-chained tactic, each step optional."""
        if not self.prep:
            return "skip"
        return " <;> ".join(f"(try {p})" for p in self.prep)

    @property
    def close_tactic(self) -> str:
        return "first | " + " | ".join(self.closers)

    @property
    def state_case_tactic(self) -> str:
        """Enumerate the state, as the body of a macro over its binder `$v`.

        `decide` is in the closer list for a finite state but can never fire
        while the state is a free variable -- it evaluates closed
        propositions, and every goal here is quantified over `s`. Worse, a
        branch whose hypotheses are contradictory *only because the state is
        finite* (`x ≠ 0#1` together with `x ≠ 1#1`) reduces to a bare `False`
        that nothing in the chain can discharge: `contradiction` needs a
        literal `h`/`¬h` pair, and omega and simp know nothing about the
        cardinality of `BitVec 1`.

        Splitting each element into its two values first makes every later
        goal closed, which is what the rest of the plan was already built
        for. `skip` when the state is not finite, or is too wide to fan out.
        """
        if not self.features.two_valued:
            return "skip"
        steps = [
            f"rcases {lemma} (($v){acc}) with h{k} | h{k}"
            for k, (acc, lemma) in enumerate(self.features.two_valued)
        ]
        names = ", ".join(f"h{k}" for k in range(len(self.features.two_valued)))
        steps.append(f"try simp only [{names}] at *")
        return " <;>\n     ".join(steps)

    @property
    def facts_tactic(self) -> str:
        """Discharge the branch conditions the invariant already settles.

        `split_ifs` fans a goal out into 2^k over k conditions and each
        branch then pays for the whole prep chain. A condition cvc5 decided
        under `inv` need not be split at all: prove it once and rewrite the
        `if` away. Every step is wrapped in `try`, so a `have` the tactics
        cannot reproduce -- cvc5 is the stronger prover -- costs nothing
        and leaves the `split_ifs` route intact.

        The condition is rendered by the same walker that printed the
        definition, so `if_pos` matches on the nose -- but only until
        something rewrites the goal. Hence the call site: after `simp_defs`,
        which unfolds `inv` so the `have` is provable at all, and before
        `simp_mat`, which is where the arithmetic starts moving.
        """
        determined = getattr(self.hints, "determined", None)
        if not determined:
            return "skip"
        steps = []
        for k, (text, value) in enumerate(determined):
            prop = text if value else f"¬ ({text})"
            lemma = "if_pos" if value else "if_neg"
            steps.append(
                f"try (have hf{k} : {prop} := by "
                f"(first | omega | linarith | norm_num); "
                f"try simp only [{lemma} hf{k}])"
            )
        # One line, whatever the length. A `;`-separated tactic sequence
        # inside a `(tactic| ...)` quotation does not survive being broken
        # across lines: Lean reports `unexpected token 'try'; expected ')'`
        # at the end of the first one. Only shows up from two conditions on,
        # which is why a single-condition case built happily.
        return "; ".join(steps)

    @property
    def clear_tactic(self) -> str:
        """Drop a precondition no refutation needed, before anything runs.

        `simp_all` is superlinear in the hypothesis count and `nlinarith`
        multiplies pairs of them, so an unused hypothesis is not free.

        `step_inv` and `hrank` only: both bind `update_pre`, which is what
        the cores were taken over. `init_inv`'s `hpre` is `init_pre`, a
        different predicate that nothing here has looked at.
        """
        if getattr(self.hints, "pre_unused", False):
            return "try clear hpre"
        return "skip"

    def why(self) -> str:
        """One line per decision, emitted as a comment above the macros."""
        f = self.features
        bits = [
            f"state: {'+'.join(sorted(f.theories)) or 'none'}"
            f", {f.n_slots} slot(s), {f.n_terms} term(s)",
            f"predicates: {'branching' if f.has_ite else 'flat'}"
            f"{', conjunctive' if f.has_and else ''}"
            f"{', disjunctive' if f.has_or else ''}"
            f"{', nonlinear' if f.nonlinear else ', linear'}"
            f"{', floor' if f.has_floor else ''}"
            f"{f', {f.n_branch} branch point(s)' if f.n_branch else ''}"
            f"{f' ({f.n_conditions} distinct)' if f.n_conditions != f.n_branch else ''}",
        ]
        if f.two_valued:
            bits.append(
                f"state is finite and narrow: enumerating "
                f"{len(f.two_valued)} two-valued element(s) before the closers"
            )
        out = "\n".join(f"-- {b}" for b in bits)
        if self.hints is not None and self.hints.any:
            out += "\n" + self.hints.why()
        return out


def plan_for(ctx, pred_text: str, facts=None, hints=None) -> TacticPlan:
    """Build the tactic plan for this module's proof obligations."""
    f = features_for(ctx, pred_text, facts)

    # ── prep: canonicalise the goal before anything tries to close it ──
    prep: list[str] = []
    if f.has_ite:
        prep.append("split_ifs")
    if f.has_and:
        prep.append("casesm* _ ∧ _")
    if f.has_and or f.has_or:
        prep.append("simp only [not_and_or] at *")
    if f.has_real:
        # Profiled, this is the single most expensive thing in the plan, and
        # over the reals it is also load-bearing: dropping it to the goal
        # alone leaves two obligations of NN2RealAllPos4 unproved, because
        # `linarith` needs the hypotheses normalised.
        #
        # For an integer state it is neither. `omega` normalises numerals
        # itself and reads `min`/`max`/`Int.toNat` natively, so `norm_num`
        # only re-traverses a goal omega is about to take apart anyway.
        # Measured on NN2Deep5, a five-layer net: 75.9 s with it, **7.0 s**
        # without, same proof, no errors -- `norm_num` was 47.3 s of the
        # 48.7 s of tactic execution while `omega`, which actually closed
        # the goal, took 0.79 s. Restricting it to the goal saves nothing
        # (77.9 s): the cost is the goal, not the context.
        #
        # `(norm_num; done)` stays in the closers either way, so a goal that
        # really does need it still gets it.
        prep.append("norm_num at *")
    if f.has_real and f.has_eq:
        # Over the reals an inductive invariant has to pin exact values, so
        # the hypotheses are equalities and the goal is a numeral once they
        # are substituted. `omega` would have done this for integers; nothing
        # does it here unless simp_all runs first.
        prep.append("simp_all")
    if f.has_eq:
        # `¬(x = c)` carries a strict bound only through integrality, which
        # linarith and nlinarith do not do. Split it so the `casesm*` below
        # hands each branch a usable inequality. Must follow `norm_num at *`
        # where that runs at all, since it renormalises `≠` back to `¬ =`.
        prep.append("simp only [← ne_eq, ne_iff_lt_or_gt] at *")
    if f.has_and or f.has_or or f.has_eq:
        prep.append("casesm* _ ∧ _, _ ∨ _")

    # ── closers: cheapest first, and only what this state can use ──
    # `linarith` stays in every plan regardless of theory. It is cheap, and
    # gating it on Real cost two integer certificates that were closing on it
    # — an integer goal that omega cannot phrase (a `Mat`-valued comparison
    # simp left behind, say) is still ordinary linear arithmetic.
    closers: list[str] = []
    # Nearly free, and a branch whose hypotheses are already contradictory is
    # common once `split_ifs` has fanned the transition out.
    closers.append("contradiction")
    if f.has_int:
        closers.append("omega")
    closers.append("linarith")
    if f.has_int:
        closers.append("(norm_cast; omega)")
    if f.has_real:
        closers.append("(norm_cast; linarith)")
    closers.append("(norm_num; done)")
    if f.nonlinear:
        closers += ["nlinarith", "positivity"]

    # Conjunctive goals: `ranking`'s `Int.toNat` comparison alone unfolds to
    # `a < b ∧ 0 < b`, so this fires far more often than the invariant shape
    # suggests. Give the branch every prover the plan already carries.
    per_conjunct = [c for c in ("omega", "linarith", "nlinarith") if c in closers]
    closers.append("(constructor <;> first | " + " | ".join(per_conjunct) + ")")

    if getattr(hints, "nlinarith_hints", None):
        # cvc5's refutation multiplied these; naming the squares up front
        # is the difference between `nlinarith` checking a certificate and
        # searching for one. It goes *before* the bare `nlinarith` and adds
        # to the list rather than replacing anything -- a `first` chain
        # costs nothing for the alternatives it never reaches, so there is
        # no case for taking a closer away on the solver's say-so.
        hint_list = ", ".join(hints.nlinarith_hints)
        idx = closers.index("nlinarith") if "nlinarith" in closers else len(closers)
        closers.insert(idx, f"nlinarith [{hint_list}]")

    closers.append("tauto")
    # Re-simplify with the hypotheses, then retry the provers this state can
    # use. `done` guards the ones that can make progress without closing,
    # which would otherwise end the chain on an open goal.
    retry = [c for c in ("omega", "linarith", "nlinarith") if c in closers]
    closers.append("(simp_all; done)")
    closers += [f"(simp_all; {c})" for c in retry]
    if f.has_real:
        closers.append("(simp_all; norm_num; done)")

    # `decide` goes last: on a finite state it settles goals nothing else
    # can, but it is the most expensive thing in the plan.
    #
    # `bv_decide` is deliberately *not* here. It only ever reaches goals
    # every cheaper prover has already failed on, and on those it bit-blasts
    # for minutes: including it took a BitVec certificate from 10s to 977s
    # while closing nothing `decide` had not already closed. The one
    # obligation it might have helped with — `hrank`, which mixes BitVec with
    # `Int.toNat` — it answers with "a potentially spurious counterexample".
    if f.finite_state:
        closers.append("decide")

    # ── budgets ──
    # `simp_mat` unfolds the whole transition at once, so recursion depth
    # tracks the term count rather than the state width.
    max_rec_depth = max(4096, 1500 * f.n_terms)
    # The base budget is low, but not to make failures fail fast -- measured,
    # it does not. When a tactic hits the cap it throws, `first` catches that
    # like any other failure and moves on to a *more expensive* alternative,
    # which burns up to the cap again; a low cap multiplies the wasted work
    # instead of cutting it short. Two Real cases below are faster at the
    # higher budget than at the lower one, succeeding rather than failing.
    # So raise it wherever the shape is known to need it.
    slow = f.finite_state or f.has_bitvec or f.n_slots > 8 or f.n_terms > 32
    max_heartbeats = 2000000 if slow else 400000
    # A wide or deep net puts far more branch points in one predicate than the
    # module's own size suggests, and the whole cost of a neural certificate
    # lives there. Measured on nets built from explicit weight matrices: 14
    # branch points (a 12-unit layer, or three dense hidden layers) close
    # inside the base budget, 30 needs 2M, 62 needs 4M. Raising the cap costs
    # nothing on a proof that closes; it only makes a failing one give up
    # later, and a predicate this size was never going to fail fast anyway.
    if f.n_branch > 32:
        max_heartbeats = max(max_heartbeats, 8000000)
    elif f.n_branch > 16:
        max_heartbeats = max(max_heartbeats, 2000000)
    if f.has_real and f.n_branch:
        # A branchy Real predicate cannot use the `min`/`max` folding above --
        # `linarith` has no support for either -- so it still pays a
        # `split_ifs` branch per unit, and each branch is closed by linarith
        # rather than omega. Measured on a 4-unit net over the reals: 106 s to
        # *fail* at the base budget, 71 s to succeed at this one.
        max_heartbeats = max(max_heartbeats, 2000000)

    return TacticPlan(
        prep=prep,
        closers=closers,
        max_rec_depth=max_rec_depth,
        max_heartbeats=max_heartbeats,
        features=f,
        hints=hints,
    )
