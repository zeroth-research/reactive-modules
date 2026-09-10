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
  slow, so a failing proof fails fast.

`plan_for` returns a `TacticPlan`; `Certificate.lean.j2` renders it as two
macros, `cert_prep` and `cert_close`, which the three proofs then share.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from zrth import Bool, Int, Real, BitVec

from .common import dtype_shape, itype_name


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


def features_for(ctx, pred_text: str) -> Features:
    """Read the shape of `ctx`'s module and its certificate predicates."""
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

    @property
    def prep_tactic(self) -> str:
        """The prep steps as one `<;>`-chained tactic, each step optional."""
        if not self.prep:
            return "skip"
        return " <;> ".join(f"(try {p})" for p in self.prep)

    @property
    def close_tactic(self) -> str:
        return "first | " + " | ".join(self.closers)

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
            f"{', floor' if f.has_floor else ''}",
        ]
        return "\n".join(f"-- {b}" for b in bits)


def plan_for(ctx, pred_text: str) -> TacticPlan:
    """Build the tactic plan for this module's proof obligations."""
    f = features_for(ctx, pred_text)

    # ── prep: canonicalise the goal before anything tries to close it ──
    prep: list[str] = []
    if f.has_ite:
        prep.append("split_ifs")
    if f.has_and:
        prep.append("casesm* _ ∧ _")
    if f.has_and or f.has_or:
        prep.append("simp only [not_and_or] at *")
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
        # hands each branch a usable inequality. Must follow `norm_num at *`,
        # which renormalises `≠` back to `¬ =`.
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
    # Leave the heartbeat budget low so a failing proof fails fast; raise it
    # only where something in the plan is known to be slow.
    slow = f.finite_state or f.has_bitvec or f.n_slots > 8 or f.n_terms > 32
    max_heartbeats = 2000000 if slow else 400000

    return TacticPlan(
        prep=prep,
        closers=closers,
        max_rec_depth=max_rec_depth,
        max_heartbeats=max_heartbeats,
        features=f,
    )
