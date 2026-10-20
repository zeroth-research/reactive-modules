"""What the certificate's terms say about the proof its tactics attempt.

`predicate_facts` reads the predicates' shape off their cvc5 terms -- how
they branch, whether a product is nonlinear, where they floor -- and
`solver_hints` asks cvc5 what it can settle in advance: a branch condition
the invariant decides, and the products a refutation multiplied. The tactic
plan (`proof.tactics.plan_for`) reads both. Neither changes what is proved, only
how the proof is attempted, and either may come back empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..smt import obligations as ob
from ..smt.terms import ite_conditions, min_max_of, nodes, state_dependent

if TYPE_CHECKING:
    from ..smt.limits import SmtBudget
    from ..smt.precheck import ModuleQueries


# ══════════════════════════════════════════════════════════════════════
# Predicate shape, read off the terms
# ══════════════════════════════════════════════════════════════════════


@dataclass
class PredicateFacts:
    """What the certificate predicates contain, from the cvc5 terms.

    `proof.tactics.features_for` reads the same things off the *rendered Lean*
    with a regex. For anything the printer emits verbatim -- how many `max`
    applications end up in the file -- that is the right measurement and
    stays. These are the ones text cannot get right:

    * `nonlinear` -- the regex hand-rolls operand scanning around each `*`
      and deliberately over-reports, so a net's affine layers can drag
      `nlinarith` into a plan that has no use for it. A `MULT` with two
      state-dependent children is exact.
    * `n_branch` -- how many branch points the predicate really contains,
      counted over the expanded tree. The regex counts `max` and `if` in
      the *printer's output*, which moves whenever the printer moves:
      teaching it to share subterms cut a five-layer net from 62 `max` to
      10 and silently dropped that case two budget tiers. The term's own
      count does not move.
    * `n_conditions` -- `split_ifs` splits once per *distinct* condition and
      reuses the hypothesis for repeats, so counting occurrences
      overestimates its fan-out, sometimes by a lot.
    * `has_and` / `has_or` / `has_eq` -- structure, rather than a substring
      search for `" ∧ "` over a string that also holds names and comments.
    """

    nonlinear: bool = False
    n_branch: int = 0
    n_conditions: int = 0
    has_and: bool = False
    has_or: bool = False
    has_eq: bool = False
    has_ite: bool = False
    mentioned: frozenset = frozenset()
    # The argument of each outermost `to_int` (printed `⌊x⌋`), distinct,
    # in the order the predicates are printed: what the plan bridges from
    # the floor's `Int` to the `Real` it was taken of.
    floors: tuple = ()

    def why(self) -> str:
        bits = [f"{self.n_branch} branch point(s)"]
        if self.n_conditions != self.n_branch:
            bits.append(f"{self.n_conditions} distinct condition(s)")
        bits.append("nonlinear" if self.nonlinear else "linear")
        if self.mentioned:
            bits.append(f"mentions slot(s) {sorted(self.mentioned)}")
        return "cvc5: " + ", ".join(bits)


def _facts_walk(t, acc: dict) -> None:
    """Accumulate shape over the DAG of one predicate."""
    from cvc5 import Kind

    for node in nodes([t], order="lifo"):
        k = node.getKind()
        if k == Kind.AND:
            acc["and"] = True
        elif k == Kind.OR:
            acc["or"] = True
        elif k == Kind.EQUAL:
            acc["eq"] = True
        elif k == Kind.CONSTANT:
            name = node.getSymbol()
            if name.startswith("s") and name[1:].isdigit():
                acc["mentioned"].add(int(name[1:]))
        elif k == Kind.ITE:
            # The condition is what `split_ifs` splits on, so identical
            # conditions in different `ite`s cost one split between them.
            # A folded `min`/`max` costs no split at all.
            if min_max_of(node) is None:
                acc["ite"] = True
                acc["conds"].add(node[0].getId())
        elif k == Kind.MULT:
            if sum(1 for c in node if state_dependent(c)) > 1:
                acc["nonlinear"] = True


def _floor_args(t, out: list) -> None:
    """Append the argument of each outermost `to_int` in `t`, left to
    right, skipping one already in `out`.

    Walked over the DAG, each node once: a term repeats its subterms by
    sharing them, and walked as a tree `(+ x2 x2)` over `(+ x1 x1)` over
    ... is 2^n visits of one node. A node met again appends nothing new,
    since its first visit appended whatever it holds.
    """
    from cvc5 import Kind

    stack, seen = [t], set()
    while stack:
        node = stack.pop()
        if node.getId() in seen:
            continue
        seen.add(node.getId())
        if node.getKind() == Kind.TO_INTEGER:
            if all(node[0] != seen for seen in out):
                out.append(node[0])
            continue
        stack.extend(reversed(list(node)))


def _tree_branch_points(t) -> int:
    """Branch points in the unfolded goal, memoised over the DAG.

    Two printer behaviours pull in opposite directions here and only one of
    them belongs in a cost model:

    * folding `ite (e ≥ 0) e 0` into `max e 0` drops the branches and keeps
      the condition's operands, so the subterm `e` is emitted *once* rather
      than twice. That is a real reduction -- it is the same term the goal
      will carry -- so this count follows the fold. Ignoring it inflates a
      five-layer net from 62 branch points to 682.
    * `let`-sharing emits a subterm once and refers to it by name, but
      `simp` is zeta-reducing by default, so the goal the closers see is the
      expanded one either way. This count therefore ignores sharing, which
      is what keeps the heartbeat budget steady across a change to the
      printer that moved a five-layer net two budget tiers.
    """
    from cvc5 import Kind

    memo: dict[int, int] = {}

    def count(node) -> int:
        key = node.getId()
        hit = memo.get(key)
        if hit is not None:
            return hit
        if node.getKind() == Kind.ITE:
            cond = node[0]
            if min_max_of(node) is not None:
                total = 1 + count(cond[0]) + count(cond[1])
            else:
                total = 1 + sum(count(c) for c in node)
        else:
            total = sum(count(c) for c in node)
        memo[key] = total
        return total

    return count(t)


def predicate_facts(q: "ModuleQueries | None") -> PredicateFacts | None:
    """Read the shape of every predicate `q` parsed. `None` if cvc5 is out."""
    if q is None:
        return None
    acc = {
        "and": False,
        "or": False,
        "eq": False,
        "ite": False,
        "nonlinear": False,
        "branch": 0,
        "conds": set(),
        "mentioned": set(),
    }
    floors: list = []
    try:
        for t in (q.prp, q.inv, q.ranking, q.init_pre, q.update_pre):
            if t is not None:
                _facts_walk(t, acc)
                acc["branch"] += _tree_branch_points(t)
        for t in (q.init_pre, q.update_pre, q.inv, q.prp, q.ranking):
            if t is not None:
                _floor_args(t, floors)
    except Exception:
        return None
    return PredicateFacts(
        nonlinear=acc["nonlinear"],
        n_branch=acc["branch"],
        n_conditions=len(acc["conds"]),
        has_and=acc["and"],
        has_or=acc["or"],
        has_eq=acc["eq"],
        has_ite=acc["ite"],
        mentioned=frozenset(acc["mentioned"]),
        floors=tuple(floors),
    )


# ══════════════════════════════════════════════════════════════════════
# Solver-informed tactics
#
# Everything above reads the terms. This asks the solver questions whose
# answers change what the certificate's tactics do -- so it is the part
# that has to be careful, and every query here shares one phase budget.
#
# Two kinds of answer are worth having:
#
#   * a branch condition the invariant already settles. `split_ifs` fans a
#     goal out into 2^k over k conditions; a condition cvc5 can decide
#     under `inv` is one the proof can rewrite away instead of splitting.
#   * what cvc5's own refutation needed. When nonlinear rules appear, the
#     products they multiply are exactly the hint terms that turn
#     `nlinarith`'s search into a check.
#
# Both are strictly *additive*. It is tempting to read "cvc5's proof was
# linear" as licence to drop `nlinarith` from the closer list, and that is
# a bad trade twice over: `first | a | b` costs nothing for `b` when `a`
# closes the goal, so removing a closer only speeds up the runs that fail
# anyway -- and cvc5 reasoning linearly about a goal is no promise that
# Mathlib's `linarith` can. So nothing here ever takes a tactic away.
#
# cvc5's proofs are not translatable into Lean -- this build offers
# alethe, cpc, dot and lfsc, none of which Mathlib reads -- but nothing
# here needs the proof itself, only what it mentions.
# ══════════════════════════════════════════════════════════════════════

# Two queries per condition, so the phase budget still has to cover the
# obligations' own proofs. Conditions past this are left to `split_ifs`.
MAX_DECIDED_CONDITIONS = 8

_NONLINEAR_RULES = {
    "ARITH_MULT_POS",
    "ARITH_MULT_NEG",
    "ARITH_MULT_SIGN",
    "ARITH_MULT_TANGENT",
    "ARITH_MULT_ABS_COMPARISON",
}


@dataclass
class SolverHints:
    """What cvc5 established about the obligations, as terms over the state
    variables; `proof.render.print_hints` prints them for the tactic plan."""

    # (a branch condition, the value `inv` forces it to), in the order the
    # printer meets the conditions.
    determined: list = field(default_factory=list)
    # None when no obligation was proved, so nothing may be concluded.
    linear_proof: "bool | None" = None
    # (a, b) for each product of two state-dependent factors a refutation
    # multiplied; meaningful when `linear_proof` is False.
    products: list = field(default_factory=list)
    # A real precondition that no obligation's refutation needed.
    pre_unused: bool = False
    notes: list = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(
            self.determined or self.products or self.pre_unused
        ) or (self.linear_proof is not None)


def _proof_facts(solver) -> tuple[bool, list]:
    """`(is_linear, multiplied_terms)` from cvc5's refutation.

    Walks the proof for the nonlinear-arithmetic rules. Their conclusions
    name the products cvc5 had to reason about, which is precisely what
    `nlinarith` would otherwise have to guess.

    A rule's name is not enough: `ARITH_MULT_POS` / `_NEG` also scale an
    inequality by a numeral -- `-1 < 0 ∧ x ≥ 1 → -1·x ≤ -1` -- which is
    linear reasoning and which a linear refutation is full of. A step
    counts when its conclusion multiplies two factors neither of which is
    a numeral.
    """
    from cvc5 import Kind

    products: dict[int, object] = {}
    linear = True
    # A proof is a DAG, and `getChildren()` wraps each child afresh: an
    # `id()` names a wrapper, which a shared step does not keep and a freed
    # one hands on to the next. A `Proof` hashes and compares by the step
    # it wraps, so the steps themselves are what is kept.
    stack, seen = list(solver.getProof()), set()
    while stack:
        step = stack.pop()
        if step in seen:
            continue
        seen.add(step)
        if str(step.getRule()).rsplit(".", 1)[-1] in _NONLINEAR_RULES:
            for node in nodes([step.getResult()], order="lifo"):
                if node.getKind() == Kind.MULT and sum(
                        1 for c in node
                        if not (c.isIntegerValue() or c.isRealValue())) > 1:
                    linear = False
                    products.setdefault(node.getId(), node)
        stack.extend(step.getChildren())
    return linear, list(products.values())


def _factor_pairs(products) -> list:
    """`(a, b)` for each product with two state-dependent factors: what
    `nlinarith` is handed squares of (`proof.render.print_hints`)."""
    out = []
    for prod in products:
        args = [c for c in prod if state_dependent(c)]
        if len(args) >= 2:
            out.append((args[0], args[1]))
    return out


def _in_core(solver, term) -> bool:
    try:
        return any(term == c for c in solver.getUnsatCore())
    except Exception:
        return True  # unknown: assume it is needed


def _core_note(solver, labels: dict) -> "str | None":
    """Which of the named hypotheses cvc5's refutation actually used."""
    try:
        core = solver.getUnsatCore()
    except Exception:
        return None
    used = [name for name, term in labels.items() if any(term == c for c in core)]
    if not used or len(used) == len(labels):
        return None
    return f"the core needs only {', '.join(sorted(used))} of {len(labels)}"


def solver_hints(q: "ModuleQueries | None", budget: SmtBudget) -> SolverHints:
    """Ask cvc5 the questions whose answers change the tactic plan.

    Bounded and total: whatever is not answered inside the budget is simply
    absent from the result, and the plan falls back to what it would have
    been. Never raises.
    """
    hints = SolverHints()
    if q is not None and q.inv is not None:
        budget.start_phase()
        try:
            _branch_polarity(q, budget, hints)
            _refutation_shape(q, budget, hints)
        except Exception as e:  # pragma: no cover -- the point is to not raise
            hints.notes.append(f"gave up: {type(e).__name__}")
    return hints


def _branch_polarity(q: "ModuleQueries", budget: SmtBudget, hints: SolverHints) -> None:
    """Conditions the invariant already settles, so `split_ifs` need not."""
    from cvc5 import Kind

    # Conditions of the `ite`s the printer does not fold into min/max, in
    # this order: it is the order of `hints.determined`, which reaches Lean.
    conds: dict = {}
    for t in (q.inv, q.ranking, q.prp):
        if t is None:
            continue
        for cond in ite_conditions([t], order="lifo",
                                   key=lambda c: c.getId(),
                                   where=lambda ite: min_max_of(ite) is None):
            conds.setdefault(cond.getId(), cond)
    for cond in list(conds.values())[:MAX_DECIDED_CONDITIONS]:
        if budget.exhausted:
            return
        s, _, _ = q.enc.fresh("d_")
        inv_s = q.enc.at(q.inv, s)
        cond_s = cond.substitute(q.state_vars, s)
        value = None
        if _unsat(q, ob.conjunction(q.tm, [inv_s, cond_s]), budget):
            value = False
        elif _unsat(q, ob.conjunction(
                q.tm, [inv_s, q.tm.mkTerm(Kind.NOT, cond_s)]), budget):
            value = True
        if value is None:
            continue
        hints.determined.append((cond, value))


def _refutation_shape(
    q: "ModuleQueries", budget: SmtBudget, hints: SolverHints
) -> None:
    """Prove each obligation with proofs on, and read what they needed.

    Both obligations, not just `step_inv`: the ranking is usually where the
    nonlinearity lives, so a module whose invariant is inductive by linear
    reasoning alone can still need `nlinarith` for `hrank`. Concluding
    "linear" from `step_inv` on its own would take the closer away from the
    obligation that needs it. On the safety route the same applies to
    `inv_imp_P`, where the property itself is the thing being reasoned about.
    """
    linear_all = True
    products: list = []
    proved_any = False
    pre_offered = False
    pre_used = False
    f = q.frame("h_")
    for name, labels in _obligation_labels(q, f):
        if budget.exhausted:
            break
        try:
            solver = q.solver(budget)
            solver.setOption("produce-proofs", "true")
            solver.setOption("produce-unsat-cores", "true")
            for term in labels.values():
                solver.assertFormula(term)
            if not solver.checkSat().isUnsat():
                continue
            linear, prods = _proof_facts(solver)
        except Exception:
            continue
        proved_any = True
        linear_all = linear_all and linear
        # Restated over the state the predicates are printed over: the
        # obligation reads the frame's own `h_s0..`, which the printer does
        # not know. A product of inputs or of the successor still names
        # the frame, and is left out when it is printed.
        products += [t.substitute(f.s, q.state_vars) for t in prods]
        if "pre" in labels:
            pre_offered = True
            pre_used = pre_used or _in_core(solver, labels["pre"])
        note = _core_note(solver, labels)
        if note:
            hints.notes.append(f"{name}: {note}")
    if not proved_any:
        return
    hints.linear_proof = linear_all
    hints.pre_unused = pre_offered and not pre_used
    if not linear_all:
        hints.products = _factor_pairs(products)


def _obligation_labels(q: "ModuleQueries", f):
    """`(name, {hypothesis name: term})` for each obligation cvc5 can state,
    over the frame `f`.

    `step_inv` always, and then whichever third obligation the certificate's
    kind states -- `hrank` for `G (F P)`, `inv_imp_P` for `G P`. `init_inv`
    is left out of both: it binds no state, so its proof says nothing about
    the shapes the plan chooses between.

    A hypothesis that is literally `true` is left out: it would be in no
    unsat core and would make every core look like a strict subset.
    """
    inv = [("inv", q.inv)]
    out = [("step_inv", ob.step_inv(f, inv, [q.inv]).labelled(q.tm))]
    if q.kind == "safety":
        if q.prp is not None:
            out.append(("inv_imp_P", ob.inv_imp_P(f, inv, q.prp).labelled(q.tm)))
        return out
    if q.ranking is not None and q.prp is not None:
        out.append(("hrank", ob.hrank(f, inv, q.prp, q.ranking).labelled(q.tm)))
    return out


def _unsat(q: "ModuleQueries", formula, budget: SmtBudget) -> bool:
    if budget.exhausted:
        return False
    try:
        solver = q.solver(budget)
        solver.assertFormula(formula)
        return solver.checkSat().isUnsat()
    except Exception:
        return False
