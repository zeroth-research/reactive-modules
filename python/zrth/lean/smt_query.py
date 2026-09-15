"""Bounded cvc5 queries about a module and its certificate.

`verith` already builds a complete symbolic encoding of every module
(`smt_module.ModuleSMT`) but, outside the `--infer ai-cegar` loop, never asks
the solver anything: cvc5 is used only to parse the SMT-LIB predicates and
hand them to the Lean printer. That leaves the two questions that matter
unanswered before a `lake build` that takes tens of seconds --

  * is the obligation even true, or is the certificate wrong?
  * what shape is it, so the tactic plan can be chosen rather than guessed?

Both are cheap. Measured on the case matrix, all three obligations resolve in
3-13 ms per module where `lake build` takes 9-79 s.

Everything here is on a leash. Each query gets `per_call_ms`, each *phase*
(a group of queries serving one codegen decision) gets `phase_ms`, and both
are enforced by cvc5's own `tlimit`, so a hard problem returns `unknown`
rather than hanging the generator. Any failure at all -- a timeout, an
`unknown`, an unsupported operator, a cvc5 exception -- is reported as
`Status.UNKNOWN` and the caller falls back to what it did before. Nothing
here may change what `verith` emits except by *adding* information.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from .common import Refused

if TYPE_CHECKING:
    from .cert import CertificateData as _CertificateData  # noqa: F401


# ══════════════════════════════════════════════════════════════════════
# Budget
# ══════════════════════════════════════════════════════════════════════

DEFAULT_CALL_MS = 5000
DEFAULT_PHASE_MS = 20000


@dataclass
class SmtBudget:
    """A per-call and a per-phase wall-clock bound on solving.

    `phase_ms` is spent down by every query in the phase, so a phase that
    fans out over many small queries (one per branch condition, say) cannot
    add up to more than one obligation check's worth of time. Call
    `start_phase` between phases to refill it.
    """

    per_call_ms: int = DEFAULT_CALL_MS
    phase_ms: int = DEFAULT_PHASE_MS
    _phase_start: float = field(default_factory=time.perf_counter, repr=False)

    def start_phase(self) -> None:
        self._phase_start = time.perf_counter()

    @property
    def phase_left_ms(self) -> int:
        spent = (time.perf_counter() - self._phase_start) * 1000
        return max(0, int(self.phase_ms - spent))

    @property
    def exhausted(self) -> bool:
        return self.phase_left_ms <= 0

    def next_call_ms(self) -> int:
        """The limit for the next query: whichever bound bites first."""
        return min(self.per_call_ms, self.phase_left_ms)


# ══════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════


class Status(str, Enum):
    HOLDS = "holds"          # negation UNSAT -- the obligation is true
    REFUTED = "refuted"      # negation SAT   -- there is a counterexample
    UNKNOWN = "unknown"      # timeout, incompleteness, or anything went wrong


@dataclass
class Verdict:
    name: str
    status: Status
    elapsed_ms: float
    detail: str | None = None   # counterexample, or why it is unknown

    @property
    def line(self) -> str:
        mark = {
            Status.HOLDS: "holds   ",
            Status.REFUTED: "REFUTED ",
            Status.UNKNOWN: "unknown ",
        }[self.status]
        out = f"   {self.name:<9} {mark} {self.elapsed_ms:6.0f} ms"
        if self.detail:
            out += f"   {self.detail}"
        return out


# ══════════════════════════════════════════════════════════════════════
# Queries
# ══════════════════════════════════════════════════════════════════════


class ModuleQueries:
    """A cvc5 view of one module together with its certificate predicates.

    Construct with `build`, which returns `None` rather than raising when the
    module or the predicates are outside what the encoder covers -- the whole
    point is that the caller can carry on without cvc5. A caller that cannot
    carry on -- `smt_predicates_to_lean`, whose output *is* the Lean the
    predicates become -- constructs directly and reports what went wrong.
    """

    def __init__(self, module, cert_data):
        import cvc5

        from .smt_module import ModuleSMT, keep_alive
        from .smt_prompt import CegarPromptEnv, parse_predicate

        self.cvc5 = cvc5
        # `artifacts/` is written through this, when a caller supplies one.
        # Left off by default so that constructing the queries stays a pure
        # encoding step -- `predicate_facts` and `solver_hints` build these
        # too, and neither is the run's record of what it asked.
        self.artifacts = None
        self.tm = cvc5.TermManager()
        self.msmt = ModuleSMT(tm=self.tm, module=module)
        self.env = CegarPromptEnv(self.msmt)
        self.state_vars = self.env.state_vars

        def opt(src, flag, want: str = "Bool"):
            """Parse one predicate source, or refuse naming the flag it came
            from. cvc5's parser reports the token it stopped at and nothing
            about where the text came from, and the caller that prints this
            has five sources in hand.

            `want` is the sort the field is declared as and the Lean printer
            assumes: a property, an invariant and a precondition become the
            body of a `Prop`, a ranking function the body of a `Nat`. An Int
            where a Prop is expected elaborates to nothing; a Bool ranking
            prints as `(((s 0 0) = 0) : Int).toNat`, which is not Lean. cvc5
            knows the sort here, so the flag can be told it is the wrong one
            rather than Lean being shown the result.
            """
            if not isinstance(src, str):
                return None
            try:
                term = parse_predicate(self.env, src)
            except Exception as e:
                raise Refused(f"{flag}: cannot read `{src}`: {e}") from e
            got = term.getSort()
            if not (got.isBoolean() if want == "Bool" else got.isInteger()):
                raise Refused(f"{flag}: must have sort {want}, got {got}")
            return term

        # After `--infer`, `inv` and `ranking` hold Lean printed from a cvc5
        # term, which nothing parses back; `inv_smt` / `ranking_smt` keep the
        # source the solver was handed, and that is what it can read.
        self.kind = getattr(cert_data, "kind", "buchi")
        prp_flag = "--safety" if self.kind == "safety" else "--buchi"
        self.prp = opt(cert_data.prp, prp_flag)
        self.inv = opt(
            getattr(cert_data, "inv_smt", None) or cert_data.inv, "--invariant"
        )
        self.ranking = opt(
            getattr(cert_data, "ranking_smt", None) or cert_data.ranking,
            "--ranking",
            want="Int",
        )
        self.init_pre = opt(cert_data.init_pre, "--pre")
        self.update_pre = opt(cert_data.update_pre, "--pre")
        # Retained whole, solvers included: `smt_module.keep_alive` owns the
        # shutdown-ordering workaround for every cvc5 object this package
        # mints, and the instance is what ties this term manager to the
        # solvers built from it.
        self._solvers: list = []
        keep_alive(self)

    @classmethod
    def build(cls, module, cert_data) -> "ModuleQueries | None":
        try:
            return cls(module, cert_data)
        except Exception:
            return None

    # ── plumbing ──────────────────────────────────────────────────────

    def _true(self):
        return self.tm.mkBoolean(True)

    def _and(self, *ts):
        from cvc5 import Kind

        ts = [t for t in ts if t is not None]
        if not ts:
            return self._true()
        return ts[0] if len(ts) == 1 else self.tm.mkTerm(Kind.AND, *ts)

    def _not(self, t):
        from cvc5 import Kind

        return self.tm.mkTerm(Kind.NOT, t)

    def _sub_inputs(self, term, el, en):
        """Rebind the env's shared `e*`/`el*` consts to a query's own vars."""
        if term is None:
            return None
        old = list(self.env.extl_next_vars) + list(self.env.extl_latched_vars)
        new = list(en) + list(el)
        return term.substitute(old, new) if old else term

    def _sub_state(self, term, s):
        return None if term is None else term.substitute(self.state_vars, s)

    def _solver(self, budget: SmtBudget):
        s = self.cvc5.Solver(self.tm)
        self._solvers.append(s)
        s.setLogic("ALL")
        limit = budget.next_call_ms()
        s.setOption("tlimit-per", str(limit))
        s.setOption("tlimit", str(limit))
        return s

    def _ask(self, name, build, budget):
        """UNSAT on the negation means the obligation holds.

        `build` is a thunk returning `(negation, witness)` rather than the
        formula itself, because encoding the transition is where an
        unsupported operator surfaces -- `Uninterpreted` has no SMT form and
        raises from `update_state`, well before any solver is involved. The
        contract is that nothing in here escapes to the caller.
        """
        if budget.exhausted:
            return Verdict(name, Status.UNKNOWN, 0.0, "phase budget spent")
        t0 = time.perf_counter()
        try:
            negation, witness = build()
            self._dump(name, negation)
            solver = self._solver(budget)
            if witness:
                solver.setOption("produce-models", "true")
            solver.assertFormula(negation)
            res = solver.checkSat()
        except Exception as e:  # cvc5 raises on unsupported terms and options
            dt = (time.perf_counter() - t0) * 1000
            return Verdict(name, Status.UNKNOWN, dt, f"cannot encode: {e}"[:120])
        dt = (time.perf_counter() - t0) * 1000
        if res.isUnsat():
            return Verdict(name, Status.HOLDS, dt)
        if not res.isSat():
            return Verdict(name, Status.UNKNOWN, dt, res.getUnknownExplanation())
        return Verdict(name, Status.REFUTED, dt, self._witness(solver, witness))

    def _dump(self, name: str, negation) -> None:
        """Leave this obligation in `artifacts/`, as the script cvc5 is given.

        Inside `_ask`'s `try`, so its own failure would otherwise read as the
        obligation being unencodable: a record of the question is worth having
        and worth nothing at the price of changing the answer."""
        if self.artifacts is None:
            return
        try:
            text = _script([negation])
        except Exception as e:                       # noqa: BLE001
            text = f"; this obligation could not be written out: {e}\n"
        self.artifacts.encoded(
            "obligation", f"obligation-{name}", text,
            what=_OBLIGATIONS.get(name, f"The `{name}` obligation, negated."),
        )

    def dump_system(self, artifacts) -> None:
        """Leave the module itself in `artifacts/`, as cvc5 encoded it.

        One `define-fun` per state component per block: `init<i>` over the
        inputs awaited at tick 0, `next<i>` over the latched state and both
        ends of the inputs. Together they are the transition every obligation
        above is stated against, which is the thing a reader most often wants
        to check by eye when an answer surprises them."""
        if artifacts is None:
            return
        try:
            s = self.msmt.fresh_ctrl("s")
            el = self.msmt.fresh_extl_l("el")
            en = self.msmt.fresh_extl_n("en")
            ins = [(str(t), str(t.getSort())) for t in en]
            step = [(str(t), str(t.getSort())) for t in s + el + en]
            state = [(str(t), str(t.getSort())) for t in self.state_vars]
            defines = [
                (f"init{i}", ins, str(t.getSort()), t)
                for i, t in enumerate(self.msmt.init_state(en))
            ] + [
                (f"next{i}", step, str(t.getSort()), t)
                for i, t in enumerate(self.msmt.update_state(s, el, en))
            ] + [
                # The certificate's own predicates, encoded over the same
                # state. The `.smt` files beside this one are their source;
                # these are the form the obligations are actually stated in,
                # and the form a reader can apply to a state by hand.
                (name, state, str(term.getSort()), term)
                for name, term in (("P", self.prp), ("inv", self.inv),
                                   ("ranking", self.ranking))
                if term is not None
            ]
            text = _script([], defines=defines)
        except Exception as e:                       # noqa: BLE001
            # `update_state` raises on an op with no SMT form -- an
            # uninterpreted symbol, say. The run carries on; the note says why
            # the file a reader expects is not here.
            text = f"; this module has no SMT encoding: {e}\n"
        artifacts.encoded(
            "system", "system", text,
            what="The whole problem as cvc5 encoded it. `init<i>` is state "
                 "component `i` after tick 0 and `next<i>` the same component "
                 "after one round, over the latched state `s*` and the inputs "
                 "`el*` / `en*` latched and awaited; `P`, `inv` and `ranking` "
                 "are the certificate's predicates over that state. No "
                 "assertion -- this file is what the obligations beside it "
                 "are stated against, and what to apply to a state by hand.",
        )

    def _witness(self, solver, witness) -> str | None:
        if not witness:
            return None
        try:
            bits = [f"{label} = {solver.getValue(t)}" for label, t in witness]
        except Exception:
            return "counterexample: <unreadable>"
        return "counterexample: " + ", ".join(bits) if bits else None

    def _clamp(self, rank):
        """Mirror Lean's `Int.toNat` clamp on the ranking function."""
        from cvc5 import Kind

        if rank.getSort().isReal():
            rank = self.tm.mkTerm(Kind.TO_INTEGER, rank)
        zero = self.tm.mkInteger(0)
        return self.tm.mkTerm(
            Kind.ITE, self.tm.mkTerm(Kind.GEQ, rank, zero), rank, zero
        )

    def _labelled(self, s):
        return [(f"s{i}", v) for i, v in enumerate(s)]

    # ── the three obligations Certificate.lean states ─────────────────

    def check_init_inv(self, budget: SmtBudget) -> Verdict:
        """`∀ e, init_pre e → inv (init e)`."""

        def build():
            el = self.msmt.fresh_extl_l("i_el")
            en = self.msmt.fresh_extl_n("i_en")
            s0 = self.msmt.init_state(en)
            return (
                self._and(
                    self._sub_inputs(self.init_pre, el, en),
                    self._not(self._sub_state(self.inv, s0)),
                ),
                self._labelled(s0),
            )

        return self._ask("init_inv", build, budget)

    def check_step_inv(self, budget: SmtBudget) -> Verdict:
        """`∀ s e, update_pre e ∧ inv s → inv (update s e)`."""

        def build():
            s = self.msmt.fresh_ctrl("p_s")
            el = self.msmt.fresh_extl_l("p_el")
            en = self.msmt.fresh_extl_n("p_en")
            nxt = self.msmt.update_state(s, el, en)
            return (
                self._and(
                    self._sub_state(self.inv, s),
                    self._sub_inputs(self.update_pre, el, en),
                    self._not(self._sub_state(self.inv, nxt)),
                ),
                self._labelled(s),
            )

        return self._ask("step_inv", build, budget)

    def check_hrank(self, budget: SmtBudget) -> Verdict:
        """`inv s ∧ ¬P s ∧ update_pre e → ranking (update s e) < ranking s`."""
        from cvc5 import Kind

        def build():
            s = self.msmt.fresh_ctrl("r_s")
            el = self.msmt.fresh_extl_l("r_el")
            en = self.msmt.fresh_extl_n("r_en")
            nxt = self.msmt.update_state(s, el, en)
            decrease = self.tm.mkTerm(
                Kind.LT,
                self._clamp(self._sub_state(self.ranking, nxt)),
                self._clamp(self._sub_state(self.ranking, s)),
            )
            return (
                self._and(
                    self._sub_state(self.inv, s),
                    self._not(self._sub_state(self.prp, s)),
                    self._sub_inputs(self.update_pre, el, en),
                    self._not(decrease),
                ),
                self._labelled(s),
            )

        return self._ask("hrank", build, budget)

    def check_inv_imp_P(self, budget: SmtBudget) -> Verdict:
        """`inv s → P s` -- the safety route's third obligation.

        `rule_globally` needs the invariant to be a strengthening of the
        property; that is the whole difference from the Büchi route, where
        `P` is merely re-reached and a ranking says how soon.
        """

        def build():
            s = self.msmt.fresh_ctrl("p_s")
            return (
                self._and(
                    self._sub_state(self.inv, s),
                    self._not(self._sub_state(self.prp, s)),
                ),
                self._labelled(s),
            )

        return self._ask("inv_imp_P", build, budget)

    def check_obligations(self, budget: SmtBudget) -> list[Verdict]:
        """Every obligation the certificate data actually states.

        Which third obligation there is follows the certificate's kind:
        `hrank` for `G (F P)`, `inv_imp_P` for `G P`.
        """
        budget.start_phase()
        out: list[Verdict] = []
        if self.inv is None:
            return out
        out.append(self.check_init_inv(budget))
        out.append(self.check_step_inv(budget))
        if self.prp is None:
            return out
        if self.kind == "safety":
            out.append(self.check_inv_imp_P(budget))
        elif self.ranking is not None:
            out.append(self.check_hrank(budget))
        return out


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════


def _constants(term, seen: dict) -> dict:
    """Every free constant in `term`, by symbol.

    cvc5 has no "the declarations this term needs", so the term is walked. A
    constant is `Kind.CONSTANT`; an operator, a bound variable and a literal
    are none of them declared.
    """
    import cvc5

    if term.getKind() == cvc5.Kind.CONSTANT:
        seen.setdefault(term.getSymbol(), term)
        return seen
    for child in term:
        _constants(child, seen)
    return seen


def _script(asserts, *, defines=(), logic: str = "ALL") -> str:
    """`asserts` as a script that runs on its own under any SMT-LIB solver.

    `defines` are `(name, params, sort, body)` emitted as `define-fun` ahead of
    the assertions -- how the transition is written out, one function per state
    component. Declarations are collected from everything mentioned, so a
    caller states what it wants said and not what it depends on.
    """
    seen: dict = {}
    for _n, _p, _s, body in defines:
        _constants(body, seen)
    for term in asserts:
        _constants(term, seen)

    bound = {p for _n, params, _s, _b in defines for p, _ps in params}
    lines = [f"(set-logic {logic})", ""]
    lines += [f"(declare-fun {sym} () {seen[sym].getSort()})"
              for sym in sorted(seen) if sym not in bound]
    if len(lines) > 2:
        lines.append("")
    for name, params, sort, body in defines:
        args = " ".join(f"({p} {srt})" for p, srt in params)
        lines.append(f"(define-fun {name} ({args}) {sort} {body})")
    if defines:
        lines.append("")
    lines += [f"(assert {term})" for term in asserts]
    lines += ["", "(check-sat)"]
    return "\n".join(lines) + "\n"


_OBLIGATIONS = {
    "init_inv": "`init_pre e -> inv (init e)`: the invariant holds in the state "
                "the module starts in.",
    "step_inv": "`inv s /\\ update_pre e -> inv (update s e)`: every round "
                "preserves the invariant.",
    "hrank": "`inv s /\\ ~P s /\\ update_pre e -> ranking (update s e) < "
             "ranking s`: the ranking function drops on every round the "
             "property does not hold.",
    "inv_imp_P": "`inv s -> P s`: what makes the invariant a proof of the "
                 "safety property rather than merely a fact about the module.",
}


def pre_check(module, cert_data, budget: SmtBudget, log=print,
              artifacts=None) -> list[Verdict]:
    """Run the obligation pre-check and report it in a few lines.

    Never raises: a module cvc5 cannot encode reports one `unknown` line and
    generation carries on exactly as it would have.

    `artifacts` is where the module and each obligation are written out as
    runnable SMT-LIB -- see :mod:`.dump`. Nothing about the check changes when
    it is `None`.
    """
    log(
        f".. SMT pre-check (cvc5): <={budget.per_call_ms} ms per query, "
        f"<={budget.phase_ms} ms total"
    )
    q = ModuleQueries.build(module, cert_data)
    if q is None:
        # Two unrelated failures land here: the module is outside the encoder,
        # or a predicate is not something cvc5 can parse -- Lean text from
        # `--infer ai`, say, which has no SMT-LIB source beside it. Rebuilding
        # with no certificate at all separates them for the price of one
        # encoding, and nothing here is allowed to be misleading.
        from .cert import CertificateData

        if ModuleQueries.build(module, CertificateData()) is None:
            log("   cvc5 could not encode this module -- skipping the pre-check")
        else:
            log("   the certificate predicates are not SMT-LIB, so cvc5 "
                "cannot state the obligations -- skipping the pre-check")
        return []
    q.artifacts = artifacts
    q.dump_system(artifacts)
    verdicts = q.check_obligations(budget)
    if not verdicts:
        if cert_data.inv is None:
            log("   nothing to check (no invariant given)")
        else:
            log("   the invariant is not SMT-LIB, so cvc5 cannot state the "
                "obligations -- skipping the pre-check")
        return []
    for v in verdicts:
        log(v.line)
    refuted = [v.name for v in verdicts if v.status is Status.REFUTED]
    if refuted:
        log(
            f"   {len(refuted)} obligation(s) refuted ({', '.join(refuted)}): "
            "the certificate is wrong, not merely hard -- Lean cannot close it"
        )
    return verdicts


# ══════════════════════════════════════════════════════════════════════
# Predicate shape, read off the terms
# ══════════════════════════════════════════════════════════════════════


@dataclass
class PredicateFacts:
    """What the certificate predicates contain, from the cvc5 terms.

    `tactics.features_for` reads the same things off the *rendered Lean*
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

    from .smt_to_lean import min_max_of

    seen = set()
    stack = [t]
    while stack:
        node = stack.pop()
        key = node.getId()
        if key in seen:
            continue
        seen.add(key)
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
            if sum(1 for c in node if _state_dependent(c)) > 1:
                acc["nonlinear"] = True
        stack.extend(node)


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

    from .smt_to_lean import min_max_of

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


def _state_dependent(t) -> bool:
    """Does this subterm read the state at all, or is it ground arithmetic?

    The visited set is not an optimisation: without it a shared DAG is
    walked as its expanded tree, which is exponential in a deep net.
    """
    from cvc5 import Kind

    stack, seen = [t], set()
    while stack:
        node = stack.pop()
        key = node.getId()
        if key in seen:
            continue
        seen.add(key)
        if node.getKind() == Kind.CONSTANT:
            return True
        stack.extend(node)
    return False


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
    try:
        for t in (q.prp, q.inv, q.ranking, q.init_pre, q.update_pre):
            if t is not None:
                _facts_walk(t, acc)
                acc["branch"] += _tree_branch_points(t)
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
    """What cvc5 established about the obligations, for the tactic plan."""

    # (Lean text of the condition, the value `inv` forces it to).
    determined: list = field(default_factory=list)
    # None when no obligation was proved, so nothing may be concluded.
    linear_proof: "bool | None" = None
    # Lean terms to hand `nlinarith`.
    nlinarith_hints: list = field(default_factory=list)
    # A real precondition that no obligation's refutation needed.
    pre_unused: bool = False
    notes: list = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(
            self.determined or self.nlinarith_hints or self.pre_unused
        ) or (self.linear_proof is not None)

    def why(self) -> str:
        bits = []
        for text, value in self.determined:
            bits.append(f"inv forces `{text}` to be {str(value).lower()}")
        if self.linear_proof is False:
            bits.append(f"cvc5 needed {len(self.nlinarith_hints)} product hint(s)")
        if self.pre_unused:
            bits.append("no refutation needed the precondition: clearing it")
        bits += self.notes
        return "\n".join(f"-- cvc5: {b}" for b in bits)


def _kept_conditions(t, out: dict) -> None:
    """Conditions of the `ite`s the printer will *not* fold into min/max."""
    from cvc5 import Kind

    from .smt_to_lean import min_max_of

    stack, seen = [t], set()
    while stack:
        node = stack.pop()
        key = node.getId()
        if key in seen:
            continue
        seen.add(key)
        if node.getKind() == Kind.ITE and min_max_of(node) is None:
            cond = node[0]
            out.setdefault(cond.getId(), cond)
        stack.extend(node)


def _proof_facts(solver) -> tuple[bool, list]:
    """`(is_linear, multiplied_terms)` from cvc5's refutation.

    Walks the proof for the nonlinear-arithmetic rules. Their conclusions
    name the products cvc5 had to reason about, which is precisely what
    `nlinarith` would otherwise have to guess.
    """
    from cvc5 import Kind

    products: dict[int, object] = {}
    linear = True
    stack, seen = list(solver.getProof()), set()
    while stack:
        step = stack.pop()
        key = id(step)
        if key in seen:
            continue
        seen.add(key)
        if str(step.getRule()).rsplit(".", 1)[-1] in _NONLINEAR_RULES:
            linear = False
            sub, sub_seen = [step.getResult()], set()
            while sub:
                node = sub.pop()
                nid = node.getId()
                if nid in sub_seen:
                    continue
                sub_seen.add(nid)
                if node.getKind() == Kind.MULT:
                    products.setdefault(nid, node)
                sub.extend(node)
        stack.extend(step.getChildren())
    return linear, list(products.values())


def _hint_terms(products, ctrl_next) -> list:
    """`nlinarith` hint terms for the products cvc5 multiplied.

    `nlinarith` looks for degree-2 certificates by multiplying pairs of
    hypotheses; naming the squares it needs up front turns that search into
    a check. `mul_self_nonneg (a - b)` and `(a + b)` between them span the
    products of two atoms, which is what the arithmetic rules produce.
    """
    from .smt_to_lean import smt_to_lean_body

    seen: set[str] = set()
    out: list[str] = []
    for prod in products:
        args = [c for c in prod if _state_dependent(c)]
        if len(args) < 2:
            continue
        try:
            a, b = (
                smt_to_lean_body(args[0], ctrl_next),
                smt_to_lean_body(args[1], ctrl_next),
            )
        except Exception:
            continue
        for hint in (
            f"mul_self_nonneg ({a} - {b})",
            f"mul_self_nonneg ({a} + {b})",
        ):
            if hint not in seen:
                seen.add(hint)
                out.append(hint)
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


def solver_hints(q: "ModuleQueries | None", budget: SmtBudget, log=None) -> SolverHints:
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
    if log:
        reasons = [ln.removeprefix("-- cvc5: ") for ln in hints.why().splitlines()]
        if reasons:
            for reason in reasons:
                log(f"   {reason}")
        else:
            log("   nothing cvc5 can add to the plan")
    return hints


def _branch_polarity(q: "ModuleQueries", budget: SmtBudget, hints: SolverHints) -> None:
    """Conditions the invariant already settles, so `split_ifs` need not."""
    from .smt_to_lean import smt_to_lean_body

    conds: dict = {}
    for t in (q.inv, q.ranking, q.prp):
        if t is not None:
            _kept_conditions(t, conds)
    for cond in list(conds.values())[:MAX_DECIDED_CONDITIONS]:
        if budget.exhausted:
            return
        s = q.msmt.fresh_ctrl("d_s")
        inv_s = q._sub_state(q.inv, s)
        cond_s = cond.substitute(q.state_vars, s)
        value = None
        if _unsat(q, q._and(inv_s, cond_s), budget):
            value = False
        elif _unsat(q, q._and(inv_s, q._not(cond_s)), budget):
            value = True
        if value is None:
            continue
        try:
            # `($v)` is the macro's own binder: the condition is spliced into
            # a tactic, so it has to read whichever state the obligation
            # bound rather than a fixed name.
            text = smt_to_lean_body(cond, q.msmt.ctrl_next, param_name="($v)")
        except Exception:
            continue
        hints.determined.append((text, value))


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
    for name, labels in _obligation_labels(q):
        if budget.exhausted:
            break
        try:
            solver = q._solver(budget)
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
        products += prods
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
        hints.nlinarith_hints = _hint_terms(products, q.msmt.ctrl_next)


def _obligation_labels(q: "ModuleQueries"):
    """`(name, {hypothesis name: term})` for each obligation cvc5 can state.

    `step_inv` always, and then whichever third obligation the certificate's
    kind states -- `hrank` for `G (F P)`, `inv_imp_P` for `G P`. `init_inv`
    is left out of both: it binds no state, so its proof says nothing about
    the shapes the plan chooses between.

    A hypothesis that is literally `true` is left out: it would be in no
    unsat core and would make every core look like a strict subset.
    """
    from cvc5 import Kind

    out = []
    s = q.msmt.fresh_ctrl("h_s")
    el = q.msmt.fresh_extl_l("h_el")
    en = q.msmt.fresh_extl_n("h_en")
    nxt = q.msmt.update_state(s, el, en)

    def keep(d):
        return {
            k: v
            for k, v in d.items()
            if v is not None and v.getKind() != Kind.CONST_BOOLEAN
        }

    out.append(
        (
            "step_inv",
            keep(
                {
                    "inv": q._sub_state(q.inv, s),
                    "pre": q._sub_inputs(q.update_pre, el, en),
                    "goal": q._not(q._sub_state(q.inv, nxt)),
                }
            ),
        )
    )
    if q.kind == "safety":
        if q.prp is not None:
            out.append(
                (
                    "inv_imp_P",
                    keep(
                        {
                            "inv": q._sub_state(q.inv, s),
                            "goal": q._not(q._sub_state(q.prp, s)),
                        }
                    ),
                )
            )
        return out
    if q.ranking is not None and q.prp is not None:
        decrease = q.tm.mkTerm(
            Kind.LT,
            q._clamp(q._sub_state(q.ranking, nxt)),
            q._clamp(q._sub_state(q.ranking, s)),
        )
        out.append(
            (
                "hrank",
                keep(
                    {
                        "inv": q._sub_state(q.inv, s),
                        "notP": q._not(q._sub_state(q.prp, s)),
                        "pre": q._sub_inputs(q.update_pre, el, en),
                        "goal": q._not(decrease),
                    }
                ),
            )
        )
    return out


def _unsat(q: "ModuleQueries", formula, budget: SmtBudget) -> bool:
    if budget.exhausted:
        return False
    try:
        solver = q._solver(budget)
        solver.assertFormula(formula)
        return solver.checkSat().isUnsat()
    except Exception:
        return False
