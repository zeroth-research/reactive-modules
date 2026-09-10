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

if TYPE_CHECKING:
    import cvc5
    from zrth import Module
    from .cert import CertificateData


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


# See `ModuleQueries.__init__`.
_LIVE: list["ModuleQueries"] = []


class ModuleQueries:
    """A cvc5 view of one module together with its certificate predicates.

    Construct with `build`, which returns `None` rather than raising when the
    module or the predicates are outside what the encoder covers -- the whole
    point is that the caller can carry on without cvc5.
    """

    def __init__(self, module, cert_data):
        import cvc5

        from .smt_module import ModuleSMT
        from .smt_prompt import CegarPromptEnv, parse_predicate

        self.cvc5 = cvc5
        self.tm = cvc5.TermManager()
        self.msmt = ModuleSMT(tm=self.tm, module=module)
        self.env = CegarPromptEnv(self.msmt)
        self.state_vars = self.env.state_vars

        def opt(src):
            return parse_predicate(self.env, src) if isinstance(src, str) else None

        self.prp = opt(cert_data.prp)
        self.inv = opt(cert_data.inv)
        self.ranking = opt(cert_data.ranking)
        self.init_pre = opt(cert_data.init_pre)
        self.update_pre = opt(cert_data.update_pre)
        # Keep every solver alive: the cvc5 bindings segfault at shutdown if a
        # TermManager is collected out of order with the solvers and terms
        # minted from it. `_LIVE` extends that to the whole instance, since a
        # GC cycle would otherwise collect the two together in arbitrary
        # order. A `verith` run builds a handful of these and then exits.
        self._solvers: list = []
        _LIVE.append(self)

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

    def check_obligations(self, budget: SmtBudget) -> list[Verdict]:
        """Every obligation the certificate data actually states."""
        budget.start_phase()
        out: list[Verdict] = []
        if self.inv is not None:
            out.append(self.check_init_inv(budget))
            out.append(self.check_step_inv(budget))
        if self.inv is not None and self.ranking is not None and self.prp is not None:
            out.append(self.check_hrank(budget))
        return out


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════


def pre_check(module, cert_data, budget: SmtBudget, log=print) -> list[Verdict]:
    """Run the obligation pre-check and report it in a few lines.

    Never raises: a module cvc5 cannot encode reports one `unknown` line and
    generation carries on exactly as it would have.
    """
    log(
        f".. SMT pre-check (cvc5): <={budget.per_call_ms} ms per query, "
        f"<={budget.phase_ms} ms total"
    )
    q = ModuleQueries.build(module, cert_data)
    if q is None:
        log("   cvc5 could not encode this module -- skipping the pre-check")
        return []
    verdicts = q.check_obligations(budget)
    if not verdicts:
        log("   nothing to check (no invariant given)")
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
