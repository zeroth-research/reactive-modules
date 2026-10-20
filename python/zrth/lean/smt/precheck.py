"""Bounded cvc5 queries about a module and its certificate.

`ModuleQueries` is a claim and a certificate read against the run's
encoding of the module, and `pre_check` asks of it the question that
matters before a `lake build` that takes tens of seconds: is each
obligation even true, or is the certificate wrong? It is cheap: measured on
the case matrix, all three obligations resolve in 3-13 ms per module where
`lake build` takes 9-79 s. What shape the predicates are, so the tactic
plan can be chosen rather than guessed, is asked of the same queries by
`proof.facts`.

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
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..certificate import Certificate, Claim, LeanAnswer
from ..errors import Refused
from . import obligations as ob
from .limits import call_ms
from .encoding import encoding_of
from .script import script

if TYPE_CHECKING:
    from .limits import SmtBudget


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


# How a verdict is recorded on the obligation artifact it answers. The two
# vocabularies are separate on purpose: `Status` is what this module decides,
# `artifacts.STATUSES` is what every producer in the package shares, and a
# consumer filtering `artifacts/` should not have to know cvc5 wrote it.
_VERDICT_STATUS = {
    Status.HOLDS: "proved",
    Status.REFUTED: "refuted",
    Status.UNKNOWN: "unknown",
}


# ══════════════════════════════════════════════════════════════════════
# Queries
# ══════════════════════════════════════════════════════════════════════


class ModuleQueries:
    """A cvc5 view of one module together with its certificate predicates.

    `source` is the module or its `smt.encoding.ModuleEncoding`; a run gets these
    from `ModuleEncoding.queries`, which builds one per claim and certificate.
    Construct with `build`, which returns `None` rather than raising when the
    module or the predicates are outside what the encoder covers -- the whole
    point is that the caller can carry on without cvc5. A caller that cannot
    carry on -- `predicate_terms`, whose terms *are* what the Lean is
    printed from -- constructs directly and reports what went wrong:
    `Refused` for a predicate, anything else for the module.
    """

    def __init__(self, source, claim: Claim, cert: Certificate | None = None):
        enc = encoding_of(source)
        # Weak: `enc.queries` memoises this, and a cycle through the owner of
        # a `TermManager` is finalised in arbitrary order (`smt.module._LIVE`).
        self.enc = weakref.proxy(enc)
        self.tm = enc.tm
        self.msmt = enc.msmt
        self.env = enc.env
        self.state_vars = enc.state

        cert = cert or Certificate()
        self.kind = claim.kind
        self.prp = enc.parse(claim.prp, "property", kind=claim.kind)
        self.inv = enc.parse(cert.inv, "invariant")
        self.ranking = enc.parse(cert.ranking, "ranking", want="Int")
        self.init_pre = enc.parse(claim.init_pre, "pre")
        self.update_pre = enc.parse(claim.update_pre, "pre")

    @classmethod
    def build(cls, source, claim: Claim,
              cert: Certificate | None = None) -> "ModuleQueries | None":
        try:
            return cls(source, claim, cert)
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

    def solver(self, budget: SmtBudget):
        """A solver on this module's terms, held to `budget`'s next call."""
        return self.enc.solver(call_ms(budget), logic="ALL")

    def _ask(self, name, build, budget, artifacts=None):
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
        written = None

        def record(verdict):
            return _record(artifacts, written, verdict)

        try:
            negation, witness = build()
            written = _dump(artifacts, name, negation)
            solver = self.solver(budget)
            if witness:
                solver.setOption("produce-models", "true")
            solver.assertFormula(negation)
            res = solver.checkSat()
        except Exception as e:  # cvc5 raises on unsupported terms and options
            dt = (time.perf_counter() - t0) * 1000
            return record(
                Verdict(name, Status.UNKNOWN, dt, f"cannot encode: {e}"[:120])
            )
        dt = (time.perf_counter() - t0) * 1000
        if res.isUnsat():
            return record(Verdict(name, Status.HOLDS, dt))
        if not res.isSat():
            return record(
                Verdict(name, Status.UNKNOWN, dt, res.getUnknownExplanation())
            )
        return record(
            Verdict(name, Status.REFUTED, dt, self._witness(solver, witness))
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
            s, el, en = self.enc.fresh()
            ins = [(str(t), str(t.getSort())) for t in en]
            step = [(str(t), str(t.getSort())) for t in s + el + en]
            state = [(str(t), str(t.getSort())) for t in self.state_vars]
            defines = [
                (f"init{i}", ins, str(t.getSort()), t)
                for i, t in enumerate(self.enc.init_state(en))
            ] + [
                (f"next{i}", step, str(t.getSort()), t)
                for i, t in enumerate(self.enc.next_state(s, el, en))
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
            text = script(defines=defines)
        except Exception as e:                       # noqa: BLE001
            # The transition raises on an op with no SMT form -- an
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

    def _labelled(self, s):
        return [(f"s{i}", v) for i, v in enumerate(s)]

    def frame(self, prefix: str) -> ob.Frame:
        """Fresh constants `{prefix}s0..` and the round from them, the
        preconditions read over their inputs."""
        return ob.Frame(self.enc, prefix, init_pre=self.init_pre,
                        update_pre=self.update_pre)

    # ── the three obligations Certificate.lean states ─────────────────

    def check_init_inv(self, budget: SmtBudget, artifacts=None) -> Verdict:
        """`∀ e, init_pre e → inv (init e)`."""

        def build():
            f = self.frame("i_")
            return (ob.init_inv(f, [self.inv]).negation(self.tm),
                    self._labelled(f.entry))

        return self._ask("init_inv", build, budget, artifacts)

    def check_step_inv(self, budget: SmtBudget, artifacts=None) -> Verdict:
        """`∀ s e, update_pre e ∧ inv s → inv (update s e)`."""

        def build():
            f = self.frame("p_")
            q = ob.step_inv(f, [("inv", self.inv)], [self.inv])
            return q.negation(self.tm), self._labelled(f.s)

        return self._ask("step_inv", build, budget, artifacts)

    def check_hrank(self, budget: SmtBudget, artifacts=None) -> Verdict:
        """`inv s ∧ ¬P s ∧ update_pre e → ranking (update s e) < ranking s`."""

        def build():
            f = self.frame("r_")
            q = ob.hrank(f, [("inv", self.inv)], self.prp, self.ranking)
            return q.negation(self.tm), self._labelled(f.s)

        return self._ask("hrank", build, budget, artifacts)

    def check_inv_imp_P(self, budget: SmtBudget, artifacts=None) -> Verdict:
        """`inv s → P s` -- the safety route's third obligation.

        `rule_globally` needs the invariant to be a strengthening of the
        property; that is the whole difference from the Büchi route, where
        `P` is merely re-reached and a ranking says how soon.
        """

        def build():
            f = self.frame("p_")
            q = ob.inv_imp_P(f, [("inv", self.inv)], self.prp)
            return q.negation(self.tm), self._labelled(f.s)

        return self._ask("inv_imp_P", build, budget, artifacts)

    def check_obligations(self, budget: SmtBudget,
                          artifacts=None) -> list[Verdict]:
        """Every obligation the certificate data actually states, each
        question and its answer left in `artifacts` when given.

        Which third obligation there is follows the certificate's kind:
        `hrank` for `G (F P)`, `inv_imp_P` for `G P`.
        """
        budget.start_phase()
        out: list[Verdict] = []
        if self.inv is None:
            return out
        out.append(self.check_init_inv(budget, artifacts))
        out.append(self.check_step_inv(budget, artifacts))
        if self.prp is None:
            return out
        if self.kind == "safety":
            out.append(self.check_inv_imp_P(budget, artifacts))
        elif self.ranking is not None:
            out.append(self.check_hrank(budget, artifacts))
        return out


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════


def _record(artifacts, written, verdict: Verdict) -> Verdict:
    """Write the answer back onto the obligation `_dump` already wrote.

    The question is left in `artifacts/` before the solver runs, so that
    one that cannot be solved still says what was asked; this is the other
    half, and without it `index.json` records only which obligations a run
    asked about and never which of them came back refuted.

    Returned rather than assigned, so the call reads as a pass-through on
    each of `_ask`'s return paths. There are four, and a verdict recorded
    on three of them would be worse than one recorded on none: a stale
    `encoded` beside three answered obligations reads as "not asked".
    """
    if artifacts is None or written is None:
        # No artifact for this obligation: `--artifacts ignore`, or the
        # encoding failed before `_dump` ran. Either way there is no
        # question on disk for this to be the answer to.
        return verdict
    why = verdict.detail or ""
    if verdict.status is Status.HOLDS:
        # Nothing was wrong with it. `_describe` prints no status line for
        # `proved` and no prose for an empty `why`, so an obligation that
        # holds says nothing beyond what it is -- which is right.
        why = ""
    elif verdict.status is Status.REFUTED and not why:
        why = "refuted, but the counterexample could not be read back"
    try:
        artifacts.resolve(
            written, status=_VERDICT_STATUS[verdict.status], why=why
        )
    except Exception:      # pragma: no cover -- never worth the run
        pass
    return verdict

def _dump(artifacts, name: str, negation):
    """Leave this obligation in `artifacts/`, as the script cvc5 is given.

    Inside `_ask`'s `try`, so its own failure would otherwise read as the
    obligation being unencodable: a record of the question is worth having
    and worth nothing at the price of changing the answer. What came back
    is written onto it afterwards, by `_record`."""
    if artifacts is None:
        return None
    try:
        text = script([negation])
    except Exception as e:                       # noqa: BLE001
        text = f"; this obligation could not be written out: {e}\n"
    return artifacts.encoded(
        "obligation", f"obligation-{name}", text,
        what=_OBLIGATIONS.get(name, f"The `{name}` obligation, negated."),
    )



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


_NOT_SMT = ("   the certificate predicates are not SMT-LIB, so cvc5 cannot "
            "state the obligations -- skipping the pre-check")


def pre_check(smt, claim: Claim, answer: "Certificate | LeanAnswer | None",
              budget: SmtBudget, log=print, artifacts=None) -> list[Verdict]:
    """Run the obligation pre-check and report it in a few lines.

    `smt` is the module, its encoding or a `smt.encoding.lazy` factory for one.
    Never raises: a module cvc5 cannot encode, or an answer it cannot read,
    reports one line and generation carries on exactly as it would have.

    `artifacts` is where the module and each obligation are written out as
    runnable SMT-LIB -- see :mod:`.dump`. Nothing about the check changes when
    it is `None`.
    """
    log(
        f".. SMT pre-check (cvc5): <={budget.per_call_ms} ms per query, "
        f"<={budget.phase_ms} ms total"
    )
    if isinstance(answer, LeanAnswer):
        log(_NOT_SMT)
        return []
    try:
        q = encoding_of(smt).queries(claim, answer)
    except Refused:
        log(_NOT_SMT)
        return []
    except Exception:
        log("   cvc5 could not encode this module -- skipping the pre-check")
        return []
    q.dump_system(artifacts)
    verdicts = q.check_obligations(budget, artifacts)
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


# ══════════════════════════════════════════════════════════════════════
# The claim, read against the module
# ══════════════════════════════════════════════════════════════════════


def predicate_terms(
    smt, claim: Claim, cert: Certificate = Certificate()
) -> "ModuleQueries":
    """`claim` and `cert` parsed against the module: what the Lean is printed
    from, and what `predicate_facts` and `solver_hints` read. `smt` is the
    module, its encoding or a `smt.encoding.lazy` factory for one.

    A predicate that does not parse is refused by its role; anything
    else is the module's, and refused as that.
    """
    try:
        return encoding_of(smt).queries(claim, cert)
    except Refused:
        raise
    except Exception as e:
        raise Refused(
            f"the certificate predicates cannot be read for this module: {e}"
        ) from e
