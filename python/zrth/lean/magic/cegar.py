"""CEGAR-based TA2Magic: LLM proposes invariant/ranking as SMT-LIB
expressions, cvc5 verifies, counterexamples are fed back to the LLM.
"""

# TODO: this is CEGIS, not CEGAR -- the counterexample refines the *candidate*,
# and there is no abstraction to refine; the obligations are checked over the
# concrete transition relation. The route was renamed (`--infer ai-cegar` ->
# `--infer ai-cegis`); the driver was not, so `ai-cegis` still runs
# `TA2MagicCEGAR` out of `magic/cegar.py` and prints `[CEGAR] attempt 0`.
# What the rename has to reach, together, in one commit:
#   * this module (-> `magic/cegis.py`), `TA2MagicCEGAR`, the docstring above,
#     and the `magic.cegar` mentions in `infer_route/`, `smt_synth.py` and
#     `magic/sygus.py`;
#   * `smt_prompt.CegarPromptEnv`, `CEGAR_GENERATE_SYSTEM`,
#     `CEGAR_SAFETY_SYSTEM` and the `"cegar"` parser tag -- `smt_synth.py`,
#     `smt_query.py` and `tests/test_synth_routes.py` import them too;
#   * `tests/test_cegar_results.py`, and `CEGAR_FIXED` in
#     `tests/test_verith_cli.py`;
#   * the `[CEGAR] ...` lines printed below -- quoted verbatim in the
#     transcripts in `tutorials/verith.md`, `tutorials/verith_gym.md` and
#     `README.md`, and matched by `CEGAR failed after` in
#     `tests/bench_matrix/run_matrix.py`'s failure regex. Those are why this
#     is a sweep of its own rather than part of the flag rename.
# Leave `magic.learn`'s Farkas/CEGAR alone: that one really does refine an
# abstraction.

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import cvc5  # type: ignore
except ImportError:
    cvc5 = None

from zrth import Module

from ..certificate import Certificate, Claim
from ..errors import Refused
from . import TA2Magic
from .llm import DEFAULT_MODEL, describe_preconditions, make_client
from ..smt.obligations import (
    Frame,
    init_inv,
    inv_imp_P,
    rank_decrease,
    rank_positive,
    step_inv,
)
from ..smt.limits import bounded_solver, call_ms
from ..smt.run import encoding_of
from ..smt_prompt import PromptResult, prompt_inv_ranking


@dataclass
class ObligationResult:
    name: str
    ok: bool
    counterexample: str | None  # formatted CE if not ok, else None


class TA2MagicCEGAR(TA2Magic):
    """Infer invariant and ranking via LLM + cvc5 CEGAR loop.

    Each attempt:
      1. Ask LLM for SMT-LIB invariant + ranking expressions.
      2. Check four obligations with cvc5 (init, inductive, rank-decrease,
         rank-positive). Each check is: assert the negation, SAT ⇒ failure,
         each query held to the run's `--smt-timeout`.
      3. If all UNSAT, return. Else format model as feedback and retry.

    `known` is what earlier runs established about *shapes* rather than
    about candidates -- "no ranking function linear in the state satisfies
    the obligations under this invariant", proved by cvc5 and left in
    `artifacts/` by `--infer smt-linear`. It is stated in the prompt so an
    attempt is not spent proposing what is already known not to exist.

    A safety certificate (`claim.kind == "safety"`, from `--safety`) is the
    same loop over a smaller certificate: the LLM is asked for an invariant
    alone, and the obligations are init, inductive, and `inv → prp` --
    `rule_globally` has no ranking function to take, so the two ranking
    checks have nothing to check.
    """

    def __init__(
        self,
        source: str,
        module: Module,
        *,
        model: str = DEFAULT_MODEL,
        max_attempts: int = 5,
        base_url: str | None = None,
        known: tuple[str, ...] = (),
        budget_s: float | None = None,
        encoding=None,
        budget=None,
        log=print,
    ):
        self.source = source
        if cvc5 is None:                             # pragma: no cover
            raise Refused(
                "--infer ai-cegis needs the `cvc5` package, which is not "
                "importable. Install with: uv pip install cvc5"
            )
        self.module = module
        # The run's `--smt-timeout` / `--smt-budget`: each check is a phase.
        self.budget = budget
        self.log = log
        # The run's `smt.run.lazy` encoding; a library caller's module gets
        # its own.
        self.encoding = encoding
        self.max_attempts = max_attempts
        # Wall-clock seconds to keep asking for, in place of `max_attempts`.
        # A fixed count is the wrong leash for a run that has a time limit
        # around it: in the bench matrix every `ai-cegis` cell that ran out
        # of attempts did so at 75-107 s of a 180 s cap, a third of the
        # budget unspent. With a budget an attempt starts only while the
        # slowest one so far would still finish inside it, so the run gives
        # up on its own terms rather than being killed mid-reply.
        self.budget_s = budget_s
        # What another run *proved* is not there -- `--infer smt-linear`
        # refuting its template, `--infer sygus` exhausting its grammar. It
        # goes into the prompt rather than into the loop: the obligations
        # already refute a bad candidate in milliseconds, and what this saves
        # is the attempt being spent on a shape a solver has ruled out.
        self.known = tuple(known)
        self._client = (base_url, model)
        self._made = None

    def _chat(self, system: str, user: str) -> str:
        """The model, reached on the first question rather than up front.

        A certificate supplied whole is verified once and asks nothing, so
        that run needs neither the LLM package nor a key.
        """
        if self._made is None:
            self._made = make_client(*self._client)
        return self._made(system, user)

    # --- driver ---------------------------------------------------------

    def infer(self, claim: Claim,
              given: Certificate = Certificate()) -> Certificate:
        """The certificate of `claim`, as SMT-LIB. What `given` holds is
        fixed: it is shown to the model and only the rest is asked for."""
        enc = encoding_of(self.encoding or self.module)
        env = enc.env

        # An absent predicate is `true`, and every obligation states it.
        def parsed(src, role):
            return (enc.tm.mkBoolean(True) if src is None
                    else enc.parse(src, role, kind=claim.kind))

        prp_term = parsed(claim.prp, "property")
        init_pre_term = parsed(claim.init_pre, "pre")
        update_pre_term = parsed(claim.update_pre, "pre")

        preconds = describe_preconditions(claim)
        feedback: str | None = None

        safety = claim.is_safety
        fixed_inv = given.inv
        fixed_ranking = None if safety else given.ranking
        # If everything this kind of certificate needs is user-provided,
        # no LLM loop — just verify once.
        complete = bool(fixed_inv) and (safety or bool(fixed_ranking))
        start, slowest, attempt = time.monotonic(), 0.0, 0

        while self._another(attempt, complete, start, slowest):
            began = time.monotonic()
            self.log(f"[CEGAR] attempt {attempt}")
            attempt += 1
            try:
                result = prompt_inv_ranking(
                    env,
                    self._chat,
                    self.source,
                    str(claim.prp),
                    preconds,
                    feedback,
                    fixed_inv_src=fixed_inv,
                    fixed_ranking_src=fixed_ranking,
                    kind=claim.kind,
                    known=self.known,
                )
            except Refused:
                # Not a reply to be parsed: the run cannot be made at all,
                # and retrying it five times only hides the reason.
                raise
            except ValueError as e:
                self.log(f"  parse error: {e}")
                feedback = f"Your reply could not be parsed: {e}"
                slowest = max(slowest, time.monotonic() - began)
                continue

            self.log(f"  inv: {result.inv_src}")
            if not safety:
                self.log(f"  ranking: {result.ranking_src}")

            obligations = self._check_all(
                enc,
                result,
                prp_term,
                init_pre_term,
                update_pre_term,
                safety=safety,
            )
            failures = [o for o in obligations if not o.ok]
            if not failures:
                self.log("[CEGAR] all obligations UNSAT — accepted")
                return Certificate(
                    result.inv_src, None if safety else result.ranking_src
                )

            feedback = self._format_feedback(failures)
            self.log(f"[CEGAR] failures: {[o.name for o in failures]}")
            slowest = max(slowest, time.monotonic() - began)

        # `Refused` for the reason `magic.ai` gives: running out of
        # attempts is this route giving up, not a defect.
        within = (f" in {time.monotonic() - start:.0f} s of a {self.budget_s:g} s "
                  f"budget" if self.budget_s is not None else "")
        raise Refused(
            f"CEGAR failed after {attempt} attempt{'' if attempt == 1 else 's'}{within}. "
            f"Last feedback:\n{feedback}",
            searched=True,
        )

    def _another(self, attempt: int, complete: bool, start: float,
                 slowest: float) -> bool:
        """Whether to start attempt number `attempt` (counting from 0).

        A certificate supplied whole is checked once and never re-asked.
        Without a budget the count is the leash; with one, an attempt starts
        while the slowest so far would still end inside it -- the first
        always does, since nothing is known about its cost yet.
        """
        if complete:
            return attempt == 0
        if self.budget_s is None:
            return attempt < self.max_attempts
        return attempt == 0 or time.monotonic() - start + slowest <= self.budget_s

    # --- obligation checks ---------------------------------------------

    def _check_all(
        self,
        enc,
        r: PromptResult,
        prp_term,
        init_pre_term,
        update_pre_term,
        *,
        safety: bool = False,
    ) -> list[ObligationResult]:
        """`rule_buchi`'s obligations, `hrank` split in two, or
        `rule_globally`'s; each over fresh constants of its own."""

        def frame(prefix):
            return Frame(enc, prefix, init_pre=init_pre_term,
                         update_pre=update_pre_term)

        if self.budget is not None:
            self.budget.start_phase()

        checks = [
            self._check_init(frame("init_"), r),
            self._check_inductive(frame("ind_"), r),
        ]
        if safety:
            checks.append(self._check_inv_implies_prp(frame("ip_"), r, prp_term))
        else:
            checks += [
                self._check_ranking_decrease(frame("rd_"), r, prp_term),
                self._check_ranking_positive(frame("rp_"), r, prp_term),
            ]
        return checks

    def _check_inv_implies_prp(self, f: Frame, r, prp_term):
        """`inv s → prp s`: what makes the invariant a *safety* proof."""
        q = inv_imp_P(f, [("inv", r.inv_term)], prp_term)
        return self._run_query("inv_imp_P", q.negation(f.tm), f.tm,
                               extra_vars=[("s", f.s)])

    def _check_init(self, f: Frame, r):
        q = init_inv(f, [r.inv_term])
        return self._run_query(
            "init_inv", q.negation(f.tm), f.tm,
            extra_vars=[("extl_l", f.el), ("extl_n", f.en)],
        )

    def _check_inductive(self, f: Frame, r):
        q = step_inv(f, [("inv", r.inv_term)], [r.inv_term])
        return self._run_query(
            "step_inv",
            q.negation(f.tm),
            f.tm,
            extra_vars=[
                ("s", f.s),
                ("extl_l", f.el),
                ("extl_n", f.en),
                ("s_next", f.succ),
            ],
        )

    def _check_ranking_decrease(self, f: Frame, r, prp_term):
        q = rank_decrease(f, [("inv", r.inv_term)], prp_term, r.ranking_term)
        return self._run_query(
            "rank_decrease",
            q.negation(f.tm),
            f.tm,
            extra_vars=[
                ("s", f.s),
                ("extl_l", f.el),
                ("extl_n", f.en),
                ("s_next", f.succ),
                ("ranking(s)", [f.at(r.ranking_term, f.s)]),
                ("ranking(s_next)", [f.at(r.ranking_term, f.succ)]),
            ],
        )

    def _check_ranking_positive(self, f: Frame, r, prp_term):
        """`inv s ∧ ¬P s ∧ update_pre → ranking s >= 1`: with
        `rank_decrease`, exactly `hrank` over `Int.toNat ranking`."""
        q = rank_positive(f, [("inv", r.inv_term)], prp_term, r.ranking_term)
        return self._run_query(
            "rank_positive", q.negation(f.tm), f.tm,
            extra_vars=[("s", f.s), ("ranking(s)", [f.at(r.ranking_term, f.s)])],
        )

    # --- solver driver --------------------------------------------------

    def _run_query(
        self,
        name: str,
        neg_query,
        tm,
        extra_vars: list[tuple[str, list]],
    ) -> ObligationResult:
        """Check `neg_query` for SAT. UNSAT ⇒ obligation holds."""
        self.log(f"[smt] check query: {neg_query}")
        solver = bounded_solver(tm, call_ms(self.budget), logic="ALL",
                                produce_models="true")
        solver.assertFormula(neg_query)
        res = solver.checkSat()
        if res.isUnsat():
            return ObligationResult(name, True, None)
        if not res.isSat():
            # `unknown` -- a timeout, or arithmetic the solver is incomplete
            # for (a nonlinear ranking function, say). There is no model, so
            # `getValue` below would raise CVC5ApiException and take the whole
            # run down. Report it as unproven, with the solver's reason, so
            # CEGAR can carry on.
            why = res.getUnknownExplanation()
            return ObligationResult(
                name,
                False,
                f"{name}: solver returned unknown ({why}); no counterexample "
                "available. The obligation is neither proved nor refuted.",
            )
        # SAT — read the counterexample off the model
        lines = [f"{name}: obligation violated. Counterexample:"]
        for label, group in extra_vars:
            for i, v in enumerate(group):
                val = solver.getValue(v)
                lines.append(f"  {label}[{i}] = {val}")
        return ObligationResult(name, False, "\n".join(lines))

    def _format_feedback(self, failures: list[ObligationResult]) -> str:
        parts = []
        for o in failures:
            parts.append(o.counterexample or f"{o.name}: failed (no CE).")
        return "\n\n".join(parts)
