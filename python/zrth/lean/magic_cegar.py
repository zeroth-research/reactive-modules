"""CEGAR-based TA2Magic: LLM proposes invariant/ranking as SMT-LIB
expressions, cvc5 verifies, counterexamples are fed back to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import cvc5  # type: ignore
    from cvc5 import Kind
except ImportError:
    cvc5 = None
    Kind = None

from zrth import Module

from .cert import CertificateData
from .magic import TA2Magic
from .magic_ai import _make_client, _describe_preconditions
from .smt_module import ModuleSMT
from .smt_prompt import (
    CegarPromptEnv,
    PromptResult,
    parse_predicate,
    prompt_inv_ranking,
)
from .smt_to_lean import smt_to_lean, smt_to_lean_nat


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
         rank-nonneg). Each check is: assert the negation, SAT ⇒ failure.
      3. If all UNSAT, return. Else format model as feedback and retry.
    """

    def __init__(
        self,
        source: str,
        module: Module,
        *,
        model: str = "gpt-4",
        max_attempts: int = 5,
        base_url: str | None = None,
    ):
        super().__init__(source)
        if cvc5 is None:
            raise RuntimeError(
                "cvc5 package is required for TA2MagicCEGAR. "
                "Install with: uv pip install cvc5"
            )
        self.module = module
        self.max_attempts = max_attempts
        self._chat = _make_client(base_url, model)

    # --- driver ---------------------------------------------------------

    def infer(self, cd: CertificateData) -> CertificateData:
        tm = cvc5.TermManager()
        msmt = ModuleSMT(tm=tm, module=self.module)
        env = CegarPromptEnv(msmt)

        # Parse predicates — Python expression first, SMT-LIB fallback.
        prp_term = (
            parse_predicate(env, cd.prp)
            if isinstance(cd.prp, str)
            else tm.mkBoolean(True)
        )
        init_pre_term = (
            parse_predicate(env, cd.init_pre)
            if isinstance(cd.init_pre, str)
            else tm.mkBoolean(True)
        )
        update_pre_term = (
            parse_predicate(env, cd.update_pre)
            if isinstance(cd.update_pre, str)
            else tm.mkBoolean(True)
        )

        preconds = _describe_preconditions(cd)
        feedback: str | None = None

        fixed_inv = cd.inv if isinstance(cd.inv, str) else None
        fixed_ranking = cd.ranking if isinstance(cd.ranking, str) else None
        # If both user-provided, no LLM loop — just verify once.
        loop_count = 1 if (fixed_inv and fixed_ranking) else self.max_attempts

        for attempt in range(loop_count):
            print(f"[CEGAR] attempt {attempt}")
            try:
                result = prompt_inv_ranking(
                    env,
                    self._chat,
                    self.source,
                    str(cd.prp),
                    preconds,
                    feedback,
                    fixed_inv_src=fixed_inv,
                    fixed_ranking_src=fixed_ranking,
                )
            except ValueError as e:
                print(f"  parse error: {e}")
                feedback = f"Your reply could not be parsed: {e}"
                continue

            print(f"  inv: {result.inv_src}")
            print(f"  ranking: {result.ranking_src}")

            obligations = self._check_all(
                msmt,
                env,
                result,
                prp_term,
                init_pre_term,
                update_pre_term,
            )
            failures = [o for o in obligations if not o.ok]
            if not failures:
                print("[CEGAR] all obligations UNSAT — accepted")
                cd.inv = smt_to_lean(result.inv_term, msmt.ctrl_next)
                cd.ranking = smt_to_lean_nat(result.ranking_term, msmt.ctrl_next)
                return cd

            feedback = self._format_feedback(failures)
            print(f"[CEGAR] failures: {[o.name for o in failures]}")

        raise RuntimeError(
            f"CEGAR failed after {self.max_attempts} attempts. Last feedback:\n{feedback}"
        )

    # --- obligation checks ---------------------------------------------

    def _subst_inputs(
        self,
        env: CegarPromptEnv,
        term: "cvc5.Term",
        el: list,
        en: list,
    ) -> "cvc5.Term":
        """Substitute the env's `e0/el0` consts with obligation-local vars."""
        old = list(env.extl_next_vars) + list(env.extl_latched_vars)
        new = list(en) + list(el)
        return term.substitute(old, new) if old else term

    def _check_all(
        self,
        msmt: ModuleSMT,
        env: CegarPromptEnv,
        r: PromptResult,
        prp_term,
        init_pre_term,
        update_pre_term,
    ) -> list[ObligationResult]:
        return [
            self._check_init(msmt, env, r, init_pre_term),
            self._check_inductive(msmt, env, r, update_pre_term),
            self._check_ranking_decrease(msmt, env, r, prp_term, update_pre_term),
            self._check_ranking_nonneg(msmt, env, r),
        ]

    def _check_init(self, msmt, env, r, init_pre_term):
        tm = msmt.tm
        el = msmt.fresh_extl_l("init_el")
        en = msmt.fresh_extl_n("init_en")
        pre = self._subst_inputs(env, init_pre_term, el, en)
        s0 = msmt.init_state(en)
        inv_at_init = r.inv_term.substitute(env.state_vars, s0)
        neg_query = tm.mkTerm(Kind.AND, pre, tm.mkTerm(Kind.NOT, inv_at_init))
        return self._run_query(
            "init_inv", neg_query, env, extra_vars=[("extl_l", el), ("extl_n", en)]
        )

    def _check_inductive(self, msmt, env, r, update_pre_term):
        tm = msmt.tm
        s = msmt.fresh_ctrl("ind_s")
        el = msmt.fresh_extl_l("ind_el")
        en = msmt.fresh_extl_n("ind_en")
        inv_s = r.inv_term.substitute(env.state_vars, s)
        pre = self._subst_inputs(env, update_pre_term, el, en)
        s_next = msmt.update_state(s, el, en)
        inv_next = r.inv_term.substitute(env.state_vars, s_next)
        neg_query = tm.mkTerm(Kind.AND, inv_s, pre, tm.mkTerm(Kind.NOT, inv_next))
        return self._run_query(
            "step_inv",
            neg_query,
            env,
            extra_vars=[
                ("s", s),
                ("extl_l", el),
                ("extl_n", en),
                ("s_next", s_next),
            ],
        )

    def _check_ranking_decrease(self, msmt, env, r, prp_term, update_pre_term):
        tm = msmt.tm
        s = msmt.fresh_ctrl("rd_s")
        el = msmt.fresh_extl_l("rd_el")
        en = msmt.fresh_extl_n("rd_en")
        inv_s = r.inv_term.substitute(env.state_vars, s)
        pre = self._subst_inputs(env, update_pre_term, el, en)
        prp_s = prp_term.substitute(env.state_vars, s)
        s_next = msmt.update_state(s, el, en)
        rank_s = r.ranking_term.substitute(env.state_vars, s)
        rank_next = r.ranking_term.substitute(env.state_vars, s_next)
        decrease = tm.mkTerm(Kind.LT, rank_next, rank_s)
        neg_query = tm.mkTerm(
            Kind.AND,
            inv_s,
            tm.mkTerm(Kind.NOT, prp_s),
            pre,
            tm.mkTerm(Kind.NOT, decrease),
        )
        return self._run_query(
            "rank_decrease",
            neg_query,
            env,
            extra_vars=[
                ("s", s),
                ("extl_l", el),
                ("extl_n", en),
                ("s_next", s_next),
                ("ranking(s)", [rank_s]),
                ("ranking(s_next)", [rank_next]),
            ],
        )

    def _check_ranking_nonneg(self, msmt, env, r):
        tm = msmt.tm
        s = msmt.fresh_ctrl("rn_s")
        inv_s = r.inv_term.substitute(env.state_vars, s)
        rank_s = r.ranking_term.substitute(env.state_vars, s)
        nonneg = tm.mkTerm(Kind.GEQ, rank_s, tm.mkInteger(0))
        neg_query = tm.mkTerm(Kind.AND, inv_s, tm.mkTerm(Kind.NOT, nonneg))
        return self._run_query("rank_nonneg", neg_query, env, extra_vars=[("s", s)])

    # --- solver driver --------------------------------------------------

    def _run_query(
        self,
        name: str,
        neg_query,
        env: CegarPromptEnv,
        extra_vars: list[tuple[str, list]],
    ) -> ObligationResult:
        """Check `neg_query` for SAT. UNSAT ⇒ obligation holds."""
        print(f"[smt] check query: {neg_query}")
        solver = cvc5.Solver(env.tm)
        solver.setLogic("ALL")
        solver.setOption("produce-models", "true")
        solver.assertFormula(neg_query)
        res = solver.checkSat()
        if res.isUnsat():
            return ObligationResult(name, True, None)
        # SAT or unknown — extract model values
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
