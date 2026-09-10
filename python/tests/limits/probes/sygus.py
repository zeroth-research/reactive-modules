"""Proof of concept for SMT_ASSIST.md item 7: synthesise the ranking function.

No LLM, no API key, deterministic. The constraints are the two `magic_cegar`
already builds -- `inv s → rank s ≥ 0` and
`inv s ∧ ¬P s → rank (update s) < rank s` -- with `rank` a `synthFun`
instead of a parsed term.

Note the times: unrestricted LIA is fine for one variable and already 10 s
for two, which is why step 3 of the plan is "give it a grammar".

    uv run python tests/limits/probes/sygus.py
"""

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvc5
from cvc5 import Kind

from zrth.lean.smt_module import ModuleSMT
from zrth.lean.smt_prompt import CegarPromptEnv, parse_predicate

CASES = [
    ("m_countdown", "(= s0 0)", "(and (>= s0 0) (<= s0 100))"),
    ("m_toward5", "(= s0 5)", "(and (>= s0 0) (<= s0 10))"),
    ("m_twovars", "(and (= s0 0) (= s1 10))",
     "(and (>= s0 0) (<= s0 s1) (= s1 10))"),
]


def synth_rank(modname: str, prp_src: str, inv_src: str) -> str:
    module = importlib.import_module(f"mods.{modname}").module()
    tm = cvc5.TermManager()
    msmt = ModuleSMT(tm=tm, module=module)
    env = CegarPromptEnv(msmt)
    prp = parse_predicate(env, prp_src)
    inv = parse_predicate(env, inv_src)
    sv = env.state_vars

    sol = cvc5.Solver(tm)
    sol.setOption("sygus", "true")
    sol.setLogic("LIA")

    Int = tm.getIntegerSort()
    args = [tm.mkVar(Int, f"x{i}") for i in range(len(sv))]
    rank = sol.synthFun("rank", args, Int)

    svars = [sol.declareSygusVar(f"v{i}", Int) for i in range(len(sv))]
    el = [sol.declareSygusVar(f"el{i}", Int) for i in range(len(msmt.extl_latched))]
    en = [sol.declareSygusVar(f"en{i}", Int) for i in range(len(msmt.extl_next))]
    nxt = msmt.update_state(svars, el, en)

    def app(vs):
        return tm.mkTerm(Kind.APPLY_UF, rank, *vs)

    inv_s = inv.substitute(sv, svars)
    prp_s = prp.substitute(sv, svars)
    sol.addSygusConstraint(
        tm.mkTerm(Kind.IMPLIES, inv_s, tm.mkTerm(Kind.GEQ, app(svars), tm.mkInteger(0)))
    )
    sol.addSygusConstraint(
        tm.mkTerm(
            Kind.IMPLIES,
            tm.mkTerm(Kind.AND, inv_s, tm.mkTerm(Kind.NOT, prp_s)),
            tm.mkTerm(Kind.LT, app(nxt), app(svars)),
        )
    )

    t0 = time.perf_counter()
    res = sol.checkSynth()
    dt = (time.perf_counter() - t0) * 1000
    body = str(sol.getSynthSolution(rank)) if res.hasSolution() else f"none ({res})"
    return f"{body}   ({dt:.0f} ms)"


if __name__ == "__main__":
    for mod, prp, inv in CASES:
        print(f"{mod:14s} {synth_rank(mod, prp, inv)}")
