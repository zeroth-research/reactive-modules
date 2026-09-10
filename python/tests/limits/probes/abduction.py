"""Proof of concept for SMT_ASSIST.md item 6: repair a non-inductive invariant.

`getAbduct` returns a formula B with `inv ∧ B` consistent and
`inv ∧ B ⊨ inv(update s)` -- the conjunct missing from an invariant that is
not inductive. Run it to see both the promise and the catch: every answer
below is *sufficient*, none is weakest.

    uv run python tests/limits/probes/abduction.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvc5

from zrth.lean.cert import CertificateData
from zrth.lean.smt_query import ModuleQueries

CASES = [
    ("m_countdown", "(>= s0 1)"),
    ("m_countdown", "(<= s0 99)"),
    ("m_countdown", "(and (>= s0 0) (<= s0 100))"),   # already inductive
    ("m_twovars", "(>= s0 0)"),
    ("m_toward5", "(<= s0 10)"),
]


def abduce(modname: str, inv: str) -> str:
    module = importlib.import_module(f"mods.{modname}").module()
    q = ModuleQueries.build(module, CertificateData(inv=inv))
    if q is None:
        return "<cvc5 could not encode this module>"
    solver = cvc5.Solver(q.tm)
    solver.setLogic("ALL")
    solver.setOption("produce-abducts", "true")
    s = q.msmt.fresh_ctrl("a_s")
    el = q.msmt.fresh_extl_l("a_el")
    en = q.msmt.fresh_extl_n("a_en")
    nxt = q.msmt.update_state(s, el, en)
    solver.assertFormula(q._sub_state(q.inv, s))
    try:
        return str(solver.getAbduct(q._sub_state(q.inv, nxt)))
    except Exception as e:
        return f"<{type(e).__name__}: {str(e)[:70]}>"


if __name__ == "__main__":
    for mod, inv in CASES:
        print(f"{mod:14s} inv={inv:34s} -> step_inv also needs: {abduce(mod, inv)}")
