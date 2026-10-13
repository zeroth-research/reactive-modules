"""The obligations each route states, held to one meaning.

`hrank` is stated two ways: clamped, `toNat r' < toNat r` as `rule_buchi`
has it (`--pre-check`, `--infer houdini`), and split, `r >= 1 /\\ r' < r`
(`--infer ai-cegis`; `--infer vampire` writes `r > 0`). The two agree on an
Int ranking, which is every ranking that reaches them, so a builder that
states one in place of the other changes no verdict.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from zrth.lean.certificate import Certificate, Claim

cvc5 = pytest.importorskip("cvc5")
from cvc5 import Kind  # noqa: E402

LIMITS = Path(__file__).parent / "limits" / "mods"


def _module(name: str):
    path = LIMITS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_m_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.module()


def _clamps():
    """`(name, tm, clamp)` for each clamp a route states `hrank` with."""
    from zrth.lean.smt.obligations import Obligations
    from zrth.lean.smt_query import ModuleQueries
    from zrth.lean.smt_synth import SynthContext

    claim = Claim(prp="(= s0 0)")
    q = ModuleQueries(_module("m_countdown"), claim,
                      Certificate("(>= s0 0)", "s0"))
    ctx = SynthContext.build(_module("m_countdown"), claim, route="houdini",
                             takes=("int", "bool", "bv", "real", "tuple"))
    ob = Obligations(ctx)
    return [("pre-check", q.tm, q._clamp), ("houdini", ctx.tm, ob._clamp)]


@pytest.mark.parametrize("positive", [1, 0], ids=["r>=1", "r>0"])
def test_the_clamped_and_the_split_hrank_agree_on_an_int_ranking(positive):
    for name, tm, clamp in _clamps():
        r, r_next = (tm.mkConst(tm.getIntegerSort(), n) for n in ("r", "rn"))
        bound = tm.mkInteger(positive)
        clamped = tm.mkTerm(Kind.LT, clamp(r_next), clamp(r))
        split = tm.mkTerm(
            Kind.AND,
            tm.mkTerm(Kind.GEQ if positive else Kind.GT, r, bound),
            tm.mkTerm(Kind.LT, r_next, r),
        )
        solver = cvc5.Solver(tm)
        solver.setLogic("ALL")
        solver.assertFormula(
            tm.mkTerm(Kind.NOT, tm.mkTerm(Kind.EQUAL, clamped, split)))
        assert solver.checkSat().isUnsat(), f"{name}'s clamp disagrees"
