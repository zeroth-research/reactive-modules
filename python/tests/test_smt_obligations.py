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
    """`(name, tm, clamp)` for each clamp `hrank` is stated with."""
    from zrth.lean.smt.obligations import clamp
    from zrth.lean.smt_query import ModuleQueries

    claim = Claim(prp="(= s0 0)")
    q = ModuleQueries(_module("m_countdown"), claim,
                      Certificate("(>= s0 0)", "s0"))
    return [("smt.obligations", q.tm, lambda t: clamp(q.tm, t))]


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


def test_bound_is_fresh_as_variables_a_quantifier_can_close():
    """`fresh` makes constants, which a model assigns. A quantifier needs
    variables of the same sorts, which `--infer smt-linear` used to make on
    its own; `bound` is them."""
    from zrth.lean.smt.run import ModuleEncoding

    enc = ModuleEncoding(_module("m_relu_input"))
    bound, fresh = enc.bound("q_"), enc.fresh("c_")
    assert [len(vs) for vs in bound] == [1, 1, 1]
    for vs, consts in zip(bound, fresh):
        assert all(v.getKind() == Kind.VARIABLE for v in vs)
        assert [v.getSort() for v in vs] == [c.getSort() for c in consts]
