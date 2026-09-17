"""`--infer vampire`: Houdini and a ranking search, proved by Vampire.

The half that needs no prover runs everywhere: the time-limit ladder, where
the binary is looked for, the evaluator the simulation runs on (SMT-LIB's
division is Euclidean, and Python's `//` is not), and the simulation
refuting a property the module violates before any prover is started.

The half that runs Vampire is skipped when no binary can be found -- pass
it as `$VAMPIRE`, or put `vampire` on PATH. Those cases are real searches
over `tests/limits` fixtures, and what they pin is what the route is for:
the certificate that comes back is the small one the proofs used, not every
fact Houdini kept.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from zrth.lean.cert import CertificateData
from zrth.lean.common import Refused

LIMITS = Path(__file__).parent / "limits" / "mods"
FIXTURES = Path(__file__).parent / "fixtures"


def module_at(path: Path):
    spec = importlib.util.spec_from_file_location(f"_m_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.module()


def vampire_or_skip() -> str:
    from zrth.lean.magic_vampire import resolve_vampire

    try:
        return resolve_vampire(None)
    except Refused:
        pytest.skip("no Vampire binary: set $VAMPIRE or put `vampire` on PATH")


def infer(name: str, kind: str, prp: str, *, vampire: str = "/nowhere",
          timeout: float = 60, fixture: Path = LIMITS):
    from zrth.lean.magic_vampire import TA2MagicVampire

    magic = TA2MagicVampire(module_at(fixture / f"{name}.py"), vampire=vampire,
                            timeout=timeout, log=lambda *_: None)
    return magic.infer(CertificateData(prp=prp, kind=kind))


# ══════════════════════════════════════════════════════════════════════════
# Without the prover
# ══════════════════════════════════════════════════════════════════════════


def test_the_time_limit_climbs_and_ends_at_the_budget():
    from zrth.lean.magic_vampire import ladder

    assert ladder(120) == (2, 10, 60, 120)
    assert ladder(60) == (2, 10, 60)
    assert ladder(5) == (2, 5)
    assert ladder(1) == (1,)


def test_a_vampire_path_that_is_not_there_is_refused(tmp_path):
    from zrth.lean.magic_vampire import resolve_vampire

    with pytest.raises(Refused, match="--vampire: no such file"):
        resolve_vampire(str(tmp_path / "vampire"))


def test_a_directory_holding_vampire_is_accepted(tmp_path):
    """The release zip unpacks to a directory with the binary in it."""
    from zrth.lean.magic_vampire import resolve_vampire

    exe = tmp_path / "vampire"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert resolve_vampire(str(tmp_path)) == str(exe)


@pytest.mark.parametrize("a", [-7, -6, 0, 5, 7])
@pytest.mark.parametrize("b", [-3, 2, 3])
def test_the_evaluator_divides_the_way_smt_lib_does(a, b):
    """Euclidean: the remainder is never negative, whatever the signs --
    checked against cvc5's own rewriter, which is the semantics every
    obligation is stated in."""
    import cvc5
    from cvc5 import Kind

    from zrth.lean.magic_vampire import Evaluator

    tm = cvc5.TermManager()
    solver = cvc5.Solver(tm)
    x = tm.mkConst(tm.getIntegerSort(), "x")
    ev = Evaluator(tm)
    for kind in (Kind.INTS_DIVISION, Kind.INTS_MODULUS):
        term = tm.mkTerm(kind, x, tm.mkInteger(b))
        want = solver.simplify(term.substitute([x], [tm.mkInteger(a)]))
        assert ev(term, {"x": a}) == want.getIntegerValue()


def test_a_property_the_module_violates_is_refuted_by_running_it():
    """`m_countdown` starts at 100: `s0 <= 50` fails on round zero, and no
    prover is needed -- the binary passed here does not exist."""
    with pytest.raises(Refused, match="does not hold.*s0 = 100"):
        infer("m_countdown", "safety", "(<= s0 50)")


def test_a_bitvector_state_is_refused_by_name():
    with pytest.raises(Refused, match="no bitvector theory"):
        infer("twobit", "buchi", "(and (= s0 (_ bv0 1)) (= s1 (_ bv0 1)))",
              fixture=FIXTURES)


# ══════════════════════════════════════════════════════════════════════════
# With the prover
# ══════════════════════════════════════════════════════════════════════════


def test_vampire_ranks_a_countdown():
    cd = infer("m_countdown", "buchi", "(= s0 0)", vampire=vampire_or_skip())
    assert cd.ranking_smt == "s0"
    assert "(<= 0 s0)" in cd.inv_smt


def test_a_safety_invariant_is_cut_to_what_its_proofs_use():
    """Houdini keeps `0 <= s0` and `s0 <= 100`; the property is the second,
    and it is inductive on its own, so the core leaves the first out."""
    cd = infer("m_countdown", "safety", "(<= s0 100)", vampire=vampire_or_skip())
    assert cd.inv_smt == "(<= s0 100)"


def test_vampire_keeps_the_congruence_a_parity_property_needs():
    """`m_step2` counts 0, 2, ..., 10: `s0 != 5` is not inductive (3 steps
    to 5), and `s0` being even is the fact that makes it so."""
    cd = infer("m_step2", "safety", "(not (= s0 5))", vampire=vampire_or_skip())
    assert "(= (mod s0 2) 0)" in cd.inv_smt
