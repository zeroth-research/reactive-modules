"""The `--infer nuterm` route (`zrth.lean.magic_learn`).

What the route promises is that the certificate it returns has already been
proved, so the tests that matter put its output to verith's *own* obligations
(`smt_query.pre_check`, which is cvc5 restating `rule_buchi` / `rule_globally`
from scratch). A certificate the Farkas procedure certified and cvc5 refutes
would be the route lying about what it did.

The rest pin the two seams that are this module's own: the SMT-LIB it prints,
and what it refuses.
"""

import numpy as np
import pytest

from zrth import Int, LIA, Module, Real, Var, X, sugar
from zrth.analyzer import convert_method
from zrth.sugar import ite
from zrth.lean.cert import CertificateData
from zrth.lean.common import Refused
from zrth.lean.magic_learn import (
    TA2MagicLearn,
    _parse_property,
    _smt_affine,
    _smt_conjunction,
    _smt_int,
    _smt_ranking,
)
from zrth.lean.smt_query import SmtBudget, Status, pre_check

pytest.importorskip("cvc5")


# ---------------------------------------------------------------------------
# Modules under test
# ---------------------------------------------------------------------------

def _countdown() -> Module:
    """`x = 100; loop { if x == 0 then x = 100 else x = x - 1 }`.

    `x == 0` recurs, under the invariant `0 <= x <= 100` and a rank that drops
    on every round where it does not hold."""
    def init():
        return 100

    def update(old_x):
        if old_x == 0:
            return 100
        return old_x - 1

    state = Var(Int([1, 1]))
    return Module.sequential(
        [state],
        convert_method(init, {}, [X(state)], theory=LIA),
        convert_method(update, {"old_x": state}, [X(state)], theory=LIA),
    )


def _real_counter() -> Module:
    """The same shape over a real-valued component: outside the procedure."""
    def init():
        return 0.0

    def update(old_x):
        return old_x + 1.0

    state = Var(Real([1, 1]))
    from zrth import LRA
    return Module.sequential(
        [state],
        convert_method(init, {}, [X(state)], theory=LRA),
        convert_method(update, {"old_x": state}, [X(state)], theory=LRA),
    )


def _write_only_second() -> Module:
    """The countdown beside a component nothing reads latched.

    `y` is driven every round and read by no term, so it is not a column of the
    system the proof quantifies over -- while still being `s1` to verith, which
    numbers the state by the module's ctrl variables."""
    class Prog(sugar.Module):
        def init(self):
            return 100, 0

        def update(self, x, y):
            return ite(x == 0, 100, x - 1), 0

    return Prog(ctrl=(Var(Int([1, 1])), Var(Int([1, 1]))), theory=LIA)


def _learn(module, prp, kind):
    cd = CertificateData(prp=prp, kind=kind)
    return TA2MagicLearn("", module, log=lambda *a: None).infer(cd)


def _verdicts(module, cd):
    return {v.name: v.status
            for v in pre_check(module, cd, SmtBudget(), log=lambda *a: None)}


# ---------------------------------------------------------------------------
# What the route returns is a certificate verith's own checker accepts
# ---------------------------------------------------------------------------

def test_a_learned_buchi_certificate_holds_under_cvc5():
    """The rank the Farkas procedure certified discharges `rule_buchi`'s
    obligations as cvc5 states them: the invariant holds at entry and is
    preserved, and the rank drops wherever the property does not hold."""
    module = _countdown()
    cd = _learn(module, "(= s0 0)", "buchi")
    assert cd.inv and cd.ranking, "a Buchi certificate is both"
    assert cd.inv_smt and cd.ranking_smt, "cvc5 needs its own input back"
    v = _verdicts(module, cd)
    assert v == {"init_inv": Status.HOLDS, "step_inv": Status.HOLDS,
                 "hrank": Status.HOLDS}, v


def test_a_learned_safety_invariant_implies_the_property():
    """`rule_globally` takes no rank, so the certificate is the invariant --
    and what makes it a safety proof is that it implies the property."""
    module = _countdown()
    cd = _learn(module, "(<= s0 100)", "safety")
    assert cd.inv and cd.ranking is None, "a safety certificate has no rank"
    v = _verdicts(module, cd)
    assert v == {"init_inv": Status.HOLDS, "step_inv": Status.HOLDS,
                 "inv_imp_P": Status.HOLDS}, v


def test_the_invariant_is_the_one_the_program_has():
    """Houdini's lattice covers this program's bound, so the route finds it
    rather than something vacuous that happens to be inductive."""
    cd = _learn(_countdown(), "(= s0 0)", "buchi")
    assert "(<= s0 100)" in cd.inv_smt and "(>= s0 0)" in cd.inv_smt, cd.inv_smt


def test_a_property_the_route_cannot_establish_is_refused():
    """A property the route cannot prove -- false here, but a true one outside
    Houdini's lattice of signs and pairwise facts reads the same way -- is a
    refusal naming the routes that are not so limited, not a certificate that
    fails four steps later in cvc5 or in lake."""
    with pytest.raises(Refused, match="--fbk-proveit.*--infer ai-cegar"):
        _learn(_countdown(), "(<= s0 3)", "safety")


# ---------------------------------------------------------------------------
# What the route refuses
# ---------------------------------------------------------------------------

def test_a_module_the_procedure_cannot_read_is_refused():
    """The procedure reads scalar integer wires; a real component is refused
    by name at the door rather than approximated."""
    with pytest.raises(Refused, match="scalar"):
        _learn(_real_counter(), "(<= s0 100)", "safety")


def test_a_property_naming_something_that_is_not_a_column_is_refused():
    """`s1` is a state component verith numbers and the procedure has no column
    for. The refusal says which it is, rather than reading as an undeclared
    name -- so every `s_i` the module has is declared for the parse."""
    with pytest.raises(Refused, match=r"names \['s1'\].*not columns"):
        _learn(_write_only_second(), "(= s1 0)", "buchi")


def test_a_property_over_the_columns_of_such_a_module_is_certified():
    """The component that is not a column is no obstacle to a property that
    does not name one: the certificate is over the columns and verith's
    obligations are over the whole state."""
    module = _write_only_second()
    cd = _learn(module, "(= s0 0)", "buchi")
    v = _verdicts(module, cd)
    assert v == {"init_inv": Status.HOLDS, "step_inv": Status.HOLDS,
                 "hrank": Status.HOLDS}, v


def test_a_property_that_is_not_smt_lib_is_refused():
    """The route reads the property as SMT-LIB, and says so of one that is not
    -- `x == 0` is the Python spelling the prompt routes also accept."""
    with pytest.raises(Refused, match="SMT-LIB"):
        _learn(_countdown(), "x == 0", "buchi")


def test_a_run_with_no_property_is_refused():
    with pytest.raises(Refused, match="--safety or --buchi"):
        TA2MagicLearn("", _countdown()).infer(CertificateData())


# ---------------------------------------------------------------------------
# The SMT-LIB the route prints
# ---------------------------------------------------------------------------

def test_a_negative_literal_is_a_negation():
    """SMT-LIB has no negative numerals; `-3` is `(- 3)`."""
    assert _smt_int(3) == "3" and _smt_int(-3) == "(- 3)" and _smt_int(0) == "0"


@pytest.mark.parametrize("coeffs, const, want", [
    ([0], 0, "0"),                                  # nothing at all
    ([0], 5, "5"),                                  # a constant alone
    ([1], 0, "s0"),                                 # a bare column
    ([-1], 0, "(- s0)"),
    ([2], 0, "(* 2 s0)"),
    ([1, -1], 3, "(+ 3 s0 (- s1))"),
    ([1, 0, 2], -1, "(+ (- 1) s0 (* 2 s2))"),       # a zero coefficient drops
])
def test_an_affine_row_prints_as_one_term(coeffs, const, want):
    assert _smt_affine(coeffs, const, ["s0", "s1", "s2"][:len(coeffs)]) == want


def test_a_relu_unit_prints_without_a_let():
    """A ReLU names its pre-activation twice. z3's own printer would bind it
    with a `let`, which cannot be the body of the Lean definition this string
    becomes -- so the term is written out in full."""
    layers = [(np.array([[1, -1]]), np.array([2])), (np.array([[3]]), np.array([0]))]
    src = _smt_ranking(layers, ["s0", "s1"])
    assert src == "(* 3 (ite (> (+ 2 s0 (- s1)) 0) (+ 2 s0 (- s1)) 0))"
    assert "let" not in src


def test_a_unit_the_output_layer_drops_is_not_printed():
    """A zero output weight contributes nothing, so it says nothing."""
    layers = [(np.array([[1], [1]]), np.array([0, 9])),
              (np.array([[1, 0]]), np.array([0]))]
    src = _smt_ranking(layers, ["s0"])
    assert "9" not in src, src


def test_the_same_fact_twice_is_one_conjunct():
    """A property seeded as a candidate that the lattice also proposes survives
    as both, and the certificate should carry it once."""
    import z3
    s0 = z3.Int("s0")
    assert _smt_conjunction([s0 <= 100, s0 >= 0, s0 <= 100]) == \
        "(and (<= s0 100) (>= s0 0))"
    assert _smt_conjunction([s0 <= 100]) == "(<= s0 100)"
    assert _smt_conjunction([]) == "true"


def test_a_property_is_parsed_over_the_columns():
    import z3
    term = _parse_property("(and (= s0 0) (> s1 s0))", ("s0", "s1"), ("s0", "s1"))
    assert z3.is_bool(term)
    assert {str(v) for v in z3.z3util.get_vars(term)} == {"s0", "s1"}


def test_a_property_that_is_two_expressions_is_refused():
    with pytest.raises(Refused, match="one expression"):
        _parse_property("(= s0 0)) (assert (> s0 1)", ("s0",), ("s0",))
