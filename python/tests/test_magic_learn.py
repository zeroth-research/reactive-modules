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
    _prune,
    _smt_affine,
    _smt_conjunction,
    _smt_int,
    _smt_ranking,
    _smt_term,
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
    with pytest.raises(Refused, match="--fbk-proveit.*--infer ai-cegis"):
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


def test_a_rule_shape_the_procedure_cannot_read_is_a_refusal_not_a_traceback():
    """The procedure refuses by raising, and not all of it happens at the door:
    a rule the LP cannot read as linear rows is met when the obligation is cut
    up. That is still a reason this route cannot answer, so it reaches the user
    as an error line."""
    with pytest.raises(Refused, match="cannot certify this property"):
        _learn(_countdown(), "(= (mod s0 2) 0)", "safety")


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
    assert _smt_conjunction(["(<= s0 100)", "(>= s0 0)", "(<= s0 100)"]) == \
        "(and (<= s0 100) (>= s0 0))"
    assert _smt_conjunction(["(<= s0 100)"]) == "(<= s0 100)"
    assert _smt_conjunction([]) == "true"


def test_a_conjunct_z3_cannot_print_flat_is_refused():
    """z3 `let`-binds a repeated subterm, and the body of a Lean definition
    cannot carry one -- so a conjunct that prints that way is named rather than
    pasted. Houdini's own candidates never do; the property, which can, is
    printed from its own source instead."""
    import z3
    s0 = z3.Int("s0")
    big = z3.If(s0 > 0, s0, 0) + z3.If(s0 > 0, s0, 0)
    for _ in range(4):
        big = big + big
    with pytest.raises(Refused, match="let"):
        _smt_term(big >= 0)
    assert _smt_term(s0 >= 0) == "(>= s0 0)"


def test_a_net_shaped_property_gets_a_flat_certificate():
    """`relu(x) + relu(100 - x) = 100` is `0 <= x <= 100` written as a net. The
    route proves it, and what it hands over is a certificate a Lean definition
    can carry: no `let`, whether the property survives pruning (and is printed
    from its own source) or is dropped for the simpler facts that imply it."""
    module = _countdown()
    net = ("(and (>= (+ (ite (>= s0 0) s0 0) (ite (>= (- 100 s0) 0) (- 100 s0) 0)) 100)"
           " (<= (+ (ite (>= s0 0) s0 0) (ite (>= (- 100 s0) 0) (- 100 s0) 0)) 100))")
    cd = _learn(module, net, "safety")
    assert "let" not in cd.inv_smt, cd.inv_smt
    v = _verdicts(module, cd)
    assert v == {"init_inv": Status.HOLDS, "step_inv": Status.HOLDS,
                 "inv_imp_P": Status.HOLDS}, v


# ---------------------------------------------------------------------------
# What the invariant carries
# ---------------------------------------------------------------------------

def test_a_conjunct_the_others_imply_is_dropped():
    """Houdini's lattice is redundant: `s1 == 10` arrives beside five weaker
    facts about `s1`. Dropping what the rest imply leaves the same predicate,
    and the conjunct count is what the obligation's disjuncts are exponential
    in."""
    import z3
    s0, s1 = z3.Int("s0"), z3.Int("s1")
    facts = [("s1>0", s1 > 0), ("s1>=1", s1 >= 1), ("s1==10", s1 == 10),
             ("s0>=0", s0 >= 0)]
    kept = _prune(facts)
    assert [lbl for lbl, _ in kept] == ["s1==10", "s0>=0"], kept
    # ... and the conjunction is the one it started as
    solver = z3.Solver()
    solver.add(z3.And(*[t for _, t in facts]) != z3.And(*[t for _, t in kept]))
    assert solver.check() == z3.unsat


def test_nothing_is_dropped_when_nothing_is_implied():
    import z3
    facts = [("s0>=0", z3.Int("s0") >= 0), ("s1<=3", z3.Int("s1") <= 3)]
    assert [lbl for lbl, _ in _prune(facts)] == ["s0>=0", "s1<=3"]


def test_the_witness_carries_the_pruned_invariant(monkeypatch):
    """Pruning has to happen before the witness is built, not before it is
    printed.

    The conjunct count is what the obligation\'s disjuncts are exponential in:
    the fbk sweep\'s `NetTwoInput` does not finish in ten minutes carrying
    Houdini\'s ten conjuncts, and takes half a second carrying the four the
    rest do not imply. So what matters is which set reaches `certify`, and
    that is what this reads off rather than the clock."""
    import benchmarks.svcomp._farkas as farkas
    import benchmarks.svcomp._invariants as invariants

    found, carried = [], []
    real_infer, real_certify = invariants.infer_invariants, farkas.certify

    def spy_infer(system, *a, **kw):
        out = real_infer(system, *a, **kw)
        found.append(len(out))
        return out

    def spy_certify(system, claim, witness, *a, **kw):
        carried.append(len(getattr(witness, "inv", ())))
        return real_certify(system, claim, witness, *a, **kw)

    monkeypatch.setattr(invariants, "infer_invariants", spy_infer)
    monkeypatch.setattr(farkas, "certify", spy_certify)
    cd = _learn(_countdown(), "(<= s0 100)", "safety")

    assert found and carried, "the route did not reach Houdini and the procedure"
    assert carried[0] < found[0], "the witness carried every candidate Houdini kept"
    # ... and the certificate states exactly what the witness carried
    assert cd.inv_smt.count("(<=") + cd.inv_smt.count("(>=") == carried[0]


def test_the_invariant_carries_no_redundant_conjunct():
    """End to end: the certificate states each fact once, and the property it
    has to imply need not appear in it -- being implied is the obligation."""
    module = _countdown()
    cd = _learn(module, "(<= s0 100)", "safety")
    assert cd.inv_smt.count("(<= s0 100)") == 1, cd.inv_smt


def test_a_property_is_parsed_over_the_columns():
    import z3
    term = _parse_property("(and (= s0 0) (> s1 s0))", ("s0", "s1"), ("s0", "s1"))
    assert z3.is_bool(term)
    assert {str(v) for v in z3.z3util.get_vars(term)} == {"s0", "s1"}


def test_a_property_that_is_two_expressions_is_refused():
    with pytest.raises(Refused, match="one expression"):
        _parse_property("(= s0 0)) (assert (> s0 1)", ("s0",), ("s0",))


# ---------------------------------------------------------------------------
# The rank shape a wrap-around needs
# ---------------------------------------------------------------------------

def _counter(bound: int):
    """`x = 0; loop { x = (x + 1) % (bound + 1) }` -- a run that wraps around.

    `x == 0` recurs, but no rank of the learner's class drops on every round it
    fails on: a non-negative sum of ReLUs is convex in the state, and this needs
    one that falls along 1..bound and falls again from bound back to 0."""
    class Counter(sugar.Module):
        def init(self):
            return 0

        def update(self, x):
            return ite(x == bound, 0, x + 1)

    return Counter(ctrl=(Var(Int([1, 1])),), theory=LIA)


def test_a_wrap_around_gets_a_rank_zeroed_where_the_property_holds():
    """No rank in the class drops on the round that wraps, so the route falls
    back to ranking the rounds *before* the property is reached and zeroing the
    rank where it holds. That is still verith's obligation: the round that
    reaches `P` drops from `delta + V` to `0`, and `V >= 0` structurally."""
    module = _counter(9)
    cd = _learn(module, "(= s0 0)", "buchi")
    assert cd.ranking_smt.startswith("(ite (= s0 0) 0 (+ 1 "), cd.ranking_smt
    v = _verdicts(module, cd)
    assert v == {"init_inv": Status.HOLDS, "step_inv": Status.HOLDS,
                 "hrank": Status.HOLDS}, v


def test_a_rank_that_needs_no_zeroing_is_left_alone():
    """Counting *down* to the property needs no wrap, so the first shape works
    and the certificate carries the net itself."""
    cd = _learn(_countdown(), "(= s0 0)", "buchi")
    assert not cd.ranking_smt.startswith("(ite"), cd.ranking_smt


def test_the_bound_in_the_guard_is_offered_to_houdini():
    """`0 <= x <= 9` is what makes the wrap-around's rank certify, and `9` is a
    constant only the guard mentions -- and only after `x + 1 == 10` has been
    normalised onto `x`. Neither the sign candidates nor the constant-coordinate
    ones carry it."""
    import benchmarks.svcomp._farkas as farkas
    from benchmarks.svcomp._invariants import infer_invariants

    module = _counter(9)
    system = farkas.read_system(module, {v: f"s{i}" for i, v in enumerate(module.ctrl)})
    assert "s0<=9" in [lbl for lbl, _ in infer_invariants(system)]
