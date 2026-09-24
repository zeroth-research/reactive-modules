"""The SPN theory through the sugar DSL: literals, the few operators the theory maps,
`next`/`flow` as method names, `None` flows, and the birth-death example."""

import pytest

from zrth import SPN, LIA, Bool, Clock, Event, Nat, Var, X as _X, d as _d
from zrth import Module as compose
from zrth.expr import collecting, expr
from zrth.sugar import Module, X, d, ite, if_then, fired, exp, clkrate

BOOL = Bool([1, 1])


def _ops(block):
    return [type(t.itype).__name__.removeprefix("SPN_") for t in block]


# --- literals and operators ---------------------------------------------------


def test_literals_take_the_variable_sort():
    class M(Module):
        def init(self):
            return 3, 0.5, False

    n, c, b = Var(Nat()), Var(Clock()), Var(BOOL)
    m = M(theory=SPN, ctrl=(n, c, b))
    assert _ops(m.atoms[0].init)[::2] == ["Nat", "Clock", "Bool"]
    shown = m.with_varnames({})
    assert "(3 : nat)" in shown and "(0.5 : clk)" in shown and "(false : bool)" in shown


def test_zero_tests_dispatch_on_sort():
    class M(Module):
        def init(self):
            return 0, 0.0, False, False

        def next(self, n, c, p, q):
            return n, c, n == 0, c != 0

    n, c, p, q = Var(Nat()), Var(Clock()), Var(BOOL), Var(BOOL)
    ops = _ops(M(theory=SPN, ctrl=(n, c, p, q)).atoms[0].update)
    assert "IsZero" in ops and "ClkIsZero" in ops and "Not" in ops


def test_tokens_step_one_at_a_time():
    class M(Module):
        def init(self):
            return 1, 1

        def next(self, a, b):
            return a + 1, b - 1

    a, b = Var(Nat()), Var(Nat())
    ops = _ops(M(theory=SPN, ctrl=(a, b)).atoms[0].update)
    assert "Inc" in ops and "Dec" in ops


@pytest.mark.parametrize(
    "body",
    [
        lambda n, c: n == 1,           # only zero can be tested
        lambda n, c: n == c,           # no equality between values
        lambda n, c: n + 2,            # tokens move one at a time
        lambda n, c: c + 1,            # a clock is not a counter
        lambda n, c: 2 * c,            # a rate needs the time form, not a clock value
    ],
)
def test_unsupported_operators_raise(body):
    class M(Module):
        def init(self):
            return 0, 0.0

        def next(self, n, c):
            body(n, c)
            return n, c

    with pytest.raises(TypeError):
        M(theory=SPN, ctrl=(Var(Nat()), Var(Clock())))


def test_an_event_flag_takes_the_bool_sugar():
    class M(Module):
        def init(self, t):
            return exp(1.0), False           # a bool literal on an Event variable

        def next(self, clk, fired, t):
            return ite(clk == 0, exp(1.0), clk), clk == 0   # a Bool-sorted test

        def flow(self, clk, fired, t):
            return -1 * d(t), None           # the flag has the trivial tangent

    clk, fired, t = Var(Clock()), Var(Event()), Var(Clock())
    m = M(theory=SPN, ctrl=(clk, fired), extl=(t,))
    assert _ops(m.atoms[0].delay) == ["ClkMul", "Id", "Zero"]
    shown = m.with_varnames({t: "t", clk: "clk", fired: "fired"})
    assert "fired : Event" in shown and "(false : bool)" in shown


def test_if_then_guards_an_update():
    """The place counts a token in only on the steps where the transition fires: off
    those steps the update has no value at all, and `n` is left to the rest of the net."""

    class M(Module):
        def init(self, t):
            return exp(1.0), 0

        def next(self, clk, n, t):
            fires = clk == 0
            return ite(fires, exp(1.0), clk), if_then(fires, n + 1)

        def flow(self, clk, n, t):
            return -1 * d(t), None

    clk, n, t = Var(Clock()), Var(Nat()), Var(Clock())
    m = M(theory=SPN, ctrl=(clk, n), extl=(t,))
    assert "IfThen" in _ops(m.atoms[0].update)
    assert "IfThen" in m.with_varnames({t: "t", clk: "clk", n: "n"})


def test_fired_is_a_change_of_the_event():
    # `X(e) != e` in the boolean fragment: the theory has no equality
    e = Var(Event())
    with collecting() as terms:
        out = fired(expr(e, theory=SPN))
    assert _ops(terms) == ["Not", "And", "Not", "And", "Or"]
    assert {w for t in terms for w in t.read} >= {e, _X(e)}
    assert isinstance(out.dtype, Event) and out.wire == terms[-1].write[0]


def test_fired_reads_an_event_alone():
    with collecting():
        for v in (Var(BOOL), Var(Nat()), Var(Clock())):
            with pytest.raises(TypeError, match="Event"):
                fired(expr(v, theory=SPN))


def test_if_then_needs_an_expr_branch_and_an_spn_guard():
    with collecting():
        c = expr(Var(Clock()), theory=SPN)
        # the branch carries the sort, so a bare literal has nothing to agree with
        with pytest.raises(TypeError, match="must be an Expr"):
            if_then(c == 0, 1)
        # and the partial form is this theory's alone
        b = expr(True, theory=LIA)
        with pytest.raises(TypeError, match="no partial if-then"):
            if_then(b, b)


@pytest.mark.parametrize(
    "body, ops",
    [
        (lambda a, b: a >= b, ["ClkGe"]),
        (lambda a, b: a <= b, ["ClkGe"]),  # the swap is in the operands
        (lambda a, b: a < b, ["ClkGe", "Not"]),  # not (a >= b)
        (lambda a, b: a > b, ["ClkGe", "Not"]),  # not (b >= a)
        (lambda a, b: a >= 2.5, ["Clock", "ClkGe"]),  # the bound is a clock literal
        (lambda a, b: 2.5 <= a, ["Clock", "ClkGe"]),  # python reflects it to `a >= 2.5`
    ],
)
def test_clock_orderings_fold_to_clkge(body, ops):
    with collecting() as terms:
        a, b = expr(Var(Clock()), theory=SPN), expr(Var(Clock()), theory=SPN)
        body(a, b)
    assert _ops(terms) == ops


def test_le_swaps_the_operands_rather_than_negating():
    a, b = Var(Clock()), Var(Clock())
    with collecting() as terms:
        expr(a, theory=SPN) <= expr(b, theory=SPN)
    [ge] = terms
    assert list(ge.read) == [b, a]


def test_orderings_are_on_clock_values_alone():
    with collecting():
        n = expr(Var(Nat()), theory=SPN)
        c = expr(Var(Clock()), theory=SPN)
        rate = d(expr(Var(Clock()), theory=SPN))
        with pytest.raises(TypeError, match="orders clock values"):
            n >= n  # a place is tested against zero, never ordered
        with pytest.raises(TypeError, match="orders clock values"):
            c >= n
        with pytest.raises(TypeError, match="orders clock values"):
            rate >= rate  # a rate is not a time


def test_the_earlier_of_two_clocks():
    """`min(a, b)` -- the term the module docs promise, in the DSL."""

    class M(Module):
        def init(self, t):
            return exp(2.0), exp(1.0), 0.0

        def next(self, a, b, first, t):
            return a, b, ite(a >= b, b, a)

        def flow(self, a, b, first, t):
            return -1 * d(t), -1 * d(t), -1 * d(t)

    a, b, first, t = Var(Clock()), Var(Clock()), Var(Clock()), Var(Clock())
    m = M(theory=SPN, ctrl=(a, b, first), extl=(t,))
    ops = _ops(m.atoms[0].update)
    assert ops.count("ClkGe") == 1 and "Ite" in ops and "Not" not in ops


# --- flows -------------------------------------------------------------------


def test_rate_times_time_form_scales_the_time_forms_tangent():
    class M(Module):
        def init(self, t):
            return exp(2.0)

        def flow(self, c, t):
            return -1 * d(t)

    c, t = Var(Clock()), Var(Clock())
    m = M(theory=SPN, ctrl=(c,), extl=(t,))
    [rate, _] = m.atoms[0].delay
    assert _ops([rate]) == ["ClkMul"] and list(rate.read) == [_d(t)]
    assert "ClkMul(-1)" in m.with_varnames({})
    assert _d(c) in m.atoms[0].delay.write()
    assert m.open() and set(m.extl) == {t}  # the time form is read, so t is a real external


def test_none_is_the_zero_flow():
    class M(Module):
        def init(self, t):
            return exp(1.0), 0, False

        def flow(self, c, n, b, t):
            return None, None, None

    c, n, b, t = Var(Clock()), Var(Nat()), Var(BOOL), Var(Clock())
    ops = _ops(M(theory=SPN, ctrl=(c, n, b), extl=(t,)).atoms[0].delay)
    assert ops == ["ClkZero", "Zero", "Zero"]


def test_conditional_rate_freezes_a_clock():
    class M(Module):
        def init(self, n, t):
            return exp(1.0)

        def flow(self, c, n, t):
            return ite(n != 0, -1 * d(t), clkrate(0))

    c, n, t = Var(Clock()), Var(Nat()), Var(Clock())
    m = M(theory=SPN, ctrl=(c,), extl=(n, t))
    assert "Ite" in _ops(m.atoms[0].delay)
    shown = m.with_varnames({})
    assert "ClkMul(-1)" in shown and "ClkRate(0)" in shown


def test_if_then_in_a_flow_is_the_invariant():
    # the clock runs down while it is non-negative; past that the flow has no value
    class M(Module):
        def init(self, t):
            return exp(1.0)

        def flow(self, c, t):
            return if_then(c >= 0, -1 * d(t))

    c, t = Var(Clock()), Var(Clock())
    m = M(theory=SPN, ctrl=(c,), extl=(t,))
    assert _ops(m.atoms[0].delay) == ["Clock", "ClkGe", "ClkMul", "IfThen", "Id"]
    [guard] = [term for term in m.atoms[0].delay if _ops([term]) == ["IfThen"]]
    assert guard.write[0].dtype == Clock(1) and _d(c) in m.atoms[0].delay.write()


# --- method names -------------------------------------------------------------


def test_next_and_flow_are_aliases_for_update_and_delay():
    class WithOld(Module):
        def init(self, t):        return 0
        def update(self, n, t):   return n
        def delay(self, n, t):    return None

    class WithNew(Module):
        def init(self, t):        return 0
        def next(self, n, t):     return n
        def flow(self, n, t):     return None

    n, t = Var(Nat()), Var(Clock())
    old, new = WithOld(theory=SPN, ctrl=(n,), extl=(t,)), WithNew(theory=SPN, ctrl=(n,), extl=(t,))
    assert _ops(old.atoms[0].update) == _ops(new.atoms[0].update)
    assert _ops(old.atoms[0].delay) == _ops(new.atoms[0].delay)


def test_defining_both_names_raises():
    class Both(Module):
        def init(self):          return 0
        def update(self, n):     return n
        def next(self, n):       return n

    with pytest.raises(TypeError, match="only one of"):
        Both(theory=SPN, ctrl=(Var(Nat()),))


# --- the example --------------------------------------------------------------


def test_birth_death_example():
    from zrth.examples import birth_death as bd

    assert bd.system.open() and set(bd.system.extl) == {bd.t}  # the time reference
    assert set(bd.system.prvt) == {bd.bclk, bd.dclk}
    assert set(bd.system.intf) == {bd.bth, bd.dth, bd.n}
    shown = bd.system.with_varnames({bd.t: "t", bd.n: "n"})
    assert "Exp(2)" in shown and "Exp(1)" in shown
    assert "ClkMul(-1)" in shown and "ClkMul(0)" in shown  # clocks run down against t; the death clock freezes on an empty place
    assert "Zero" in shown  # the events have no flow
    for atom in (bd.birth, bd.death):
        [flow] = [term for term in atom.atoms[0].delay if _ops([term]) == ["IfThen"]]
        assert flow.write[0].dtype == Clock(1)  # `clk >= 0` guards the clock's rate: the invariant
        assert "IfThen" in _ops(atom.atoms[0].update)  # the event toggles only where the transition fires
    assert _ops(bd.place.atoms[0].update).count("Or") == 2  # a `fired` per event
