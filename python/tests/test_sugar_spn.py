"""The SPN theory through the sugar DSL: literals, the few operators the theory maps,
`next`/`flow` as method names, `None` flows, and the birth-death example."""

import pytest

from zrth import SPN, Bool, Clock, Nat, Var, X as _X, d as _d
from zrth import Module as compose
from zrth.sugar import Module, X, d, ite, exp, clkrate

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
    assert "Zero" in shown  # the flags have no flow
