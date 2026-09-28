"""The SPN simulator: exact expiries, invariants as the end of a step, the gap schedulers,
timelock, skipped guards, and the birth-death example over a long run."""

import math

import pytest

from zrth import SPN, Clock, Event, Nat, Var
from zrth import Module as compose
from zrth.simulate import FlowNotSolvable, GuardSkipped, NoFlow, first, last, simulate, uniform
from zrth.sugar import Module, d, ite, fired, exp


class Ticker(Module):
    """One clock counting its own expiries; it re-arms to 5."""

    def init(self, t):
        return 0.37, 0

    def next(self, c, k, t):
        fires = c == 0
        return ite(fires, 5.0, c), ite(fires, k + 1, k)

    def flow(self, c, k, t):
        return ite(c >= 0, -1 * d(t), None), 0


def _ticker(cls=Ticker):
    c, k, t = Var(Clock()), Var(Nat()), Var(Clock())
    return cls(theory=SPN, ctrl=(c, k), extl=(t,)), c, k, t


# --- flows and expiries -------------------------------------------------------


def test_a_clock_expires_exactly_where_the_invariant_stops_time():
    m, c, k, t = _ticker()
    trace = simulate(m, t=t, horizon=1.0)
    # init, the round at the expiry, the (stuttering) round at the horizon
    assert trace.times == [0.0, 0.37, 1.0]
    assert trace.series(k) == [0, 1, 1]
    assert trace.series(t) == trace.times
    assert math.isclose(trace.series(c)[-1], 5.0 - (1.0 - 0.37), rel_tol=1e-12)
    assert not trace.quiescent


def test_a_frozen_clock_does_not_move_at_all():
    class M(Module):
        def init(self, t):
            return 0.37, 0.5

        def next(self, c, f, t):
            return ite(c == 0, 5.0, c), f

        def flow(self, c, f, t):
            return ite(c >= 0, -1 * d(t), None), 0 * d(t)

    c, f, t = Var(Clock()), Var(Clock()), Var(Clock())
    trace = simulate(M(theory=SPN, ctrl=(c, f), extl=(t,)), t=t, horizon=1.0)
    assert trace.times == [0.0, 0.37, 1.0]
    assert all(v == 0.5 for v in trace.series(f))


def test_nothing_before_the_horizon_just_flows_to_it():
    class M(Ticker):
        def init(self, t):
            return 3.0, 0

    m, c, k, t = _ticker(M)
    trace = simulate(m, t=t, horizon=1.0)
    assert trace.times == [0.0, 1.0] and trace.series(c) == [3.0, 2.0]


def test_a_rate_may_not_switch_on_a_clock_inside_a_flow():
    class M(Ticker):
        def flow(self, c, k, t):
            return ite(c >= 0, -1 * d(t), 0 * d(t)), 0

    m, c, k, t = _ticker(M)
    with pytest.raises(FlowNotSolvable, match="must be None"):
        simulate(m, t=t, horizon=1.0)


def test_a_guard_without_an_invariant_would_be_skipped():
    class M(Ticker):
        def flow(self, c, k, t):
            return -1 * d(t), 0  # nothing stops time at the expiry

    m, c, k, t = _ticker(M)
    with pytest.raises(GuardSkipped, match="invariant"):
        simulate(m, t=t, horizon=1.0)


def test_the_only_external_is_the_time_reference():
    class M(Module):
        def init(self, t, other):
            return 1.0

        def flow(self, c, t, other):
            return ite(other != 0, -1 * d(t), 0 * d(t))

    c, t, other = Var(Clock()), Var(Clock()), Var(Nat())
    with pytest.raises(TypeError, match="time reference"):
        simulate(M(theory=SPN, ctrl=(c,), extl=(t, other)), t=t, horizon=1.0)


# --- the end of a step ------------------------------------------------------------


def test_stuck_when_time_must_stop_and_nothing_fires():
    class Stuck(Module):
        def init(self, t):
            return 1.0

        def next(self, c, t):
            return c

        def flow(self, c, t):
            return ite(c >= 0, -1 * d(t), None)

    c, t = Var(Clock()), Var(Clock())
    with pytest.raises(NoFlow, match="changes nothing") as e:
        simulate(Stuck(theory=SPN, ctrl=(c,), extl=(t,)), t=t, horizon=10.0)
    assert e.value.trace.times == [0.0, 1.0] and e.value.trace.series(c) == [1.0, 0.0]


def test_a_state_with_no_flow_cannot_be_entered():
    class M(Ticker):
        def init(self, t):
            return -1.0, 0  # past the invariant from the start

    m, c, k, t = _ticker(M)
    with pytest.raises(NoFlow, match="no value") as e:
        simulate(m, t=t, horizon=1.0)
    assert e.value.trace.times == [0.0]


def test_quiescent_when_nothing_can_ever_change():
    class Frozen(Module):
        def init(self, t):
            return 0.5

        def flow(self, c, t):
            return 0 * d(t)

    c, t = Var(Clock()), Var(Clock())
    trace = simulate(Frozen(theory=SPN, ctrl=(c,), extl=(t,)), t=t, horizon=10.0)
    assert trace.quiescent and trace.times == [0.0, 10.0] and trace.series(c) == [0.5, 0.5]


# --- schedulers -------------------------------------------------------------------


class Metronome(Module):
    """An event every unit of time, by a clock re-armed to 1."""

    def init(self, t):
        return 1.0, False

    def next(self, c, e, t):
        fires = c == 0
        return ite(fires, 1.0, c), ite(fires, ~e, e)

    def flow(self, c, e, t):
        return ite(c >= 0, -1 * d(t), None), 0


def test_first_takes_a_round_every_gap_and_at_the_expiry():
    c, e, t = Var(Clock()), Var(Event()), Var(Clock())
    m = Metronome(theory=SPN, ctrl=(c, e), extl=(t,))
    trace = simulate(m, t=t, horizon=2.0, scheduler=first(0.25))
    assert trace.times == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    # the rounds between expiries stutter; the event toggles exactly at the expiries
    assert trace.series(e) == [False] * 4 + [True] * 4 + [False]


def test_uniform_lands_on_the_expiry_after_a_few_random_rounds():
    c, e, t = Var(Clock()), Var(Event()), Var(Clock())
    m = Metronome(theory=SPN, ctrl=(c, e), extl=(t,))
    trace = simulate(m, t=t, horizon=3.0, seed=5, scheduler=uniform(0.3))
    toggles = [trace.times[i] for i in range(1, len(trace)) if trace.series(e)[i] != trace.series(e)[i - 1]]
    assert toggles == pytest.approx([1.0, 2.0, 3.0])
    gaps = [b - a for a, b in zip(trace.times, trace.times[1:])]
    assert all(g >= 0.3 or b in toggles for g, b in zip(gaps, trace.times[1:]))
    assert len(trace) > 4  # there were stuttering rounds


def test_a_window_is_where_the_guard_holds_up_to_the_invariant():
    # fires once the clock is at most 1, while it is non-negative: any round in [2, 3]
    class M(Module):
        def init(self, t):
            return 3.0, 0

        def next(self, c, k, t):
            return ite(1 >= c, 5.0, c), ite(1 >= c, k + 1, k)

        def flow(self, c, k, t):
            return ite(c >= 0, -1 * d(t), None), 0

    def fired_at(scheduler):
        m, c, k, t = _ticker(M)
        trace = simulate(m, t=t, horizon=4.0, seed=7, scheduler=scheduler)
        return next(time for time, s in trace if s[k] == 1)

    assert fired_at(last()) == 3.0 and fired_at(first(0.5)) == 2.0
    assert 2.0 <= fired_at(uniform(0.5)) <= 3.0


def test_a_scheduler_must_stay_inside_the_step():
    m, c, k, t = _ticker()
    with pytest.raises(ValueError, match="outside"):
        simulate(m, t=t, horizon=1.0, scheduler=lambda hi, rng: hi * 2)
    with pytest.raises(ValueError, match="positive"):
        first(0.0)


# --- events -----------------------------------------------------------------------


def test_consecutive_firings_are_consecutive_toggles():
    class Counter(Module):
        def init(self, e):
            return 0

        def next(self, n, e):
            return ite(fired(e), n + 1, n)

    c, e, n, t = Var(Clock()), Var(Event()), Var(Nat()), Var(Clock())
    system = compose(
        Metronome(theory=SPN, ctrl=(c, e), extl=(t,)),
        Counter(theory=SPN, ctrl=(n,), extl=(e,)),
        hide={c},
    )
    trace = simulate(system, t=t, horizon=3.5)
    assert trace.times == [0.0, 1.0, 2.0, 3.0, 3.5]
    assert trace.series(e) == [False, True, False, True, True]
    assert trace.series(n) == [0, 1, 2, 3, 3]


# --- birth-death ------------------------------------------------------------------


def _toggles(values):
    return [i for i in range(1, len(values)) if values[i] != values[i - 1]]


def test_birth_death_over_a_long_run():
    from zrth.examples import birth_death as bd

    horizon = 200.0
    trace = simulate(bd.system, t=bd.t, horizon=horizon, seed=1)
    n = trace.series(n_var := bd.n)
    assert all(v >= 0 for v in n)
    births, deaths = set(_toggles(trace.series(bd.bth))), set(_toggles(trace.series(bd.dth)))
    # the place decides: a birth alone adds a token, a death alone takes one if there is
    # one, both at once cancel out
    for i in range(1, len(n)):
        born, died = i in births, i in deaths
        assert n[i] == n[i - 1] + (born and not died) - (died and not born and n[i - 1] > 0)
    # births at 2 per unit time, deaths at 1: the place drifts up at about 1 per unit time
    assert 0.85 * 2 * horizon < len(births) < 1.15 * 2 * horizon
    assert 0.8 * horizon < len(deaths) < 1.2 * horizon
    assert 0.6 * horizon < n[-1] < 1.4 * horizon
    assert trace.times[-1] == horizon


def test_the_same_seed_gives_the_same_run():
    from zrth.examples import birth_death as bd

    a = simulate(bd.system, t=bd.t, horizon=20.0, seed=3)
    b = simulate(bd.system, t=bd.t, horizon=20.0, seed=3)
    c = simulate(bd.system, t=bd.t, horizon=20.0, seed=4)
    assert a.times == b.times and a.series(bd.n) == b.series(bd.n)
    assert a.times != c.times


def test_stuttering_rounds_do_not_change_what_fires():
    from zrth.examples import birth_death as bd

    trace = simulate(bd.system, t=bd.t, horizon=20.0, seed=3, scheduler=first(0.05))
    births = _toggles(trace.series(bd.bth))
    n = trace.series(bd.n)
    assert len(trace) > 400  # a round every 0.05 plus the expiries
    assert all(n[i] == n[i - 1] + 1 for i in births if i not in _toggles(trace.series(bd.dth)))
    assert all(n[i] == n[i - 1] for i in range(1, len(n)) if i not in births and i not in _toggles(trace.series(bd.dth)))
