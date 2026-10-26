"""Stochastic Petri nets through `zrth.petri`: arcs carry the firing and the enabling,
transitions and places are modules composed like any other, the birth-death net equals the
hand-written example, and places net their arcs."""

import pytest

from zrth import SPN, Clock, Event, Nat, Var
from zrth import Module as compose
from zrth.petri import TIME, Arc, Place, Transition
from zrth.simulate import simulate
from zrth.sugar import Module, d, ite


def birth_death(lam=2.0, mu=1.0, t=TIME):
    bth, dth = Arc("production"), Arc("consumption")
    birth = Transition(rate=lam, t=t, outputs=[bth], name="b")
    death = Transition(rate=mu, t=t, inputs=[dth], name="d")
    queue = Place(initial=0, inputs=[bth], outputs=[dth], name="n")
    return compose(birth, death, queue), queue, birth, death, bth, dth


def _ops(block):
    return sorted(type(t.itype).__name__.removeprefix("SPN_") for t in block)


def _fired(trace, i, arc):
    return trace.states[i][arc.fired] != trace.before[i][arc.fired]


def test_birth_death_is_the_example_net():
    """The example's net, with the enabling of the death as an explicit Bool: the birth
    transition op for op, and the same seeded run."""
    from zrth.examples import birth_death as bd

    net, queue, birth, death, bth, dth = birth_death()
    for block in ("init", "update", "delay"):
        assert _ops(getattr(birth.atoms[0], block)) == _ops(getattr(bd.birth.atoms[0], block))
    assert net.open() and set(net.extl) == {TIME}
    assert set(net.prvt) == {birth.clock, death.clock}
    assert set(net.intf) == {bth.fired, dth.fired, dth.enabled, queue.count}

    ours = simulate(net, horizon=50.0, seed=3)
    theirs = simulate(bd.system, t=bd.t, horizon=50.0, seed=3)
    assert ours.times == theirs.times and ours.series(queue.count) == theirs.series(bd.n)


def test_arcs_carry_the_firing_and_the_enabling():
    prod, cons = Arc("production"), Arc("consumption")
    assert prod.to_place and prod.enabled is None and prod.fired.dtype == Event()
    assert not cons.to_place and cons.consumes and cons.enabled is not None
    assert cons.multiplicity == 1 and Arc("enabling", 3).multiplicity == 3
    assert not Arc("enabling").consumes and not Arc("inhibitor").to_place
    with pytest.raises(TypeError, match="kind"):
        Arc("teleport")
    with pytest.raises(TypeError, match="multiplicity"):
        Arc("consumption", multiplicity=0)


def test_multiplicity_moves_that_many_tokens_atomically():
    # pairs come in, triples go out while there are at least three
    two, three = Arc("production", 2), Arc("consumption", 3)
    maker, taker = Transition(rate=1.0, outputs=[two]), Transition(rate=1.0, inputs=[three])
    p = Place(initial=1, inputs=[two], outputs=[three])
    trace = simulate(compose(maker, taker, p), horizon=100.0, seed=6)
    n = trace.series(p.count)
    assert all(v >= 0 for v in n) and n[0] == 1 and not trace.states[0][three.enabled]
    for i in range(1, len(trace)):
        expected = n[i - 1] + 2 * _fired(trace, i, two)
        if _fired(trace, i, three):
            assert trace.before[i][three.enabled] and n[i - 1] >= 3
            expected -= 3
        assert n[i] == expected and trace.states[i][three.enabled] == (n[i] >= 3)


def test_an_enabling_arc_needs_the_tokens_and_leaves_them():
    key, done = Arc("enabling"), Arc("production")
    work = Transition(rate=2.0, inputs=[key], outputs=[done])
    keys, finished = Place(initial=1, outputs=[key]), Place(inputs=[done])
    trace = simulate(compose(work, keys, finished), horizon=20.0, seed=1)
    assert all(v == 1 for v in trace.series(keys.count))
    assert trace.series(finished.count)[-1] > 10
    assert set(keys.extl) == set()  # nothing moves its tokens: it reads no firing


def test_an_inhibitor_arc_enables_while_there_are_fewer_tokens():
    stop, fill = Arc("inhibitor", 2), Arc("production")
    filler = Transition(rate=5.0, inputs=[stop], outputs=[fill])
    tank = Place(initial=0, inputs=[fill], outputs=[stop])
    trace = simulate(compose(filler, tank), horizon=20.0, seed=1)
    n = trace.series(tank.count)
    assert n[-1] == 2 and max(n) == 2 and n.count(2) > 5  # fills to two and stops there
    assert all(s[stop.enabled] == (s[tank.count] < 2) for s in trace.states)


def test_a_transition_and_a_place_are_modules():
    t = Var(Clock())
    a = Arc("production")
    tr = Transition(rate=1.5, t=t, outputs=[a])
    assert isinstance(tr, Transition) and tr.rate == 1.5 and tr.t is t
    assert set(tr.ctrl) == {tr.clock, a.fired} and set(tr.extl) == {t}
    assert list(tr.prvt) == [tr.clock] and list(tr.intf) == [a.fired]
    pl = Place(initial=3, inputs=[a])
    assert isinstance(pl, Place) and pl.initial == 3 and set(pl.ctrl) == {pl.count}
    assert set(pl.extl) == {a.fired} and pl.inputs == (a,) and pl.outputs == ()
    assert "Transition(" in repr(tr) and "Place(" in repr(pl) and "Arc(" in repr(a)


def test_the_time_reference_defaults_to_TIME():
    net, queue, birth, *_ = birth_death()
    assert birth.t is TIME and list(net.extl) == [TIME]
    other = Var(Clock())
    net2, *_ = birth_death(t=other)
    assert list(net2.extl) == [other]
    assert simulate(net2, horizon=5.0, seed=1).times == simulate(net, horizon=5.0, seed=1).times


def test_enabled_follows_the_count_in_the_same_round():
    net, queue, birth, death, bth, dth = birth_death()
    trace = simulate(net, horizon=30.0, seed=4)
    for s in trace.states:
        assert s[dth.enabled] == (s[queue.count] != 0)
    deaths = [i for i in range(1, len(trace)) if _fired(trace, i, dth)]
    assert deaths and all(trace.before[i][dth.enabled] for i in deaths)


def test_a_place_nets_several_arcs():
    # two producers into one place, one consumer out of it
    ins, out = (Arc("production"), Arc("production")), Arc("consumption")
    a, b = (Transition(rate=1.0, outputs=[arc]) for arc in ins)
    c = Transition(rate=1.0, inputs=[out])
    p = Place(inputs=ins, outputs=[out])
    net = compose(a, b, c, p)
    assert set(net.prvt) == {a.clock, b.clock, c.clock}
    trace = simulate(net, horizon=200.0, seed=5)
    n = trace.series(p.count)
    assert all(v >= 0 for v in n)
    for i in range(1, len(trace)):
        expected = n[i - 1] + _fired(trace, i, ins[0]) + _fired(trace, i, ins[1])
        expected -= 1 if _fired(trace, i, out) and expected > 0 else 0
        assert n[i] == expected


def test_a_join_fires_only_when_every_input_is_enabled():
    part, machine, done = Arc("consumption"), Arc("consumption"), Arc("production")
    work = Transition(rate=1.0, inputs=[part, machine], outputs=[done])
    assert set(work.extl) == {TIME, part.enabled, machine.enabled}
    parts = Place(initial=3, outputs=[part])
    machines = Place(initial=1, outputs=[machine])
    finished = Place(inputs=[done])
    trace = simulate(compose(work, parts, machines, finished), horizon=50.0, seed=2)
    # the machine is consumed by the first firing, then every expiry is a lost attempt
    assert trace.series(finished.count)[-1] == 1
    assert trace.series(parts.count)[-1] == 2 and trace.series(machines.count)[-1] == 0
    assert len(trace) > 3
    assert all(s[work.clock] > 0 for s in trace.states)  # re-armed at every expiry


def test_a_place_composes_with_any_module():
    # a place awaiting a hand-written transition's event through an arc
    class Tick(Module):
        def init(self, t):
            return 1.0, False

        def next(self, clk, ev, t):
            fires = clk == 0
            return ite(fires, 1.0, clk), ite(fires, ~ev, ev)

        def flow(self, clk, ev, t):
            return ite(clk >= 0, -1 * d(t), None), 0

    arc = Arc("production")
    clk = Var(Clock())
    tick = Tick(theory=SPN, ctrl=(clk, arc.fired), extl=(TIME,))
    p = Place(inputs=[arc])
    trace = simulate(compose(tick, p), horizon=3.5)
    assert trace.series(p.count) == [0, 1, 2, 3, 3]


def test_a_delay_variable_arms_the_clock():
    # a deterministic timer: the delay is a constant, the firings land exactly
    class Fixed(Module):
        def init(self):
            return 0.5

        def next(self, delay):
            return 0.5

    delay = Var(Clock())
    tick, out = Fixed(theory=SPN, ctrl=(delay,)), Arc("production")
    tr = Transition(rate=delay, outputs=[out])
    assert tr.rate is delay and delay in tr.extl
    p = Place(inputs=[out])
    trace = simulate(compose(tick, tr, p), horizon=2.0)
    assert trace.times == [0.0, 0.5, 1.0, 1.5, 2.0] and trace.series(p.count) == [0, 1, 2, 3, 4]


def test_a_delay_chosen_by_the_marking():
    # births at 2 while the place is empty, at 0.5 once it holds a token
    from zrth.sugar import exp

    class Dependent(Module):
        def init(self, n):
            return exp(2.0)

        def next(self, delay, n):
            return ite(n == 0, exp(2.0), exp(0.5))

    delay, bth = Var(Clock()), Arc("production")
    birth = Transition(rate=delay, outputs=[bth])
    place = Place(inputs=[bth])
    speed = Dependent(theory=SPN, ctrl=(delay,), extl=(place.count,))
    trace = simulate(compose(speed, birth, place), horizon=200.0, seed=1)
    gaps = [b - a for a, b in zip(trace.times, trace.times[1:])][1:-1]
    assert len(gaps) > 50 and 1.0 < sum(gaps) / len(gaps) < 3.0  # about 2, as 1/0.5


def test_bad_arguments_raise():
    prod, cons = Arc("production"), Arc("consumption")
    with pytest.raises(TypeError, match="rate"):
        Transition(rate=0, outputs=[prod])
    with pytest.raises(TypeError, match="delay variable"):
        Transition(rate=Var(Nat()), outputs=[prod])
    with pytest.raises(TypeError, match="time reference"):
        Transition(rate=1.0, t=Var(Clock(1)), outputs=[prod])
    with pytest.raises(TypeError, match="at least one arc"):
        Transition(rate=1.0)
    with pytest.raises(TypeError, match="cannot be a production"):
        Transition(rate=1.0, inputs=[prod])
    with pytest.raises(TypeError, match="cannot be a consumption"):
        Place(inputs=[cons])
    with pytest.raises(TypeError, match="are arcs"):
        Place(inputs=[Var(Event())])
    with pytest.raises(TypeError, match="tokens"):
        Place(initial=-1, inputs=[prod])
    with pytest.raises(TypeError, match="at least one"):
        Place()
    with pytest.raises(ValueError, match="twice"):
        Place(inputs=[prod, prod])
    with pytest.raises(TypeError):
        Place(inputs=[prod], count=Var(Nat()))
