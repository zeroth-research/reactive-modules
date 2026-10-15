"""Stochastic Petri nets through `zrth.petri`: transitions and places are modules composed
like any other, the birth-death net equals the hand-written example, and places net their
arcs."""

import pytest

from zrth import SPN, Clock, Event, Nat, Var
from zrth import Module as compose
from zrth.petri import Place, Transition
from zrth.simulate import simulate
from zrth.sugar import Module, d, ite


def birth_death(lam=2.0, mu=1.0):
    t = Var(Clock())
    birth, death = Transition(rate=lam, t=t, name="b"), Transition(rate=mu, t=t, name="d")
    queue = Place(initial=0, inputs=[birth.event], outputs=[death.event], name="n")
    net = compose(birth, death, queue, hide={birth.clock, death.clock})
    return net, t, queue, birth, death


def _ops(block):
    return sorted(type(t.itype).__name__.removeprefix("SPN_") for t in block)


def test_birth_death_is_the_example_net():
    """The same net as `examples/birth_death.py`: the transitions op for op, the place by
    its behaviour (it folds its arcs; the example spells the two cases out)."""
    from zrth.examples import birth_death as bd

    net, t, queue, birth, death = birth_death()
    for ours, theirs in zip(list(net.atoms)[:2], list(bd.system.atoms)[:2]):
        for block in ("init", "update", "delay"):
            assert _ops(getattr(ours, block)) == _ops(getattr(theirs, block))
    assert net.open() and set(net.extl) == {t}
    assert set(net.prvt) == {birth.clock, death.clock}
    assert set(net.intf) == {birth.event, death.event, queue.count}


def test_a_transition_and_a_place_are_modules():
    t = Var(Clock())
    tr = Transition(rate=1.5, t=t)
    assert isinstance(tr, Transition) and tr.rate == 1.5 and tr.t is t
    assert set(tr.ctrl) == {tr.clock, tr.event} and set(tr.extl) == {t}
    pl = Place(initial=3, inputs=[tr.event])
    assert isinstance(pl, Place) and pl.initial == 3 and set(pl.ctrl) == {pl.count}
    assert set(pl.extl) == {tr.event} and pl.inputs == (tr.event,) and pl.outputs == ()
    assert "Transition(" in repr(tr) and "Place(" in repr(pl)


def test_a_seeded_run_matches_the_example():
    from zrth.examples import birth_death as bd

    net, t, queue, *_ = birth_death()
    ours = simulate(net, t=t, horizon=50.0, seed=3)
    theirs = simulate(bd.system, t=bd.t, horizon=50.0, seed=3)
    assert ours.times == theirs.times and ours.series(queue.count) == theirs.series(bd.n)
    assert ours.times[-1] == 50.0 and ours.series(queue.count)[-1] > 0


def test_a_place_nets_several_arcs():
    # two producers into one place, one consumer out of it; nothing hidden unless asked
    t = Var(Clock())
    a, b, c = (Transition(rate=1.0, t=t) for _ in range(3))
    p = Place(inputs=[a.event, b.event], outputs=[c.event])
    net = compose(a, b, c, p)
    assert set(net.prvt) == set()
    trace = simulate(net, t=t, horizon=200.0, seed=5)
    n = trace.series(p.count)
    assert all(v >= 0 for v in n)
    for i in range(1, len(trace)):
        fired = lambda tr: trace.states[i][tr.event] != trace.before[i][tr.event]
        expected = n[i - 1] + fired(a) + fired(b)
        expected -= 1 if fired(c) and expected > 0 else 0
        assert n[i] == expected


def test_a_place_composes_with_any_module():
    # a place awaiting a hand-written transition's event: the arc is just the event
    class Tick(Module):
        def init(self, t):
            return 1.0, False

        def next(self, clk, ev, t):
            fires = clk == 0
            return ite(fires, 1.0, clk), ite(fires, ~ev, ev)

        def flow(self, clk, ev, t):
            return ite(clk >= 0, -1 * d(t), None), 0

    t, clk, ev = Var(Clock()), Var(Clock()), Var(Event())
    tick = Tick(theory=SPN, ctrl=(clk, ev), extl=(t,))
    p = Place(inputs=[ev])
    trace = simulate(compose(tick, p), t=t, horizon=3.5)
    assert trace.series(p.count) == [0, 1, 2, 3, 3]


def test_a_join_fires_only_when_every_input_holds_a_token():
    # parts and one machine feed `work`; the shared variables are declared first
    t = Var(Clock())
    part, machine, ev = Var(Nat()), Var(Nat()), Var(Event())
    work = Transition(rate=1.0, t=t, event=ev, inputs=[part, machine])
    assert work.event is ev and set(work.extl) == {t, part, machine}
    parts = Place(initial=3, count=part, outputs=[ev])
    machines = Place(initial=1, count=machine, outputs=[ev])
    done = Place(inputs=[ev])
    assert parts.count is part
    trace = simulate(compose(work, parts, machines, done), t=t, horizon=50.0, seed=2)
    # the machine is consumed by the first firing, then every expiry is a lost attempt
    assert trace.series(done.count)[-1] == 1
    assert trace.series(part)[-1] == 2 and trace.series(machine)[-1] == 0
    assert len(trace) > 3  # the clock kept re-arming after the machine was gone


def test_a_lost_attempt_re_arms_the_clock():
    t = Var(Clock())
    empty = Var(Nat())
    tr = Transition(rate=1.0, t=t, inputs=[empty])
    src = Place(initial=0, count=empty, outputs=[tr.event])
    trace = simulate(compose(tr, src), t=t, horizon=20.0, seed=1)
    assert len(trace) > 5 and all(not v for v in trace.series(tr.event))
    assert all(s[tr.clock] > 0 for s in trace.states)  # re-armed at every expiry


def test_bad_arguments_raise():
    t = Var(Clock())
    with pytest.raises(TypeError, match="rate"):
        Transition(rate=0, t=t)
    with pytest.raises(TypeError, match="time reference"):
        Transition(rate=1.0, t=Var(Clock(1)))
    with pytest.raises(TypeError, match="inputs"):
        Transition(rate=1.0, t=t, inputs=[Var(Event())])
    with pytest.raises(TypeError, match="event"):
        Transition(rate=1.0, t=t, event=Var(Nat()))
    with pytest.raises(TypeError, match="tokens"):
        Place(initial=-1, inputs=[Var(Event())])
    with pytest.raises(TypeError, match="inputs"):
        Place(inputs=[Var(Nat())])
    with pytest.raises(TypeError, match="count"):
        Place(inputs=[Var(Event())], count=Var(Clock()))
    with pytest.raises(TypeError, match="at least one"):
        Place()
    ev = Var(Event())
    with pytest.raises(ValueError, match="twice"):
        Place(inputs=[ev, ev])
