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
    assert not cons.to_place and cons.enabled is not None and cons.multiplicity == 1
    with pytest.raises(TypeError, match="kind"):
        Arc("teleport")
    for kind in ("enabling", "inhibitor"):
        with pytest.raises(NotImplementedError, match="token counts"):
            Arc(kind)
    with pytest.raises(NotImplementedError, match="multiplicity"):
        Arc("consumption", multiplicity=2)


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


def test_bad_arguments_raise():
    prod, cons = Arc("production"), Arc("consumption")
    with pytest.raises(TypeError, match="rate"):
        Transition(rate=0, outputs=[prod])
    with pytest.raises(NotImplementedError, match="variable"):
        Transition(rate=Var(Clock()), outputs=[prod])
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
