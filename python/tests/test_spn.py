"""The `spn` theory through the Python bindings.

Covers the sorts (`Nat`, `Clock`, `Zero`) and their tangents, the type checking of
individual terms, and a two-place net: a token flows from `p0` to `p1` when a
Poisson clock expires.
"""

import pytest
from zrth import SPN, Atom, Bool, Clock, Event, Module, Nat, Term, Var, Wire, X, Zero, d

# the firing rate of the transition, and the speed of its countdown clock
RATE = 2.5
COUNTDOWN = -1.0

BOOL = Bool([1, 1])  # the theory is scalar: its booleans are 1x1


# ---------------------------------------------------------------------------
# Sorts
# ---------------------------------------------------------------------------


def test_sorts():
    assert Clock() == Clock(0) != Clock(1)
    assert (str(Nat()), str(Clock()), str(Clock(1)), str(Zero())) == (
        "Nat",
        "Clock",
        "T1 Clock",
        "Zero",
    )


def test_tangents():
    # a marking changes by firing a transition, not by flowing, so its
    # derivative wire carries the trivial tangent; a clock's does not
    assert d(Var(Nat())).dtype == Zero()
    assert d(Var(BOOL)).dtype == Zero()
    assert d(Var(Clock())).dtype == Clock(1)


def test_event_is_a_sort_of_its_own():
    # a wire records which of the two names it was declared with, and that is all
    # that tells them apart
    assert str(Event()) == "Event" and Event() != BOOL
    assert d(Var(Event())).dtype == Zero()


def test_event_and_bool_are_interchangeable():
    # every operation takes the one for the other: the alias is transparent
    ev, b, c, p = Var(Event()), Var(BOOL), Var(Clock()), Var(Nat())

    Term.constant(SPN.Bool(True), [X(ev)])       # the same literal op writes both
    Term(SPN.ClkIsZero(), [X(ev)], [c])          # a zero test writes an event
    Term(SPN.IsZero(), [Wire(Event())], [p])
    Term(SPN.And(), [Wire(Event())], [ev, b])    # and they mix in one term
    Term(SPN.Not(), [Wire(BOOL)], [ev])
    Term(SPN.Id(), [X(b)], [ev])                 # each flows into the other


def test_event_is_not_a_place():
    # transparent to the boolean fragment, and to nothing else
    ev, p = Var(Event()), Var(Nat())
    with pytest.raises(Exception, match="must be Nat"):
        Term(SPN.Inc(), [X(p)], [ev])


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------


def test_terms_typecheck():
    p = Var(Nat())
    c = Var(Clock())

    Term(SPN.Inc(), [X(p)], [p])
    Term(SPN.Dec(), [X(p)], [p])
    Term(SPN.IsZero(), [Wire(BOOL)], [p])
    Term(SPN.ClkIsZero(), [Wire(BOOL)], [c])
    Term.constant(SPN.Nat(3), [X(p)])
    Term.constant(SPN.Exp(RATE), [X(c)])
    # the flows write derivative wires
    Term.constant(SPN.ClkRate(COUNTDOWN), [d(c)])
    Term.constant(SPN.ClkZero(), [d(c)])
    Term.constant(SPN.Zero(), [d(p)])
    # a clock's rate relative to another clock's tangent
    Term(SPN.ClkMul(COUNTDOWN), [Wire(Clock(1))], [d(c)])


@pytest.mark.parametrize(
    "build, message",
    [
        # a place is not a clock and the two tests do not swap
        (lambda p, c: Term(SPN.Inc(), [X(c)], [c]), "must be Nat"),
        (lambda p, c: Term(SPN.ClkIsZero(), [Wire(BOOL)], [p]), "must be Clock"),
        (lambda p, c: Term(SPN.IsZero(), [Wire(BOOL)], [c]), "must be Nat"),
        # a literal writes a value, a flow writes a rate, and never the other way
        (lambda p, c: Term.constant(SPN.Nat(1), [X(c)]), "cannot be written"),
        (lambda p, c: Term.constant(SPN.Clock(1.0), [d(c)]), "clock tangent"),
        (lambda p, c: Term.constant(SPN.ClkRate(COUNTDOWN), [X(c)]), "clock tangent"),
        (lambda p, c: Term.constant(SPN.Zero(), [X(p)]), "ZERO"),
        # `Exp` arms a clock, nothing else
        (lambda p, c: Term.constant(SPN.Exp(RATE), [X(p)]), "must be Clock"),
        # `ClkMul` scales a tangent, not a clock value
        (lambda p, c: Term(SPN.ClkMul(COUNTDOWN), [Wire(Clock(1))], [c]), "clock tangent"),
    ],
)
def test_ill_typed_terms_raise(build, message):
    p, c = Var(Nat()), Var(Clock())
    with pytest.raises(Exception, match=message):
        build(p, c)


# ---------------------------------------------------------------------------
# A two-place net
# ---------------------------------------------------------------------------


def two_place_net():
    """`p0 --[t]--> p1`: one token, one transition, one Poisson clock.

    The transition is enabled while `p0` holds a token; its clock counts down
    while it is, and freezes while it is not. The transition fires when the
    clock expires: the token moves and the clock is re-armed.
    """
    p0, p1, c = Var(Nat()), Var(Nat()), Var(Clock())

    # one token in the left place, none in the right, and a fresh clock
    init = [
        Term.constant(SPN.Nat(1), [X(p0)]),
        Term.constant(SPN.Nat(0), [X(p1)]),
        Term.constant(SPN.Exp(RATE), [X(c)]),
    ]

    # the discrete step: move a token when the clock has expired
    empty, enabled, expired, fire = (Wire(BOOL) for _ in range(4))
    taken, put, fresh = Wire(Nat()), Wire(Nat()), Wire(Clock())
    update = [
        Term(SPN.IsZero(), [empty], [p0]),
        Term(SPN.Not(), [enabled], [empty]),
        Term(SPN.ClkIsZero(), [expired], [c]),
        Term(SPN.And(), [fire], [enabled, expired]),
        Term(SPN.Dec(), [taken], [p0]),
        Term(SPN.Inc(), [put], [p1]),
        Term.constant(SPN.Exp(RATE), [fresh]),
        Term(SPN.Ite(), [X(p0)], [fire, taken, p0]),
        Term(SPN.Ite(), [X(p1)], [fire, put, p1]),
        Term(SPN.Ite(), [X(c)], [fire, fresh, c]),
    ]

    # the continuous step: the clock runs down while the transition is
    # enabled and stands still while it is not (preemptive resume); the
    # markings do not flow at all
    empty_d, enabled_d = Wire(BOOL), Wire(BOOL)
    run, stop = Wire(Clock(1)), Wire(Clock(1))
    delay = [
        Term(SPN.IsZero(), [empty_d], [p0]),
        Term(SPN.Not(), [enabled_d], [empty_d]),
        Term.constant(SPN.ClkRate(COUNTDOWN), [run]),
        Term.constant(SPN.ClkZero(), [stop]),
        Term(SPN.Ite(), [d(c)], [enabled_d, run, stop]),
        Term.constant(SPN.Zero(), [d(p0)]),
        Term.constant(SPN.Zero(), [d(p1)]),
    ]

    return (p0, p1, c), init, update, delay


def test_two_place_net_atom():
    (p0, p1, c), init, update, delay = two_place_net()

    atom = Atom.hybrid([p0, p1, c], init, update, delay)

    # the atom controls all three variables and awaits none
    assert list(atom.ctrl) == [p0, p1, c]
    assert list(atom.wait) == []
    # it reads the places and the clock it controls (its own latched values)
    assert set(atom.read) == {p0, p1, c}


def test_two_place_net_module():
    (p0, p1, c), init, update, delay = two_place_net()

    net = Module.hybrid([p0, p1, c], init, update, delay)

    assert net.closed()
    assert set(net.ctrl) == {p0, p1, c}
    shown = net.with_varnames({p0: "p0", p1: "p1", c: "c"})
    assert "Exp(2.5)" in shown and "ClkRate(-1)" in shown


def test_net_rejects_a_token_moving_into_a_clock():
    (p0, p1, c), init, update, delay = two_place_net()

    # `Inc` produces a token, so its result cannot stand in for a clock
    with pytest.raises(Exception, match="must be Nat"):
        Term(SPN.Inc(), [X(c)], [c])

    # and a place cannot be given the clock's flow
    with pytest.raises(Exception, match="clock tangent"):
        Term.constant(SPN.ClkRate(COUNTDOWN), [d(p0)])


# ---------------------------------------------------------------------------
# Two transitions racing for one token
# ---------------------------------------------------------------------------


def race_net():
    """`p1 <--[t1]-- p0 --[t2]--> p2`: two transitions, a clock each.

    Both are enabled while `p0` holds a token and both clocks count down; the
    one whose clock expires first takes it. A tie (which the continuous
    semantics makes a measure-zero event) is broken by giving `t1` priority,
    so the token is never consumed twice.

    The net assumes the executor's convention: time advances to the first
    expiry -- the minimum over the running clocks -- and the discrete step is
    taken there. The theory states no minimum, so nothing here enforces it.
    """
    p0, p1, p2 = Var(Nat()), Var(Nat()), Var(Nat())
    c1, c2 = Var(Clock()), Var(Clock())
    rate1, rate2 = RATE, 1.0

    init = [
        Term.constant(SPN.Nat(1), [X(p0)]),
        Term.constant(SPN.Nat(0), [X(p1)]),
        Term.constant(SPN.Nat(0), [X(p2)]),
        Term.constant(SPN.Exp(rate1), [X(c1)]),
        Term.constant(SPN.Exp(rate2), [X(c2)]),
    ]

    empty, enabled = Wire(BOOL), Wire(BOOL)
    exp1, exp2, fire1, fire2, ready2, alone2, any_fire = (Wire(BOOL) for _ in range(7))
    taken, put1, put2 = Wire(Nat()), Wire(Nat()), Wire(Nat())
    fresh1, fresh2 = Wire(Clock()), Wire(Clock())
    update = [
        Term(SPN.IsZero(), [empty], [p0]),
        Term(SPN.Not(), [enabled], [empty]),
        Term(SPN.ClkIsZero(), [exp1], [c1]),
        Term(SPN.ClkIsZero(), [exp2], [c2]),
        Term(SPN.And(), [fire1], [enabled, exp1]),
        # t1 wins a tie, so t2 fires only when t1 does not
        Term(SPN.And(), [ready2], [enabled, exp2]),
        Term(SPN.Not(), [alone2], [fire1]),
        Term(SPN.And(), [fire2], [ready2, alone2]),
        Term(SPN.Or(), [any_fire], [fire1, fire2]),
        Term(SPN.Dec(), [taken], [p0]),
        Term(SPN.Inc(), [put1], [p1]),
        Term(SPN.Inc(), [put2], [p2]),
        Term.constant(SPN.Exp(rate1), [fresh1]),
        Term.constant(SPN.Exp(rate2), [fresh2]),
        Term(SPN.Ite(), [X(p0)], [any_fire, taken, p0]),
        Term(SPN.Ite(), [X(p1)], [fire1, put1, p1]),
        Term(SPN.Ite(), [X(p2)], [fire2, put2, p2]),
        # race with resampling: a firing re-arms both clocks
        Term(SPN.Ite(), [X(c1)], [any_fire, fresh1, c1]),
        Term(SPN.Ite(), [X(c2)], [any_fire, fresh2, c2]),
    ]

    empty_d, enabled_d = Wire(BOOL), Wire(BOOL)
    run1, stop1, run2, stop2 = (Wire(Clock(1)) for _ in range(4))
    delay = [
        Term(SPN.IsZero(), [empty_d], [p0]),
        Term(SPN.Not(), [enabled_d], [empty_d]),
        Term.constant(SPN.ClkRate(COUNTDOWN), [run1]),
        Term.constant(SPN.ClkZero(), [stop1]),
        Term.constant(SPN.ClkRate(COUNTDOWN), [run2]),
        Term.constant(SPN.ClkZero(), [stop2]),
        Term(SPN.Ite(), [d(c1)], [enabled_d, run1, stop1]),
        Term(SPN.Ite(), [d(c2)], [enabled_d, run2, stop2]),
        Term.constant(SPN.Zero(), [d(p0)]),
        Term.constant(SPN.Zero(), [d(p1)]),
        Term.constant(SPN.Zero(), [d(p2)]),
    ]

    return (p0, p1, p2, c1, c2), init, update, delay


def race_net_m() -> Module:
    variables, init, update, delay = race_net()

    return Module.hybrid(list(variables), init, update, delay)


def test_two_transitions_one_atom():
    variables, init, update, delay = race_net()

    net = Module.hybrid(list(variables), init, update, delay)

    assert net.closed()
    assert set(net.ctrl) == set(variables)


def test_two_transitions_clocks_are_independent():
    (_, _, _, c1, c2), _, _, delay = race_net()

    # each clock has its own derivative wire, written exactly once
    written = [w for term in delay for w in term.write]
    assert d(c1) in written and d(c2) in written
    assert len(written) == len(set(written))


# The modular encoding: an atom per *variable*, not per transition. A
# transition atom owns its clock and publishes a "fires now" boolean; the
# places await those signals. Splitting the other way -- an atom per
# transition -- is what reactive modules forbid, since two transitions
# sharing an input place would both write it.


def transition_atom(clk, fire, src, rate, veto=None):
    """Owns `clk` and the firing signal `fire`; reads the input place `src`."""
    empty, enabled, expired, fires = (Wire(BOOL) for _ in range(4))
    fresh = Wire(Clock())
    update = [
        Term(SPN.IsZero(), [empty], [src]),
        Term(SPN.Not(), [enabled], [empty]),
        Term(SPN.ClkIsZero(), [expired], [clk]),
        Term(SPN.And(), [fires], [enabled, expired]),
    ]
    if veto is not None:
        # loses a tie: awaits the other transition's signal for this round
        stands, alone = Wire(BOOL), Wire(BOOL)
        update += [
            Term(SPN.Not(), [alone], [X(veto)]),
            Term(SPN.And(), [stands], [fires, alone]),
        ]
        fires = stands
    update += [
        Term(SPN.Id(), [X(fire)], [fires]),
        Term.constant(SPN.Exp(rate), [fresh]),
        Term(SPN.Ite(), [X(clk)], [fires, fresh, clk]),
    ]

    empty_d, enabled_d = Wire(BOOL), Wire(BOOL)
    run, stop = Wire(Clock(1)), Wire(Clock(1))
    delay = [
        Term(SPN.IsZero(), [empty_d], [src]),
        Term(SPN.Not(), [enabled_d], [empty_d]),
        Term.constant(SPN.ClkRate(COUNTDOWN), [run]),
        Term.constant(SPN.ClkZero(), [stop]),
        Term(SPN.Ite(), [d(clk)], [enabled_d, run, stop]),
        Term.constant(SPN.Zero(), [d(fire)]),
    ]
    init = [
        Term.constant(SPN.Exp(rate), [X(clk)]),
        Term.constant(SPN.Bool(False), [X(fire)]),
    ]
    vars_ = [clk, fire, src] + ([veto] if veto is not None else [])
    return Atom.hybrid(vars_, init, update, delay)


def place_atom(place, tokens, consumed=(), produced=()):
    """Owns one place; awaits the firing signals of the arcs touching it."""
    current, update = place, []
    for op, fires in [(SPN.Dec(), consumed), (SPN.Inc(), produced)]:
        for fire in fires:
            stepped, moved = Wire(Nat()), Wire(Nat())
            update += [
                Term(op, [moved], [current]),
                Term(SPN.Ite(), [stepped], [X(fire), moved, current]),
            ]
            current = stepped
    update.append(Term(SPN.Id(), [X(place)], [current]))

    init = [Term.constant(SPN.Nat(tokens), [X(place)])]
    delay = [Term.constant(SPN.Zero(), [d(place)])]
    return Atom.hybrid([place, *consumed, *produced], init, update, delay)


def test_two_transitions_modular():
    p0, p1, p2 = Var(Nat()), Var(Nat()), Var(Nat())
    c1, c2 = Var(Clock()), Var(Clock())
    f1, f2 = Var(BOOL), Var(BOOL)

    net = Module(
        transition_atom(c1, f1, p0, RATE),
        transition_atom(c2, f2, p0, 1.0, veto=f1),
        place_atom(p0, 1, consumed=(f1, f2)),
        place_atom(p1, 0, produced=(f1,)),
        place_atom(p2, 0, produced=(f2,)),
    )

    assert net.closed()
    assert set(net.ctrl) == {p0, p1, p2, c1, c2, f1, f2}


def test_transitions_cannot_share_a_place():
    # the encoding that does *not* work: an atom per transition, both of them
    # writing the input place they share
    p0, p1, p2 = Var(Nat()), Var(Nat()), Var(Nat())
    c1, c2 = Var(Clock()), Var(Clock())

    def consumer(clk, dst, rate):
        expired, fires = Wire(BOOL), Wire(BOOL)
        taken, put, fresh = Wire(Nat()), Wire(Nat()), Wire(Clock())
        empty, enabled = Wire(BOOL), Wire(BOOL)
        return Atom.hybrid(
            [p0, dst, clk],
            [
                Term.constant(SPN.Nat(1), [X(p0)]),
                Term.constant(SPN.Nat(0), [X(dst)]),
                Term.constant(SPN.Exp(rate), [X(clk)]),
            ],
            [
                Term(SPN.IsZero(), [empty], [p0]),
                Term(SPN.Not(), [enabled], [empty]),
                Term(SPN.ClkIsZero(), [expired], [clk]),
                Term(SPN.And(), [fires], [enabled, expired]),
                Term(SPN.Dec(), [taken], [p0]),
                Term(SPN.Inc(), [put], [dst]),
                Term.constant(SPN.Exp(rate), [fresh]),
                Term(SPN.Ite(), [X(p0)], [fires, taken, p0]),
                Term(SPN.Ite(), [X(dst)], [fires, put, dst]),
                Term(SPN.Ite(), [X(clk)], [fires, fresh, clk]),
            ],
            [
                Term.constant(SPN.ClkZero(), [d(clk)]),
                Term.constant(SPN.Zero(), [d(p0)]),
                Term.constant(SPN.Zero(), [d(dst)]),
            ],
        )

    with pytest.raises(Exception, match="doubly controlled"):
        Module(consumer(c1, p1, RATE), consumer(c2, p2, 1.0))
