"""Stochastic Petri nets as reactive modules, without writing the blocks.

A transition and a place are modules; you compose them like any other modules. The arcs
are variables shared between them: a transition fires its `event`, which the places it
touches take as an input or an output; a transition with input places reads their `count`s
and fires only when all hold a token, so a join is atomic:

    from zrth import Clock, Var, Module as compose
    from zrth.petri import Place, Transition
    from zrth.simulate import simulate

    t = Var(Clock())                                  # the time reference the clocks run against
    birth, death = Transition(rate=2.0, t=t), Transition(rate=1.0, t=t)
    queue = Place(initial=0, inputs=[birth.event], outputs=[death.event])
    net = compose(birth, death, queue)

    trace = simulate(net, t=t, horizon=100.0, seed=1)
    trace.series(queue.count)

A variable is created by the module that owns it unless passed in (`event=`, `count=`),
which is how a transition and a place that each need the other's variable are written:
declare the shared variables first, then the modules. A transition's clock is its own
business: it is private, reachable as `transition.clock` for plotting a trace.

    part, machine, ev = Var(Nat()), Var(Nat()), Var(Event())
    work = Transition(rate=1.0, t=t, event=ev, inputs=[part, machine])
    parts = Place(initial=3, count=part, outputs=[ev])
    machines = Place(initial=1, count=machine, outputs=[ev])

A transition owns a clock, armed with an exponential delay at its rate and run down against
the time reference while it is non-negative; whenever it expires the clock is re-armed, and
the transition fires if every input place holds a token (an expiry on an empty place is a
lost attempt; for exponential rates this is the same process as a clock that waits). A place
owns a token count and applies the firings: one token in per firing input, one out per
firing output, never below zero. Two transitions firing in one round on a place with one
token consume in declaration order, the second firing dropped there.

Only plain stochastic nets for now: every arc moves one token; read and inhibitor arcs,
weights and capacities wait for a comparison on token counts in the theory.
"""

import inspect

from .zrth import SPN, Var
from .sort import Clock, Event, Nat
from .sugar import Module, d, exp, fired, ite


def _signature(names):
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    return inspect.Signature([inspect.Parameter(n, kind) for n in names])


def _var(v, sort, what):
    if v is None:
        return Var(sort)
    if not isinstance(v, Var) or v.dtype != sort:
        raise TypeError(f"{what} is a Var({sort}), got {v!r}")
    return v


def _vars(vs, sort, what):
    vs = tuple(_var(v, sort, what) for v in vs)
    if len(set(vs)) != len(vs):
        raise ValueError(f"a variable appears twice among the {what}")
    return vs


class Transition(Module):
    """A stochastic transition. `event` is its variable, `clock` its private one, `t` the
    time reference, `inputs` the counts of the places it takes a token from."""

    def __new__(cls, rate, *, t, inputs=(), event=None, name=None):
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not rate > 0:
            raise TypeError(f"a transition's rate is a positive number, got {rate!r}")
        rate = float(rate)
        t = _var(t, Clock(), "the time reference")
        inputs = _vars(inputs, Nat(), "inputs")
        clock, event = Var(Clock()), _var(event, Event(), "event")

        def init(self, t, *counts):
            return exp(rate), False

        def next(self, clk, ev, t, *counts):
            expired = clk == 0
            fires = expired
            for n in counts:
                fires = fires & (n != 0)
            return ite(expired, exp(rate), clk), ite(fires, ~ev, ev)

        def flow(self, clk, ev, t, *counts):
            return ite(clk >= 0, -1 * d(t), None), 0

        # sugar binds the variables positionally and checks the arity, so the methods
        # declare one parameter per input place
        names = [f"n{i}" for i in range(len(inputs))]
        init.__signature__ = _signature(["self", "t", *names])
        next.__signature__ = _signature(["self", "clk", "ev", "t", *names])
        flow.__signature__ = _signature(["self", "clk", "ev", "t", *names])
        built = type(cls.__name__, (cls,), {"init": init, "next": next, "flow": flow})
        self = Module.__new__(
            built, ctrl=(clock, event), extl=(t, *inputs), hide={clock}, theory=SPN
        )
        self.rate, self.t, self.name, self.inputs = rate, t, name, inputs
        self.clock, self.event = clock, event
        return self

    def __repr__(self):
        return f"Transition({self.name or self.event.id}, rate={self.rate})"


class Place(Module):
    """A place. `count` is its variable; `inputs` and `outputs` are the events of the
    transitions that put a token in and take one out."""

    def __new__(cls, initial=0, *, inputs=(), outputs=(), count=None, name=None):
        if isinstance(initial, bool) or not isinstance(initial, int) or initial < 0:
            raise TypeError(f"a place holds a non-negative number of tokens, got {initial!r}")
        inputs, outputs = _vars(inputs, Event(), "inputs"), _vars(outputs, Event(), "outputs")
        if not inputs and not outputs:
            raise TypeError("a place needs at least one input or output")
        count = _var(count, Nat(), "count")

        def init(self, *events):
            return initial

        def next(self, n, *events):
            acc = n
            for ev in events[: len(inputs)]:
                acc = ite(fired(ev), acc + 1, acc)
            for ev in events[len(inputs) :]:
                acc = ite(fired(ev) & (acc != 0), acc - 1, acc)
            return acc

        names = [f"e{i}" for i in range(len(inputs) + len(outputs))]
        init.__signature__ = _signature(["self", *names])
        next.__signature__ = _signature(["self", "n", *names])
        built = type(cls.__name__, (cls,), {"init": init, "next": next})
        self = Module.__new__(built, ctrl=(count,), extl=inputs + outputs, theory=SPN)
        self.initial, self.name, self.count = initial, name, count
        self.inputs, self.outputs = inputs, outputs
        return self

    def __repr__(self):
        return f"Place({self.name or self.count.id}, initial={self.initial})"
