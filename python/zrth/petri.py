"""Stochastic Petri nets as reactive modules, without writing the blocks.

A transition and a place are modules; you compose them like any other modules. An arc is
the variables between a transition and a place: the event the transition toggles when it
fires, and for an arc from a place to a transition the Bool the place keeps equal to "this
arc lets it fire". A production arc goes from a transition into a place (the place's input,
the transition's output) and adds `multiplicity` tokens per firing. The other kinds go from
a place to a transition (the place's output, the transition's input): a consumption arc
needs `multiplicity` tokens and takes them, an enabling arc needs them and leaves them, an
inhibitor arc needs fewer than `multiplicity` and leaves them.

    from zrth import Module as compose
    from zrth.petri import Arc, Place, Transition
    from zrth.simulate import simulate

    bth, dth = Arc("production"), Arc("consumption")
    birth, death = Transition(rate=2.0, outputs=[bth]), Transition(rate=1.0, inputs=[dth])
    queue = Place(initial=0, inputs=[bth], outputs=[dth])
    net = compose(birth, death, queue)

    trace = simulate(net, horizon=100.0, seed=1)
    trace.series(queue.count)

A transition owns a clock, armed with an exponential delay at its rate and run down against
the time reference `TIME` (`t=` for another) while it is non-negative; whenever it expires
the clock is re-armed, and the transition fires if every input arc is enabled (an expiry
otherwise is a lost attempt; for exponential rates this is the same process as a clock that
waits). The rate may instead be a `Var(Clock())` holding the time to the next expiry, driven
by a module you compose: a constant for a deterministic delay, `exp(...)` chosen by the
marking for a rate that depends on it; the clock is armed with its value of the round. The
clock is private, reachable as `transition.clock` to plot a trace. A place owns its token
count: the tokens of each firing input come in, those of each firing consumption go out
while there are enough, and it keeps each output arc's `enabled` current in the same round
the count changes. Two transitions firing in one round and competing for the tokens are
served in the order of `outputs`, a firing the count cannot serve dropped there.

A transition reads `enabled` on its input arcs without caring who writes it, so an enabling
arc with no place behind it is a control: whatever drives its `enabled`, a composed
controller module or the inputs of a `Simulation` step, decides whether the transition may
fire. That is how a controllable transition is written; one without such an arc fires
whenever it can.
"""

from .zrth import SPN, Var
from .sort import Bool, Clock, Event, Nat
from .sugar import Module, X, d, exp, fired, ite

TIME = Var(Clock())
"""The time reference every clock runs against unless told otherwise: a net's one external."""

KINDS = ("production", "consumption", "enabling", "inhibitor")


class Arc:
    """An arc of the net. `fired` is the event its transition toggles when it fires; on an
    arc from a place to a transition `enabled` is the Bool the place writes."""

    def __init__(self, kind, multiplicity=1):
        if kind not in KINDS:
            raise TypeError(f"an arc's kind is one of {KINDS}, got {kind!r}")
        if isinstance(multiplicity, bool) or not isinstance(multiplicity, int) or multiplicity < 1:
            raise TypeError(f"an arc's multiplicity is a positive integer, got {multiplicity!r}")
        self.kind, self.multiplicity = kind, multiplicity
        self.fired = Var(Event())
        self.enabled = None if kind == "production" else Var(Bool([1, 1]))

    @property
    def to_place(self) -> bool:
        return self.kind == "production"

    @property
    def consumes(self) -> bool:
        return self.kind == "consumption"

    def __repr__(self):
        return f"Arc({self.kind!r}, {self.multiplicity}, #{self.fired.id})"


def _arcs(arcs, to_place, what):
    arcs = tuple(arcs)
    for a in arcs:
        if not isinstance(a, Arc):
            raise TypeError(f"{what} are arcs, got {a!r}")
        if a.to_place != to_place:
            raise TypeError(f"{what} cannot be a {a.kind} arc")
    if len(set(arcs)) != len(arcs):
        raise ValueError(f"an arc appears twice among the {what}")
    return arcs


def _build(cls, blocks, **config):
    """The module of `blocks` (`init`/`next`/`flow`), as an instance of a subclass of `cls`."""
    return Module.__new__(type(cls.__name__, (cls,), blocks), theory=SPN, **config)


class Transition(Module):
    """A stochastic transition: `inputs` are the arcs it takes tokens through, `outputs` the
    arcs it produces through; it fires `arc.fired` on every one of them. `rate` is a
    number, the rate of an exponential delay, or a `Var(Clock())` holding the delay."""

    def __new__(cls, rate, *, t=TIME, inputs=(), outputs=(), name=None):
        delay = rate if isinstance(rate, Var) else None
        if delay is not None:
            if delay.dtype != Clock():
                raise TypeError(f"a delay variable is a Var(Clock()), got {delay!r}")
        elif isinstance(rate, bool) or not isinstance(rate, (int, float)) or not rate > 0:
            raise TypeError(f"a transition's rate is a positive number, got {rate!r}")
        if not isinstance(t, Var) or t.dtype != Clock():
            raise TypeError(f"the time reference is a Var(Clock()), got {t!r}")
        inputs = _arcs(inputs, False, "a transition's inputs")
        outputs = _arcs(outputs, True, "a transition's outputs")
        if not inputs and not outputs:
            raise TypeError("a transition needs at least one arc")
        rate = rate if delay is not None else float(rate)
        clock = Var(Clock())
        events = [a.fired for a in inputs + outputs]
        k = len(events)
        m = len(inputs)

        # the arming value: a fresh exponential sample, or the delay variable as the
        # driving module writes it this round
        def armed(rest):
            return exp(rate) if delay is None else X(rest[-1])

        # the variables bind positionally: clk, *events, t, *enabled, [delay]
        def init(self, t, *rest):
            return armed(rest), *[False] * k

        def next(self, clk, *rest):
            events, enabled = rest[:k], rest[k + 1 : k + 1 + m]
            expired = clk == 0
            fires = expired
            for e in enabled:
                fires = fires & e
            return ite(expired, armed(rest), clk), *[ite(fires, ~ev, ev) for ev in events]

        def flow(self, clk, *rest):
            return ite(clk >= 0, -1 * d(rest[k]), None), *[0] * k

        self = _build(
            cls,
            {"init": init, "next": next, "flow": flow},
            ctrl=(clock, *events),
            extl=(t, *[a.enabled for a in inputs], *([delay] if delay is not None else [])),
            hide={clock},
        )
        self.rate, self.t, self.name = rate, t, name
        self.inputs, self.outputs, self.clock = inputs, outputs, clock
        return self

    def __repr__(self):
        return f"Transition({self.name or self.clock.id}, rate={self.rate})"


class Place(Module):
    """A place: `count` is its token count, `inputs` the arcs tokens come in through,
    `outputs` the arcs they leave through, whose `enabled` it writes."""

    def __new__(cls, initial=0, *, inputs=(), outputs=(), name=None):
        if isinstance(initial, bool) or not isinstance(initial, int) or initial < 0:
            raise TypeError(f"a place holds a non-negative number of tokens, got {initial!r}")
        inputs = _arcs(inputs, True, "a place's inputs")
        outputs = _arcs(outputs, False, "a place's outputs")
        if not inputs and not outputs:
            raise TypeError("a place needs at least one input or output")
        count = Var(Nat())
        k = len(outputs)
        moving = [a for a in inputs + outputs if a.to_place or a.consumes]

        def lets_fire(n, arc):
            return ~(n >= arc.multiplicity) if arc.kind == "inhibitor" else n >= arc.multiplicity

        # the variables bind positionally: n, *enabled, *events (the moving arcs')
        def init(self, *events):
            return initial, *[(initial < a.multiplicity) == (a.kind == "inhibitor") for a in outputs]

        def next(self, n, *rest):
            events = rest[k:]
            for arc, ev in zip(moving, events):
                f = fired(ev)
                moved = n
                for _ in range(arc.multiplicity):
                    moved = moved + 1 if arc.to_place else moved - 1
                n = ite(f if arc.to_place else f & (n >= arc.multiplicity), moved, n)
            return n, *[lets_fire(n, a) for a in outputs]

        self = _build(
            cls,
            {"init": init, "next": next},
            ctrl=(count, *[a.enabled for a in outputs]),
            extl=tuple(a.fired for a in moving),
        )
        self.initial, self.name, self.count = initial, name, count
        self.inputs, self.outputs = inputs, outputs
        return self

    def __repr__(self):
        return f"Place({self.name or self.count.id}, initial={self.initial})"
