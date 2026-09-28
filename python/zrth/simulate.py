"""Simulate a hybrid SPN module: exact flows, a round per step, a scheduler.

A run is a sequence of steps. A step flows for a while, then takes a round: every atom's
`next` block runs once on the same latched state, and an atom whose next has no value (an
`ite` branch that is `None`) stutters. How long the step flows is decided in two parts:

- the **oracle** asks each atom until when it may flow: the first instant at which its
  `flow` has no value (`ite(cond, rate, None)` past `cond`: the invariant). Clocks are
  affine in time, so that instant is
  where a clock predicate flips, and the oracle finds it exactly. A step may last at most
  until the earliest of them, `hi`: time cannot pass an invariant;
- the **scheduler** picks how long the step lasts, in `(0, hi]`: `last()` flows to `hi`,
  `first(gap)` and `uniform(gap)` take shorter steps, whose rounds mostly stutter.

The oracle never reads the guards: a transition that must fire at an instant says so with
an invariant (`ite(clk >= 0, rate, None)` stops time exactly where `clk == 0` fires). A guard
that holds strictly before `hi` and no longer at `hi` would be passed over by every
scheduler, so it raises `GuardSkipped` instead. A flow whose rate cannot be reduced to a
number between rounds raises `FlowNotSolvable` rather than being integrated approximately.

    from zrth.simulate import simulate, first
    trace = simulate(module, t=t, horizon=100.0, seed=1)                # a round per expiry
    trace = simulate(module, t=t, horizon=100.0, scheduler=first(0.1))  # a round every 0.1
    trace.series(n)                                                     # one variable, round by round
"""

import math
import random
from dataclasses import dataclass

from .zrth import SPN, Combinatorial, Differential, Sequential, X, d
from .sort import Clock, Nat


class FlowNotSolvable(Exception):
    """A flow, or a guard on a flow, that the simulator cannot solve exactly."""


class GuardSkipped(Exception):
    """A `next` guard that holds before the flows stop and no longer where they do: no
    scheduler can land on it. Say when the round must happen with an invariant."""


class NoFlow(Exception):
    """The run cannot go on: the flow has no value in the current state (an `ite` branch
    that is `None` marks a state one cannot be in), or time cannot pass and the round
    changes nothing. `trace` holds the run up to there."""

    trace = None


class _NoValue:
    def __repr__(self):
        return "NOVALUE"


NOVALUE = _NoValue()
"""The value of an `ite` branch that is `None` (the theory's `IfThen` off its guard): a
`next` with it stutters, a `flow` with it forbids the passage of time."""


class _TimeDependent:
    def __repr__(self):
        return "TDEP"


TDEP = _TimeDependent()
"""In the affine pass: a value that turns with time in a way the pass does not follow."""


@dataclass(frozen=True)
class _Aff:
    """A clock value `c + k*delta` over the time `delta` since the latched state; `var` is
    the clock variable when the value is one, so a breakpoint can set it exactly."""

    c: float
    k: float
    var: object = None


@dataclass
class Trace:
    """The snapshots of a run: `states[i]` maps every controlled variable (and the time
    reference) to its value at `times[i]`; the first is the initial state, then one per
    round, stuttering rounds included."""

    times: list
    states: list
    quiescent: bool = False

    def series(self, var) -> list:
        return [s[var] for s in self.states]

    def __len__(self) -> int:
        return len(self.times)

    def __iter__(self):
        return iter(zip(self.times, self.states))


# ---------------------------------------------------------------------------
# The numeric interpreter: one SPN op on Python scalars
# ---------------------------------------------------------------------------


def _eval(op, r, rng):
    """`int` for a Nat, `bool` for a Bool/Event, `float` for a clock or a rate."""
    match op:
        case SPN.Ite():
            if r[0] is NOVALUE:
                return NOVALUE
            return r[1] if r[0] else r[2]
        case SPN.IfThen():
            if r[0] is NOVALUE:
                return NOVALUE
            return r[1] if r[0] else NOVALUE
    if any(v is NOVALUE for v in r):
        return NOVALUE
    match op:
        # the generators the constructors fill a missing block with
        case Sequential.SKIP():
            return r[0]
        case Differential.ZERO():
            return 0
        case Combinatorial.HAVOC():
            raise NotImplementedError("HAVOC has no simulation semantics")
        case SPN.Nat(v):
            return v
        case SPN.Bool(b):
            return b
        case SPN.Clock(x):
            return x
        case SPN.And():
            return r[0] and r[1]
        case SPN.Or():
            return r[0] or r[1]
        case SPN.Not():
            return not r[0]
        case SPN.IsZero():
            return r[0] == 0
        case SPN.ClkIsZero():
            return r[0] == 0.0
        case SPN.ClkGe():
            return r[0] >= r[1]
        case SPN.Inc():
            return r[0] + 1
        case SPN.Dec():
            return r[0] - 1  # speculative: an `Ite` may discard it; the latch checks the sign
        case SPN.Id():
            return r[0]
        case SPN.Exp(rate):
            if rng is None:
                raise FlowNotSolvable("a fresh sample inside a flow")
            return rng.expovariate(rate)
        case SPN.ClkRate(k):
            return k
        case SPN.ClkZero():
            return 0.0
        case SPN.ClkMul(k):
            return k * r[0]
        case SPN.Zero():
            return 0
        case SPN.Nondet(_):
            raise NotImplementedError("Nondet has no simulation semantics")
    raise FlowNotSolvable(f"cannot evaluate {op}")


# ---------------------------------------------------------------------------
# The affine pass: where do the clock predicates flip?
# ---------------------------------------------------------------------------


def _same(a, b) -> bool:
    if isinstance(a, _Aff) and isinstance(b, _Aff):
        return (a.c, a.k) == (b.c, b.k)
    return a is b or (type(a) is type(b) and a == b)


def _clock(v, op) -> _Aff:
    if isinstance(v, _Aff):
        return v
    raise FlowNotSolvable(
        f"{op} compares a clock value the oracle cannot track: one chosen by a "
        "time-dependent condition, or freshly sampled"
    )


def _breakpoint(breakpoints, delta, pins):
    if delta < 0.0:
        return
    breakpoints.setdefault(abs(delta), []).extend(pins)


def _affine(op, r, in_flow, breakpoints):
    """Like `_eval`, over `_Aff` clock values and `TDEP` truth values. Records into
    `breakpoints` (delta -> what to pin exactly at that instant) where every clock
    predicate flips. A rate that switches on such a predicate cannot be solved: it raises."""
    match op:
        case SPN.Ite():
            g = r[0]
            if g is NOVALUE:
                return NOVALUE
            if g is TDEP:
                if in_flow:
                    raise FlowNotSolvable(
                        "a rate switches on a clock condition inside a flow; "
                        "past a condition on a clock the flow must be None"
                    )
                return r[1] if _same(r[1], r[2]) else TDEP
            return r[1] if g else r[2]
        case SPN.IfThen():
            g = r[0]
            if g is NOVALUE:
                return NOVALUE
            if g is TDEP:
                return r[1]
            return r[1] if g else NOVALUE
    if any(v is NOVALUE for v in r):
        return NOVALUE
    match op:
        case SPN.Clock(x):
            return _Aff(x, 0.0)
        case SPN.Exp(_):
            return TDEP
        case SPN.ClkIsZero():
            a = _clock(r[0], op)
            if a.k == 0.0:
                return a.c == 0.0
            _breakpoint(breakpoints, -a.c / a.k, [(a.var, 0.0)] if a.var is not None else [])
            return TDEP
        case SPN.ClkGe():
            a, b = _clock(r[0], op), _clock(r[1], op)
            c, k = a.c - b.c, a.k - b.k
            if k == 0.0:
                return c >= 0.0
            delta = -c / k
            if a.var is not None and b.k == 0.0:
                pins = [(a.var, b.c)]
            elif b.var is not None and a.k == 0.0:
                pins = [(b.var, a.c)]
            elif a.var is not None and b.var is not None:
                pins = [(b.var, a.c + a.k * delta)]
            else:
                pins = []
            _breakpoint(breakpoints, delta, pins)
            return TDEP
        case SPN.IsZero():
            if r[0] is TDEP:
                raise FlowNotSolvable("a place count that depends on time is tested")
            return r[0] == 0
        case SPN.And():
            if r[0] is False or r[1] is False:
                return False
        case SPN.Or():
            if r[0] is True or r[1] is True:
                return True
    if any(v is TDEP for v in r):
        return TDEP
    return _eval(op, r, None)


# ---------------------------------------------------------------------------
# Blocks, rounds, state
# ---------------------------------------------------------------------------


def _walk(terms, state, fn):
    for term in terms:
        state[term.write[0]] = fn(term.itype, [state[w] for w in term.read])


def _round(module, state, fn):
    """Every atom's `next` once, in await order; an atom whose next has no value
    stutters, and the atoms awaiting it read the latched value."""
    for atom in module.atoms:
        _walk(atom.update, state, fn)
        for v in atom.ctrl:
            if state[X(v)] is NOVALUE:
                state[X(v)] = state[v]


def _latch(state, ctrl) -> bool:
    changed = False
    for v in ctrl:
        nxt = state[X(v)]
        if isinstance(v.dtype, Nat) and nxt < 0:
            raise ValueError(f"a place went negative: Dec on an empty place ({v})")
        if v not in state or nxt != state[v]:
            changed = True
        state[v] = nxt
    return changed


def _advanced(state, clocks, rates, t, delta, pins):
    """The state `delta` later: clocks moved at their rates, then the ones a breakpoint
    at `delta` pins set exactly (a clock expiring is exactly 0.0, not 1e-17)."""
    s = dict(state)
    for v in clocks:
        s[v] = state[v] + rates[v] * delta
    for w, val in pins.get(delta, ()):
        s[w] = val
    if t is not None:
        s[t] = state[t] + delta
        s[X(t)] = s[t]
    return s


def _snapshot(state, ctrl, t):
    snap = {v: state[v] for v in ctrl}
    if t is not None:
        snap[t] = state[t]
    return snap


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


def _rates(module, state, clocks) -> dict:
    s = dict(state)
    for atom in module.atoms:
        _walk(atom.delay, s, lambda op, r: _eval(op, r, None))
    rates = {}
    for v in clocks:
        k = s[d(v)]
        if k is NOVALUE:
            raise NoFlow(f"the flow of {v} has no value in this state: it cannot be here")
        if isinstance(k, bool) or not isinstance(k, (int, float)):
            raise FlowNotSolvable(f"the rate of {v} is not a number: {k!r}")
        rates[v] = float(k)
    return rates


def _breakpoints(module, state, clocks, rates, t) -> dict:
    s = dict(state)
    for v in clocks:
        s[v] = _Aff(state[v], rates[v], v)
    if t is not None:
        s[t] = _Aff(state[t], 1.0, t)
        s[X(t)] = s[t]
    breakpoints = {}
    for atom in module.atoms:
        _walk(atom.delay, s, lambda op, r: _affine(op, r, True, breakpoints))
    for atom in module.atoms:
        _walk(atom.update, s, lambda op, r: _affine(op, r, False, breakpoints))
        for v in atom.ctrl:
            if s[X(v)] is NOVALUE:
                s[X(v)] = s[v]
    return breakpoints


@dataclass(frozen=True)
class _Probe:
    delta: float
    at_breakpoint: bool
    flowing: tuple  # per atom: does its flow still have a value here?
    changing: tuple  # per atom: would a round here change its state?


def _probe(module, state, clocks, rates, t, delta, pins, at_breakpoint) -> _Probe:
    s = _advanced(state, clocks, rates, t, delta, pins)
    for atom in module.atoms:
        _walk(atom.delay, s, lambda op, r: _eval(op, r, None))
    flowing = tuple(all(s[d(v)] is not NOVALUE for v in atom.ctrl) for atom in module.atoms)
    scratch = random.Random(0)
    _round(module, s, lambda op, r: _eval(op, r, scratch))
    changing = tuple(any(s[X(v)] != s[v] for v in atom.ctrl) for atom in module.atoms)
    return _Probe(delta, at_breakpoint, flowing, changing)


def _probes(module, state, clocks, rates, t, pins) -> list:
    """At 0, at every breakpoint, and in between: the predicates are constant on each
    open interval, so one point per interval tells everything about it."""
    deltas = [0.0]
    for b in sorted(pins):
        if b > deltas[-1]:
            deltas.append(b)
    points = [(0.0, True)]
    for prev, b in zip(deltas, deltas[1:]):
        points.append(((prev + b) / 2, False))
        points.append((b, True))
    points.append((deltas[-1] + 1.0, False))
    return [_probe(module, state, clocks, rates, t, dl, pins, bp) for dl, bp in points]


def _stops_at(probes, i) -> float:
    """The last instant atom `i` may still flow: the breakpoint before the first probe
    where its flow has no value."""
    for j, p in enumerate(probes):
        if not p.flowing[i]:
            return p.delta if p.at_breakpoint else probes[j - 1].delta
    return math.inf


def _oracle(module, state, clocks, t):
    """`(hi, rates, pins)`: how long the flows may go on, the rates the clocks run at
    until then, and what a breakpoint pins exactly."""
    rates = _rates(module, state, clocks)
    pins = _breakpoints(module, state, clocks, rates, t)
    probes = _probes(module, state, clocks, rates, t, pins)
    atoms = range(len(module.atoms))
    hi = min((_stops_at(probes, i) for i in atoms), default=math.inf)
    at_hi = probes[-1] if hi == math.inf else next(p for p in probes if p.delta == hi)
    for i in atoms:
        if not at_hi.changing[i] and any(p.changing[i] for p in probes if p.delta < hi):
            raise GuardSkipped(
                f"atom {i} would fire before the flows stop (at {hi}) but not there: "
                "no scheduler lands on that instant; give it an invariant"
            )
    return hi, rates, pins


# ---------------------------------------------------------------------------
# Schedulers and the run
# ---------------------------------------------------------------------------


def last():
    """Flow as long as the invariants allow: a round exactly where they stop."""
    return lambda hi, rng: hi


def first(gap):
    """A round every `gap`, and where the invariants stop."""
    if not gap > 0:
        raise ValueError("the gap must be positive")
    return lambda hi, rng: min(gap, hi)


def uniform(gap):
    """A round at a uniformly random instant at least `gap` away, and where the invariants
    stop."""
    if not gap > 0:
        raise ValueError("the gap must be positive")
    return lambda hi, rng: hi if gap >= hi else rng.uniform(gap, hi)


def simulate(module, *, t=None, horizon, seed=0, scheduler=None) -> Trace:
    """Run `module` from its initial state up to time `horizon`.

    `t` is the module's time reference, its only allowed external: it reads the current
    time and its rate is 1. `seed` feeds the exponential clocks and the `uniform`
    scheduler. `scheduler` is `last()` (the default), `first(gap)`, `uniform(gap)`, or a
    function `(hi, rng) -> delta` choosing how long the step flows, `0 < delta <= hi`,
    where `hi` is how long the invariants allow (already clipped to the horizon).

    The run ends at the horizon. It is quiescent when no clock runs and nothing stops
    time: it then flows straight to the horizon. It raises `NoFlow` when it cannot go on,
    with the trace so far attached: the flow has no value in the current state, or time
    cannot pass and the round changes nothing.
    """
    if set(module.extl) != ({t} if t is not None else set()):
        raise TypeError("simulate(): the module's only external may be the time reference `t`")
    if t is not None and not isinstance(t.dtype, Clock):
        raise TypeError(f"simulate(): the time reference must be a Clock, got {t.dtype}")
    pick = last() if scheduler is None else scheduler
    rng = random.Random(seed)
    ctrl = list(module.ctrl)
    clocks = [v for v in ctrl if isinstance(v.dtype, Clock)]

    state = {}
    if t is not None:
        state[t] = state[X(t)] = 0.0
        state[d(t)] = 1.0
    for atom in module.atoms:
        _walk(atom.init, state, lambda op, r: _eval(op, r, rng))
    _latch(state, ctrl)
    now = 0.0
    trace = Trace([now], [_snapshot(state, ctrl, t)])

    while now < horizon:
        try:
            hi, rates, pins = _oracle(module, state, clocks, t)
        except NoFlow as e:
            e.trace = trace
            raise
        left = horizon - now
        if hi == math.inf and all(k == 0.0 for k in rates.values()):
            trace.quiescent = True
            delta = left
        elif hi == 0.0:
            delta = 0.0
        else:
            delta = pick(min(hi, left), rng)
            if not 0.0 < delta <= min(hi, left):
                raise ValueError(f"the scheduler chose {delta}, outside (0, {min(hi, left)}]")
        state = _advanced(state, clocks, rates, t, delta, pins)
        now += delta
        if trace.quiescent:
            trace.times.append(now)
            trace.states.append(_snapshot(state, ctrl, t))
            break
        _round(module, state, lambda op, r: _eval(op, r, rng))
        changed = _latch(state, ctrl)
        if hi == 0.0 and not changed:
            e = NoFlow(f"stuck at {now}: time cannot pass and the round changes nothing")
            e.trace = trace
            raise e
        trace.times.append(now)
        trace.states.append(_snapshot(state, ctrl, t))
    return trace
