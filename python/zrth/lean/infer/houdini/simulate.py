"""The module run from its entry states (`simulate`), its inputs drawn
as `--pre` allows (`Draws`), and rounds sampled from states near the
ones it reached (`sample_rounds`).

The runs only *filter*: a candidate they keep is still proved or
dropped by the solver.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from fractions import Fraction

from ...smt.claim import SynthContext
from ...smt.obligations import Obligations
from ..common.synth import component_slots
from .evaluate import Evaluator, Unevaluable

# How much of the module the simulation may evaluate, and for how long. The
# runs only *filter*: a fact they keep is still proved or dropped by the
# solver.
_MAX_ROUNDS = 4000
_SIM_SECONDS = 3.0
_TRIES = 64                     # input draws per round before a run is ended
_OVERFLOW = 10**9               # a run past this is diverging, not informing

# How long one batch of sampled rounds may take, and how far a wide draw
# reaches past the program's own constants.
SAMPLE_SECONDS = 1.0
_WIDE = 10**6


@dataclass
class Runs:
    """What the simulation reached: states, and the rounds between them.

    A state is a tuple of component values, in `ctx.state` order.
    """

    states: list
    rounds: list
    # Under `--safety`, a reached state where the property fails, and how
    # many rounds it took.
    violation: "tuple | None" = None
    depth: int = 0


class Draws:
    """Input values, and states near the ones a run reached.

    Integers come from the program's own constants half the time -- the
    value a guard compares against is the one worth hitting -- and from a
    range a little wider than those constants otherwise.

    A real is drawn on the lattice the program lives on -- multiples of
    `1/scale`, plus the rationals it mentions -- most of the time, and off
    it the rest: a state with a denominator the program never writes is what
    shows an interval is not inductive over the reals, and a state on the
    lattice is what a fact about the values a run takes has to be tested at.
    """

    def __init__(self, ctx: SynthContext, constants, seed: int,
                 wide: bool = False, rationals=(), scale: int = 1):
        self.rng = random.Random(seed)
        self.pool = sorted(set(constants) | {-1, 0, 1})
        self.reach = min(1000, 2 * max(abs(c) for c in self.pool) + 10)
        # `wide` draws integers far past anything the program mentions. A
        # shape fitted to states within the program's own range can hold
        # only *because* they were small -- `11 - x` is positive on every
        # state with `x <= 10` -- and states from a wider range are what tell
        # that apart from a shape the invariant actually bounds.
        if wide:
            self.reach = _WIDE
        self.fresh = 0.5 if wide else 0.25
        self.sorts = [v.getSort() for v in ctx.state]
        # Where a state can be moved, and in what sort: one place per
        # *element*, so a matrix-shaped component is as many as it has.
        self.slots = [(i, slot, srt)
                      for i in range(len(self.sorts))
                      for slot, srt in component_slots(ctx, i)]
        self.rpool = sorted(set(rationals) | {Fraction(c) for c in self.pool})
        self.step = Fraction(1, scale)

    def value(self, sort):
        rng = self.rng
        if sort.isTuple():
            return tuple(self.value(s) for s in sort.getTupleSorts())
        if sort.isBoolean():
            return rng.random() < 0.5
        if sort.isReal():
            roll = rng.random()
            if roll < 0.4:
                return rng.choice(self.rpool)
            if roll < 0.8:
                return self.step * rng.randint(-self.reach, self.reach)
            return Fraction(rng.randint(-self.reach, self.reach),
                            rng.choice((3, 5, 7, 8)))
        if rng.random() < 0.5:
            return rng.choice(self.pool)
        return rng.randint(-self.reach, self.reach)

    def near(self, seeds: list) -> tuple:
        """A state: a reached one with some elements moved, or a fresh one.

        Elements rather than components: a 3-vector moved as a unit is three
        coordinates that always change together, and the state that tells a
        fact about `v[0]` apart from one about the whole vector is the state
        where only `v[0]` moved.
        """
        rng = self.rng
        if not seeds or rng.random() < self.fresh:
            return tuple(self.value(srt) for srt in self.sorts)
        s = list(rng.choice(seeds))
        for i, slot, srt in rng.sample(self.slots,
                                       k=rng.randint(1, len(self.slots))):
            old = s[i] if slot is None else s[i][slot]
            if srt.isBoolean():
                new = not old
            elif rng.random() < 0.6:
                new = old + (self.step if srt.isReal() else 1) * rng.choice(
                    (-2, -1, 1, 2))
            else:
                new = self.value(srt)
            if slot is None:
                s[i] = new
            else:
                s[i] = s[i][:slot] + (new,) + s[i][slot + 1:]
        return tuple(s)


def simulate(ctx: SynthContext, ob: Obligations, ev: Evaluator,
             constants: tuple[int, ...], *, safety: bool, seed: int = 0,
             rationals=(), scale: int = 1) -> Runs:
    """Run the module from its entry states, drawing inputs as `--pre` allows.

    A run is a run of the module: each round's latched inputs are the values
    the round before awaited, and only the awaited ones are drawn. So every
    state reached is reachable -- a fact it violates is not an invariant, and
    a `--safety` property it violates does not hold.
    """
    draws = Draws(ctx, constants, seed, rationals=rationals, scale=scale)
    names = ctx.names
    latched = [n for n, _ in ob.slots(ob.el)]
    awaited = ob.slots(ob.en)

    def inputs(pre, base, before):
        """Awaited inputs `pre` allows. At entry nothing is latched yet, so
        the latched ones are drawn too -- `init_inv` quantifies them."""
        for _ in range(_TRIES):
            env = dict(base)
            if before is None:
                env.update({n: draws.value(srt)
                            for n, (_, srt) in zip(latched, awaited)})
            else:
                env.update(zip(latched, before))
            env.update({n: draws.value(srt) for n, srt in awaited})
            try:
                if ev(pre, env):
                    return env
            except Unevaluable:
                return None
        return None

    def prop_fails(state) -> bool:
        try:
            return not ev(ctx.prp, dict(zip(names, state)))
        except Unevaluable:
            return False

    states: dict = {}
    rounds: dict = {}
    deterministic = not awaited
    trajectories = 1 if deterministic else 32
    length = _MAX_ROUNDS if deterministic else 128
    t_end = time.monotonic() + _SIM_SECONDS
    for _ in range(trajectories):
        if len(rounds) >= _MAX_ROUNDS or time.monotonic() > t_end:
            break
        env = inputs(ob.init_pre, {}, None)
        if env is None:
            continue
        try:
            s = tuple(ev.many(ob.init, env))
        except Unevaluable:
            continue
        before = [env[n] for n, _ in awaited]
        seen_here = set()
        for depth in range(length + 1):
            states.setdefault(s, None)
            if safety and prop_fails(s):
                return Runs(list(states), list(rounds), violation=s, depth=depth)
            if deterministic and s in seen_here:
                break
            seen_here.add(s)
            if (len(rounds) >= _MAX_ROUNDS or time.monotonic() > t_end
                    or any(not isinstance(v, bool) and abs(v) > _OVERFLOW
                           for v in scalars(s))):
                break
            base = ob.env_of(s)
            env = inputs(ob.update_pre, base, before)
            if env is None:
                break
            try:
                sp = tuple(ev.many(ob.next, env))
            except Unevaluable:
                break
            before = [env[n] for n, _ in awaited]
            rounds.setdefault((s, sp), None)
            s = sp
    return Runs(list(states), list(rounds))


def sample_rounds(ctx: SynthContext, ob: Obligations, ev: Evaluator,
                  draws: Draws, seeds: list, keep, *, want: int = 300,
                  seconds: float = SAMPLE_SECONDS) -> list:
    """Rounds from states `keep` accepts, not necessarily reachable ones.

    What an obligation quantifies over is every state the invariant admits,
    with any inputs `--pre` allows -- latched ones included -- so a round
    from such a state that breaks a fact, or that a ranking function does not
    drop on, is a counterexample to that obligation as a solver would be
    asked it. Finding one here is free, where a call costs milliseconds at
    best and, on a solver that cannot refute, a whole time limit to learn
    nothing.
    """
    ins = ob.slots(ob.el) + ob.slots(ob.en)
    out: list = []
    t_end = time.monotonic() + seconds
    for _ in range(40 * want):
        if len(out) >= want or time.monotonic() > t_end:
            break
        s = draws.near(seeds)
        try:
            if not keep(s):
                continue
            base = ob.env_of(s)
            for _ in range(8):
                env = dict(base)
                env.update({n: draws.value(srt) for n, srt in ins})
                if ev(ob.update_pre, env):
                    out.append((s, tuple(ev.many(ob.next, env))))
                    break
        except Unevaluable:
            continue
    return out


def scalars(state):
    """Every scalar in a simulated state, matrix-shaped ones flattened."""
    for v in state:
        if isinstance(v, tuple):
            yield from v
        else:
            yield v
