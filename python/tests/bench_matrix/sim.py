"""Run a row's module, so `Row.truth` is measured rather than asserted.

`Row.truth` says whether a property is known to hold of its module, and it is
what makes a route's answer *correct*: a `REFUTED` is right only where the
property fails. For the four original suites that field is transcribed from
something that already paid for it -- an fbk probe's `expect`, a benchmark's
own Houdini invariant. The `hybrid` and `petri` suites have no such upstream,
so their `truth` is checked here instead, by stepping the module through
`zrth.eval` and looking at the states it actually visits.

What that buys is one-sided and the asymmetry matters:

  `truth="fails"`  is *established*. A concrete reachable state falsifying the
                   property is a counterexample, and finding one is proof.
  `truth="holds"`  is only *not refuted*. A bounded run over sampled inputs
                   cannot certify a property -- that is what the matrix's
                   routes are for. It catches the mistake this file exists to
                   catch: a row whose stated property is simply false, which
                   would make every honest `REFUTED` look like a route bug.

**A recurrence is refuted by a cycle, not by a quiet tail.** `G F p` cannot
be refuted by any finite prefix, so the only thing a simulation can offer is
a run that comes back to a configuration it has already been in with `p`
false all the way round: the inputs along that loop are the ones the run
drew, so replaying them from the start repeats it for ever and `p` holds
finitely often. That is a proof, and `Observation.period` carries it.

This used to be a tail count -- `p` fewer than four times in the last
quarter of a 200-tick run -- which assumes a recurrence period of about
twelve ticks or less. That is true of the family it was written for and
false elsewhere, and it made the rule a false-refutation machine on the
rest: `m_countdown` counts 100 down to 0 and resets, period 101, so it hits
its property once in a 50-tick tail and was reported `fails` while five
routes carried Lean proofs that it holds. Measured over the 19 `hybrid` and
`petri` recurrence rows the cycle rule reproduces every declared `truth`,
all four `fails` among them; over the 75 runnable rows with none declared it
agrees with the tail count on 71 and differs on exactly the four the tail
count had wrong.

What it costs is reach, and it is the honest direction: a module whose
period is longer than the run, or whose state never repeats at all, closes
no cycle and is not refuted rather than refuted on a guess.
`Observation.looped` is the difference between "no cycle avoids `p`" and
"no cycle was reached", so a caller need not read the absence as evidence.

Modules here are all 1x1-component, and the nondeterministic ones take their
choices from external inputs, sampled under the row's `--pre` via z3.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import z3

from zrth import X
from zrth.eval import execute_init, execute_update
from zrth.sort import Bool, Int, Real

# Defaults sized for `just py-test`: deep enough to reach every counterexample
# the `fails` rows name -- the deepest is at tick 115 -- and short enough that
# all sixty rows take under a minute. Evaluating the property, not stepping the
# module, is what costs; `holds` is the claim that wants more, so the bounds
# quoted in the two `cases.py` notes come from runs twenty times longer, worth
# repeating by hand (`steps=40000, trials=1`) after changing any constant.
#
# `STEPS` bounds the recurrence rule too, and there in one direction only: a
# cycle longer than the run is one this file cannot close, so raising it can
# turn a `holds` into a `fails` and can never do the reverse.
STEPS = 200          # ticks per trial
TRIALS = 8           # independent input profiles per row


# ══════════════════════════════════════════════════════════════════════════
# Sorts, properties, inputs
# ══════════════════════════════════════════════════════════════════════════

def _scalar(dtype):
    """`dtype` as a z3 sort. 1x1 only -- a row with matrix state needs the
    tuple-select spelling of `--pre`, and none of these have one."""
    match dtype:
        case Real([1, 1]):
            return z3.RealSort()
        case Int([1, 1]):
            return z3.IntSort()
        case Bool([1, 1]):
            return z3.BoolSort()
        case _:
            raise NotImplementedError(f"sim: {dtype} is not a 1x1 component")


def _consts(prefix, dtypes):
    return [z3.Const(f"{prefix}{i}", _scalar(d)) for i, d in enumerate(dtypes)]


def parse(prop: str, consts):
    """An SMT-LIB 2 property as one z3 expression over `consts`."""
    decls = {c.decl().name(): c for c in consts}
    asserted = z3.parse_smt2_string(f"(assert {prop})", decls=decls)
    return z3.And(*asserted) if len(asserted) != 1 else asserted[0]


def _lit(value, dtype):
    """A 1x1 tensor as the z3 literal of `dtype`."""
    v = value.flatten()[0].item()
    match dtype:
        case Real([1, 1]):
            return z3.RealVal(v)
        case Int([1, 1]):
            return z3.IntVal(int(v))
        case Bool([1, 1]):
            return z3.BoolVal(bool(v))
        case _:
            raise NotImplementedError(f"sim: {dtype} is not a 1x1 component")


def holds(prop, consts, dtypes, state, memo=None) -> bool:
    """Whether `prop` is true of the concrete `state` (one value per const).

    Substituting and simplifying costs far more than stepping the module does,
    and a run revisits states -- every marking of a bounded net, over and over
    -- so `memo` keys the answer on the state itself. It is the difference
    between seconds and minutes over the suite.
    """
    if memo is not None:
        key = tuple(v.flatten()[0].item() for v in state)
        if key in memo:
            return memo[key]
    sub = [(c, _lit(v, d)) for c, v, d in zip(consts, state, dtypes)]
    answer = z3.is_true(z3.simplify(z3.substitute(prop, *sub)))
    if memo is not None:
        memo[key] = answer
    return answer


def _tensor(value, dtype):
    match dtype:
        case Real([1, 1]):
            return torch.tensor([[float(value)]], dtype=torch.float32)
        case Int([1, 1]):
            return torch.tensor([[int(value)]], dtype=torch.int64)
        case Bool([1, 1]):
            return torch.tensor([[bool(value)]], dtype=torch.bool)
        case _:
            raise NotImplementedError(f"sim: {dtype} is not a 1x1 component")


class Inputs:
    """A source of input vectors satisfying `pre`.

    A row's `--pre` is the only thing constraining a reactive module's inputs,
    and it is almost always a box -- a selector's range, a disturbance's
    bound -- so the admissible box is asked of z3 once (`Optimize`, one bound
    per input per direction) and candidates are then drawn inside it: integers
    uniformly, Reals on the eighths every constant in these modules is written
    at, Bools by coin flip. A candidate is still checked against `pre` before
    it is used, because the box is an over-approximation whenever `pre` is not
    one.

    A precondition no candidate satisfies falls back to whatever model z3
    hands back -- the same vector every tick, a degenerate profile rather than
    a wrong one. No row here needs it.
    """

    WIDTH = 8           # how far to explore an input `pre` leaves unbounded

    def __init__(self, dtypes, pre: str, rng: random.Random):
        self.dtypes, self.rng = dtypes, rng
        self.consts = _consts("e", dtypes)
        self.pre = parse(pre, self.consts) if pre else z3.BoolVal(True)
        self.box = [self._range(c, d) for c, d in zip(self.consts, dtypes)]
        self._fallback = None

    @staticmethod
    def _finite(v):
        """`v` as a float, or None when z3 reports it unbounded."""
        try:
            return float(v.as_fraction()) if v.is_real() else float(v.as_long())
        except Exception:
            return None

    def _range(self, c, dtype):
        """What `pre` allows `c` to be, as (lo, hi); None where unbounded."""
        if dtype == Bool([1, 1]):
            return None
        out = []
        for maximise in (False, True):
            opt = z3.Optimize()
            opt.add(self.pre)
            handle = opt.maximize(c) if maximise else opt.minimize(c)
            if opt.check() != z3.sat:
                raise ValueError("sim: --pre is unsatisfiable")
            out.append(self._finite(opt.upper(handle) if maximise
                                    else opt.lower(handle)))
        return tuple(out)

    def _draw(self):
        out = []
        for d, box in zip(self.dtypes, self.box):
            lo, hi = box if box else (None, None)
            lo = -self.WIDTH if lo is None else lo
            hi = self.WIDTH if hi is None else hi
            match d:
                case Real([1, 1]):
                    eighths = range(int(lo * 8) - 1, int(hi * 8) + 2)
                    out.append(self.rng.choice(list(eighths)) / 8)
                case Int([1, 1]):
                    out.append(self.rng.randint(int(lo) - 1, int(hi) + 1))
                case Bool([1, 1]):
                    out.append(self.rng.random() < 0.5)
                case _:
                    raise NotImplementedError(f"sim: {d} is not a 1x1 component")
        return out

    def _any_model(self):
        if self._fallback is None:
            s = z3.Solver()
            s.add(self.pre)
            if s.check() != z3.sat:
                raise ValueError("sim: --pre is unsatisfiable")
            m = s.model()
            self._fallback = []
            for c, d in zip(self.consts, self.dtypes):
                v = m.eval(c, model_completion=True)
                self._fallback.append(z3.is_true(v) if d == Bool([1, 1])
                                      else self._finite(v) or 0.0)
        return self._fallback

    def next(self) -> list:
        """One input vector, as tensors."""
        if not self.dtypes:
            return []
        for _ in range(256):
            cand = [_tensor(v, d) for v, d in zip(self._draw(), self.dtypes)]
            if holds(self.pre, self.consts, self.dtypes, cand):
                return cand
        return [_tensor(v, d) for v, d in zip(self._any_model(), self.dtypes)]


# ══════════════════════════════════════════════════════════════════════════
# Stepping
# ══════════════════════════════════════════════════════════════════════════

def trace(module, steps: int, inputs: Inputs) -> list:
    """The run after init and after each of `steps` updates, tick by tick.

    A tick is `(state, held)` and not the state alone. The next update reads
    the latched input beside the control variables, so two ticks continue
    alike only if both agree -- which is what makes a repeat a cycle rather
    than a coincidence. A module with no inputs has `held` empty and the
    pair is the state, which is the same statement.
    """
    extl = list(module.extl)
    state = {}
    fresh = inputs.next()
    for e, v in zip(extl, fresh):
        state[e], state[X(e)] = v, v
    execute_init(state, module.atoms)
    held = fresh
    out = [([state[X(v)] for v in module.ctrl], held)]
    for _ in range(steps):
        nxt = {v: state[X(v)] for v in module.ctrl}
        prev, fresh = held, inputs.next()
        for e, was, new in zip(extl, prev, fresh):
            nxt[e], nxt[X(e)] = was, new
        execute_update(nxt, module.atoms)
        state, held = nxt, fresh
        out.append(([state[X(v)] for v in module.ctrl], held))
    return out


def _values(vals) -> tuple:
    """A tick's tensors as plain Python, which is what can be a dict key."""
    return tuple(v.flatten()[0].item() for v in vals)


def lasso(run: list, prop, consts, dtypes, memo=None) -> "tuple | None":
    """The first stretch of `run` that loops with `prop` false throughout.

    Returned as `(start, end)`: `run[start]` and `run[end]` are the same
    configuration and no tick in between satisfies `prop`. Replaying the
    inputs of `run[start:end]` from `run[start]` therefore repeats it for
    ever, which refutes `G F prop`.

    Only *consecutive* occurrences of a configuration are compared, and that
    misses nothing: a loop spanning three occurrences is `prop`-free only if
    both halves are, and the first half was already checked.
    """
    seen: dict = {}
    for i, (state, held) in enumerate(run):
        here = (_values(state), _values(held))
        was = seen.get(here)
        if was is not None and not any(
            holds(prop, consts, dtypes, s, memo) for s, _h in run[was:i]
        ):
            return was, i
        seen[here] = i
    return None


# ══════════════════════════════════════════════════════════════════════════
# One row
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Observation:
    """What the runs saw, in the terms `Row.truth` is written in."""

    refuted: bool = False           # a reachable state falsifies the property
    witness: tuple = ()             # that state, as plain Python values
    step: int = -1                  # the tick it was reached on
    bounds: list = field(default_factory=list)   # (min, max) per component
    # `--buchi` only. `period` is the length of the refuting cycle, and
    # `looped` whether any cycle was closed at all -- which is what tells a
    # property no cycle avoids from one no cycle was reached to test.
    period: int = 0
    looped: bool = False

    @property
    def verdict(self) -> str:
        return "fails" if self.refuted else "holds"


def observe(module, prop: str, kind: str, pre: str = "", *, steps: int = STEPS,
            trials: int = TRIALS, seed: int = 0) -> Observation:
    """Step `module` and report whether the runs refute `prop`.

    For `safety` a refutation is one reachable state where the property is
    false. For `buchi` it is a cycle the property is false all the way
    round -- see `lasso`, and the module docstring for why a quiet tail is
    not one.
    """
    if kind not in ("safety", "buchi"):
        raise ValueError(f"sim: unknown property kind {kind!r}")
    dtypes = [v.dtype for v in module.ctrl]
    consts = _consts("s", dtypes)
    parsed = parse(prop, consts)
    lo = [float("inf")] * len(dtypes)
    hi = [float("-inf")] * len(dtypes)
    memo: dict = {}
    out = Observation()
    for t in range(trials):
        rng = random.Random(seed * 1000 + t)
        run = trace(module, steps, Inputs([v.dtype for v in module.extl], pre, rng))
        for i, (s, _held) in enumerate(run):
            for j, v in enumerate(s):
                x = float(v.flatten()[0].item())
                lo[j], hi[j] = min(lo[j], x), max(hi[j], x)
            if kind == "safety" and not out.refuted and not holds(parsed, consts, dtypes, s, memo):
                out.refuted, out.step = True, i
                out.witness = _values(s)
        if kind == "buchi":
            out.looped = out.looped or _repeats(run)
            loop = lasso(run, parsed, consts, dtypes, memo)
            if loop is not None and not out.refuted:
                start, end = loop
                out.refuted, out.step, out.period = True, start, end - start
                out.witness = _values(run[start][0])
    out.bounds = list(zip(lo, hi))
    return out


def _repeats(run: list) -> bool:
    """Whether `run` ever returns to a configuration -- whether, that is,
    there was any cycle for the property to be tested against."""
    seen = set()
    for state, held in run:
        here = (_values(state), _values(held))
        if here in seen:
            return True
        seen.add(here)
    return False
