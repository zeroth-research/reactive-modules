"""The bounds each mode keeps: candidate facts for Houdini, computed.

Houdini's own bounds are *read*: the tightest program constant around what
the runs reached. That misses the bound a hybrid automaton is held by, which
is usually a constant no program text names. `m_thermostat` heats as
`T' = 0.875 T + 3.75` while `T < 22`, so with the heater on the room never
passes `0.875 * 22 + 3.75 = 23`, and with it off never passes the 23.875 one
more tick of heating gives. `band` (`15 <= T <= 25`) is proved by exactly
those two facts, and by no interval that ignores the mode: from `T = 25`
with the heater on the next temperature is 25.5.

So this module *computes* them, by abstract interpretation over the state
space split by its Bool columns. A **mode** is a valuation of the Bool
columns; the abstract state gives each mode either "unreached" or an
interval for every numeric column. The successor of an abstract state is
read off the round exactly -- for each mode the round can lead to and each
column, the least and greatest value the column can take there, as an
optimisation query to z3 -- and the reachable states are the least fixed
point of that, from the initial ones:

* **ascending**, iterates are joined, and after a few of them a bound still
  moving is widened to the next *threshold* -- a literal of the program or
  the property, or a power-of-two multiple of the largest -- or dropped.
* **descending**, the post-fixed point is narrowed back towards the least
  one by re-applying the round. A contraction such as the thermostat's gets
  there only in the limit, so the descent stops once a pass moves no bound
  by more than :data:`_SETTLED`.
* **rounding**, the bounds are moved outward onto a grid, finest first, and
  the result kept only if it is inductive: its successor and the initial
  states both stay within it. That is checked exactly, so what comes back
  is an inductive invariant by z3's account, and Houdini then proves or
  drops each fact again with its own solver -- these are candidates, and
  nothing here is trusted.

z3 rather than cvc5 because the descent needs the least upper bound of a
term exactly, `23` and not some model's `22.99`, and z3's optimiser answers
that over linear real arithmetic with `ite` and Bools in one call.

What it cannot see is a relation: `m_twotanks` holds tank 2 by how much of
tank 1 is left to pump, `x2 + 2 x1`, and an interval per mode is too coarse
for that however it is computed.
"""

from __future__ import annotations

import math
import time
from fractions import Fraction
from itertools import product

try:
    import z3                                        # type: ignore
except ImportError:                                  # pragma: no cover
    z3 = None

try:
    from cvc5 import Kind
except ImportError:                                  # pragma: no cover
    Kind = None

# A mode is a valuation of the Bool columns, so the modes are exponential in
# them; past this many, the split is not worth the queries.
_MAX_MODE_BITS = 3
# Ascending passes before a moving bound is widened, and the most passes
# before the ascent is abandoned.
_WIDEN_AFTER = 2
_MAX_ASCENT = 40
# Descending passes, at most, and the movement below which one is the last.
_MAX_DESCENT = 80
_SETTLED = Fraction(1, 10_000)
# The grids the bounds are rounded onto, coarsest first: `1/2**k`.
_GRIDS = tuple(Fraction(1, 2 ** k) for k in range(11))
# One optimisation query's time limit.
_QUERY_MS = 2000

Box = "list[tuple[Fraction | None, Fraction | None]] | None"


class _OutOfTime(Exception):
    pass


def mode_bounds(ctx, ob, *, seconds: float = 8.0, log=print) -> list[str]:
    """Facts over `s0..` that the modes keep, or none.

    Each fact is a bound on one numeric column under one mode --
    `(=> (not s1) (<= s0 23.875))` -- or a mode no run can be in, as
    `(not ...)`; and the bounds every mode shares, unguarded. Two grids'
    worth when both are inductive: the finest, which is the tightest the
    descent reached, and whole numbers, which are what a reader of the
    certificate expects and often all a proof needs.
    """
    if z3 is None:                                   # pragma: no cover
        return []
    t0 = time.monotonic()
    try:
        model = _Model.of(ctx, ob, t0 + seconds)
        if model is None:
            return []
        found = model.invariants()
    except _OutOfTime:
        log(f"[houdini] mode bounds: none within {seconds:g} s")
        return []
    except z3.Z3Exception as e:
        log(f"[houdini] mode bounds: none, z3 could not read the round ({e})")
        return []
    if not found:
        log(f"[houdini] mode bounds: none inductive ({time.monotonic() - t0:.1f} s)")
        return []
    facts = list(dict.fromkeys(f for a in found for f in model.facts(a)))
    log(f"[houdini] mode bounds: {len(facts)} facts over {len(model.modes)} "
        f"mode(s) ({time.monotonic() - t0:.1f} s)")
    return facts


class _Model:
    """The round as z3 reads it, and the abstract domain over it."""

    @classmethod
    def of(cls, ctx, ob, deadline: float) -> "_Model | None":
        sorts = list(ctx.env.state_sorts)
        if not all(s.isBoolean() or s.isInteger() or s.isReal() for s in sorts):
            return None
        bits = [i for i, s in enumerate(sorts) if s.isBoolean()]
        nums = [i for i, s in enumerate(sorts) if not s.isBoolean()]
        if not nums or len(bits) > _MAX_MODE_BITS:
            return None
        seen: dict = {}
        from ..smt_query import _constants

        terms = list(ob.next) + list(ob.init) + [ob.update_pre, ob.init_pre]
        for t in terms:
            _constants(t, seen)
        if any(not (v.getSort().isBoolean() or v.getSort().isInteger()
                    or v.getSort().isReal()) for v in seen.values()):
            return None
        return cls(ctx, ob, sorts, bits, nums, seen, terms, deadline)

    def __init__(self, ctx, ob, sorts, bits, nums, seen, terms, deadline):
        from ..houdini_solver import decimals

        self.sorts, self.bits, self.nums = sorts, bits, nums
        self.deadline = deadline
        self.z = z = z3.Context()
        lines = [f"(declare-fun {s} () {seen[s].getSort()})" for s in sorted(seen)]
        for tag, ts in (("N", ob.next), ("I", ob.init)):
            for i, t in enumerate(ts):
                lines.append(f"(declare-fun {tag}{i} () {t.getSort()})")
                lines.append(f"(assert (= {tag}{i} {t}))")
        lines.append(f"(declare-fun PRE () Bool) (assert (= PRE {ob.update_pre}))")
        lines.append(f"(declare-fun IPRE () Bool) (assert (= IPRE {ob.init_pre}))")
        self.round = list(z3.parse_smt2_string(decimals("\n".join(lines)), ctx=z))

        def var(name, s):
            if s.isBoolean():
                return z3.Bool(name, z)
            return z3.Int(name, z) if s.isInteger() else z3.Real(name, z)

        n = len(sorts)
        self.S = [var(f"v_s{i}", sorts[i]) for i in range(n)]
        self.N = [var(f"N{i}", sorts[i]) for i in range(n)]
        self.I = [var(f"I{i}", sorts[i]) for i in range(n)]
        self.pre, self.ipre = z3.Bool("PRE", z), z3.Bool("IPRE", z)
        self.modes = list(product((False, True), repeat=len(bits)))
        lits = _literals(terms + [ctx.prp])
        big = max([abs(x) for x in lits] + [Fraction(1)])
        ladder = {big * 2 ** k for k in range(1, 6)}
        self.thresholds = sorted(lits | {-x for x in lits} | {Fraction(0)}
                                 | ladder | {-x for x in ladder})

    # --- the domain -------------------------------------------------------

    def _mode(self, vs, m):
        if not self.bits:
            return z3.BoolVal(True, self.z)
        return z3.And([vs[b] if v else z3.Not(vs[b])
                       for b, v in zip(self.bits, m)])

    def _within(self, vs, box):
        cs = []
        for i, (lo, hi) in zip(self.nums, box):
            if lo is not None:
                cs.append(vs[i] >= _z3_num(lo, self.sorts[i], self.z))
            if hi is not None:
                cs.append(vs[i] <= _z3_num(hi, self.sorts[i], self.z))
        return z3.And(cs) if cs else z3.BoolVal(True, self.z)

    def _image(self, given, vs) -> dict:
        """For each mode, the box `vs` stays within when `given` holds."""
        out = {}
        for m in self.modes:
            self._clock()
            s = z3.Solver(ctx=self.z)
            s.set("timeout", _QUERY_MS)
            s.add(self.round)
            s.add(given, self._mode(vs, m))
            answer = s.check()
            if answer == z3.unsat:
                out[m] = None
                continue
            box = []
            for i in self.nums:
                box.append((self._extreme(given, vs, m, i, low=True),
                            self._extreme(given, vs, m, i, low=False)))
            out[m] = box
        return out

    def _extreme(self, given, vs, m, i, *, low: bool) -> "Fraction | None":
        """The infimum or supremum of `vs[i]`, or None when there is none
        or z3 did not find it -- either way no bound, which is sound."""
        self._clock()
        o = z3.Optimize(ctx=self.z)
        o.set("timeout", _QUERY_MS)
        o.add(self.round)
        o.add(given, self._mode(vs, m))
        h = o.minimize(vs[i]) if low else o.maximize(vs[i])
        if o.check() != z3.sat:
            return None
        inf, val, _eps = h.lower_values() if low else h.upper_values()
        if not (z3.is_rational_value(inf) or z3.is_int_value(inf)) \
                or _fraction(inf) != 0:
            return None
        if not (z3.is_rational_value(val) or z3.is_int_value(val)):
            return None
        v = _fraction(val)
        if self.sorts[i].isInteger():
            v = Fraction(math.ceil(v) if low else math.floor(v))
        return v

    def _post(self, a: dict) -> dict:
        reached = [z3.And(self._mode(self.S, m), self._within(self.S, a[m]))
                   for m in self.modes if a[m] is not None]
        if not reached:
            return {m: None for m in self.modes}
        return self._image(z3.And(z3.Or(reached), self.pre), self.N)

    def _clock(self):
        if time.monotonic() > self.deadline:
            raise _OutOfTime

    # --- the iteration ----------------------------------------------------

    def invariants(self) -> list[dict]:
        """The inductive abstract states, finest grid first; possibly none."""
        start = self._image(self.ipre, self.I)
        a = start
        for k in range(_MAX_ASCENT):
            nxt = {m: _join(a[m], b) for m, b in self._post(a).items()}
            if nxt == a:
                break
            a = ({m: self._widen(a[m], nxt[m]) for m in self.modes}
                 if k >= _WIDEN_AFTER else nxt)
        else:
            return []
        for _ in range(_MAX_DESCENT):
            back = {m: _join(b, start[m]) for m, b in self._post(a).items()}
            nxt = {m: _meet(a[m], back[m]) for m in self.modes}
            moved = _moved(a, nxt)
            a = nxt
            if moved <= _SETTLED:
                break
        out = []
        for q in reversed(_GRIDS):
            r = {m: self._rounded(a[m], q) for m in self.modes}
            if self._inductive(r, start):
                out.append(r)
                break
        whole = {m: self._rounded(a[m], Fraction(1)) for m in self.modes}
        if out and whole != out[0] and self._inductive(whole, start):
            out.append(whole)
        return out

    def _inductive(self, a: dict, start: dict) -> bool:
        image = self._post(a)
        return all(_within(image[m], a[m]) and _within(start[m], a[m])
                   for m in self.modes)

    def _widen(self, old: Box, new: Box) -> Box:
        if old is None or new is None:
            return new if old is None else old
        out = []
        for (ol, oh), (nl, nh) in zip(old, new):
            if nl is not None and ol is not None and nl >= ol:
                lo = ol
            else:
                lo = None if nl is None else max(
                    (t for t in self.thresholds if t <= nl), default=None)
            if nh is not None and oh is not None and nh <= oh:
                hi = oh
            else:
                hi = None if nh is None else min(
                    (t for t in self.thresholds if t >= nh), default=None)
            out.append((lo, hi))
        return out

    def _rounded(self, box: Box, q: Fraction) -> Box:
        if box is None:
            return None
        out = []
        for i, (lo, hi) in zip(self.nums, box):
            g = max(q, Fraction(1)) if self.sorts[i].isInteger() else q
            out.append((None if lo is None else math.floor(lo / g) * g,
                        None if hi is None else math.ceil(hi / g) * g))
        return out

    # --- out --------------------------------------------------------------

    def facts(self, a: dict) -> list[str]:
        from ..houdini_solver import smt_lit

        out: list[str] = []
        reached = [m for m in self.modes if a[m] is not None]
        everywhere = None
        for m in reached:
            everywhere = _join(everywhere, a[m])
        for i, (lo, hi) in zip(self.nums, everywhere or []):
            out += _bound(f"s{i}", lo, hi, self.sorts[i], smt_lit)
        if not self.bits:
            return out
        for m in self.modes:
            lits = [f"s{b}" if v else f"(not s{b})" for b, v in zip(self.bits, m)]
            guard = lits[0] if len(lits) == 1 else f"(and {' '.join(lits)})"
            if a[m] is None:
                out.append(f"(not {guard})")
                continue
            for i, (lo, hi), (glo, ghi) in zip(self.nums, a[m], everywhere):
                out += [f"(=> {guard} {f})" for f in _bound(
                    f"s{i}", lo if lo != glo else None,
                    hi if hi != ghi else None, self.sorts[i], smt_lit)]
        return out


def _bound(col, lo, hi, sort, smt_lit) -> list[str]:
    out = []
    if lo is not None:
        out.append(f"(<= {smt_lit(lo, sort)} {col})")
    if hi is not None:
        out.append(f"(<= {col} {smt_lit(hi, sort)})")
    return out


def _join(a: Box, b: Box) -> Box:
    if a is None or b is None:
        return b if a is None else a
    return [(None if al is None or bl is None else min(al, bl),
             None if ah is None or bh is None else max(ah, bh))
            for (al, ah), (bl, bh) in zip(a, b)]


def _meet(a: Box, b: Box) -> Box:
    """`a` tightened to `b`'s bounds; a mode `b` does not reach is dropped."""
    if a is None or b is None:
        return None
    return [(bl if al is None or (bl is not None and bl > al) else al,
             bh if ah is None or (bh is not None and bh < ah) else ah)
            for (al, ah), (bl, bh) in zip(a, b)]


def _within(inner: Box, outer: Box) -> bool:
    if inner is None:
        return True
    if outer is None:
        return False
    for (il, ih), (ol, oh) in zip(inner, outer):
        if ol is not None and (il is None or il < ol):
            return False
        if oh is not None and (ih is None or ih > oh):
            return False
    return True


def _moved(a: dict, b: dict) -> Fraction:
    """The most any bound moved from `a` to `b`; a bound appearing is
    infinitely far, and a mode dropping out is a move too."""
    far = Fraction(10 ** 9)
    worst = Fraction(0)
    for m in a:
        if (a[m] is None) != (b[m] is None):
            worst = far
            continue
        if a[m] is None:
            continue
        for x, y in zip(a[m], b[m]):
            for u, v in zip(x, y):
                if (u is None) != (v is None):
                    worst = far
                elif u is not None:
                    worst = max(worst, abs(u - v))
    return worst


def _literals(terms) -> set:
    """Every numeric literal in `terms`."""
    out: set = set()
    seen: set = set()
    stack = list(terms)
    while stack:
        t = stack.pop()
        if t.getId() in seen:
            continue
        seen.add(t.getId())
        if t.getKind() == Kind.CONST_INTEGER:
            out.add(Fraction(t.getIntegerValue()))
        elif t.getKind() == Kind.CONST_RATIONAL:
            out.add(Fraction(t.getRealValue()))
        stack.extend(t)
    return out


def _fraction(v) -> Fraction:
    return Fraction(v.as_long()) if z3.is_int_value(v) else Fraction(
        v.numerator_as_long(), v.denominator_as_long())


def _z3_num(v: Fraction, sort, z):
    if sort.isInteger():
        return z3.IntVal(int(v), z)
    return z3.RealVal(f"{v.numerator}/{v.denominator}", z)
