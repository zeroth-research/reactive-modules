"""Ranking functions fitted to rounds, in a few fixed shapes (`Ranks`),
and what it is for one to drop across a round (`drops`).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from itertools import product

from ...core.errors import PredicateRefused
from ...smt.claim import SynthContext
from ...smt.script import affine_smt, smt_int, smt_real
from ..common.synth import Column, columns
from .candidates import reading
from .evaluate import Evaluator, Unevaluable

# The shapes are fitted to at most this many rounds, with at most this many
# forms on each side of a branch (`infer.common.synth.split_conditions` says
# which branches).
_MAX_OUTSIDE = 1000
_MAX_BRANCH_FORMS = 6

# Up to this many columns, every {-1, 0, 1} combination is a form; past it,
# one or two columns at a time. There is no third rung, and a cap on the
# pairs was measured and dropped: they are quadratic in the columns, but on
# the widest module here -- a 32-element vector, so 2048 forms against 64 --
# dropping them saves 2.3 s of a 5.5 s search and changes no answer on any
# of the seven matrix-shaped modules. A guard that costs reach and buys
# nothing measurable is not worth the rung.
_MAX_DENSE = 5

# Constant-derived shifts offered above a fitted one, per form.
_MAX_SHIFTS = 2


class Ranks:
    """Ranking functions fitted to rounds, in a few fixed shapes.

    Each shape is an affine form of the state's integer readings -- one
    column, a sum or difference of two, or `K*x + y` for a lexicographic
    pair -- and the *shift* that makes it at least one on every round that
    has to drop is read off those rounds rather than enumerated. A piecewise
    rank `(ite c f g)` fits a form to each side of `c` the same way, and the
    offset between the sides from the rounds that cross from one to the
    other. Every shape is also offered zeroed where the property holds,
    since `hrank` never constrains those states.
    """

    def __init__(self, ctx: SynthContext, ev: Evaluator, prp_src: str,
                 splits: list[str], constants: tuple[int, ...] = (),
                 rationals: tuple = (), scale: int = 1):
        self.ctx, self.ev, self.prp_src = ctx, ev, prp_src
        # What a real-valued form is multiplied by before it is floored, and
        # whether it has to be: a state of `Int` and `Bool` is read as it is.
        self.scale = max(1, int(scale))
        self.cols = columns(ctx)
        self.real = any(c.sort.isReal() for c in self.cols)
        # Shifts a guard's constant suggests: `c` and one past it. A shift
        # fitted to sampled rounds is only as low as the lowest one sampled,
        # and the round that needs the largest shift is usually the corner
        # the guard names -- `y <= 100 && z <= x` needs `101 + x - y - z`,
        # where samples that never hit `y = 100, z = x` exactly fit `95`.
        # A real literal suggests one at the scale the rank is floored at.
        shifts = set(constants) | {math.floor(self.scale * Fraction(q))
                                   for q in rationals}
        self.shift_pool = sorted({c + d for c in shifts for d in (0, 1)})
        # The env the evaluator wants is keyed by *component*, which is what
        # a state is a tuple of; the forms below are over columns, which is
        # what a matrix-shaped component has several of.
        self.names = ctx.names
        n = len(self.cols)
        self.readings = [self._reading(c) for c in self.cols]
        if n <= _MAX_DENSE:
            # Every form with coefficients in {-1, 0, 1}: `100 - y + x - z`
            # ranks `ColonSipma-TACAS2001-Fig1`, and no pair of its three
            # columns does.
            forms = [v for v in product((1, -1, 0), repeat=n) if any(v)]
        else:
            forms = []
            for i in range(n):
                for sign in (1, -1):
                    forms.append(tuple(sign if k == i else 0 for k in range(n)))
            for i in range(n):
                for j in range(i + 1, n):
                    for si, sj in ((1, -1), (-1, 1), (1, 1), (-1, -1)):
                        forms.append(tuple(si if k == i else sj if k == j else 0
                                           for k in range(n)))
        forms.sort(key=lambda v: sum(map(abs, v)))
        self.forms = forms
        self.splits = []
        for src in splits:
            try:
                self.splits.append(
                    (src, ctx.enc.parse(src, "branch condition")))
            except PredicateRefused:
                continue
        self._truth: dict = {}

    def _reading(self, col: Column) -> str:
        """`col` in the arithmetic this rank's forms are summed in.

        Integer, unless some column is Real -- then every reading is
        lifted to Real, because a sum has one sort and `to_real` is the only
        way an `Int` column joins it.
        """
        if not self.real:
            return reading(col)
        if col.sort.isBoolean():
            return f"(ite {col.src} 1.0 0.0)"
        return col.src if col.sort.isReal() else f"(to_real {col.src})"

    def _src(self, shift: int, form) -> str:
        """The shape as the certificate carries it: an `Int`-sorted term.

        Over a real form that is `(to_int (scale*form + shift))`. Flooring
        after the shift is the same number as flooring before it -- the
        shift is a whole number -- which is what lets everything downstream
        fit shifts and offsets in integers.
        """
        if not self.real:
            return affine_smt(shift, form, self.readings)
        coeffs = [self.scale * c for c in form]
        if not any(coeffs):
            return smt_int(shift)
        return f"(to_int {affine_smt(shift, coeffs, self.readings, lit=smt_real)})"

    def holds(self, state, term=None) -> bool:
        key = (state, None if term is None else term.getId())
        if key not in self._truth:
            try:
                self._truth[key] = bool(self.ev(
                    self.ctx.prp if term is None else term,
                    dict(zip(self.names, state))))
            except Unevaluable:
                self._truth[key] = term is None
        return self._truth[key]

    def fit(self, rounds: list, keep=()) -> list[str]:
        """Every shape that drops on all of `rounds` and `keep`, smallest
        first. `rounds` may be thinned to :data:`_MAX_OUTSIDE`; `keep` --
        the rounds a shape was already seen to fail on -- never is."""
        outside = [(s, sp) for s, sp in rounds if not self.holds(s)]
        if len(outside) > _MAX_OUTSIDE:
            outside = random.Random(0).sample(outside, _MAX_OUTSIDE)
        outside += [(s, sp) for s, sp in keep if not self.holds(s)]
        if not outside:
            # Nothing to rank on. A constant is then the first thing worth
            # asking: it proves the property holds wherever the invariant does.
            return ["0"]
        forms = list(self.forms) + self._lex(outside)
        out: list[str] = []
        for form in dict.fromkeys(forms):
            shift = 1 - min(self._at(form, s) for s, _ in outside)
            above = [c for c in self.shift_pool if c > shift][:_MAX_SHIFTS]
            for sh in dict.fromkeys((0, shift, *above)):
                src = self._src(sh, form)
                out += self._accept(src, lambda st, f=form, sh=sh: self._at(f, st) + sh,
                                    outside)
        for c_src, c_term in self.splits:
            out += self._piecewise(c_src, c_term, outside)
        out = list(dict.fromkeys(out))
        out.sort(key=len)
        return out

    # --- shapes -----------------------------------------------------------

    def _lex(self, outside) -> list:
        """`K*x + y`, with `K` wider than the range `y` takes on these rounds."""
        n = len(self.cols)
        out = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                unit = tuple(1 if k == j else 0 for k in range(n))
                ys = [self._at(unit, s) for s, _ in outside]
                width = max(ys) - min(ys) + 1
                if width > 1000:
                    continue
                K = max(2, width + 1)
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    out.append(tuple(si * K if k == i else sj if k == j else 0
                                     for k in range(n)))
        return out

    def _piecewise(self, c_src, c_term, outside) -> list[str]:
        side = {b: [(s, sp) for s, sp in outside
                    if self.holds(s, c_term) == b] for b in (True, False)}
        if not side[True] or not side[False]:
            return []
        zero = tuple(0 for _ in self.cols)
        fits = {}
        for b in (True, False):
            stays = [(s, sp) for s, sp in side[b] if self.holds(sp, c_term) == b]
            fits[b] = [f for f in [zero] + self.forms
                       if all(self._at(f, sp) < self._at(f, s) for s, sp in stays)
                       ][:_MAX_BRANCH_FORMS]
        out = []
        for f in fits[True]:
            for g in fits[False]:
                sf = 1 - min(self._at(f, s) for s, _ in side[True])
                sg = 1 - min(self._at(g, s) for s, _ in side[False])
                offsets = self._offsets(c_term, f, g, sf, sg, side)
                if offsets is None:
                    continue
                sf, sg = sf + offsets[0], sg + offsets[1]
                src = (f"(ite {c_src} {self._src(sf, f)} "
                       f"{self._src(sg, g)})")

                def rank(st, f=f, g=g, sf=sf, sg=sg):
                    if self.holds(st, c_term):
                        return self._at(f, st) + sf
                    return self._at(g, st) + sg

                out += self._accept(src, rank, outside)
        return out

    def _offsets(self, c_term, f, g, sf, sg, side):
        """How far to lift each side so that crossing to the other drops."""
        of = og = 0
        for _ in range(6):
            moved = False
            for s, sp in side[True]:
                if not self.holds(sp, c_term):
                    need = max(self._at(g, sp) + sg + og, 0) + 1 - (self._at(f, s) + sf)
                    if need > of:
                        of, moved = need, True
            for s, sp in side[False]:
                if self.holds(sp, c_term):
                    need = max(self._at(f, sp) + sf + of, 0) + 1 - (self._at(g, s) + sg)
                    if need > og:
                        og, moved = need, True
            if not moved:
                return of, og
        return None                                  # each side lifts the other

    def _accept(self, src, rank, outside) -> list[str]:
        """`src`, and its zeroed form, for whichever drops on every round."""
        out = []
        before = [(rank(s), sp) for s, sp in outside]
        if all(drops(r, rank(sp)) for r, sp in before):
            out.append(src)
        if all(drops(r, 0 if self.holds(sp) else rank(sp)) for r, sp in before):
            out.append(f"(ite {self.prp_src} 0 {src})")
        return out

    def _at(self, form, state) -> int:
        """What `_src(0, form)` evaluates to at `state` -- floored, if real."""
        total = 0
        for c, col in zip(form, self.cols):
            if c:
                v = col.at(state)
                total += c * (int(v) if isinstance(v, bool) else v)
        return math.floor(self.scale * total) if self.real else int(total)


def drops(before: int, after: int) -> bool:
    """`Int.toNat after < Int.toNat before`."""
    return max(after, 0) < max(before, 0)
