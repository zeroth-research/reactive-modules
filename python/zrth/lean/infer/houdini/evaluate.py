"""A term's value at a concrete state, without a solver.

What the simulation runs the module with, and what a candidate is
checked against on a sampled round. `Evaluator` compiles the terms
once. A term with no value here -- a division by zero, an op the table
does not cover -- raises `Unevaluable`, and nothing is claimed about
that state.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ...smt.terms import free_constants

import cvc5
from cvc5 import Kind


def _term(tm, sort, value):
    """`value` as a cvc5 literal of `sort`.

    A matrix-shaped value is a Python tuple, element by element, which is
    how the simulation carries one and the only shape `tuple.select` reads
    back.
    """
    if sort.isTuple():
        return tm.mkTuple([_term(tm, s, v)
                           for s, v in zip(sort.getTupleSorts(), value)])
    if sort.isBoolean():
        return tm.mkBoolean(bool(value))
    if sort.isReal():
        q = Fraction(value)
        return tm.mkReal(q.numerator, q.denominator)
    return tm.mkInteger(int(value))


class Unevaluable(Exception):
    """A term with no value here: a division by zero, or an op not covered."""


# The value of a division by zero. SMT-LIB leaves it unspecified, so nothing
# is claimed about a state that needs one: it propagates, and a result that
# depends on it is `Unevaluable`.
_UNDEF = object()


def _strict(fn):
    def apply(vals):
        return _UNDEF if any(v is _UNDEF for v in vals) else fn(vals)
    return apply


def _ite(vals):
    c = vals[0]
    return _UNDEF if c is _UNDEF else vals[1] if c else vals[2]


def _and(vals):
    if any(v is False for v in vals):
        return False
    return _UNDEF if any(v is _UNDEF for v in vals) else True


def _or(vals):
    if any(v is True for v in vals):
        return True
    return _UNDEF if any(v is _UNDEF for v in vals) else False


def _product(vals):
    out = 1
    for v in vals:
        out *= v
    return out


def _chain(op):
    return _strict(lambda vs: all(op(a, b) for a, b in zip(vs, vs[1:])))


def _division(vals):
    a, b = vals
    if a is _UNDEF or b is _UNDEF or b == 0:
        return _UNDEF
    return (a - a % abs(b)) // b


def _modulus(vals):
    a, b = vals
    if a is _UNDEF or b is _UNDEF or b == 0:
        return _UNDEF
    # SMT-LIB's is Euclidean: the remainder is never negative.
    return a % abs(b)


def _quotient(vals):
    """Real `/`. Exact, on `Fraction`: a float would round a guard's corner."""
    a, b = vals
    if a is _UNDEF or b is _UNDEF or b == 0:
        return _UNDEF
    return Fraction(a) / Fraction(b)


def _tuple(vals):
    """A matrix-shaped value: its elements, in the tuple's own order."""
    return _UNDEF if any(v is _UNDEF for v in vals) else tuple(vals)


def _selector(j: int):
    """Element `j` of one, which is what `((_ tuple.select j) x)` means."""
    def pick(vals):
        return _UNDEF if vals[0] is _UNDEF else vals[0][j]
    return pick


def _selected(t) -> int:
    """Which element `t`, an `APPLY_SELECTOR`, reads.

    By identity against the datatype's own selectors rather than by reading
    the index out of the printed name: the name is cvc5's to spell.
    """
    dt = t[1].getSort().getDatatype()[0]
    for j in range(dt.getNumSelectors()):
        if t[0] == dt[j].getTerm():
            return j
    raise Unevaluable(f"no element for {t}")      # pragma: no cover


_OPS = {
    Kind.ITE: _ite,
    Kind.AND: _and,
    Kind.OR: _or,
    Kind.NOT: _strict(lambda vs: not vs[0]),
    Kind.IMPLIES: lambda vs: _or([_strict(lambda v: not v[0])(vs[:1]), vs[1]]),
    Kind.XOR: _strict(lambda vs: vs[0] != vs[1]),
    Kind.ADD: _strict(sum),
    Kind.SUB: _strict(lambda vs: vs[0] - sum(vs[1:])),
    Kind.NEG: _strict(lambda vs: -vs[0]),
    Kind.MULT: _strict(_product),
    Kind.ABS: _strict(lambda vs: abs(vs[0])),
    Kind.EQUAL: _strict(lambda vs: all(v == vs[0] for v in vs[1:])),
    Kind.DISTINCT: _strict(lambda vs: len(set(vs)) == len(vs)),
    Kind.LT: _chain(lambda a, b: a < b),
    Kind.LEQ: _chain(lambda a, b: a <= b),
    Kind.GT: _chain(lambda a, b: a > b),
    Kind.GEQ: _chain(lambda a, b: a >= b),
    Kind.INTS_DIVISION: _division,
    Kind.INTS_MODULUS: _modulus,
    Kind.DIVISION: _quotient,
    # The Real/Int boundary. `to_real` is the identity on a value this
    # evaluator already keeps exactly; `to_int` is the floor `Int.toNat`
    # sees, and the reason a ranking function over a Real component has to
    # be scaled before it is read.
    Kind.TO_REAL: _strict(lambda vs: Fraction(vs[0])),
    Kind.TO_INTEGER: _strict(lambda vs: math.floor(vs[0])),
    Kind.IS_INTEGER: _strict(lambda vs: Fraction(vs[0]).denominator == 1),
}


class Evaluator:
    """cvc5 terms at concrete values, in Python.

    A term is compiled once into a flat list of operations, one per distinct
    subterm, so a transition that shares its subterms -- the `let`s a deep
    network prints with -- is evaluated in time linear in its DAG rather than
    its tree. The kinds a module's integer transition is made of are Python
    arithmetic; anything else is handed to cvc5's rewriter with the values
    substituted in, which evaluates a ground term without deciding anything.
    """

    def __init__(self, tm):
        self.tm = tm
        self._solver = None
        self._programs: dict = {}

    def __call__(self, term, env: dict):
        return self.many([term], env)[0]

    def many(self, terms, env: dict) -> list:
        """Several terms at one assignment, sharing what they share."""
        key = tuple(t.getId() for t in terms)
        program = self._programs.get(key)
        if program is None:
            program = self._programs[key] = self._compile(terms)
        vals: list = []
        for kind, arg in program[0]:
            if kind == 0:
                vals.append(arg)
            elif kind == 1:
                try:
                    vals.append(env[arg])
                except KeyError as e:
                    raise Unevaluable(f"no value for {arg}") from e
            elif kind == 2:
                vals.append(self._rewrite(arg, env))
            else:
                vals.append(kind([vals[j] for j in arg]))
        out = [vals[i] for i in program[1]]
        if any(v is _UNDEF for v in out):
            raise Unevaluable("division by zero")
        return out

    def _compile(self, terms):
        slot: dict = {}
        ops: list = []
        stack = [(t, False) for t in reversed(list(terms))]
        while stack:
            t, expanded = stack.pop()
            tid = t.getId()
            if tid in slot:
                continue
            k = t.getKind()
            if k == Kind.CONST_INTEGER:
                op = (0, t.getIntegerValue())
            elif k == Kind.CONST_RATIONAL:
                op = (0, t.getRealValue())
            elif k == Kind.CONST_BOOLEAN:
                op = (0, t.getBooleanValue())
            elif k == Kind.CONSTANT:
                op = (1, t.getSymbol())
            elif k in (Kind.APPLY_SELECTOR, Kind.APPLY_CONSTRUCTOR):
                # The tuples a matrix-shaped component and the wires around
                # it are encoded as. Child 0 is the selector or constructor
                # symbol, which stands for the operation rather than for a
                # value, so it is the one child not compiled.
                kids = list(t)[1:]
                if not expanded:
                    stack.append((t, True))
                    stack.extend((c, False) for c in reversed(kids))
                    continue
                fn = (_selector(_selected(t)) if k == Kind.APPLY_SELECTOR
                      else _tuple)
                op = (fn, tuple(slot[c.getId()] for c in kids))
            elif k not in _OPS:
                op = (2, t)
            elif not expanded:
                stack.append((t, True))
                stack.extend((c, False) for c in reversed(list(t)))
                continue
            else:
                op = (_OPS[k], tuple(slot[c.getId()] for c in t))
            slot[tid] = len(ops)
            ops.append(op)
        return ops, [slot[t.getId()] for t in terms]

    def _rewrite(self, t, env):
        tm = self.tm
        if self._solver is None:
            self._solver = cvc5.Solver(tm)
        consts = list(free_constants([t]).values())
        vals = []
        for c in consts:
            v = env.get(c.getSymbol())
            if v is None:
                raise Unevaluable(f"no value for {c.getSymbol()}")
            vals.append(_term(tm, c.getSort(), v))
        r = self._solver.simplify(t.substitute(consts, vals) if consts else t)
        if r.getKind() == Kind.CONST_INTEGER:
            return r.getIntegerValue()
        if r.getKind() == Kind.CONST_RATIONAL:
            return r.getRealValue()
        if r.getKind() == Kind.CONST_BOOLEAN:
            return r.getBooleanValue()
        raise Unevaluable(f"{t.getKind()} does not evaluate to a literal")
