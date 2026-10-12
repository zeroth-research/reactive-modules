"""SMT-LIB out: the script printer, and the literals written into one.

Every script handed to a solver outside this process, or left in
`artifacts/` for a reader, is printed by :func:`script`.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from re import Match, compile as _compile

from .terms import free_constants


# ══════════════════════════════════════════════════════════════════════════
# Literals
# ══════════════════════════════════════════════════════════════════════════
#
# Vampire's SMT-LIB front end sorts literals strictly, and cvc5's printer does
# not write for it: a rational prints as `(/ 1 2)`, whose arguments are Int
# ("invalid sort $int for interpretation /"), and an Int numeral anywhere a
# Real is expected is a parse error rather than a coercion. So every real
# literal written for a script is a decimal, and every term cvc5 prints is
# passed through `decimals` on its way into one. cvc5 reads a decimal as
# happily as it reads `(/ 1 2)`, so a candidate's source uses them too.


def smt_int(value: int) -> str:
    """An integer literal. SMT-LIB has no negative numerals, so one is a negation."""
    value = int(value)
    return str(value) if value >= 0 else f"(- {-value})"


def smt_real(value) -> str:
    """A Real literal: a decimal where one is exact, else a quotient of two.

    `1/2` is `0.5` and `1/3` is `(/ 1.0 3.0)` -- a decimal cannot say the
    second, and rounding it would state a different obligation.
    """
    q = Fraction(value)
    sign, q = ("(- ", -q) if q < 0 else ("", -(-q))
    close = ")" if sign else ""
    rest = q.denominator
    for factor in (2, 5):
        while rest % factor == 0:
            rest //= factor
    if rest != 1:
        return f"{sign}(/ {q.numerator}.0 {q.denominator}.0){close}"
    places = 0
    while 10 ** places % q.denominator:
        places += 1
    digits = str(q.numerator * 10 ** places // q.denominator).rjust(places + 1, "0")
    whole, frac = digits[:len(digits) - places], digits[len(digits) - places:]
    return f"{sign}{whole}.{frac or '0'}{close}"


def smt_lit(value, sort) -> str:
    """`value` as a literal of `sort`: an Int numeral, or a decimal."""
    return smt_real(value) if sort.isReal() else smt_int(int(value))


# `(/ 1 2)`, `(/ (- 1) 4)`: a rational as cvc5 prints one. Integer division
# prints as `div`, so a `/` whose arguments are both numerals is always this.
_RATIONAL = _compile(r"\(/ (?:\(- (\d+)\)|(\d+)) (\d+)\)")


def decimals(text: str) -> str:
    """`text` with cvc5's rational literals rewritten as Vampire reads them."""
    def fix(m: Match) -> str:
        neg, num, den = m.groups()
        return smt_real(Fraction(-int(neg) if neg else int(num), int(den)))

    return _RATIONAL.sub(fix, text)


# ══════════════════════════════════════════════════════════════════════════
# Scripts
# ══════════════════════════════════════════════════════════════════════════


def declarations(terms: Iterable, *, skip=()) -> list[str]:
    """A `declare-fun` for each free constant `terms` mention, by symbol,
    less the names in `skip`."""
    seen = free_constants(terms)
    return [f"(declare-fun {sym} () {seen[sym].getSort()})"
            for sym in sorted(seen) if sym not in skip]


def script(asserts: Iterable = (), *, defines: Iterable = (),
           mentions: "Iterable | None" = None, logic: str = "ALL",
           command: str = "assert", spaced: bool = True) -> str:
    """`asserts` as a script that runs on its own under any SMT-LIB solver.

    `defines` are `(name, params, sort, body)`, each `params` a list of
    `(name, sort)`, emitted as `define-fun` ahead of the assertions.
    Declarations are collected from `mentions` -- by default the define
    bodies and the assertions, which are then terms -- less every name a
    define binds, so a caller states what it wants said and not what it
    depends on. An assertion may be text, for a form the caller writes
    itself (a `:named` one, say), and `mentions` then says what it reads.
    `command` is what each assertion is put under; `spaced` separates the
    sections with a blank line, for a reader.
    """
    asserts, defines = list(asserts), list(defines)
    if mentions is None:
        mentions = [body for _n, _p, _s, body in defines] + asserts
    bound = ({name for name, _p, _s, _b in defines}
             | {p for _n, params, _s, _b in defines for p, _ps in params})
    decls = declarations(mentions, skip=bound)
    gap = [""] if spaced else []
    lines = [f"(set-logic {logic})", *gap, *decls]
    if decls:
        lines += gap
    for name, params, sort, body in defines:
        args = " ".join(f"({p} {srt})" for p, srt in params)
        lines.append(f"(define-fun {name} ({args}) {sort} {body})")
    if defines:
        lines += gap
    lines += [f"({command} {term})" for term in asserts]
    lines += [*gap, "(check-sat)"]
    return "\n".join(lines) + "\n"
