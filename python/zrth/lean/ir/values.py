"""The values a module's constants hold, as the backends read them."""

from __future__ import annotations

from fractions import Fraction


def decimal_fraction(value) -> Fraction:
    """`value` as an exact fraction, a float read by its decimal form:
    `0.1` is 1/10, what the number was written as and what Lean makes of
    a decimal literal. The SMT encoding builds its constants from this
    too, so the two state the same number."""
    if isinstance(value, float):
        return Fraction(repr(value))
    return Fraction(value)
