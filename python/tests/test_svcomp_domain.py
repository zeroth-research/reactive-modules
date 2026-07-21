"""Regression tests for the derived loop guard (``benchmarks.svcomp._domain``).

The DSL encodes a loop as ``update = ite(guard, body, self)``, and ``_domain``
derives the verification domain from that update — there is no declared
``domain``. These tests pin the derivation so that a future encoding which
breaks the ``ite(guard, body, self)`` convention fails loudly rather than
silently producing the wrong (or no) guard.
"""
import z3
import pytest

from benchmarks.svcomp import discover
from benchmarks.svcomp._domain import domain

_BENCHES = {b.name: b for b in discover()}


def _guard(name: str):
    """The derived loop guard of a benchmark, over z3.Int(state-name) symbols."""
    b = _BENCHES[name]
    return domain(b)({n: z3.Int(n) for n in b.state})


def test_guard_derivable_for_every_benchmark():
    """Every encoding must yield a boolean loop guard (extraction never fails)."""
    for b in discover():
        g = domain(b)({n: z3.Int(n) for n in b.state})
        assert z3.is_bool(g), f"{b.name}: derived guard is not boolean"


@pytest.mark.parametrize("name, expected", [
    ("AliasDarteFeautrierGonnord-SAS2010-ndecr", z3.Int("i") > 1),
    ("HeizmannHoenickeLeikePodelski-ATVA2013-Fig8", z3.Int("x") >= 0),
    ("HeizmannHoenickeLeikePodelski-ATVA2013-Fig9",
     z3.And(z3.Int("x") >= 0, z3.Int("z") == 1)),
])
def test_known_guards(name, expected):
    """Derived guards match the known loop conditions (z3-equivalence)."""
    g = _guard(name)
    s = z3.Solver()
    s.add(g != expected)
    assert s.check() == z3.unsat, f"{name}: derived {z3.simplify(g)} != {expected}"
