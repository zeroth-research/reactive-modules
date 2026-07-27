"""Regression tests for the Farkas verifier (``benchmarks.svcomp._farkas``).

These pin the properties the emitted proofs rest on:

  * ``affine_coeffs`` returns coefficients only for genuinely affine input. Its
    0/1 sampling alone would accept an ``ite`` (each sample is a constant) and
    return a linear fit for a branching transition, which would then be
    certified — so the rejection is a soundness property, not a nicety.
  * ``enumerate_paths`` expands a branching body into ite-free paths whose
    guards partition the original guard.
  * a certificate returned by ``certify_decrease`` satisfies the three Farkas
    conditions on its own system.
"""
import numpy as np
import pytest
import z3

from benchmarks.svcomp._farkas import (
    affine_coeffs,
    atom_rows,
    certify_decrease,
    enumerate_paths,
    find_infeasibility_certificate,
)

x, y = z3.Ints("x y")


# --- affine_coeffs: exact on affine input, rejects everything else ----------

@pytest.mark.parametrize("expr, coeffs, const", [
    (x + 1, [1, 0], 1),
    (2 * x + 3 * y - 4, [2, 3], -4),
    (z3.IntVal(7), [0, 0], 7),
    (-x, [-1, 0], 0),
])
def test_affine_coeffs_exact(expr, coeffs, const):
    assert affine_coeffs(expr, [x, y]) == (coeffs, const)


@pytest.mark.parametrize("expr", [
    z3.If(x > 0, x - 1, x + 1),              # in-loop branch: samples to a constant
    z3.If(x > 0, z3.If(y > 0, x, y), x),     # nested branch
    x * y,                                   # nonlinear
])
def test_affine_coeffs_rejects_non_affine(expr):
    """Sampling at 0/1 alone would return a bogus linear fit for these."""
    with pytest.raises(ValueError):
        affine_coeffs(expr, [x, y])


def test_affine_coeffs_rejects_free_symbol():
    """A per-iteration nondet input is not affine over the state columns."""
    with pytest.raises(ValueError):
        affine_coeffs(x + z3.Int("nondet"), [x, y])


def test_atom_rows_drops_non_linear_atom():
    """Non-half-space atoms are skipped, which only weakens the LP."""
    assert atom_rows(x * y <= 0, [x, y]) == []
    assert atom_rows(x != 0, [x, y]) == []
    assert atom_rows(x <= 3, [x, y]) == [([1, 0], 3)]


# --- enumerate_paths -------------------------------------------------------

def test_enumerate_paths_splits_branch():
    """`if (x>0) x-1 else x+1` under `x != 0` becomes two ite-free paths whose
    guards partition the original guard."""
    guard = x != 0
    body = [z3.If(x > 0, x - 1, x + 1)]
    paths = enumerate_paths(body, guard)
    assert len(paths) == 2
    for pguard, pbody in paths:
        assert affine_coeffs(pbody[0], [x])          # affine: does not raise
        s = z3.Solver(); s.add(pguard, z3.Not(guard))
        assert s.check() == z3.unsat                 # each path refines the guard
    s = z3.Solver(); s.add(guard, z3.Not(z3.Or([g for g, _ in paths])))
    assert s.check() == z3.unsat                     # together they cover it


def test_enumerate_paths_keeps_affine_body_single():
    guard = x > 0
    paths = enumerate_paths([x - 1], guard)
    assert len(paths) == 1
    assert paths[0][0] is guard


# --- certificates ----------------------------------------------------------

def test_find_infeasibility_certificate_on_infeasible_system():
    """x <= -1 and -x <= -1 is infeasible; the multipliers witness it."""
    A, b = [[1], [-1]], [-1, -1]
    y_cert = find_infeasibility_certificate(A, b)
    assert y_cert is not None
    assert all(v >= 0 for v in y_cert)                                  # y >= 0
    assert sum(y_cert[i] * A[i][0] for i in range(2)) == 0              # Aᵀy = 0
    assert sum(y_cert[i] * b[i] for i in range(2)) < 0                  # b·y < 0


def test_find_infeasibility_certificate_none_when_feasible():
    assert find_infeasibility_certificate([[1]], [5]) is None


def test_certify_decrease_certificates_are_valid():
    """Every cell certificate of a certified loop satisfies the three Farkas
    conditions on its own system — the facts ``farkas_sound`` consumes."""
    # V(s) = relu(x) over the loop `while (x > 0) x = x - 1`
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    s = [z3.Int("x")]
    sp = [z3.If(z3.Int("x") > 0, z3.Int("x") - 1, z3.Int("x"))]
    res = certify_decrease(layers, s, sp, z3.Int("x") > 0, (), 1.0)
    assert res.verified, res.status
    cells = [c for p in res.certificates for c in p.cells]
    assert cells
    for c in cells:
        n = len(c.A[0])
        assert all(v >= 0 for v in c.y)
        for j in range(n):
            assert sum(c.y[i] * c.A[i][j] for i in range(len(c.A))) == 0
        assert sum(c.y[i] * c.b[i] for i in range(len(c.b))) < 0
