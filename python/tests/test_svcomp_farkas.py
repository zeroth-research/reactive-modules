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
  * the cells of a certified path partition the guard, and every mask
    under-approximates ``V`` — the two facts the asymmetric bound rests on.
"""
import itertools

import numpy as np
import pytest
import z3

from benchmarks.svcomp import _farkas
from benchmarks.svcomp._farkas import (
    Net,
    _pre_activations,
    affine_coeffs,
    atom_rows,
    certify_decrease,
    enumerate_paths,
    find_infeasibility_certificate,
    masked_value,
    output_weights,
    split_guard,
    strict_signs,
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


def test_split_guard_covers_exactly_and_does_not_overlap():
    """The emitted ``RawStep`` is the union of the paths, so the pieces must cover
    the guard exactly — under-covering would state termination of a subset of the
    program's steps — and stay disjoint, so no state is certified twice."""
    x, m, i, j, n = z3.Ints("x m i j n")
    for guard in (x != m, z3.Or(i < m, j < n), z3.And(x > 0, z3.Or(i < m, j < n))):
        pieces = split_guard(guard)
        assert len(pieces) > 1, f"{guard} should split"
        union = z3.Solver()
        union.add(z3.Xor(z3.Or(*pieces), guard))
        assert union.check() == z3.unsat, f"{guard}: pieces are not exactly the guard"
        for a in range(len(pieces)):
            for b in range(a + 1, len(pieces)):
                overlap = z3.Solver()
                overlap.add(pieces[a], pieces[b])
                assert overlap.check() == z3.unsat, f"{guard}: pieces {a},{b} overlap"


def test_split_guard_leaves_a_convex_guard_alone():
    """A conjunction of half-spaces already reaches the LP intact, so it is not
    split — the path count (and the emitted proof) stays unchanged."""
    x, m = z3.Ints("x m")
    for guard in (x > 0, z3.And(x > 0, x < m)):
        assert split_guard(guard) == [guard]


def test_disjunctive_guard_is_certifiable_only_once_split(monkeypatch):
    """``atom_rows`` drops a disjunction, so an un-split disjunctive guard reaches
    the LP with no rows at all and its cells cannot be certified. Pinned against
    the un-split domain, since that gap is the reason the split exists."""
    i, m, k = z3.Ints("i m k")
    # V = relu(m - i) + relu(k - i) over (i, m, k), for
    # `while (i < m or i < k) { i = i + 1 }`
    layers = [(np.array([[-1, 1, 0], [-1, 0, 1]]), np.array([0, 0])),
              (np.array([[1, 1]]), np.array([0]))]
    s = [i, m, k]
    guard = z3.Or(i < m, i < k)
    sp = [z3.If(guard, i + 1, i), m, k]

    assert certify_decrease(Net.from_layers(layers), s, sp, guard, (), 1.0).verified

    monkeypatch.setattr(_farkas, "split_guard", lambda g, max_pieces=16: [g])
    unsplit = certify_decrease(Net.from_layers(layers), s, sp, guard, (), 1.0)
    assert not unsplit.verified, "the un-split disjunctive guard should not certify"


def _conditional_loop():
    """`while (x <= n) { if (b >= 1) x = x + t; else x = x - t; }` with `t` set to
    `1` / `-1` before the loop according to `b`. `x` climbs by one either way, so
    `V = relu(n - x + 1)` ranks it once the conditional facts about `t` are usable."""
    x, n, b, t = z3.Ints("x n b t")
    layers = [(np.array([[-1, 1, 0, 0]]), np.array([1])),
              (np.array([[1]]), np.array([0]))]
    guard = x <= n
    s = [x, n, b, t]
    sp = [z3.If(guard, z3.If(b >= 1, x + t, x - t), x), n, b, t]
    invariants = (z3.Or(b <= 0, t == 1), z3.Or(b >= 1, t == -1))
    return layers, s, sp, guard, invariants


def test_conditional_invariant_gives_rows_once_the_guard_picks_a_side():
    """`atom_rows` drops a conditional invariant, so the entailed side is what the LP
    gets instead — and it must genuinely follow from the domain."""
    _, s, _, guard, invariants = _conditional_loop()
    x, n, b, t = s
    got = _farkas._entailed_atoms(z3.And(z3.And(guard, b >= 1), *invariants),
                                 invariants, s)
    solver = z3.Solver()
    solver.add(z3.And(guard, b >= 1, *invariants), z3.Not(z3.And(*got)))
    assert got, "the guard settles b, so t == 1 should be entailed"
    assert solver.check() == z3.unsat, "entailed atoms must follow from the domain"
    # and on the other side the opposite fact is the entailed one
    other = _farkas._entailed_atoms(z3.And(z3.And(guard, b <= 0), *invariants),
                                    invariants, s)
    assert any("-1" in str(z3.simplify(a)) for a in other), other


def test_conditional_invariant_is_what_certifies(monkeypatch):
    """Pinned against the same run without the entailed rows, which is the state the
    LP was in before."""
    layers, s, sp, guard, invariants = _conditional_loop()
    assert certify_decrease(Net.from_layers(layers), s, sp, guard, invariants, 1.0).verified

    monkeypatch.setattr(_farkas, "_entailed_atoms", lambda *a, **k: ())
    without = certify_decrease(Net.from_layers(layers), s, sp, guard, invariants, 1.0)
    assert not without.verified, "without the entailed rows this should not certify"


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


def test_cells_partition_the_domain():
    """A cell is pinned by the successor signs alone, strict on one side and
    non-strict on the other, so the cells cover the guard and no state lies in two
    of them. Coverage in the emitted proof rests on this."""
    layers = [(np.array([[1], [1]]), np.array([0, -3])),
              (np.array([[1, 1]]), np.array([0]))]
    res = certify_decrease(Net.from_layers(layers), [x], [z3.If(x > 0, x - 1, x)], x > 0, (), 1.0)
    assert res.verified, res.status
    for path in res.certificates:
        regions = []
        for c in path.cells:
            lits = list(strict_signs(_pre_activations(Net.from_layers(layers), list(path.body)),
                                     c.pattern_sp))
            if c.pattern_s is not None:
                lits += list(strict_signs(_pre_activations(Net.from_layers(layers), [x]),
                                          c.pattern_s))
            regions.append(z3.And(*lits))
        covered = z3.Solver()
        covered.add(path.guard, z3.Not(z3.Or(*regions)))
        assert covered.check() == z3.unsat, "cells leave part of the guard uncovered"
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                overlap = z3.Solver()
                overlap.add(path.guard, regions[i], regions[j])
                assert overlap.check() == z3.unsat, f"cells {i} and {j} overlap"


def test_mask_never_exceeds_V():
    """Every mask under-approximates ``V``, with no condition on the state. That
    is what lets a cell leave the pre-state activations unconstrained, so it holds
    for masks that match no reachable pattern too."""
    layers = [(np.array([[1], [2]]), np.array([0, -3])),
              (np.array([[1, 2]]), np.array([4]))]
    pres = _pre_activations(Net.from_layers(layers), [x])
    weights = output_weights(Net.from_layers(layers))
    coeffs, bias = weights
    exact = z3.Sum([c * z3.If(p > 0, p, z3.IntVal(0))
                    for c, p in zip(coeffs, pres)]) + bias
    for mask in itertools.product((True, False), repeat=len(pres)):
        solver = z3.Solver()
        solver.add(masked_value(pres, weights, mask) > exact)
        assert solver.check() == z3.unsat, f"mask {mask} exceeds V"


def test_mixed_output_weights_and_bias_are_carried():
    """The trained nets only ever have uniform output weights and no output bias,
    so nothing else pins the per-unit scaling. Both affine forms are recomputed
    here from the raw matrices: the floor over the units its mask keeps, the
    ceiling over the successor pattern. Getting the ceiling wrong overstates the
    drop, so it is checked too."""
    layers = [(np.array([[1], [1]]), np.array([0, -3])),
              (np.array([[1, 2]]), np.array([5]))]
    (W1, b1), (W2, b2) = layers
    res = certify_decrease(Net.from_layers(layers), [x], [z3.If(x > 0, x - 1, x)], x > 0, (), 1.0)
    assert res.verified, res.status
    for c in (c for p in res.certificates for c in p.cells):
        # z_j(s) = W1[j]·x + b1[j];  z_j(s') is the same at x - 1
        for pattern, affine, shift in ((c.mu, c.pre_affine, 0),
                                       (c.pattern_sp, c.post_affine, -1)):
            kept = [j for j, on in enumerate(pattern) if on]
            coeff = sum(int(W2[0][j]) * int(W1[j][0]) for j in kept)
            const = int(b2[0]) + sum(int(W2[0][j]) * (int(b1[j]) + shift)
                                     for j in kept)
            assert affine == ((coeff,), const), (pattern, affine, coeff, const)


def test_margin_the_rank_cannot_meet_is_rejected():
    """``relu(x)`` drops by 1 at ``x = 1``, so a margin of 2 is a real
    counterexample and has to be reported as one. Two checks can catch it — the
    witness-level one in ``_certify_path`` and the region-wide prune in
    ``_certify_cell`` — and the guarantee holds as long as either does."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    res = certify_decrease(Net.from_layers(layers), [x], [z3.If(x > 0, x - 2, x)], x > 0, (), 2.0)
    assert not res.verified
    assert res.status == "FAILED(decrease)"


def test_certify_decrease_certificates_are_valid():
    """Every cell certificate of a certified loop satisfies the three Farkas
    conditions on its own system — the facts ``farkas_sound`` consumes."""
    # V(s) = relu(x) over the loop `while (x > 0) x = x - 1`
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    s = [z3.Int("x")]
    sp = [z3.If(z3.Int("x") > 0, z3.Int("x") - 1, z3.Int("x"))]
    res = certify_decrease(Net.from_layers(layers), s, sp, z3.Int("x") > 0, (), 1.0)
    assert res.verified, res.status
    cells = [c for p in res.certificates for c in p.cells]
    assert cells
    for c in cells:
        n = len(c.A[0])
        assert all(v >= 0 for v in c.y)
        for j in range(n):
            assert sum(c.y[i] * c.A[i][j] for i in range(len(c.A))) == 0
        assert sum(c.y[i] * c.b[i] for i in range(len(c.b))) < 0
