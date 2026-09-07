"""Regression tests for the Farkas verifier (``benchmarks.svcomp._farkas``).

These pin the properties the emitted proofs rest on:

  * ``affine_coeffs`` returns coefficients only for genuinely affine input. Its
    0/1 sampling alone would accept an ``ite`` (each sample is a constant) and
    return a linear fit for a branching transition, which would then be
    certified — so the rejection is a soundness property, not a nicety.
  * ``expand_cases`` expands a branching body and a non-convex guard into cases
    whose guards partition the original guard — what the emitted ``RawStep``
    union rests on.
  * a certificate the verifier returns satisfies the three Farkas conditions on
    its own system.
  * the cells of a certified path partition the guard, and every mask
    under-approximates ``V`` — the two facts the asymmetric bound rests on.
"""
import itertools

import numpy as np
import pytest
import z3

from benchmarks.svcomp import _farkas
from benchmarks.svcomp._bench import INT
from benchmarks.svcomp._verify_ranking import (_v_module, build_obligation,
                                               farkas_cell, system_of)
from tests._fixtures import loop_bench
from zrth import LIA, Module, Sort, Wire, sugar
from zrth.sugar import argmax as dsl_argmax
from zrth.sugar import expr as dsl_expr
from zrth.sugar import ite as dsl_ite
from benchmarks.svcomp._farkas import certify_decrease, read_system
from benchmarks.svcomp._property import Fixpoint
from benchmarks.svcomp._nodes import Node, Unsupported, node_view
from benchmarks.svcomp._farkas import (
    Net,
    _pre_activations,
    affine_coeffs,
    atom_rows,
    expand_cases,
    find_infeasibility_certificate,
    masked_value,
    output_weights,
    strict_signs,
)

x, y = z3.Ints("x y")


def _obligation(state, update, layers, invariants=(), delta=1.0):
    """A real obligation for a compact loop spec — the module is built and walked
    exactly as the pipeline does it (see :mod:`tests._fixtures`)."""
    named = [(f"inv{k}", (lambda st, p=p: p)) for k, p in enumerate(invariants)]
    return build_obligation(loop_bench(state, update), layers, delta, named)


def _decrement(layers, delta=1.0, step=1):
    """`while (x > 0) x = x - step` as an obligation."""
    return _obligation(("x",), lambda v: dsl_ite(v > 0, v - step, v), layers, delta=delta)


def _certify(ob, prop=None):
    """``ob``'s system under ``prop``, defaulting to the obligation's own
    termination property."""
    return certify_decrease(ob.system, prop or ob.prop, ob.net, ob.delta)


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


# --- expand_cases ----------------------------------------------------------

def test_expand_cases_splits_a_branch():
    """`if (x>0) x-1 else x+1` becomes ite-free cases whose guards partition the
    original guard. The `x != 0` guard is split too, so there are more than two."""
    guard = x != 0
    body = [z3.If(x > 0, x - 1, x + 1)]
    paths = expand_cases(guard, body)
    assert len(paths) >= 2
    for pguard, pbody in paths:
        assert affine_coeffs(pbody[0], [x])          # affine: does not raise
        s = z3.Solver(); s.add(pguard, z3.Not(guard))
        assert s.check() == z3.unsat                 # each path refines the guard
    s = z3.Solver(); s.add(guard, z3.Not(z3.Or([g for g, _ in paths])))
    assert s.check() == z3.unsat                     # together they cover it


def test_expand_cases_covers_exactly_and_does_not_overlap():
    """The emitted ``RawStep`` is the union of the cases, so they must cover the
    guard exactly — under-covering would state termination of a subset of the
    program's steps — and stay disjoint, so no state is certified twice."""
    x, m, i, j, n = z3.Ints("x m i j n")
    for guard in (x != m, z3.Or(i < m, j < n), z3.And(x > 0, z3.Or(i < m, j < n))):
        pieces = [g for g, _ in expand_cases(guard, [x])]
        assert len(pieces) > 1, f"{guard} should split"
        union = z3.Solver()
        union.add(z3.Xor(z3.Or(*pieces), guard))
        assert union.check() == z3.unsat, f"{guard}: pieces are not exactly the guard"
        for a in range(len(pieces)):
            for b in range(a + 1, len(pieces)):
                overlap = z3.Solver()
                overlap.add(pieces[a], pieces[b])
                assert overlap.check() == z3.unsat, f"{guard}: pieces {a},{b} overlap"


def test_expand_cases_leaves_a_convex_guard_and_affine_body_alone():
    """A conjunction of half-spaces with an ite-free body already reaches the LP
    intact, so it yields one case — the emitted proof stays unchanged."""
    x, m = z3.Ints("x m")
    for guard in (x > 0, z3.And(x > 0, x < m)):
        assert expand_cases(guard, [x - 1]) == [(guard, [x - 1])]


def test_disjunctive_guard_is_certifiable_only_once_split(monkeypatch):
    """``atom_rows`` drops a disjunction, so an un-split disjunctive guard reaches
    the LP with no rows at all and its cells cannot be certified. Pinned against
    the un-split domain, since that gap is the reason the split exists."""
    i, m, k = z3.Ints("i m k")
    # V = relu(m - i) + relu(k - i) over (i, m, k), for
    # `while (i < m or i < k) { i = i + 1 }`
    layers = [(np.array([[-1, 1, 0], [-1, 0, 1]]), np.array([0, 0])),
              (np.array([[1, 1]]), np.array([0]))]
    ob = _obligation(("i", "m", "k"),
                     lambda c: (dsl_ite((c[0] < c[1]) | (c[0] < c[2]), c[0] + 1, c[0]),
                                c[1], c[2]),
                     layers)
    assert farkas_cell(ob).verified

    monkeypatch.setattr(_farkas, "_convex_alternatives", lambda a: None)
    assert not farkas_cell(ob).verified, \
        "the un-split disjunctive guard should not certify"


def _conditional_loop():
    """`while (x <= n) { if (b >= 1) x = x + t; else x = x - t; }` with `t` set to
    `1` / `-1` before the loop according to `b`. `x` climbs by one either way, so
    `V = relu(n - x + 1)` ranks it once the conditional facts about `t` are usable."""
    x, n, b, t = z3.Ints("x n b t")
    layers = [(np.array([[-1, 1, 0, 0]]), np.array([1])),
              (np.array([[1]]), np.array([0]))]
    invariants = (z3.Or(b <= 0, t == 1), z3.Or(b >= 1, t == -1))
    return layers, invariants


def test_atoms_the_lp_cannot_express_are_reported():
    """Dropping a disjunctive invariant is legitimate — it still shapes the domain
    and reaches the emitted proof — but it is information the LP does not get, so
    the result names it rather than proceeding silently. A trivially true conjunct
    is no information and is not reported."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    plain = _certify(_decrement(layers))
    assert plain.verified and plain.unused == (), plain.unused

    lay, invariants = _conditional_loop()
    ob = _obligation(("x", "n", "b", "t"),
                     lambda c: (dsl_ite(c[0] <= c[1],
                                        dsl_ite(c[2] >= 1, c[0] + c[3], c[0] - c[3]),
                                        c[0]), c[1], c[2], c[3]),
                     lay, invariants=invariants)
    res = _certify(ob)
    assert res.verified, res.status
    assert res.unused, "the disjunctive invariants are not expressible as rows"
    assert all("Or" in u or "Implies" in u for u in res.unused), res.unused


def test_the_vocabulary_covers_the_theory():
    """``OPS`` answers for every LIA operation, so one added to the theory shows up
    as a failure here rather than as a refusal on some benchmark. ``Uninterpreted``
    has no linear reading at all, so it is excluded by name."""
    theory = {f"LIA_{n}" for n in dir(LIA) if not n.startswith("_")}
    assert set(_farkas.OPS) - theory == set(), "declared but not in the theory"
    assert theory - set(_farkas.OPS) == {"LIA_Uninterpreted"}


def test_an_itype_outside_the_vocabulary_is_refused():
    """``OPS`` is the procedure's declaration of what it understands, so the walk
    refuses an operation absent from it by name rather than guessing a kind."""
    bench = loop_bench(("x",), lambda x: dsl_ite(x > 0, x - 1, x))
    prog, ctrl, _ = bench.build()
    ops = {k: v for k, v in _farkas.OPS.items() if k != "LIA_Ite"}
    with pytest.raises(Unsupported, match="LIA_Ite"):
        node_view(prog, {ctrl[n][0]: [z3.Int(n)] for n in bench.state}, ops)


def _prog(update, *, extl=()):
    """A one-variable program module, for the tests that need an operation or a
    wiring the fixture's DSL spec cannot express. ``update`` takes the latched
    variable and the awaited inputs, as a DSL update block does."""
    pair = (Wire(INT), Wire(INT))

    class Program(sugar.Module):
        def init(self, *a):
            return (0,)

        def update(self, ctrl, extl):
            return update(ctrl, extl)

    return Program(theory=LIA, ctrl=(pair,), extl=extl), pair


def test_an_untranslatable_itype_is_refused():
    """An op the theory has but Z3 cannot read is refused by name in the walk,
    rather than surfacing as a backend error from underneath it."""
    for build in (lambda c, _e: (c - 1)._unop(LIA.Min(), out=c.dtype),
                  lambda c, _e: dsl_argmax(c)):
        prog, _ = _prog(build)
        with pytest.raises(Unsupported, match="no Z3 translation"):
            read_system(prog, ("x",))


def test_a_vector_wire_is_refused():
    """The reader takes scalar integer wires — one symbol per wire is what the
    rows, the regions and the proof's state quantify over — so a vector-valued
    wire is refused by name at the door rather than read element by element."""
    vec = Sort.Int([2, 1])
    pair = (Wire(vec), Wire(vec))

    class Program(sugar.Module):
        def init(self):
            return (dsl_expr(np.zeros((2, 1), dtype=int), theory=LIA, sort=vec),)

        def update(self, ctrl):
            return ctrl

    with pytest.raises(Unsupported, match="scalar"):
        read_system(Program(theory=LIA, ctrl=(pair,)), ("v",))


def test_the_property_owns_the_domain():
    """The engine asks the property which steps the obligation must hold on, so a
    different domain is a property change rather than an engine change."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    ob = _decrement(layers)

    class NoSteps(Fixpoint):
        def domain(self, system):
            return z3.BoolVal(False)

    empty = _certify(ob, NoSteps(over=ob.prop.over))
    assert not empty.verified and "no step" in empty.status, empty.status
    # and the real property does find steps on the same obligation
    full = _certify(ob, ob.prop)
    assert full.verified, full.status


def test_conditional_loop_certifies():
    """No negative arm: the domain is now ``T(s) != s``, and on the ``b >= 1``
    branch that says ``x + t != x`` — so the sign of ``t`` is in the domain
    without any entailment step deducing it."""
    layers, invariants = _conditional_loop()
    ob = _obligation(("x", "n", "b", "t"),
                     lambda c: (dsl_ite(c[0] <= c[1],
                                        dsl_ite(c[2] >= 1, c[0] + c[3], c[0] - c[3]),
                                        c[0]), c[1], c[2], c[3]),
                     layers, invariants=invariants)
    assert farkas_cell(ob).verified


def test_expand_cases_keeps_an_affine_body_single():
    guard = x > 0
    paths = expand_cases(guard, [x - 1])
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
    ob = _decrement(layers)
    res = farkas_cell(ob)
    assert res.verified, res.status
    for path in res.certificate:
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
    res = farkas_cell(_decrement(layers))
    assert res.verified, res.status
    for c in (c for p in res.certificate for c in p.cells):
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
    counterexample and has to be reported as one."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    res = farkas_cell(_decrement(layers, delta=2.0, step=2))
    assert not res.verified
    assert res.status == "FAILED(decrease)"


def test_certificates_are_valid():
    """Every cell certificate of a certified loop satisfies the three Farkas
    conditions on its own system — the facts ``farkas_sound`` consumes."""
    # V(s) = relu(x) over the loop `while (x > 0) x = x - 1`
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    res = farkas_cell(_decrement(layers))
    assert res.verified, res.status
    cells = [c for p in res.certificate for c in p.cells]
    assert cells
    for c in cells:
        n = len(c.A[0])
        assert all(v >= 0 for v in c.y)
        for j in range(n):
            assert sum(c.y[i] * c.A[i][j] for i in range(len(c.A))) == 0
        assert sum(c.y[i] * c.b[i] for i in range(len(c.b))) < 0


