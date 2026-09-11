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
  * the regions of a certified path partition the guard, which is what the
    emitted coverage proof rests on.
"""
import numpy as np
import pytest
import z3

from benchmarks.svcomp import _farkas
from benchmarks.svcomp._bench import INT
from benchmarks.svcomp._termination import _v_module, system_of
from tests._fixtures import candidate, loop_bench
from zrth import LIA, Module, Sort, Wire, sugar
from zrth.sugar import argmax as dsl_argmax
from zrth.sugar import expr as dsl_expr
from zrth.sugar import ite as dsl_ite
from benchmarks.svcomp._farkas import (certify, check_supported,
                                      decrease, inductive, lex_decrease,
                                      read_system)
from benchmarks.svcomp._property import Liveness, Safety
from benchmarks.svcomp._termination import terminates
from benchmarks.svcomp._nodes import Unsupported, node_view
from benchmarks.svcomp._farkas import (
    Net,
    _pre_activations,
    affine_coeffs,
    atom_rows,
    expand_cases,
    find_infeasibility_certificate,
    mode_region,
    piece_value,
)

x, y = z3.Ints("x y")


def _obligation(state, update, layers, invariants=(), delta=1.0):
    """A real obligation for a compact loop spec — the module is built and walked
    exactly as the pipeline does it (see :mod:`tests._fixtures`)."""
    named = [(f"inv{k}", (lambda st, p=p: p)) for k, p in enumerate(invariants)]
    return candidate(loop_bench(state, update), layers, delta, named)


def _decrement(layers, delta=1.0, step=1):
    """`while (x > 0) x = x - step` as an obligation."""
    return _obligation(("x",), lambda v: dsl_ite(v > 0, v - step, v), layers, delta=delta)


def _cell(c):
    """``c`` under its own claim and witness: the termination path."""
    return certify(c.system, c.claim, c.witness)


def _certify(ob, prop=None, rule=None):
    """``ob``'s system under ``prop`` (default: the obligation's termination
    claim) by ``rule`` (default: the obligation's decrease witness; for a Safety
    claim given without one, its own predicate as the inductive invariant)."""
    if rule is None:
        rule = ob.witness if prop is None else inductive((prop.holds,))
    return certify(ob.system, prop or ob.claim, rule)


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


def test_a_disjunctive_guard_reaches_the_lp_only_once_split(monkeypatch):
    """``atom_rows`` drops a disjunction, so an un-split disjunctive guard reaches
    the LP as no rows at all and is reported as an atom it cannot express. The
    split is what turns it into rows, which is the reason the split exists."""
    i, m, k = z3.Ints("i m k")
    # V = relu(m - i) + relu(k - i) over (i, m, k), for
    # `while (i < m or i < k) { i = i + 1 }`
    layers = [(np.array([[-1, 1, 0], [-1, 0, 1]]), np.array([0, 0])),
              (np.array([[1, 1]]), np.array([0]))]
    ob = _obligation(("i", "m", "k"),
                     lambda c: (dsl_ite((c[0] < c[1]) | (c[0] < c[2]), c[0] + 1, c[0]),
                                c[1], c[2]),
                     layers)
    split = _cell(ob)
    assert split.verified and split.unused == (), split.unused

    monkeypatch.setattr(_farkas, "_convex_alternatives", lambda a: None)
    unsplit = _cell(ob)
    assert unsplit.unused, \
        "the un-split disjunction should be reported as unusable by the LP"
    assert all("Or" in u for u in unsplit.unused), unsplit.unused


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


def test_the_kinds_without_a_rule_are_the_ones_the_walk_refuses():
    """The vocabulary's kinds with neither a cell rule nor a case split are exactly
    the ones Z3 cannot read, so the walk refuses a module carrying one by name
    (see below) before the engine could reason over it. A kind declared here
    before its rule shows up in this set."""
    from benchmarks.svcomp._farkas import OPS
    unruled = {op.kind for op in OPS.values() if op.kind and not op.mode and not op.split}
    assert unruled == {"min", "max", "argmax"}


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


def test_a_nondeterministic_transition_is_refused():
    """A next value reading an awaited input is nondeterminism, which this procedure
    has no rule for — so it says so, naming the input."""
    prog, _ = _prog(lambda c, e: c + e, extl=((Wire(INT), Wire(INT)),))
    system = read_system(prog, ("x",))
    with pytest.raises(Unsupported, match="_in0"):
        check_supported(system)


def test_a_supported_module_passes_the_door():
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    ob = _decrement(layers)
    check_supported(ob.system)                                 # does not raise


def test_a_second_property_runs_through_the_same_engine():
    """``Safety(pred)`` is a property of the *program*, discharged by an inductive
    invariant through the same region engine — with no ReLU-bearing wire named,
    there is one region per path, and the same Farkas rows close it: one per
    disjunct of the rule's negation (the invariant not preserved, the invariant
    not implying the predicate)."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    ob = _decrement(layers)                    # while (x > 0) x = x - 1, from x = 0

    ok = _certify(ob, Safety(lambda W, S: S["x"] >= 0))
    assert ok.verified, ok.status
    assert ok.witness.inv, "pred serves as its own invariant"
    assert all({c.disjunct for c in p.cells} == {0, 1} and len(p.cells) == 2
               for p in ok.certificates), "no device: one region, two disjuncts"

    # false at entry: x starts at 0
    bad = _certify(ob, Safety(lambda W, S: S["x"] >= 1))
    assert not bad.verified and bad.status == "FAILED(initiation)", bad.status

    # true at entry, preserved by the step, but not what was asked: a state the
    # invariant admits violates the predicate, and the exact check finds it
    weak = _certify(ob, Safety(lambda W, S: S["x"] <= 5),
                    inductive((lambda W, S: S["x"] >= 0,)))
    assert not weak.verified and weak.status == "FAILED(violated)", weak.status


def test_a_witness_refuses_a_claim_it_cannot_use():
    """A witness is a strategy for one kind of claim: ``inductive`` needs a predicate
    to imply, which a ``Liveness`` claim has not, and says so by name. Nothing
    picks a witness on the caller's behalf — a rank is not a thing to guess."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    ob = _decrement(layers)
    with pytest.raises(Unsupported, match="has none"):
        certify(ob.system, ob.claim, inductive(()))
    with pytest.raises(TypeError):
        certify(ob.system, ob.claim)


def test_the_property_owns_the_domain():
    """The engine asks the property which steps the obligation must hold on, so a
    different domain is a property change rather than an engine change."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    ob = _decrement(layers)

    empty = _certify(ob, Liveness(lambda W, S: z3.BoolVal(False)), ob.witness)
    assert not empty.verified and "no step" in empty.status, empty.status
    # and the real property does find steps on the same obligation
    full = _certify(ob, ob.claim, ob.witness)
    assert full.verified, full.status


def test_the_rule_owns_the_goal():
    """The obligation reaches the LP as the rule's negated row, so tightening the
    margin is a rule change and is rejected for the same net."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    ob = _decrement(layers)
    v_s, v_sp = ob.witness.ranks[0]
    assert _certify(ob, rule=decrease(v_s, v_sp, 1.0)).verified
    hard = _certify(ob, rule=decrease(v_s, v_sp, 2.0))
    assert not hard.verified, "x decreases by 1, so a margin of 2 cannot hold"


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
    assert _cell(ob).verified


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
    """A region fixes every node's mode, strict on one side and non-strict on the
    other, so the regions cover the guard and no state lies in two of them.
    Coverage in the emitted proof rests on this."""
    layers = [(np.array([[1], [1]]), np.array([0, -3])),
              (np.array([[1, 1]]), np.array([0]))]
    ob = _decrement(layers)
    res = _cell(ob)
    assert res.verified, res.status
    for path in res.certificates:
        acts = [z3.IntVal(k) + c[0] * x for c, k in path.units]
        regions = [z3.And(*mode_region((("relu", tuple(acts)),), c.pattern))
                   for c in path.cells]
        covered = z3.Solver()
        covered.add(path.guard, z3.Not(z3.Or(*regions)))
        assert covered.check() == z3.unsat, "cells leave part of the guard uncovered"
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                overlap = z3.Solver()
                overlap.add(path.guard, regions[i], regions[j])
                assert overlap.check() == z3.unsat, f"cells {i} and {j} overlap"


def test_mixed_output_weights_and_bias_are_carried():
    """The trained nets only ever have uniform output weights and no output bias,
    so nothing else pins the per-unit scaling. Each device's affine piece is
    recomputed here from the raw matrices, over the slice of the region's pattern
    that device owns."""
    layers = [(np.array([[1], [1]]), np.array([0, -3])),
              (np.array([[1, 2]]), np.array([5]))]
    (W1, b1), (W2, b2) = layers
    res = _cell(_decrement(layers))
    assert res.verified, res.status
    devs = res.devices          # V(s) reads x, V(s') reads x - 1
    shifts = {0: 0, 1: -1}
    for c in (c for p in res.certificates for c in p.cells):
        for d, dev in enumerate(devs):
            m = len(W1)
            pattern = c.pattern[dev.offset:dev.offset + m]
            kept = [j for j, on in enumerate(pattern) if on]
            coeff = sum(int(W2[0][j]) * int(W1[j][0]) for j in kept)
            const = int(b2[0]) + sum(int(W2[0][j]) * (int(b1[j]) + shifts[d])
                                     for j in kept)
            assert c.affines[d] == ((coeff,), const), (pattern, c.affines[d])


def test_margin_the_rank_cannot_meet_is_rejected():
    """``relu(x)`` drops by 1 at ``x = 1``, so a margin of 2 is a real
    counterexample and has to be reported as one. Two checks can catch it — the
    witness-level one in ``_certify_path`` and the region-wide prune in
    ``_certify_cell`` — and the guarantee holds as long as either does."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    res = _cell(_decrement(layers, delta=2.0, step=2))
    assert not res.verified
    assert res.status == "FAILED(violated)"


def test_certificates_are_valid():
    """Every cell certificate of a certified loop satisfies the three Farkas
    conditions on its own system — the facts ``farkas_sound`` consumes."""
    # V(s) = relu(x) over the loop `while (x > 0) x = x - 1`
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    res = _cell(_decrement(layers))
    assert res.verified, res.status
    cells = [c for p in res.certificates for c in p.cells]
    assert cells
    for c in cells:
        n = len(c.A[0])
        assert all(v >= 0 for v in c.y)
        for j in range(n):
            assert sum(c.y[i] * c.A[i][j] for i in range(len(c.A))) == 0
        assert sum(c.y[i] * c.b[i] for i in range(len(c.b))) < 0


def _two_ranks(update):
    """A two-variable program composed with two ranks, ``relu(i)`` and ``relu(j)``,
    each read at both ends of a step. Returns the system, the property, and the
    two ``(V(s), V(s'))`` wire pairs."""
    prog = system_of(loop_bench(("i", "j"), update))
    mods, ranks = [], []
    for W in ([[1, 0]], [[0, 1]]):
        layers = [(np.array(W), np.array([0])), (np.array([[1]]), np.array([0]))]
        vs_mod, vs = _v_module(prog.pairs, layers, read_next=False)
        vsp_mod, vsp = _v_module(prog.pairs, layers, read_next=True)
        mods += [vs_mod, vsp_mod]; ranks.append((vs[1], vsp[1]))
    system = read_system(Module.parallel(prog.module, *mods), prog.names)
    assert len(system.pairs) == 2 and len(system.all_pairs) == 6
    return system, terminates(), tuple(ranks)


def test_lexicographic_rank_where_no_single_rank_works():
    """``while (i > 0) { if (j > 0) j--; else { i--; j = 3; } }``: the inner step
    keeps ``i`` and the outer resets ``j``, so neither rank drops on every step —
    but ``(i, j)`` drops lexicographically."""
    system, prop, (r0, r1) = _two_ranks(
        lambda c: (dsl_ite(c[0] > 0, dsl_ite(c[1] > 0, c[0], c[0] - 1), c[0]),
                   dsl_ite(c[0] > 0, dsl_ite(c[1] > 0, c[1] - 1, 3), c[1])))
    assert certify(system, prop, decrease(*r0)).status == "FAILED(violated)"
    assert certify(system, prop, decrease(*r1)).status == "FAILED(violated)"
    res = certify(system, prop, lex_decrease((r0, r1)))
    assert res.verified, res.status
    # ¬lexDec has two disjuncts (the first rank does not drop and: it increases /
    # the second does not drop); every region refutes both
    for p in res.certificates:
        assert {c.disjunct for c in p.cells} == {0, 1}, p.cells


def test_lexicographic_rank_must_prove_earlier_ranks_do_not_increase():
    """``while (i > 0) { if (j > 0) { i = i + j - 2; j--; } else { i--; j = 3; } }``.
    Across an inner run ``i`` goes +1, 0, −1, so ``(i, j)`` is *not* a lexicographic
    rank: at ``j = 3`` the first rank increases while the second drops. A region's
    witness may lie at ``j = 1`` where lex holds, so an encoding that *assumed*
    "earlier ranks do not increase" while proving the later one drops certified the
    region vacuously and reported VERIFIED. Non-increase is part of the formula,
    never an assumption."""
    system, prop, (r0, r1) = _two_ranks(
        lambda c: (dsl_ite(c[0] > 0, dsl_ite(c[1] > 0, c[0] + c[1] - 2, c[0] - 1), c[0]),
                   dsl_ite(c[0] > 0, dsl_ite(c[1] > 0, c[1] - 1, 3), c[1])))
    res = certify(system, prop, lex_decrease((r0, r1)))
    assert not res.verified, "an invalid lexicographic rank was certified"
    assert res.status == "FAILED(violated)", res.status


# --- the rule as a formula: disjunction, disequality, wires in the property ---

def test_a_disjunctive_invariant_goes_through_the_disjuncts():
    """The rule's negation is cut into disjuncts of rows, so a disjunctive
    invariant is one more shape of formula rather than a refusal: ``while (x > 0)
    x--`` from ``x = 0`` keeps ``x >= 0 ∨ x <= -5`` — each disjunct of its negation
    (each side of the invariant at ``s`` with both sides false at ``s'``, and each
    side with the predicate false) is refuted on the one region per path."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    ob = _decrement(layers)
    res = _certify(ob, Safety(lambda W, S: z3.Or(S["x"] >= 0, S["x"] <= -5)))
    assert res.verified, res.status
    assert all(len({c.disjunct for c in p.cells}) == 4 for p in res.certificates), \
        [len(p.cells) for p in res.certificates]
    bad = _certify(ob, Safety(lambda W, S: z3.Or(S["x"] >= 1, S["x"] <= -5)))
    assert not bad.verified and bad.status == "FAILED(initiation)", bad.status


def test_a_disequality_in_the_rule_splits_into_its_two_sides():
    """``x != -1`` is no half-space, but as a formula it is ``x < -1 ∨ x > -1``,
    and its negation the equality's two rows — so it certifies where before it
    was refused as \"not a linear comparison\"."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    ob = _decrement(layers)
    res = _certify(ob, Safety(lambda W, S: S["x"] != -1))
    assert res.verified, res.status


def _computed(state, update, layers, *, init=None):
    """``state``'s program composed with one sequential atom computing a network
    of the latched state — a wire of the graph a property may name. Returns the
    system and that wire."""
    prog = system_of(loop_bench(state, update, init=init))
    mod, out = _v_module(prog.pairs, layers, read_next=False)
    return read_system(Module.parallel(prog.module, mod), prog.names), out[1]


def test_a_property_may_name_a_computed_wire():
    """``while (i < n) i++`` beside ``d = relu(n - i)``: the claim ``d == n - i``
    names the computed wire, and holds because ``i <= n`` is invariant. The
    equality pins ``d``, so the engine regions over its pattern and refutes the
    two sides of the disequality on each; the invariant is a state predicate the
    same rule carries. Without it the claim must hold outright, and does not."""
    system, d = _computed(("i", "n"),
                          lambda c: (dsl_ite(c[0] < c[1], c[0] + 1, c[0]), c[1]),
                          [(np.array([[-1, 1]]), np.array([0])),
                           (np.array([[1]]), np.array([0]))],
                          init=lambda: (0, 5))
    prop = Safety(lambda W, S: W[d] == S["n"] - S["i"])
    res = certify(system, prop, inductive((lambda W, S: S["i"] <= S["n"],)))
    assert res.verified, res.status
    assert len(res.devices) == 1, res.devices
    assert res.claim.holds is not None and len(res.witness.inv) == 1
    assert any(len(p.cells) >= 2 for p in res.certificates), "the wire's two sides"
    alone = certify(system, prop, inductive(()))      # no invariant: hold outright
    assert not alone.verified and alone.status == "FAILED(violated)", alone.status


def test_a_safety_claim_may_speak_of_the_step():
    """A round holds a state and its successor, so a safety claim may relate the
    two — ``x`` never increases — and the engine treats it as any other predicate
    over the round: certified where it holds, refuted where it does not, including
    at the stuttering rounds a run claim would not count."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    ob = _decrement(layers)                    # while (x > 0) x = x - 1, from x = 0
    down = _certify(ob, Safety(lambda W, S: S.next["x"] <= S["x"]), inductive(()))
    assert down.verified, down.status
    v_sp = ob.witness.ranks[0][1]                 # V read at the next state: a wire too
    bounded = _certify(ob, Safety(lambda W, S: W[v_sp] >= 0), inductive(()))
    assert bounded.verified, bounded.status
    up = _certify(ob, Safety(lambda W, S: S.next["x"] >= S["x"] + 1), inductive(()))
    assert not up.verified and up.status == "FAILED(violated)", up.status


def test_an_invariant_may_not_name_a_wire():
    """Invariants are state predicates — a wire belongs in the property or the
    rule — and a rule whose atom is not linear is refused by name too."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    ob = _decrement(layers)
    v_s = ob.witness.ranks[0][0]
    with pytest.raises(Unsupported, match="over the state"):
        _certify(ob, Safety(lambda W, S: S["x"] >= 0),
                 inductive((lambda W, S: W[v_s] >= 0,)))
    with pytest.raises(Unsupported, match="linear"):
        _certify(ob, Safety(lambda W, S: S["x"] * S["x"] >= 0))
