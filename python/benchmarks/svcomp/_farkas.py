"""Farkas-certified ranking verification (cell / CEGAR).

A ranking net ``V(s) = W2·relu(W1·s + b1) + b2`` is piecewise-affine: on a fixed
ReLU **activation pattern** (a *cell*) it is an affine function of the state, and
the cell is the polyhedron where each hidden pre-activation ``W1[j]·s + b1[j]``
has the sign the pattern dictates. Over one cell the decrease obligation
``V(s) - V(s') >= delta`` becomes a linear entailment, discharged by proving the
system ``cell ∧ guard ∧ invariants ∧ ¬decrease`` infeasible via Farkas' lemma
(over z3's exact LRA) — which yields an exact, checkable integer certificate
(the Farkas multipliers ``y``: ``y >= 0``, ``Aᵀy = 0``, ``b·y < 0``). ``V >= 0``
is *not* Farkas-certified: it holds structurally because the output layer is
non-negative (a positive sum of ReLUs), which the caller asserts.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm

import numpy as np
import z3

from ._domain import guard_ite


@dataclass(frozen=True)
class Cell:
    """One ReLU activation pattern (a linear region of the net)."""
    affine: z3.ArithRef                 # V on this cell, affine in the inputs
    constraints: tuple[z3.BoolRef, ...] # sign rows: pre>=0 (active) / pre<=0 (inactive)
    pattern: tuple[bool, ...]           # activation of each hidden ReLU


def _pre_activations(layers, inputs):
    """Symbolic hidden pre-activations ``W1[j]·inputs + b1[j]`` (a z3 expr each)."""
    (W1, b1), _ = layers
    W1 = np.asarray(W1); b1 = np.asarray(b1)
    pres = []
    for j in range(W1.shape[0]):
        e = z3.IntVal(int(b1[j]))
        for k in range(W1.shape[1]):
            c = int(W1[j][k])
            if c:
                e = e + c * inputs[k]
        pres.append(e)
    return pres


def symbolic_forward(layers, inputs, input_vals) -> Cell:
    """The cell of ``V`` at the witness ``input_vals`` (concrete ints), symbolic
    over ``inputs`` (z3 exprs). ``inputs``/``input_vals`` are the state fed to V
    — the pre-state ``s`` or the successor ``s' = T(s)``."""
    (W1, b1), (W2, b2) = layers
    W1 = np.asarray(W1); b1 = np.asarray(b1)
    W2 = np.asarray(W2); b2 = np.asarray(b2)
    pres = _pre_activations(layers, inputs)

    vals = np.asarray(input_vals, dtype=object)
    constraints, pattern, hidden = [], [], []
    for j, pre in enumerate(pres):
        active = int(W1[j] @ vals + b1[j]) >= 0     # sign at the witness
        pattern.append(active)
        constraints.append(pre >= 0 if active else pre <= 0)
        hidden.append(pre if active else z3.IntVal(0))

    out = z3.IntVal(int(b2[0]))
    for j in range(W2.shape[1]):
        c = int(W2[0][j])
        if c:
            out = out + c * hidden[j]
    return Cell(out, tuple(constraints), tuple(pattern))


# ---------------------------------------------------------------------------
# Affine extraction and the integer infeasibility system
# ---------------------------------------------------------------------------

def _int(e) -> int:
    """The integer value of a z3 numeral (v1 is integer-linear throughout)."""
    e = z3.simplify(e)
    if z3.is_int_value(e):
        return int(e.as_long())
    if z3.is_rational_value(e):
        f = e.as_fraction()
        if f.denominator == 1:
            return int(f.numerator)
    raise ValueError(f"non-integer / non-constant numeral: {e}")


def affine_coeffs(expr, syms):
    """(alpha, beta) with ``expr == sum(alpha_k * syms_k) + beta`` — by 0/1
    substitution. Raises ``ValueError`` if ``expr`` is not integer-affine over
    ``syms``.

    The 0/1 sampling alone is *not* a soundness check: a non-affine term (e.g.
    an ``ite`` from an in-loop branch, ``ite(x>0, x-1, x+1)``) evaluates to a
    constant at each integer sample, so sampling silently returns a bogus linear
    fit. Trusting it would certify the decrease of the wrong (linearised)
    transition. So we verify the fit: the reconstructed form must equal ``expr``
    on all inputs (z3-valid). Only a genuinely affine ``expr`` passes; anything else is rejected here rather than certified
    downstream."""
    zeros = [(s, z3.IntVal(0)) for s in syms]
    beta = _int(z3.substitute(expr, *zeros))
    alpha = []
    for sk in syms:
        subs = [(s, z3.IntVal(1 if s.eq(sk) else 0)) for s in syms]
        alpha.append(_int(z3.substitute(expr, *subs)) - beta)

    recon = z3.IntVal(beta)
    for a, s in zip(alpha, syms):
        if a:
            recon = recon + a * s
    diff = z3.simplify(expr - recon)
    if not (z3.is_int_value(diff) and diff.as_long() == 0):
        solver = z3.Solver()
        solver.add(expr != recon)
        if solver.check() != z3.unsat:      # not provably equal to its linear fit
            raise ValueError(f"expression is not affine over the given symbols: {expr}")
    return alpha, beta


def _flatten_and(pred):
    if z3.is_and(pred):
        out = []
        for c in pred.children():
            out += _flatten_and(c)
        return out
    return [pred]


_CMP = {z3.Z3_OP_GE: ">=", z3.Z3_OP_LE: "<=", z3.Z3_OP_GT: ">",
        z3.Z3_OP_LT: "<", z3.Z3_OP_EQ: "=="}
_FLIP = {">=": "<", "<=": ">", ">": "<=", "<": ">=", "==": None}


def atom_rows(atom, syms):
    """Integer rows ``A·s <= b`` for a single linear atom, or ``[]`` if it is not
    a linear half-space (disjunction / != / boolean / nonlinear — soundly skipped).
    Strict inequalities are integer-tightened."""
    a = atom
    neg = False
    while z3.is_not(a):
        a = a.arg(0); neg = not neg
    if not z3.is_app(a) or a.decl().kind() not in _CMP:
        return []
    op = _CMP[a.decl().kind()]
    if neg:
        op = _FLIP[op]
        if op is None:            # Not(==) is !=  -> skip
            return []
    try:
        alpha, beta = affine_coeffs(a.arg(0) - a.arg(1), syms)
    except ValueError:
        return []                 # nonlinear / free symbol -> skip
    pos = list(alpha)             # a·s + b <= 0  row
    neg = [-c for c in alpha]     # a·s + b >= 0  row
    if op == "==":                # both directions
        return [(pos, -beta), (neg, beta)]
    if op == ">=":                # a·s+b >= 0  ->  -a·s <= b
        return [(neg, beta)]
    if op == "<=":                # a·s+b <= 0  ->   a·s <= -b
        return [(pos, -beta)]
    if op == ">":                 # a·s+b >= 1  ->  -a·s <= b-1
        return [(neg, beta - 1)]
    if op == "<":                 # a·s+b <= -1 ->   a·s <= -b-1
        return [(pos, -beta - 1)]
    return []


def build_integer_system(cell_s, cell_sp, guard, invariants, syms, delta):
    """Rows of ``cell_s ∧ cell_sp ∧ guard ∧ invariants ∧ ¬(V(s)-V(s') >= delta)``
    as an integer system ``A·s <= b`` with per-row labels. Infeasibility of this
    system certifies the decrease on the cell."""
    rows, labels, seen = [], [], set()

    def add(atom, label):
        for A_row, b in atom_rows(atom, syms):
            if all(c == 0 for c in A_row) and b >= 0:
                continue                      # always-true constant row (redundant)
            key = (tuple(A_row), b)
            if key in seen:
                continue                      # duplicate row
            seen.add(key)
            rows.append((A_row, b)); labels.append(label)

    for j, c in enumerate(cell_s.constraints):
        add(c, f"cell_s[{j}]")
    for j, c in enumerate(cell_sp.constraints):
        add(c, f"cell_sp[{j}]")
    for i, g in enumerate(_flatten_and(guard)):
        add(g, f"guard[{i}]")
    for i, inv in enumerate(invariants):
        add(inv, f"inv[{i}]")

    # negated decrease: V(s) - V(s') <= delta - 1  (integer). Kept unconditionally.
    alpha, beta = affine_coeffs(cell_s.affine - cell_sp.affine, syms)
    rows.append((list(alpha), int(delta) - 1 - beta))
    labels.append("neg_decrease")

    return [r[0] for r in rows], [r[1] for r in rows], labels


def find_infeasibility_certificate(A, b):
    """Farkas multipliers ``y >= 0`` with ``Aᵀy = 0`` and ``b·y < 0`` (integer),
    or ``None`` if the system is feasible. Uses z3 LRA (exact rationals)."""
    m = len(A)
    if m == 0:
        return None
    n = len(A[0])
    ys = [z3.Real(f"y_{i}") for i in range(m)]
    s = z3.Solver()
    for y in ys:
        s.add(y >= 0)
    for j in range(n):
        s.add(z3.Sum([ys[i] * A[i][j] for i in range(m)]) == 0)
    s.add(z3.Sum([ys[i] * b[i] for i in range(m)]) < 0)
    if s.check() != z3.sat:
        return None
    model = s.model()
    vals = []
    for y in ys:
        v = model.eval(y, model_completion=True)
        vals.append(v.as_fraction() if z3.is_rational_value(v)
                    else Fraction(v.as_long()))
    den = 1
    for f in vals:
        den = lcm(den, f.denominator)
    ints = [int(f * den) for f in vals]
    g = 0
    for x in ints:
        g = gcd(g, x)
    if g > 1:
        ints = [x // g for x in ints]
    return ints


# ---------------------------------------------------------------------------
# CEGAR driver: certify the decrease over every cell that meets the domain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellCert:
    """A per-cell decrease certificate: the integer system and its Farkas
    multipliers, the activation patterns, and V's collapsed affine form at each
    end of the step (all consumed by the Lean emitter).

    ``pre_affine``/``post_affine`` are ``(coeffs, const)`` over the state columns:
    on this cell ``V(s) = pre_affine`` and ``V(s') = post_affine`` (affine, since
    the pattern is fixed and the transition is affine). The emitter renders them
    as the right-hand sides of the two V-collapse lemmas."""
    A: tuple
    b: tuple
    y: tuple
    labels: tuple
    pattern_s: tuple
    pattern_sp: tuple
    pre_affine: tuple = ((), 0)
    post_affine: tuple = ((), 0)


@dataclass
class FarkasResult:
    verified: bool
    certificates: list           # list[PathCert]
    counterexample: object = None
    status: str = ""


@dataclass(frozen=True)
class PathCert:
    """One affine path of the loop body: its path condition ``guard`` (the loop
    guard strengthened by the branch literals taken along the path), its affine
    next-state ``body`` (z3 exprs over the pre-state symbols), and the per-cell
    decrease certificates on it. The paths partition the loop guard, so the union
    of their ``Step`` relations is the loop's transition — hence termination of
    the union (every path strictly drops ``V``) is termination of the program."""
    guard: object
    body: tuple
    cells: tuple


def _on_guard_body(sp_syms, s_syms):
    """The loop body's next-state (on-guard): the ``then`` branch of each
    guard-``ite`` (``ite(guard, body, self)``), so the transition is affine."""
    body = []
    for sp, s in zip(sp_syms, s_syms):
        ite = guard_ite(sp, s)
        body.append(ite.arg(1) if ite is not None else sp)
    return body


# ---------------------------------------------------------------------------
# Path splitting: expand in-loop branches into affine paths
# ---------------------------------------------------------------------------

def _find_ite_cond(e):
    """The condition of some ``ite`` node in ``e`` (depth-first), or ``None``."""
    if z3.is_app(e) and e.decl().kind() == z3.Z3_OP_ITE:
        return e.arg(0)
    for c in e.children():
        r = _find_ite_cond(c)
        if r is not None:
            return r
    return None


def _select(e, cond, truth: bool):
    """``e`` with ``cond`` pinned to ``truth`` — every ``ite(cond, ..)`` collapses
    to its taken branch (substitute the condition, then simplify)."""
    return z3.simplify(z3.substitute(e, (cond, z3.BoolVal(truth))))


def enumerate_paths(body, guard):
    """Expand the (possibly branching) next-state ``body`` into affine paths.

    An in-loop branch appears as a nested ``ite`` in ``body``. Splitting on each
    ``ite`` condition both ways yields, at the leaves, a next-state with no
    ``ite`` left — a single affine map — under the path condition ``guard ∧
    branch-literals``. Returns ``[(path_guard, affine_body), ...]``; the paths
    partition the guard, so their union is exactly the loop's transition."""
    cond = None
    for e in body:
        cond = _find_ite_cond(e)
        if cond is not None:
            break
    if cond is None:
        return [(guard, body)]
    then_body = [_select(e, cond, True) for e in body]
    else_body = [_select(e, cond, False) for e in body]
    return (enumerate_paths(then_body, z3.And(guard, cond))
            + enumerate_paths(else_body, z3.And(guard, z3.Not(cond))))


def _feasible(pred) -> bool:
    s = z3.Solver()
    s.add(pred)
    return s.check() == z3.sat


def _certify_path(layers, s_syms, body, guard, invariants, delta, max_iters):
    """CEGAR over one affine path: certify the decrease on every cell of
    ``guard ∧ invariants`` under next-state ``body``. Repeatedly find an
    uncovered in-domain state, certify its cell via Farkas, and block it — until
    the path's domain is exhausted. Returns ``(ok, cells, counterexample, status)``."""
    dom = z3.And(guard, *invariants) if invariants else guard
    solver = z3.Solver()
    solver.add(dom)
    cells: list[CellCert] = []

    for _ in range(max_iters):
        r = solver.check()
        if r == z3.unsat:
            return True, cells, None, "VERIFIED"
        if r == z3.unknown:
            return False, cells, None, "UNKNOWN"

        m = solver.model()
        s_val = [m.eval(x, model_completion=True).as_long() for x in s_syms]
        sp_val = [m.eval(e, model_completion=True).as_long() for e in body]
        cell_s = symbolic_forward(layers, s_syms, s_val)
        cell_sp = symbolic_forward(layers, body, sp_val)

        # genuine counterexample: the real V does not decrease at this state
        d_concrete = m.eval(cell_s.affine - cell_sp.affine,
                            model_completion=True).as_long()
        if d_concrete < delta:
            return False, cells, np.array(s_val, dtype=np.float64), "FAILED(decrease)"

        try:
            A, b, labels = build_integer_system(cell_s, cell_sp, guard,
                                                list(invariants), s_syms, delta)
        except ValueError:        # non-affine even after splitting (nondet, etc.)
            return False, cells, np.array(s_val, dtype=np.float64), "FAILED(non-affine)"
        y = find_infeasibility_certificate(A, b)
        if y is None:
            return False, cells, np.array(s_val, dtype=np.float64), "FAILED(uncertifiable)"
        pre_c, pre_k = affine_coeffs(cell_s.affine, s_syms)
        post_c, post_k = affine_coeffs(cell_sp.affine, s_syms)
        cells.append(CellCert(tuple(map(tuple, A)), tuple(b), tuple(y),
                              tuple(labels), cell_s.pattern, cell_sp.pattern,
                              (tuple(pre_c), pre_k), (tuple(post_c), post_k)))

        # block this joint cell so the next witness lies in a different region
        solver.add(z3.Not(z3.And(*(cell_s.constraints + cell_sp.constraints))))

    return False, cells, None, "FAILED(max_iters)"


def certify_decrease(layers, s_syms, sp_syms, guard, invariants, delta,
                     max_iters: int = 1000) -> FarkasResult:
    """Certify the loop's decrease by splitting its body into affine paths
    (:func:`enumerate_paths`) and certifying each with the cell/CEGAR engine
    (:func:`_certify_path`). Every feasible path must strictly drop ``V``; the
    result carries one :class:`PathCert` per path (a non-branching loop is a
    single path, so this subsumes the scalar single-path case)."""
    body = _on_guard_body(sp_syms, s_syms)
    paths: list[PathCert] = []
    for pguard, pbody in enumerate_paths(body, guard):
        dom = z3.And(pguard, *invariants) if invariants else pguard
        if not _feasible(dom):
            continue                       # dead path (guard unsat) — never taken
        ok, cells, cex, status = _certify_path(
            layers, s_syms, pbody, pguard, invariants, delta, max_iters)
        if not ok:
            return FarkasResult(False, [], cex, status)
        paths.append(PathCert(pguard, tuple(pbody), tuple(cells)))
    if not paths:
        return FarkasResult(False, [], None, "FAILED(no feasible path)")
    return FarkasResult(True, paths, None, "VERIFIED")
