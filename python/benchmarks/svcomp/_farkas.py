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
    substitution. Raises if ``expr`` is not integer-affine over ``syms``."""
    zeros = [(s, z3.IntVal(0)) for s in syms]
    beta = _int(z3.substitute(expr, *zeros))
    alpha = []
    for sk in syms:
        subs = [(s, z3.IntVal(1 if s.eq(sk) else 0)) for s in syms]
        alpha.append(_int(z3.substitute(expr, *subs)) - beta)
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
    multipliers, plus the activation patterns (for the Lean emitter)."""
    A: tuple
    b: tuple
    y: tuple
    labels: tuple
    pattern_s: tuple
    pattern_sp: tuple


@dataclass
class FarkasResult:
    verified: bool
    certificates: list           # list[CellCert]
    counterexample: object = None
    status: str = ""


def _on_guard_body(sp_syms, s_syms):
    """The loop body's next-state (on-guard): the ``then`` branch of each
    guard-``ite`` (``ite(guard, body, self)``), so the transition is affine."""
    body = []
    for sp, s in zip(sp_syms, s_syms):
        ite = guard_ite(sp, s)
        body.append(ite.arg(1) if ite is not None else sp)
    return body


def certify_decrease(layers, s_syms, sp_syms, guard, invariants, delta,
                     max_iters: int = 1000) -> FarkasResult:
    """CEGAR: repeatedly find an uncovered in-domain state, certify the decrease
    on its cell via Farkas, and block that cell — until the domain is exhausted
    (VERIFIED) or a cell yields a real counterexample / cannot be certified."""
    body = _on_guard_body(sp_syms, s_syms)
    dom = z3.And(guard, *invariants) if invariants else guard
    solver = z3.Solver()
    solver.add(dom)
    certs: list[CellCert] = []

    for _ in range(max_iters):
        r = solver.check()
        if r == z3.unsat:
            return FarkasResult(True, certs, status="VERIFIED")
        if r == z3.unknown:
            return FarkasResult(False, certs, status="UNKNOWN")

        m = solver.model()
        s_val = [m.eval(x, model_completion=True).as_long() for x in s_syms]
        sp_val = [m.eval(e, model_completion=True).as_long() for e in body]
        cell_s = symbolic_forward(layers, s_syms, s_val)
        cell_sp = symbolic_forward(layers, body, sp_val)

        # genuine counterexample: the real V does not decrease at this state
        d_concrete = m.eval(cell_s.affine - cell_sp.affine,
                            model_completion=True).as_long()
        if d_concrete < delta:
            return FarkasResult(False, certs, np.array(s_val, dtype=np.float64),
                                "FAILED(decrease)")

        try:
            A, b, labels = build_integer_system(cell_s, cell_sp, guard,
                                                list(invariants), s_syms, delta)
        except ValueError:        # non-affine transition (in-loop branching, v1)
            return FarkasResult(False, certs, np.array(s_val, dtype=np.float64),
                                "FAILED(non-affine)")
        y = find_infeasibility_certificate(A, b)
        if y is None:
            return FarkasResult(False, certs, np.array(s_val, dtype=np.float64),
                                "FAILED(uncertifiable)")
        certs.append(CellCert(tuple(map(tuple, A)), tuple(b), tuple(y),
                              tuple(labels), cell_s.pattern, cell_sp.pattern))

        # block this joint cell so the next witness lies in a different region
        solver.add(z3.Not(z3.And(*(cell_s.constraints + cell_sp.constraints))))

    return FarkasResult(False, certs, status="FAILED(max_iters)")
