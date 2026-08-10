"""Farkas-certified ranking verification (cell / CEGAR).

A ranking net ``V(s) = W2·relu(W1·s + b1) + b2`` is piecewise-affine: on a fixed
ReLU **activation pattern** it is an affine function of the state, the polyhedron
where each hidden pre-activation ``W1[j]·s + b1[j]`` has the sign the pattern
dictates. The decrease obligation ``V(s) - V(s') >= delta`` needs the two ends of
a step treated differently, because it needs a *lower* bound on ``V(s)`` and an
*upper* bound on ``V(s')``:

  * at the successor, ``V(s')`` must be exact, so the pattern there is pinned —
    this is what a **cell** constrains, strictly (``> 0`` active, ``<= 0``
    inactive), so the cells partition rather than overlap;
  * at the pre-state, any mask under-approximates ``V(s)`` (each dropped unit
    contributes ``relu >= 0``, each kept one ``relu(z) >= z``), so no constraint
    is needed at all. The mask is a free parameter of the certificate: sound for
    any choice, and exact where it matches the true pattern.

So a cell fixes the successor signs alone, and one cell covers what the joint
pre/post patterns would have split into many. Over a cell the obligation is a
linear entailment, discharged by proving ``cell ∧ guard ∧ invariants ∧ ¬decrease``
infeasible via Farkas' lemma (over z3's exact LRA) — an exact, checkable integer
certificate (the multipliers ``y``: ``y >= 0``, ``Aᵀy = 0``, ``b·y < 0``).

``V >= 0`` is *not* Farkas-certified: it holds structurally because the output
layer is non-negative (a positive sum of ReLUs). The mask bound needs that same
non-negativity — see :func:`output_weights`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm

import numpy as np
import z3

from ._domain import guard_ite


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


def output_weights(layers):
    """The output layer's weights and bias, as ints: ``V = sum_j c_j relu(z_j) + k``.

    The mask bound needs ``c_j >= 0``, which holds structurally because the output
    layer is frozen non-negative. The assert fails fast on a net that breaks that;
    the binding check is in Lean, where ``affine_mask_le``'s ``hW`` side goal is
    ``0 <= W i j`` and would not close."""
    _, (W2, b2) = layers
    W2 = np.asarray(W2); b2 = np.asarray(b2)
    c = [int(W2[0][j]) for j in range(W2.shape[1])]
    assert all(x >= 0 for x in c), f"output layer must be non-negative: {c}"
    return c, int(b2[0])


def masked_value(pres, weights, pattern):
    """``V`` with ``pattern`` deciding which hidden units contribute:
    ``sum_j c_j * pres_j`` over the marked units, plus the output bias.

    Where ``pattern`` is the true activation pattern at these pre-activations this
    is ``V`` exactly; for any other pattern it is a lower bound, since each dropped
    unit contributes ``relu >= 0`` and each kept one ``relu(z) >= z``."""
    c, k = weights
    out = z3.IntVal(k)
    for cj, pre, on in zip(c, pres, pattern):
        if cj and on:
            out = out + cj * pre
    return out


def strict_signs(pres, pattern):
    """The region ``pattern`` names, as complementary literals: ``0 < e`` where
    active, ``e <= 0`` where not, nothing where the pattern leaves a unit open
    (``None``). Complementary, so the regions partition."""
    return tuple((p > 0) if a else (p <= 0)
                 for p, a in zip(pres, pattern) if a is not None)


def exact_value(pres, weights):
    """``V`` with the ReLUs left in: ``sum_j c_j * relu(pres_j) + k``.

    The strongest bound any mask can reach, since a mask matching the true signs
    is exactly this. Used to reject a class no mask could certify without paying
    for the search."""
    c, k = weights
    out = z3.IntVal(k)
    for cj, pre in zip(c, pres):
        if cj:
            out = out + cj * z3.If(pre > 0, pre, z3.IntVal(0))
    return out


def _sign_status(region, exprs):
    """Which of ``exprs`` keep one sign throughout ``region``.

    Returns ``(fixed, free)``: ``fixed[j]`` is ``True`` where the expression stays
    ``>= 0`` and ``False`` where it stays ``<= 0``; ``free`` lists the indices that
    take both signs. A mask bit is settled for every fixed unit — keeping a
    non-negative one only raises the bound, keeping a negative one only lowers it —
    so only the free ones are ever a choice."""
    fixed, free = {}, []
    for j, e in enumerate(exprs):
        if not _feasible(z3.And(region, e < 0)):
            fixed[j] = True
        elif not _feasible(z3.And(region, e > 0)):
            fixed[j] = False
        else:
            free.append(j)
    return fixed, free


def _mask_candidates(fixed, free, hint, m):
    """Masks worth trying, the witness's own first: the settled bits from
    ``fixed``, the ``free`` ones enumerated."""
    for combo in itertools.product(*[(hint[j], not hint[j]) for j in free]):
        mask = [fixed.get(j, False) for j in range(m)]
        for j, v in zip(free, combo):
            mask[j] = v
        yield tuple(mask)


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


def _convex_alternatives(atom):
    """Disjoint alternatives replacing a non-convex ``atom``, or ``None`` if it is
    left alone. A disjunction becomes its disjuncts, each excluding the earlier
    ones; ``a != b`` becomes ``a < b`` and ``a > b``."""
    if z3.is_or(atom):
        alts, earlier = [], []
        for d in atom.children():
            alts.append(z3.And(*earlier, d) if earlier else d)
            earlier.append(z3.simplify(z3.Not(d)))
        return alts
    eq = atom.arg(0) if z3.is_not(atom) and z3.is_eq(atom.arg(0)) else None
    if eq is not None:
        return [eq.arg(0) < eq.arg(1), eq.arg(0) > eq.arg(1)]
    if z3.is_distinct(atom) and atom.num_args() == 2:
        return [atom.arg(0) < atom.arg(1), atom.arg(0) > atom.arg(1)]
    return None


def split_guard(guard, max_pieces: int = 16):
    """Split ``guard`` into disjoint conjunctions whose union is exactly ``guard``.

    ``atom_rows`` keeps only linear half-spaces, so a disjunction or a ``!=`` in
    the guard reaches the LP as no rows at all — the domain the certificate is
    proved over is then weaker than the loop's own guard. Splitting recovers those
    rows, the way :func:`enumerate_paths` splits an ``ite`` in the body.

    Returns ``[guard]`` unchanged when it is already a conjunction of half-spaces,
    and also when the split would exceed ``max_pieces`` (a weaker domain only
    costs certificates, never soundness)."""
    pieces = [guard]
    for _ in range(max_pieces):
        out, changed = [], False
        for g in pieces:
            conj = _flatten_and(g)
            for i, c in enumerate(conj):
                alts = _convex_alternatives(c)
                if alts is None:
                    continue
                rest = conj[:i] + conj[i + 1:]
                out += [z3.And(*rest, a) if rest else a for a in alts]
                changed = True
                break
            else:
                out.append(g)
        if not changed:
            return pieces
        if len(out) > max_pieces:
            return [guard]
        pieces = out
    return [guard]


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


def build_integer_system(sp_signs, s_signs, guard, invariants, lower, post,
                         syms, delta):
    """Rows of ``signs ∧ guard ∧ invariants ∧ ¬(lower - post >= delta)`` as an
    integer system ``A·s <= b`` with per-row labels. Infeasibility of this system
    certifies the decrease on the cell.

    ``lower`` under-approximates ``V(s)`` (the mask bound, valid with no
    pre-state signs) and ``post`` is ``V(s')`` exactly on ``sp_signs``, so
    ``lower - post <= V(s) - V(s')`` and the decrease follows. ``s_signs`` is
    empty unless the cell needed pre-state literals to certify.

    Each row is also emitted gcd-tightened (divided through by the gcd of its
    coefficients, constant floored): valid over the integers but out of reach of the
    rational LP, as ``2y <= 1`` gives ``y <= 0``. Lean proves every row by ``omega``,
    which is integer-complete, so a tightened row needs no extra machinery."""
    rows, labels, seen = [], [], set()

    def add(A_row, b, label):
        if all(c == 0 for c in A_row) and b >= 0:
            return                            # always-true constant row (redundant)
        key = (tuple(A_row), b)
        if key in seen:
            return                            # duplicate row
        seen.add(key)
        rows.append((list(A_row), b)); labels.append(label)

    def add_atom(atom, label):
        for A_row, b in atom_rows(atom, syms):
            add(A_row, b, label)
            g = 0
            for x in A_row:
                g = gcd(g, abs(x))
            if g > 1:
                add([x // g for x in A_row], b // g, label + "/int")

    for j, c in enumerate(sp_signs):
        add_atom(c, f"cell_sp[{j}]")
    for j, c in enumerate(s_signs):
        add_atom(c, f"cell_s[{j}]")
    for i, g in enumerate(_flatten_and(guard)):
        add_atom(g, f"guard[{i}]")
    for i, inv in enumerate(invariants):
        add_atom(inv, f"inv[{i}]")

    # negated decrease: lower - post <= delta - 1  (integer). Kept unconditionally.
    alpha, beta = affine_coeffs(z3.simplify(lower - post), syms)
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
    multipliers, the successor activation pattern that defines the cell, and V's
    two affine forms (all consumed by the Lean emitter).

    ``pattern_sp`` is the successor pattern whose signs define the cell; ``mu`` the
    mask the certificate chose for the bound on ``V(s)``.

    ``pre_affine``/``post_affine`` are ``(coeffs, const)`` over the state columns,
    with ``pre_affine <= V(s)`` and ``V(s') = post_affine`` on this cell — the
    right-hand sides of the emitted mask-bound and collapse lemmas.

    ``pattern_s`` is ``None`` unless the cell was narrowed by pre-state literals. It
    is then *partial*: a sign per unit the narrowing pinned, ``None`` for the rest,
    since splitting stops as soon as some mask certifies."""
    A: tuple
    b: tuple
    y: tuple
    labels: tuple
    pattern_sp: tuple
    mu: tuple
    pre_affine: tuple = ((), 0)
    post_affine: tuple = ((), 0)
    pattern_s: tuple | None = None


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
    the union (every path strictly drops ``V``) is termination of the program.

    ``units`` holds each hidden unit's pre-activation as ``(coeffs, const)`` over
    the pre-state columns: the first half evaluated at ``s``, the second at the
    successor. Read them through :attr:`pre_units` / :attr:`post_units`. A cell is
    the region where the successor expressions take the signs its pattern names,
    which is what lets the emitter case-split on them."""
    guard: object
    body: tuple
    cells: tuple
    units: tuple = ()

    @property
    def pre_units(self) -> tuple:
        return self.units[:len(self.units) // 2]

    @property
    def post_units(self) -> tuple:
        return self.units[len(self.units) // 2:]


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


@dataclass(frozen=True)
class _Path:
    """What every cell attempt on one path reads, built once by
    :func:`_certify_path`: the state symbols, the path's guard and invariants and
    their conjunction ``dom``, the decrease margin, the hidden pre-activations at
    each end of a step, and the output-layer weights."""
    s_syms: list
    guard: object
    invariants: tuple
    dom: object
    delta: float
    z_s: tuple
    z_sp: tuple
    weights: tuple

    @staticmethod
    def of(layers, s_syms, body, guard, invariants, delta) -> "_Path":
        invariants = tuple(invariants)
        return _Path(s_syms=s_syms, guard=guard, invariants=invariants,
                     dom=z3.And(guard, *invariants) if invariants else guard,
                     delta=delta,
                     z_s=tuple(_pre_activations(layers, s_syms)),
                     z_sp=tuple(_pre_activations(layers, body)),
                     weights=output_weights(layers))


def _signs_at(model, exprs):
    """The sign pattern ``model`` gives ``exprs``: ``True`` where positive."""
    return tuple(model.eval(e, model_completion=True).as_long() > 0
                 for e in exprs)


def _try_cell(p: _Path, lam, mu, pat_s):
    """Farkas-certify one cell: the successor pattern ``lam``, the mask ``mu`` for
    the lower bound on ``V(s)``, and any pre-state literals ``pat_s`` narrowing it.
    Returns a :class:`CellCert`, or ``None`` if the LP is feasible."""
    s_signs = strict_signs(p.z_s, pat_s) if pat_s is not None else ()
    lower = masked_value(p.z_s, p.weights, mu)
    post = masked_value(p.z_sp, p.weights, lam)
    A, b, labels = build_integer_system(strict_signs(p.z_sp, lam), s_signs,
                                        p.guard, list(p.invariants),
                                        lower, post, p.s_syms, p.delta)
    y = find_infeasibility_certificate(A, b)
    if y is None:
        return None
    pre_c, pre_k = affine_coeffs(z3.simplify(lower), p.s_syms)
    post_c, post_k = affine_coeffs(z3.simplify(post), p.s_syms)
    return CellCert(tuple(map(tuple, A)), tuple(b), tuple(y), tuple(labels),
                    tuple(lam), tuple(mu), (tuple(pre_c), pre_k),
                    (tuple(post_c), post_k),
                    tuple(pat_s) if s_signs else None)


def _certify_cell(p: _Path, lam, hint):
    """Certify the successor-pattern class ``lam``, searching for a mask.

    ``hint`` — the witness's own pre-state pattern, where the bound is exact — is
    tried alone first, before the sign-status queries :func:`_narrow` needs; that
    settles most cells in a single LP. Failing that the class is checked against
    the *exact* decrease, which no mask can beat, so a failure there is a genuine
    counterexample rather than a bound too weak to certify. Only then does
    :func:`_narrow` search masks and, where it must, split.

    Returns ``(cells, status)``: the cells on ``"ok"``, else ``None`` with status
    ``"decrease"`` (the real V does not drop somewhere in the class) or
    ``"uncertifiable"``."""
    cert = _try_cell(p, lam, hint, None)
    if cert is not None:
        return [cert], "ok"

    region = z3.And(p.dom, *strict_signs(p.z_sp, lam))
    if _feasible(z3.And(region, exact_value(p.z_s, p.weights)
                        - exact_value(p.z_sp, p.weights) < p.delta)):
        return None, "decrease"

    cells = _narrow(p, lam, hint, region, (None,) * len(lam))
    return cells, "ok" if cells is not None else "uncertifiable"


def _narrow(p: _Path, lam, hint, region, pinned):
    """The cells certifying ``region`` — the class narrowed by whatever pre-state
    literals ``pinned`` names — or ``None`` if some part of it stays uncertifiable.

    Splitting is lazy: pin a single sign-indefinite unit, then retry the mask
    search on each half. Pinning a unit makes it sign-definite there, so its mask
    bit settles and its slack vanishes; often that is enough and the sibling units
    stay open. With every unit pinned the mask matches the true pattern and the
    bound is exact, so the recursion bottoms out at the joint activation cell."""
    fixed, free = _sign_status(region, p.z_s)
    for mu in _mask_candidates(fixed, free, hint, len(p.z_s)):
        cert = _try_cell(p, lam, mu, pinned)
        if cert is not None:
            return [cert]
    if not free:
        return None                   # fully pinned and still no certificate

    j = free[0]
    cells = []
    for truth in (True, False):
        sub_pinned = list(pinned)
        sub_pinned[j] = truth
        sub = z3.And(region, (p.z_s[j] > 0) if truth else (p.z_s[j] <= 0))
        if not _feasible(sub):
            continue
        got = _narrow(p, lam, hint, sub, tuple(sub_pinned))
        if got is None:
            return None
        cells.extend(got)
    return cells


def _certify_path(layers, s_syms, body, guard, invariants, delta, max_iters):
    """CEGAR over one affine path: certify the decrease on every cell of
    ``guard ∧ invariants`` under next-state ``body``. Repeatedly find an
    uncovered in-domain state, certify the cell of successor signs it lies in, and
    block that cell — until the path's domain is exhausted. Because the blocked
    regions are complementary, exhausting the domain *is* the coverage guarantee.

    Returns ``(ok, cells, counterexample, status)``."""
    p = _Path.of(layers, s_syms, body, guard, invariants, delta)
    solver = z3.Solver()
    solver.add(p.dom)
    cells: list[CellCert] = []

    for _ in range(max_iters):
        r = solver.check()
        if r == z3.unsat:
            return True, cells, None, "VERIFIED"
        if r == z3.unknown:
            return False, cells, None, "UNKNOWN"
        model = solver.model()
        s_val = [model.eval(x, model_completion=True).as_long() for x in s_syms]
        lam, hint = _signs_at(model, p.z_sp), _signs_at(model, p.z_s)

        # genuine counterexample: the real V does not decrease at this state
        # (both masks match the true signs here, so both values are exact)
        drop = (masked_value(p.z_s, p.weights, hint)
                - masked_value(p.z_sp, p.weights, lam))
        if model.eval(drop, model_completion=True).as_long() < delta:
            return False, cells, np.array(s_val, dtype=np.float64), "FAILED(decrease)"

        try:
            new, status = _certify_cell(p, lam, hint)
        except ValueError:        # non-affine even after splitting (nondet, etc.)
            return False, cells, np.array(s_val, dtype=np.float64), "FAILED(non-affine)"
        if status != "ok":
            return (False, cells, np.array(s_val, dtype=np.float64),
                    f"FAILED({status})")
        cells.extend(new)

        # block the whole successor-sign class so the next witness lies elsewhere
        solver.add(z3.Not(z3.And(*strict_signs(p.z_sp, lam))))

    return False, cells, None, "FAILED(max_iters)"


def certify_decrease(layers, s_syms, sp_syms, guard, invariants, delta,
                     max_iters: int = 1000) -> FarkasResult:
    """Certify the loop's decrease by splitting the guard into convex pieces
    (:func:`split_guard`), each of those into affine paths
    (:func:`enumerate_paths`), and certifying each with the cell/CEGAR engine
    (:func:`_certify_path`). Every feasible path must strictly drop ``V``; the
    result carries one :class:`PathCert` per path (a non-branching loop under a
    conjunctive guard is a single path, so this subsumes the scalar case)."""
    body = _on_guard_body(sp_syms, s_syms)
    paths: list[PathCert] = []
    for pguard, pbody in [p for g in split_guard(guard)
                          for p in enumerate_paths(body, g)]:
        dom = z3.And(pguard, *invariants) if invariants else pguard
        if not _feasible(dom):
            continue                       # dead path (guard unsat) — never taken
        ok, cells, cex, status = _certify_path(
            layers, s_syms, pbody, pguard, invariants, delta, max_iters)
        if not ok:
            return FarkasResult(False, [], cex, status)
        units = tuple(affine_coeffs(e, s_syms)
                      for e in (_pre_activations(layers, s_syms)
                                + _pre_activations(layers, pbody)))
        paths.append(PathCert(pguard, tuple(pbody), tuple(cells), units))
    if not paths:
        return FarkasResult(False, [], None, "FAILED(no feasible path)")
    return FarkasResult(True, paths, None, "VERIFIED")
