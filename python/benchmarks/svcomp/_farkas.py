"""A decision procedure over a reactive module: Farkas-certified regions (CEGAR).

Given a :class:`System` — a module read once, by :func:`read_system` — and a
property over it, :func:`certify` proves the property on every step of its domain
by a :class:`Step` rule: one boolean formula over the graph's *wires* that must
hold on each step. The rule names wires (``W[wire]``) and columns (``S[name]``,
``S.next[name]``); it knows nothing of programs or ranks. Termination is the
client :func:`decrease` — ``V(s) - V(s') >= delta`` over two wires computing the
same function at each end of a step — plus the substrate's well-foundedness
theorem.

The procedure is sound and incomplete, and what it cannot handle it refuses by
name: :data:`OPS` for the theory's operations, :func:`check_supported` for the
module, :func:`rule_for` for the property, :func:`_dnf` and :func:`_polarity` for
the rule's shape.

The method
==========
A wire behind a ReLU network is piecewise-affine: on a fixed **activation
pattern** it is an affine function of the columns, the polyhedron where each
pre-activation has the sign the pattern dictates. The rule's negation is cut into
**disjuncts** of linear rows (:func:`_dnf`); each row needs the wires it names
either *exact* or *bounded from below*, and which is read off its coefficients
(:func:`_polarity`): where a wire's coefficient is non-negative in every row
``A·x <= b`` a lower bound only weakens the row, so a mask suffices; everywhere
else the wire is pinned to a region — strictly (``> 0`` active, ``<= 0``
inactive), so regions partition rather than overlap.

  * a pinned wire's pattern is what a **region** constrains; over the region its
    value is one affine piece;
  * a bounded wire takes a mask — each dropped unit contributes ``relu >= 0``, each
    kept one ``relu(z) >= z`` — a free parameter of the certificate, sound for
    any choice and exact where it matches the true pattern.

Over a region each disjunct is a linear infeasibility, discharged by proving
``region ∧ domain ∧ invariants ∧ disjunct`` infeasible via Farkas' lemma (over
z3's exact LRA) — an exact, checkable integer certificate (the multipliers ``y``:
``y >= 0``, ``Aᵀy = 0``, ``b·y < 0``). Every disjunct refuted is the rule proved
on the region. CEGAR finds the regions: an uncovered state names one by its
pinned pattern; certifying and blocking it until none is left is the coverage
proof, since regions are complementary.

``V >= 0`` for a ranking network is *not* Farkas-certified: it holds structurally
because the output layer is non-negative. The mask bound needs that same
non-negativity — see :func:`output_weights`.
"""
from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd, lcm

import numpy as np
import z3
from zrth import Sort

from ._nodes import ModeKind, Op, Unsupported, free_symbols, node_view
from ._property import Fixpoint


# ---------------------------------------------------------------------------
# The procedure's vocabulary
# ---------------------------------------------------------------------------

def _relu_at(model, e):
    """A ReLU's mode at a witness: active where the pre-activation is positive."""
    return model.eval(e, model_completion=True).as_long() > 0


def _relu_region(e, mode):
    """A ReLU's mode as constraints: strictly ``> 0`` active, ``<= 0`` inactive."""
    return ((e > 0,) if mode else (e <= 0,))


# This procedure's vocabulary: what it understands of the theory's operations, in
# one place. An itype absent here is refused by :func:`._nodes.node_view`; a kind
# with neither a cell rule nor a case split is refused by :func:`check_supported`.
# Adding support for an operation is an entry here and nothing else.
OPS = {
    # affine arithmetic — evaluated straight through
    "LIA_Linear": Op(), "LIA_Add": Op(), "LIA_Sub": Op(), "LIA_Const": Op(),
    "LIA_Id": Op(), "LIA_Transpose": Op(),
    # boolean structure — z3 handles it, and PRED_MODES splits what the LP cannot
    "LIA_And": Op(), "LIA_Or": Op(), "LIA_Xor": Op(), "LIA_Not": Op(),
    "LIA_Le": Op(), "LIA_Lt": Op(), "LIA_Ge": Op(), "LIA_Gt": Op(),
    "LIA_Eq": Op(), "LIA_Ne": Op(),
    # piecewise-linear
    "LIA_ReLU": Op("relu", mode=ModeKind(_relu_at, _relu_region)),
    "LIA_Ite": Op("ite", split=True),
    # recognised, no cell rule — and no Z3 translation either, so in practice a
    # module carrying one is refused by the walk before `check_supported` sees it
    "LIA_Min": Op("min"),
    "LIA_Max": Op("max"),
    "LIA_Argmax": Op("argmax"),
}

# Derived, so the table above stays the single source of truth.
_MODE_OF = {op.kind: op.mode for op in OPS.values() if op.kind and op.mode}
_SPLIT = {op.kind for op in OPS.values() if op.split}


@dataclass(frozen=True)
class Net:
    """``V`` as the cell engine needs it, independent of how it was built.

    ``units`` is each ReLU's pre-activation as ``(coeffs, const)`` over the state
    columns; ``out`` is ``(coeffs, const)`` from the unit outputs to ``V``, so
    ``V = sum_j out_c[j] * relu(unit_j) + out_k``. Both are read off the graph by
    :func:`reading`, so unit count and wiring are not assumed — only that every
    pre-activation is affine in the columns and ``V`` is affine in the unit
    outputs."""
    units: tuple
    out: tuple

    @staticmethod
    def from_layers(layers) -> "Net":
        """A one-hidden-layer ``[(W1,b1),(W2,b2)]`` as a :class:`Net`."""
        (W1, b1), (W2, b2) = layers
        W1, b1, W2, b2 = map(np.asarray, (W1, b1, W2, b2))
        units = tuple((tuple(int(c) for c in W1[j]), int(b1[j]))
                      for j in range(W1.shape[0]))
        return Net(units, (tuple(int(c) for c in W2[0]), int(b2[0])))


def _affine_pair(e, syms):
    """``affine_coeffs`` with tuple coefficients, so a :class:`Net` is hashable."""
    coeffs, const = affine_coeffs(e, syms)
    return tuple(coeffs), const


def _pre_activations(net: Net, inputs):
    """Each unit's pre-activation as a z3 expr over ``inputs``."""
    pres = []
    for coeffs, const in net.units:
        e = z3.IntVal(const)
        for c, x in zip(coeffs, inputs):
            if c:
                e = e + c * x
        pres.append(e)
    return pres


def output_weights(net: Net):
    """``net.out``, checked non-negative: ``V = sum_j c_j relu(z_j) + k``.

    The mask bound needs ``c_j >= 0``, which holds structurally because the output
    layer is frozen non-negative. The assert fails fast on a net that breaks that;
    the binding check is in Lean, where ``affine_mask_le``'s ``hW`` side goal is
    ``0 <= W i j`` and would not close."""
    c, k = net.out
    assert all(x >= 0 for x in c), f"output layer must be non-negative: {c}"
    return list(c), k


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


def pinned_nodes(pinned):
    """``(kind, expr)`` for every node the groups in ``pinned`` name, in order."""
    return tuple((kind, e) for kind, exprs in pinned for e in exprs)


def mode_region(pinned, pattern):
    """The region a mode assignment names, each node through its own kind. Nodes the
    pattern leaves open (``None``) contribute nothing."""
    out = []
    for (kind, e), m in zip(pinned_nodes(pinned), pattern):
        if m is not None:
            out += list(_MODE_OF[kind].region(e, m))
    return tuple(out)


def modes_at(pinned, model):
    """The mode each pinned node is in at ``model``."""
    return tuple(_MODE_OF[kind].at(model, e) for kind, e in pinned_nodes(pinned))


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


def _or_alts(atom):
    """A disjunction as disjoint alternatives, each excluding the earlier ones."""
    if not z3.is_or(atom):
        return None
    alts, earlier = [], []
    for d in atom.children():
        alts.append(z3.And(*earlier, d) if earlier else d)
        earlier.append(z3.simplify(z3.Not(d)))
    return alts


def _ne_alts(atom):
    """``a != b`` as ``a < b`` and ``a > b``."""
    eq = atom.arg(0) if z3.is_not(atom) and z3.is_eq(atom.arg(0)) else None
    if eq is not None:
        return [eq.arg(0) < eq.arg(1), eq.arg(0) > eq.arg(1)]
    if z3.is_distinct(atom) and atom.num_args() == 2:
        return [atom.arg(0) < atom.arg(1), atom.arg(0) > atom.arg(1)]
    return None


def _implies_alts(atom):
    """``a -> c`` as the disjoint ``¬a`` and ``a ∧ c``."""
    if not (z3.is_app(atom) and atom.decl().kind() == z3.Z3_OP_IMPLIES):
        return None
    a, c = atom.arg(0), atom.arg(1)
    return [z3.Not(a), z3.And(a, c)]


def _not_and_alts(atom):
    """``¬(a ∧ b ∧ …)`` as the disjunction of the negations (De Morgan).

    A ``!=`` on a state component simplifies to this shape, so without it the
    conjunct is not a half-space and ``atom_rows`` drops it."""
    if z3.is_not(atom) and z3.is_and(atom.arg(0)):
        return _or_alts(z3.Or(*[z3.Not(c) for c in atom.arg(0).children()]))
    return None


# Predicate kinds the domain splitter knows. A kind maps an atom to disjoint
# alternatives whose union is the atom, or ``None`` if it does not apply — so a
# new kind is an entry here rather than a new splitter.
PRED_MODES = (_or_alts, _ne_alts, _implies_alts, _not_and_alts)


def _convex_alternatives(atom):
    """Disjoint alternatives replacing a non-convex ``atom``, or ``None`` if no
    kind in :data:`PRED_MODES` applies and it is left alone."""
    for kind in PRED_MODES:
        alts = kind(atom)
        if alts is not None:
            return alts
    return None


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


def build_integer_system(pin_signs, bnd_signs, guard, invariants, atoms, syms):
    """Rows of ``signs ∧ guard ∧ invariants ∧ atoms`` as an integer system
    ``A·s <= b`` with per-row labels, plus the atoms that carry information the LP
    cannot express. ``atoms`` is one disjunct of the rule's negation, already
    substituted for the region; infeasibility refutes that disjunct on it.
    ``bnd_signs`` is empty unless the region needed pre-state literals to certify.

    Each row is also emitted gcd-tightened (divided through by the gcd of its
    coefficients, constant floored): valid over the integers but out of reach of the
    rational LP, as ``2y <= 1`` gives ``y <= 0``. Lean proves every row by ``omega``,
    which is integer-complete, so a tightened row needs no extra machinery."""
    rows, labels, seen, unused = [], [], set(), []

    def add(A_row, b, label):
        if all(c == 0 for c in A_row) and b >= 0:
            return                            # always-true constant row (redundant)
        key = (tuple(A_row), b)
        if key in seen:
            return                            # duplicate row
        seen.add(key)
        rows.append((list(A_row), b)); labels.append(label)

    def add_atom(atom, label):
        got = atom_rows(atom, syms)
        if not got and not z3.is_true(z3.simplify(atom)):
            unused.append(atom)      # carries information the LP cannot express
        for A_row, b in got:
            add(A_row, b, label)
            g = 0
            for x in A_row:
                g = gcd(g, abs(x))
            if g > 1:
                add([x // g for x in A_row], b // g, label + "/int")

    for j, c in enumerate(pin_signs):
        add_atom(c, f"cell_sp[{j}]")
    for j, c in enumerate(bnd_signs):
        add_atom(c, f"cell_s[{j}]")
    for i, g in enumerate(_flatten_and(guard)):
        add_atom(g, f"guard[{i}]")
    for i, inv in enumerate(invariants):
        add_atom(inv, f"inv[{i}]")
    for i, a in enumerate(atoms):
        if not atom_rows(a, syms):
            raise ValueError(f"rule atom is not linear over the columns: {a}")
        add_atom(a, f"rule[{i}]")
    return [r[0] for r in rows], [r[1] for r in rows], labels, tuple(unused)


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
    """One Farkas certificate: a region's rows and one disjunct of the rule's
    negation, infeasible together (the integer system and its multipliers, for
    the Lean emitter).

    ``pattern_sp`` is the pinned group's activation pattern, which defines the
    region; ``mu`` the mask the certificate chose on the bounded group.
    ``pattern_s`` is ``None`` unless the cell was narrowed by bounded-group
    literals. It is then *partial*: a sign per unit the narrowing pinned, ``None``
    for the rest, since splitting stops as soon as some mask certifies.

    ``disjunct`` indexes the rule's disjuncts. ``affines`` gives, per device, the
    ``(coeffs, const)`` of its value under the region and mask over the state
    columns — exact for a pinned device, a lower bound for a bounded one — the
    right-hand sides of the emitted collapse and bound lemmas."""
    A: tuple
    b: tuple
    y: tuple
    labels: tuple
    pattern_sp: tuple
    mu: tuple
    pattern_s: tuple | None = None
    disjunct: int = 0
    affines: tuple = ()


@dataclass(frozen=True)
class Device:
    """A wire the rule names with a reading behind it, as the proof renders it:
    ``net`` indexes the result's distinct networks; ``inputs`` says which end of
    each column the reading takes (``"latched"``, ``"next"`` or ``"unread"``);
    ``pinned`` whether regions fix its pattern or a mask bounds it; ``offset``
    where its units start within that group."""
    wire_id: int
    net: int
    inputs: tuple
    pinned: bool
    offset: int


@dataclass
class FarkasResult:
    """What a ``certify`` run established, and what the proof of it needs: the
    per-path certificates, the rule with its formula ``ok`` resolved over the
    columns and the wire symbols, and the named wires as devices over the
    distinct networks they read through."""
    verified: bool
    certificates: list           # list[PathCert]
    counterexample: object = None
    status: str = ""
    unused: tuple = ()           # domain atoms the LP could not express
    rule: object = None
    ok: object = None            # the rule's formula, z3 over columns and wires
    devices: tuple = ()          # Device, in the order the rule named them
    nets: tuple = ()             # the distinct Net each device reads through


@dataclass(frozen=True)
class PathCert:
    """One affine path of the loop body: its path condition ``guard`` (the loop
    guard strengthened by the branch literals taken along the path), its affine
    next-state ``body`` (z3 exprs over the pre-state symbols), and the per-cell
    certificates on it. The paths partition the loop guard, so the union of their
    ``Step`` relations is the loop's transition — hence a property of every path's
    steps is a property of the program's.

    ``pinned_units`` / ``bounded_units`` hold each hidden unit's pre-activation at
    this path's round as ``(coeffs, const)`` over the pre-state columns, for the
    pinned and the bounded group; ``device_units`` the same per device. A cell is
    the region where the pinned group's expressions take the signs its pattern
    names, which is what lets the emitter case-split on them."""
    guard: object
    body: tuple
    cells: tuple
    pinned_units: tuple = ()
    bounded_units: tuple = ()
    device_units: tuple = ()


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


def lift_ites(pred):
    """``pred`` with every ``ite`` *term* case-split away, leaving a Boolean
    combination of linear atoms.

    An ``ite`` inside a comparison is not a half-space, so ``atom_rows`` drops the
    conjunct holding it. The domain ``T(s) != s`` puts the transition's ``ite``s
    there by construction, so they have to be lifted before the domain is split."""
    cond = _find_ite_cond(pred)
    if cond is None:
        return pred
    return z3.Or(*[z3.And(cond if truth else z3.Not(cond),
                          lift_ites(_select(pred, cond, truth)))
                   for truth in (True, False)])


# ---------------------------------------------------------------------------
# The system: a module read once
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class System:
    """A reactive module as read, plus what is known about its states.

    Built by :func:`read_system`, the one place a module is walked; everything
    here is a value, so refining a system costs nothing and re-reads nothing.

    ``pairs`` are the *columns*: the ctrl pairs whose latched wire some term
    reads. That is what a free input to the round is — a value the round depends
    on and carries in — so it is the state the proofs quantify over. A latched
    wire nothing reads (a ranking function's own previous value, composed
    alongside) is not a column; its next value is still a wire of the graph and
    a rule may name it. Nothing is classified by kind: only by data flow.

    ``W`` gives each ctrl next wire a symbol a rule's predicate can name; the
    engine resolves it, per region, to the wire's value at that round.
    ``assume`` is what may be assumed of the entry state (a ``state_map ->
    predicates`` callable, or ``None``) and ``invariants`` an over-approximation
    of the reachable states. Both describe the module's state space, not any
    property of it. ``names`` is provenance only, defaulting to the wire ids."""
    module: object
    view: object
    pairs: tuple
    all_pairs: tuple
    s_syms: tuple
    seed: dict
    W: dict
    atom_of: dict = field(default_factory=dict)
    entry_inputs: tuple = ()
    names: tuple = ()
    assume: object = None
    invariants: tuple = ()

    @property
    def sp_syms(self) -> list:
        """The transition, read off the columns' next wires."""
        return [self.view.values[pr[1]][0] for pr in self.pairs]

    def index(self, pair) -> int:
        for k, pr in enumerate(self.pairs):
            if pr[0].id == pair[0].id:
                return k
        raise Unsupported(f"wire pair {pair[0].id} is not a column of this system")

    def sym_of(self, pair):
        return self.s_syms[self.index(pair)]

    def next_of(self, pair):
        return self.view.values[pair[1]][0]

    @property
    def entry(self) -> dict:
        """Each column's value at tick 0, by name, from the init block."""
        return {n: self.view.entry[pr[1]][0] for n, pr in zip(self.names, self.pairs)}

    @property
    def s_map(self) -> dict:
        return dict(zip(self.names, self.s_syms))

    @property
    def sp_map(self) -> dict:
        return dict(zip(self.names, self.sp_syms))


_SCALAR = Sort.Int([1, 1])


def read_system(module, names=()) -> System:
    """``module`` walked once, as a :class:`System`.

    Every latched ctrl wire is seeded with a symbol, and awaited inputs with
    theirs; the columns are the latched wires some update term actually reads.
    ``names`` labels the columns — a tuple in column order, or a mapping from a
    latched wire to its name.

    Wires are scalar integers: a symbol per wire is what the rows, the regions
    and the proof's ``Vector n Int`` state quantify over, so anything else is
    refused here by name rather than read element by element."""
    all_pairs = tuple(tuple(pr) for pr in module.ctrl)
    inputs = tuple(tuple(pr) for pr in module.extl)
    for w in (w for pr in all_pairs + inputs for w in pr):
        if w.dtype != _SCALAR:
            raise Unsupported(f"wire {w.id} has sort Int{w.dtype[0]}; only scalar "
                              f"integer wires are supported")
    read = {w.id for a in module.atoms for w in a.read}
    pairs = tuple(pr for pr in all_pairs if pr[0].id in read)
    if isinstance(names, dict):
        names = tuple(names.get(pr[0], names.get(pr[0].id, f"w{pr[0].id}")) for pr in pairs)
    else:
        names = tuple(names) or tuple(f"w{pr[0].id}" for pr in pairs)
    if len(names) != len(pairs):
        raise Unsupported(f"{len(names)} names for {len(pairs)} columns")
    syms = tuple(z3.Int(n) for n in names)
    seed = {pr[0]: [s] for pr, s in zip(pairs, syms)}
    for pr in all_pairs:                       # unread latched wires: seeded, never used
        seed.setdefault(pr[0], [z3.Int(f"_w{pr[0].id}")])
    for i, pr in enumerate(inputs):
        seed[pr[0]] = [z3.Int(f"_in{i}")]
        seed[pr[1]] = [z3.Int(f"_in{i}_next")]
    entry_seed = {pr[1]: [z3.Int(f"_entry{i}")] for i, pr in enumerate(inputs)}
    W = {pr[1].id: z3.Int(f"_W{pr[1].id}") for pr in all_pairs}
    atom_of = {w.id: a for a in module.atoms for w in a.ctrl}
    return System(module, node_view(module, seed, OPS, entry_seed), pairs, all_pairs,
                  syms, seed, W, atom_of=atom_of,
                  entry_inputs=tuple(v[0] for v in entry_seed.values()), names=names)


@dataclass(frozen=True)
class Reading:
    """A ctrl next wire as a function of the round: its network over the atom's
    inputs, and where each input comes from — ``("latched", k)`` the column
    ``k``'s pre-state value, ``("next", k)`` its value after the step,
    ``("unread", k)`` a column the atom does not read. A wire with no ReLU behind
    it is a reading with no units, its ``out`` affine in the inputs."""
    wire: object
    net: Net
    inputs: tuple

    def args(self, s_syms, body):
        """The reading's inputs at the round ``(s_syms, body)``."""
        return tuple(body[k] if kind == "next" else s_syms[k] for kind, k in self.inputs)

    def at(self, s_syms, body):
        """The reading's pre-activations at the round ``(s_syms, body)``."""
        return tuple(_pre_activations(self.net, self.args(s_syms, body)))

    def value(self, s_syms, body, pattern=None):
        """The wire's value at the round: exact with the ReLUs in when ``pattern``
        is ``None``, else the mask ``pattern`` applied."""
        if not self.net.units:
            return _affine_value(self.net.out, self.args(s_syms, body))
        acts = self.at(s_syms, body)
        weights = output_weights(self.net)
        return (exact_value(acts, weights) if pattern is None
                else masked_value(acts, weights, pattern))


def _affine_value(weights, args):
    """``Σ cⱼ·argsⱼ + k`` for ``weights = (c, k)`` — a ReLU-free wire's value."""
    c, k = weights
    out = z3.IntVal(k)
    for cj, a in zip(c, args):
        if cj:
            out = out + cj * a
    return out


def reading(system: System, wire) -> Reading:
    """The :class:`Reading` behind ``wire``, a ctrl next wire.

    Its atom is walked alone with the wires its **update block** reads seeded as
    the corresponding column symbols, so its structure comes out over the columns
    whatever round it is applied to; ``inputs`` records which end of each column
    the update takes, so the engine can evaluate it at either. (The atom's
    ``wait`` interface also lists what its *init* awaits, which is not this.)"""
    atom = system.atom_of.get(wire.id)
    if atom is None:
        raise Unsupported(f"wire {wire.id} is not a ctrl wire of this system")
    latched = {pr[0].id: k for k, pr in enumerate(system.pairs)}
    nexts = {pr[1].id: k for k, pr in enumerate(system.pairs)}
    rd, wr = atom.update.read(), atom.update.write()
    reads = [rd[i] for i in range(len(rd))]
    at, kind_of = {}, {}
    written = {wr[i].id for i in range(len(wr))}
    for w in reads:
        if w.id in written:
            continue                                   # an internal wire of the block
        if w.id in latched:
            at[w] = [system.s_syms[latched[w.id]]]; kind_of[latched[w.id]] = "latched"
        elif w.id in nexts:
            at[w] = [system.s_syms[nexts[w.id]]]; kind_of[nexts[w.id]] = "next"
        else:
            raise Unsupported(f"the atom behind wire {wire.id} reads wire {w.id}, "
                              f"which is not a column of this system")
    view = node_view(system.module, at, OPS, atoms=(atom,))
    value = view.opaque[wire][0]
    relus = view.feeding(value, "relu")
    col_syms = list(system.s_syms)
    if relus:
        net = Net(tuple(_affine_pair(n.args[0], col_syms) for n in relus),
                  _affine_pair(value, [n.sym for n in relus]))
    else:
        net = Net((), _affine_pair(value, col_syms))
    inputs = tuple((kind_of.get(k, "unread"), k) for k in range(len(system.pairs)))
    return Reading(wire, net, inputs)


def entry_predicate(system: System):
    """The entry state as a predicate over ``system``'s pre-state symbols.

    ``system.entry`` is the init block's value per column, with each nondet input
    a fresh symbol; this conjoins ``v == <v's entry value>`` per column plus
    ``assume``. Inputs are substituted away first (see below), so the result is a
    relation between columns, e.g. ``i == n - 1``."""
    init_vals, s_map = system.entry, system.s_map
    # A column initialised to a bare input (``v := input``) holds that input at
    # entry, so the input symbol can be replaced by v's state symbol.
    sub = []
    for insym in system.entry_inputs:
        for v in system.names:
            if z3.eq(z3.simplify(init_vals[v]), insym):
                sub.append((insym, s_map[v]))
                break
    conj = []
    for n in system.names:
        e = z3.substitute(init_vals[n], *sub) if sub else init_vals[n]
        conj.append(s_map[n] == e)
    conj += list((system.assume or (lambda st: []))(s_map))
    return z3.And(*conj) if conj else z3.BoolVal(True)


# ---------------------------------------------------------------------------
# The rule: one formula over wires
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    """How this procedure discharges a property: a formula that holds on every
    step of the property's domain.

    ``ok`` is ``(W, S) -> BoolRef``, a boolean combination of linear comparisons —
    ``W[wire]`` the value a ctrl next wire takes this round, ``S[name]`` a
    column's latched value and ``S.next[name]`` its value after the step — so a
    rule is written over the graph and knows nothing of programs or ranks. Which
    wires must be pinned to a region and which a bound suffices for is read off
    the formula's coefficients, not declared.

    ``ranks`` pairs ``(at s, at s')`` wires the emitter renders as one network
    ``V``."""
    ok: object
    proves: type
    ranks: tuple = ()


def decrease(v_s, v_sp, delta: float = 1.0) -> Step:
    """Termination by one rank: ``V(s) - V(s') >= delta`` on every step. ``v_s`` and
    ``v_sp`` are the next wires carrying the rank at the pre- and post-state — two
    readings of the same function, one of the latched state, one of the next.
    ``V >= 0`` is structural (a non-negative output layer) and checked before."""
    d = int(delta)
    return Step(ok=lambda W, S: W[v_s] - W[v_sp] >= d, proves=Fixpoint,
                ranks=((v_s, v_sp),))


def rule_for(prop, system, rule=None) -> Step:
    """``rule`` checked against ``prop`` on ``system``.

    :class:`Fixpoint` has nothing to guess a rank from, so a rule is asked for
    rather than invented."""
    if rule is None:
        if isinstance(prop, Fixpoint):
            raise Unsupported("Fixpoint needs a rank: pass decrease(v_s, v_sp)")
        raise Unsupported(f"no rule for property {type(prop).__name__!r}")
    if not isinstance(prop, rule.proves):
        raise Unsupported(f"the rule proves {rule.proves.__name__}, "
                          f"not {type(prop).__name__}")
    return rule


class _WireMap:
    """``W`` as a predicate sees it: indexable by a ctrl next wire, handing out the
    system's symbol for it and recording which wires were named. ``next`` is the
    same map one round on, which this procedure cannot yet evaluate — so it
    refuses by name, as does a map built with ``refuse`` for a predicate that may
    not name wires at all."""
    def __init__(self, system, used=None, refuse=None):
        self.system, self.refuse = system, refuse
        self.used = [] if used is None else used

    def __getitem__(self, wire):
        if self.refuse:
            raise Unsupported(f"wire {wire.id}: {self.refuse}")
        sym = self.system.W.get(wire.id)
        if sym is None:
            raise Unsupported(f"wire {wire.id} is not a ctrl wire of this system")
        if all(w.id != wire.id for w in self.used):
            self.used.append(wire)
        return sym

    @property
    def next(self):
        return _WireMap(self.system, self.used,
                        refuse="a wire's value one round ahead is not available")


class _StateMap:
    """``S`` as a predicate sees it: a column's latched value by name, and
    ``next`` the same columns after the step — each the ``W`` of the column's
    next wire, so naming one is recorded like naming a wire."""
    def __init__(self, system, W, ahead: bool = False):
        self.system, self.W, self.ahead = system, W, ahead

    def __getitem__(self, name):
        if name not in self.system.names:
            raise Unsupported(f"{name!r} is not a column of this system")
        k = self.system.names.index(name)
        if self.ahead:
            return self.W[self.system.pairs[k][1]]
        return self.system.s_syms[k]

    @property
    def next(self):
        if self.ahead:
            raise Unsupported("the state two rounds ahead is not available")
        return _StateMap(self.system, self.W, ahead=True)


def _resolve(rule: Step, system: System):
    """The rule's formula as z3 over the columns' symbols and the wire symbols
    ``system.W`` — the form the engine cuts into rows and substitutes into — with
    the wires it named, in order."""
    used = []
    W = _WireMap(system, used)
    return lift_ites(rule.ok(W, _StateMap(system, W))), tuple(used)


def _is_cmp(e) -> bool:
    return z3.is_app(e) and e.decl().kind() in _CMP


def _dnf(e) -> list:
    """The disjuncts of ``e``: each a list of linear atoms whose conjunction is
    one disjunct of ``e``'s disjunctive normal form. Negation is pushed to the
    atoms (an atom may stay negated — :func:`atom_rows` flips it), ``a != b``
    becomes its two strict sides, an implication its disjunction. Anything that
    is not a boolean combination of comparisons is refused by name."""
    if z3.is_true(e):
        return [[]]
    if z3.is_false(e):
        return []
    if z3.is_and(e):
        out = [[]]
        for c in e.children():
            out = [a + b for a in out for b in _dnf(c)]
        return out
    if z3.is_or(e):
        return [d for c in e.children() for d in _dnf(c)]
    kind = e.decl().kind() if z3.is_app(e) else None
    if kind == z3.Z3_OP_IMPLIES:
        return _dnf(z3.Or(z3.Not(e.arg(0)), e.arg(1)))
    if kind == z3.Z3_OP_IFF:
        a, b = e.arg(0), e.arg(1)
        return _dnf(z3.Or(z3.And(a, b), z3.And(z3.Not(a), z3.Not(b))))
    if kind == z3.Z3_OP_ITE:
        c, t, f = e.arg(0), e.arg(1), e.arg(2)
        return _dnf(z3.Or(z3.And(c, t), z3.And(z3.Not(c), f)))
    if z3.is_distinct(e) and e.num_args() == 2:
        return _dnf(z3.Or(e.arg(0) < e.arg(1), e.arg(0) > e.arg(1)))
    if z3.is_not(e):
        x = e.arg(0)
        if z3.is_not(x):
            return _dnf(x.arg(0))
        if z3.is_true(x) or z3.is_false(x) or z3.is_and(x) or z3.is_or(x):
            return _dnf(z3.simplify(z3.Not(x))) if z3.is_true(x) or z3.is_false(x) else (
                _dnf(z3.Or(*[z3.Not(c) for c in x.children()])) if z3.is_and(x)
                else _dnf(z3.And(*[z3.Not(c) for c in x.children()])))
        xk = x.decl().kind() if z3.is_app(x) else None
        if xk == z3.Z3_OP_IMPLIES:
            return _dnf(z3.And(x.arg(0), z3.Not(x.arg(1))))
        if xk == z3.Z3_OP_IFF:
            a, b = x.arg(0), x.arg(1)
            return _dnf(z3.Or(z3.And(a, z3.Not(b)), z3.And(z3.Not(a), b)))
        if xk == z3.Z3_OP_ITE:
            c, t, f = x.arg(0), x.arg(1), x.arg(2)
            return _dnf(z3.Or(z3.And(c, z3.Not(t)), z3.And(z3.Not(c), z3.Not(f))))
        if z3.is_eq(x):
            return _dnf(z3.Or(x.arg(0) < x.arg(1), x.arg(0) > x.arg(1)))
        if z3.is_distinct(x) and x.num_args() == 2:
            return [[x.arg(0) == x.arg(1)]]
        if _is_cmp(x):
            return [[e]]
    elif _is_cmp(e):
        return [[e]]
    raise Unsupported("rule predicate is not a boolean combination of linear "
                      f"comparisons: {e}")


def _polarity(disjuncts, wires, syms):
    """Which named wires must be pinned to a region for the rule's rows to be
    sound. A wire is substituted by a *lower bound* where that only weakens the
    row — a non-negative coefficient in the row ``A·x <= b`` — and must be exact
    (pinned) where any row has it negative; an equality pins. Refuses by name a
    rule atom that is not linear over the columns and the named wires."""
    pin = set()
    n_cols = len(syms) - len(wires)
    for atoms in disjuncts:
        for a in atoms:
            rows = atom_rows(a, syms)
            if not rows:
                raise Unsupported("rule predicate is not a linear comparison over the "
                                  f"columns and the named wires: {a}")
            for A_row, _ in rows:
                for w, c in zip(wires, A_row[n_cols:]):
                    if c < 0:
                        pin.add(w.id)
    return pin


# ---------------------------------------------------------------------------
# Doors
# ---------------------------------------------------------------------------

def check_kinds(nodes) -> None:
    """Raise :class:`Unsupported` for a node whose kind this procedure has no rule
    for — neither a cell rule nor a case split.

    Defensive: every such kind in :data:`OPS` also has no Z3 translation, so the
    walk refuses it first. It is what catches a kind added to the vocabulary before
    its rule."""
    for node in nodes:
        if node.kind not in _MODE_OF and node.kind not in _SPLIT:
            raise Unsupported(
                f"node kind {node.kind!r} is recognised but has neither a cell "
                f"rule nor a case split; pinnable kinds are {sorted(_MODE_OF)}")


def check_supported(system: System, prop, rule=None) -> None:
    """Raise :class:`Unsupported`, naming the reason, if this procedure has no rule
    for something in ``system``'s module or for ``prop``.

    Checked before any work so the interior can assume its preconditions. The
    alternative — proceeding with whatever it happens to understand — reports a
    proof that will not close rather than the thing it could not use."""
    check_kinds(system.view.nodes)
    names = [str(s) for s in system.s_syms]
    for name, e in zip(names, system.sp_syms):
        extra = free_symbols(e) - set(names)
        if extra:
            raise Unsupported(
                f"the next value of {name!r} reads {sorted(extra)}, which are not "
                f"columns — nondeterministic inputs are not supported")
    rule_for(prop, system, rule)


def expand_cases(guard, body, budget: int = 16):
    """Every ``(guard, body)`` case of the loop, by one recursion over the kinds
    that need a case split.

    Two things need splitting and they are the same operation on different
    subjects: a guard conjunct that is not a half-space (:data:`PRED_MODES` — a
    disjunction, a ``!=``), and an ``ite`` term in the body. Both replace their
    subject by disjoint alternatives and recurse, so at the leaves the guard is a
    conjunction of half-spaces and the body is ite-free — a single affine map under
    a convex domain, which is what the LP needs.

    Guard conjuncts go first, so a body branch is only split once the domain is
    convex. ``budget`` caps how far the guard is split; past it the unsplit guard
    is kept, which only weakens the domain. The cases partition the guard, so the
    union of their ``Step`` relations is exactly the loop's transition."""
    conj = _flatten_and(guard)
    for i, c in enumerate(conj):
        alts = _convex_alternatives(c)
        if alts is None:
            continue
        if len(alts) > budget:
            break                     # stop splitting; a weaker domain is still sound
        rest = conj[:i] + conj[i + 1:]
        out = []
        for a in alts:
            g = z3.And(*rest, a) if rest else a
            out += expand_cases(g, body, budget - len(alts))
        return out
    cond = None
    for e in body:
        cond = _find_ite_cond(e)
        if cond is not None:
            break
    if cond is None:
        return [(guard, body)]
    return (expand_cases(z3.And(guard, cond),
                         [_select(e, cond, True) for e in body], budget)
            + expand_cases(z3.And(guard, z3.Not(cond)),
                           [_select(e, cond, False) for e in body], budget))


def _feasible(pred) -> bool:
    s = z3.Solver()
    s.add(pred)
    return s.check() == z3.sat


# ---------------------------------------------------------------------------
# The engine: regions, masks, one Farkas system per disjunct
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Device:
    """One named wire on a path: its :class:`Device`, its symbol, its reading, and
    the inputs, pre-activations and output weights at this path's round."""
    spec: Device
    sym: object
    reading: Reading
    args: tuple
    acts: tuple
    weights: tuple


@dataclass(frozen=True)
class _Path:
    """What every region attempt on one path reads, built once by
    :func:`certify`: the columns, the path's guard and invariants and their
    conjunction ``dom``, the rule's formula and the disjuncts of its negation, the
    named wires as devices, and the pinned / bounded unit groups.

    A wire is pinned if any row needs it exact (:func:`_polarity`); the pinned
    group is what a region enumerates modes over, the bounded group is where a
    mask suffices."""
    s_syms: tuple
    body: tuple
    guard: object
    invariants: tuple
    dom: object
    ok: object
    disjuncts: tuple
    devices: tuple           # _Device, in the order the rule named the wires
    columns: dict            # wire id -> column index, for a column's own next wire
    pinned: tuple
    bounded: tuple
    unused: set

    @staticmethod
    def of(system, ok, disjuncts, devices, readings, columns, body, guard,
           invariants) -> "_Path":
        invariants = tuple(invariants)
        dom = z3.And(guard, *invariants) if invariants else guard
        s_syms = tuple(system.s_syms)
        devs, pin_acts, bnd_acts = [], [], []
        for spec in devices:
            rd = readings[spec.wire_id]
            args = rd.args(s_syms, body)
            acts = tuple(_pre_activations(rd.net, args))
            weights = output_weights(rd.net) if rd.net.units else rd.net.out
            devs.append(_Device(spec, system.W[spec.wire_id], rd, tuple(args), acts, weights))
            (pin_acts if spec.pinned else bnd_acts).extend(acts)
        return _Path(s_syms=s_syms, body=tuple(body), guard=guard,
                     invariants=invariants, dom=dom, ok=ok, disjuncts=tuple(disjuncts),
                     devices=tuple(devs), columns=dict(columns),
                     pinned=(("relu", tuple(pin_acts)),) if pin_acts else (),
                     bounded=tuple(bnd_acts), unused=set())

    def _values(self, system, value_of) -> list:
        out = [(system.W[wid], self.body[k]) for wid, k in self.columns.items()]
        return out + [(d.sym, value_of(d)) for d in self.devices]

    def subst(self, system, lam, mu) -> list:
        """``(W symbol, value)`` for every named wire under the region ``lam`` and
        mask ``mu``: pinned wires exact on the region, bounded ones from below,
        a column's next wire its body on this path, a ReLU-free wire its value."""
        def value(d):
            n = len(d.acts)
            if not n:
                return _affine_value(d.weights, d.args)
            pat = (lam if d.spec.pinned else mu)[d.spec.offset:d.spec.offset + n]
            return masked_value(d.acts, d.weights, pat)
        return self._values(system, value)

    def exact(self, system) -> list:
        """``(W symbol, value)`` with every ReLU left in — the rule's truth."""
        return self._values(system, lambda d: (exact_value(d.acts, d.weights) if d.acts
                                               else _affine_value(d.weights, d.args)))

    def violation(self, system):
        """The rule false at a state, on exact values."""
        sub = self.exact(system)
        return z3.Not(z3.substitute(self.ok, *sub) if sub else self.ok)

    def affines(self, sub) -> tuple:
        """Per device, the ``(coeffs, const)`` of its substituted value over the
        columns — what the emitter's bound and collapse lemmas state."""
        smap = {str(k): v for k, v in sub}
        return tuple(_affine_pair(z3.simplify(smap[str(d.sym)]), self.s_syms)
                     for d in self.devices)


def _signs_at(model, exprs):
    """The sign pattern ``model`` gives ``exprs``: ``True`` where positive."""
    return tuple(model.eval(e, model_completion=True).as_long() > 0
                 for e in exprs)


def _try_cell(system, p: _Path, lam, mu, pat_b):
    """Farkas-certify one region: the pinned pattern ``lam``, the mask ``mu``, and
    any literals ``pat_b`` narrowing the bounded group.

    One LP per disjunct of the rule's negation — the rows of the region, the
    guard, the invariants and the disjunct. Returns one :class:`CellCert` per
    disjunct, or ``None`` if some disjunct stays feasible on the region."""
    bnd_signs = strict_signs(p.bounded, pat_b) if pat_b is not None else ()
    sub = p.subst(system, lam, mu)
    affines = p.affines(sub)
    certs = []
    for d, atoms in enumerate(p.disjuncts):
        rows = [z3.substitute(a, *sub) for a in atoms] if sub else list(atoms)
        A, b, labels, unused = build_integer_system(
            mode_region(p.pinned, lam), bnd_signs, p.guard, list(p.invariants),
            rows, p.s_syms)
        p.unused.update(str(a) for a in unused)
        y = find_infeasibility_certificate(A, b)
        if y is None:
            return None
        certs.append(CellCert(tuple(map(tuple, A)), tuple(b), tuple(y), tuple(labels),
                              tuple(lam), tuple(mu),
                              tuple(pat_b) if bnd_signs else None,
                              disjunct=d, affines=affines))
    return certs


def _certify_cell(system, p: _Path, lam, hint):
    """Certify the pinned-pattern class ``lam``, searching for a mask.

    ``hint`` — the witness's own bounded-group pattern, where the bound is exact —
    is tried alone first; that settles most regions in a single LP. Failing that
    the class is checked against the rule's *exact* violation, which no mask can
    beat, so a failure there is a genuine counterexample rather than a bound too
    weak to certify. Only then does :func:`_narrow` search masks and, where it
    must, split."""
    certs = _try_cell(system, p, lam, hint, None)
    if certs is not None:
        return certs, "ok"
    region = z3.And(p.dom, *mode_region(p.pinned, lam))
    if _feasible(z3.And(region, p.violation(system))):
        return None, "violated"
    cells = _narrow(system, p, lam, hint, region, (None,) * len(lam))
    return cells, "ok" if cells is not None else "uncertifiable"


def _narrow(system, p: _Path, lam, hint, region, pinned):
    """The cells certifying ``region`` — the class narrowed by whatever bounded
    literals ``pinned`` names — or ``None`` if some part of it stays uncertifiable.

    Splitting is lazy: pin a single sign-indefinite unit, then retry the mask
    search on each half. With every unit pinned the mask matches the true pattern
    and the bound is exact, so the recursion bottoms out at the joint cell."""
    fixed, free = _sign_status(region, p.bounded)
    for mu in _mask_candidates(fixed, free, hint, len(p.bounded)):
        certs = _try_cell(system, p, lam, mu, pinned)
        if certs is not None:
            return certs
    if not free:
        return None
    j = free[0]
    cells = []
    for truth in (True, False):
        sub_pinned = list(pinned)
        sub_pinned[j] = truth
        sub = z3.And(region, (p.bounded[j] > 0) if truth else (p.bounded[j] <= 0))
        if not _feasible(sub):
            continue
        got = _narrow(system, p, lam, hint, sub, tuple(sub_pinned))
        if got is None:
            return None
        cells.extend(got)
    return cells


def _certify_path(system, p: _Path, max_iters):
    """CEGAR over one affine path: discharge the rule on every region of
    ``guard ∧ invariants`` under next-state ``body``. Repeatedly find an uncovered
    in-domain state, certify the region its pinned pattern names, and block that
    region — until the path's domain is exhausted. Because the blocked regions are
    complementary, exhausting the domain *is* the coverage guarantee. With nothing
    pinned there is one region, so one pass."""
    solver = z3.Solver()
    solver.add(p.dom)
    cells: list[CellCert] = []
    s_syms = p.s_syms
    for _ in range(max_iters):
        r = solver.check()
        if r == z3.unsat:
            return True, cells, None, "VERIFIED"
        if r == z3.unknown:
            return False, cells, None, "UNKNOWN"
        model = solver.model()
        s_val = [model.eval(x, model_completion=True).as_long() for x in s_syms]
        lam, hint = modes_at(p.pinned, model), _signs_at(model, p.bounded)
        if z3.is_true(model.eval(p.violation(system), model_completion=True)):
            return False, cells, np.array(s_val, dtype=np.float64), "FAILED(violated)"
        try:
            new, status = _certify_cell(system, p, lam, hint)
        except ValueError as exc:            # non-affine even after splitting
            return (False, cells, np.array(s_val, dtype=np.float64),
                    f"FAILED(non-affine: {exc})")
        if status != "ok":
            return False, cells, np.array(s_val, dtype=np.float64), f"FAILED({status})"
        cells.extend(new)
        block = mode_region(p.pinned, lam)
        solver.add(z3.Not(z3.And(*block)) if block else z3.BoolVal(False))
    return False, cells, None, "FAILED(max_iters)"


def certify(system: System, prop, rule=None, max_iters: int = 1000) -> FarkasResult:
    """Certify ``prop`` of ``system`` on every step of the property's domain, by
    ``rule`` — one formula over the graph's wires.

    Preconditions first (:func:`check_supported`); the rule's formula is resolved
    over the columns and the named wires, its negation cut into disjuncts of rows
    and the wires classified by their coefficients; then the property's domain is expanded into cases (:func:`expand_cases`),
    each a convex region with an affine transition, and each certified by the
    region/CEGAR engine. Nothing here knows what a program or a rank is."""
    check_supported(system, prop, rule)
    rule = rule_for(prop, system, rule)
    ok, wires = _resolve(rule, system)
    W_syms = [system.W[w.id] for w in wires]
    disjuncts = _dnf(z3.Not(ok))
    pin_ids = _polarity(disjuncts, wires, list(system.s_syms) + W_syms)
    nexts = {pr[1].id: k for k, pr in enumerate(system.pairs)}
    columns = {w.id: nexts[w.id] for w in wires if w.id in nexts}
    readings = {w.id: reading(system, w) for w in wires if w.id not in nexts}
    nets, devices, offset = [], [], {True: 0, False: 0}
    for w in wires:
        if w.id in columns:
            continue
        rd = readings[w.id]
        if rd.net not in nets:
            nets.append(rd.net)
        pinned = w.id in pin_ids
        devices.append(Device(w.id, nets.index(rd.net), rd.inputs, pinned, offset[pinned]))
        offset[pinned] += len(rd.net.units)
    nets, devices = tuple(nets), tuple(devices)

    def result(verified, paths, cex, status, unused=()):
        return FarkasResult(verified, paths, cex, status, tuple(sorted(unused)), rule,
                            ok, devices, nets)

    s_syms, sp_syms = system.s_syms, system.sp_syms
    invariants = system.invariants
    paths: list[PathCert] = []
    unused: set = set()
    dom = lift_ites(prop.domain(system))
    for pguard, pbody in expand_cases(dom, list(sp_syms)):
        region = z3.And(pguard, *invariants) if invariants else pguard
        if not _feasible(region):
            continue                       # dead case — no state takes this step
        p = _Path.of(system, ok, disjuncts, devices, readings, columns, pbody, pguard,
                     invariants)
        verified, cells, cex, status = _certify_path(system, p, max_iters)
        unused |= p.unused
        if not verified:
            return result(False, [], cex, status, unused)
        pin_acts = p.pinned[0][1] if p.pinned else ()
        paths.append(PathCert(
            pguard, tuple(pbody), tuple(cells),
            pinned_units=tuple(_affine_pair(e, s_syms) for e in pin_acts),
            bounded_units=tuple(_affine_pair(e, s_syms) for e in p.bounded),
            device_units=tuple(tuple(_affine_pair(e, s_syms) for e in d.acts)
                               for d in p.devices)))
    if not paths:
        return result(False, [], None, "FAILED(no step in the domain)", unused)
    return result(True, paths, None, "VERIFIED", unused)
