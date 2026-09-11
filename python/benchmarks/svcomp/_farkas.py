"""A decision procedure over a reactive module: Farkas-certified regions (CEGAR).

Given a :class:`System` — a module read once, by :func:`read_system` — a claim
over it and a witness, :func:`certify` proves the witness's obligation on every
round of the claim's domain: one boolean formula over the graph's *wires*. The
formula names wires (``W[wire]``) and columns (``S[name]``, ``S.next[name]``); it
knows nothing of programs or ranks. A :class:`._property.Liveness` claim is
discharged by :func:`lex_decrease` (:func:`decrease` for one rank) plus the
substrate's well-foundedness theorem; a :class:`._property.Safety` claim by
:func:`inductive`.

The procedure is sound and incomplete, and what it cannot handle it refuses by
name: :data:`OPS` for the theory's operations, :func:`check_supported` for the
module, :class:`Obligation` for what a witness may ask, :func:`_dnf` and
:func:`_check_linear` for the rule's shape.

The method
==========
A wire behind a piecewise-linear node is affine once that node's **mode** is
fixed: a ReLU is the identity or zero according to the sign of its input. A
**region** fixes the mode of every node the rule's wires read through, strictly
(``> 0`` active, ``<= 0`` inactive), so regions partition rather than overlap,
and over one region every named wire is a single affine function of the columns.

The rule's negation is cut into **disjuncts** of linear rows (:func:`_dnf`), and
each row is then linear in the columns alone.

Over a region each disjunct is a linear infeasibility, discharged by proving
``region ∧ domain ∧ invariants ∧ disjunct`` infeasible via Farkas' lemma (over
z3's exact LRA) — an exact, checkable integer certificate (the multipliers ``y``:
``y >= 0``, ``Aᵀy = 0``, ``b·y < 0``). Every disjunct refuted is the rule proved
on the region. CEGAR finds the regions: an uncovered state names one by its
pinned pattern; certifying and blocking it until none is left is the coverage
proof, since regions are complementary.

``V >= 0`` for a ranking network is *not* Farkas-certified: it holds structurally
because the output layer is non-negative, and the client checks it before asking.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd, lcm

import numpy as np
import z3
from zrth import Sort

from ._nodes import ModeKind, Op, Unsupported, free_symbols, node_view
from ._property import Safety


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
# with neither a cell rule nor a case split is one Z3 cannot read either, so the
# walk refuses it too — a test pins that those are the only such kinds.
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
    # recognised, no cell rule — and no Z3 translation either, so the walk refuses
    # a module carrying one by name
    "LIA_Min": Op("min"),
    "LIA_Max": Op("max"),
    "LIA_Argmax": Op("argmax"),
}

# Derived, so the table above stays the single source of truth.
_MODE_OF = {op.kind: op.mode for op in OPS.values() if op.kind and op.mode}


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


def piece_value(pres, weights, pattern):
    """The wire's affine piece where ``pattern`` is the active set:
    ``sum_j c_j * pres_j`` over the active units, plus the output bias.

    A region fixes the pattern to the true one, so on that region this is the
    wire's value exactly."""
    c, k = weights
    out = z3.IntVal(k)
    for cj, pre, on in zip(c, pres, pattern):
        if cj and on:
            out = out + cj * pre
    return out


def mode_nodes(modes):
    """``(kind, expr)`` for every node the groups in ``modes`` name, in order."""
    return tuple((kind, e) for kind, exprs in modes for e in exprs)


def mode_region(modes, pattern):
    """The region a mode assignment names, each node through its own kind.

    The modes a kind offers are complementary, so the regions partition the state
    space and a state lies in exactly one."""
    out = []
    for (kind, e), m in zip(mode_nodes(modes), pattern):
        out += list(_MODE_OF[kind].region(e, m))
    return tuple(out)


def modes_at(modes, model):
    """The mode each node is in at ``model``."""
    return tuple(_MODE_OF[kind].at(model, e) for kind, e in mode_nodes(modes))


def exact_value(pres, weights):
    """``V`` with the ReLUs left in: ``sum_j c_j * relu(pres_j) + k``.

    The wire's truth, whatever region a state lies in. Used to tell a region the
    LP could not close from one where the rule genuinely fails."""
    c, k = weights
    out = z3.IntVal(k)
    for cj, pre in zip(c, pres):
        if cj:
            out = out + cj * z3.If(pre > 0, pre, z3.IntVal(0))
    return out


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


def build_integer_system(region, guard, invariants, atoms, syms):
    """Rows of ``region ∧ guard ∧ invariants ∧ atoms`` as an integer system
    ``A·s <= b`` with per-row labels, plus the atoms that carry information the LP
    cannot express. ``atoms`` is one disjunct of the rule's negation, already
    substituted for the region; infeasibility refutes that disjunct on it.

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

    for j, c in enumerate(region):
        add_atom(c, f"cell[{j}]")
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

    ``pattern`` is the mode of every node the named wires read through, which is
    what defines the region. ``disjunct`` indexes the rule's disjuncts.
    ``affines`` gives, per device, the ``(coeffs, const)`` of its value on this
    region over the state columns — the right-hand sides of the emitted collapse
    lemmas."""
    A: tuple
    b: tuple
    y: tuple
    labels: tuple
    pattern: tuple
    disjunct: int = 0
    affines: tuple = ()


@dataclass(frozen=True)
class Device:
    """A wire the rule names with a reading behind it, as the proof renders it:
    ``net`` indexes the result's distinct networks; ``inputs`` says which end of
    each column the reading takes (``"latched"``, ``"next"`` or ``"unread"``);
    ``offset`` where its nodes start in the region's pattern."""
    wire_id: int
    net: int
    inputs: tuple
    offset: int


@dataclass
class Proof:
    """What a ``certify`` run established, and what the proof layer needs: the
    per-path certificates; the claim and the witness, whose own data (an
    invariant, the ranks) the proof layer reads off them; the obligation's formula
    resolved over the columns and the wire symbols; and the named wires as devices
    over the distinct networks they read through."""
    verified: bool
    certificates: list           # list[PathCert]
    counterexample: object = None
    status: str = ""
    unused: tuple = ()           # domain atoms the LP could not express
    claim: object = None
    witness: object = None
    formula: object = None       # the obligation's formula, z3 over columns and wires
    devices: tuple = ()          # Device, in the order the formula named them
    nets: tuple = ()             # the distinct Net each device reads through
    columns: tuple = ()          # the column names this was certified over


@dataclass(frozen=True)
class PathCert:
    """One affine path of the loop body: its path condition ``guard`` (the loop
    guard strengthened by the branch literals taken along the path), its affine
    next-state ``body`` (z3 exprs over the pre-state symbols), and the per-cell
    certificates on it. The paths partition the loop guard, so the union of their
    ``Step`` relations is the loop's transition — hence a property of every path's
    steps is a property of the program's.

    ``device_units`` holds, per named wire, its nodes' pre-activations at this
    path's round as ``(coeffs, const)`` over the pre-state columns; ``units`` is
    their concatenation. A region is where those expressions take the signs its
    pattern names, which is what lets the emitter case-split on them."""
    guard: object
    body: tuple
    cells: tuple
    device_units: tuple = ()

    @property
    def units(self) -> tuple:
        return sum(self.device_units, ())


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
    precondition: object = None
    invariant_proofs: tuple = ()   # verified Proofs of Safety claims: the facts assumed

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

    def assuming(self, precondition) -> "System":
        """This system with ``precondition`` assumable at entry: ``state_map ->
        [BoolRef]``, the facts the entry state may be taken to satisfy."""
        return dataclasses.replace(self, precondition=precondition)

    def knowing(self, *proofs) -> "System":
        """This system assuming the facts ``proofs`` establish: each a verified
        :class:`Proof` of a :class:`._property.Safety` claim over these columns.
        The engine narrows a claim's obligation by them and the proof layer proves
        them in the same file and cites them, so what is assumed and what is proved
        are one object. Anything else is refused by name — an assumption without a
        proof has no place here."""
        for p in proofs:
            claim = getattr(p, "claim", None)
            if not isinstance(claim, Safety):
                raise Unsupported("knowing takes proofs of Safety claims; got a proof of "
                                  f"{type(claim).__name__ if claim is not None else type(p).__name__}")
            if not p.verified:
                raise Unsupported("knowing takes a verified proof; this one is not verified: "
                                  f"{p.status}")
            if tuple(p.columns) != tuple(self.names):
                raise Unsupported(f"the proof is over columns {tuple(p.columns)}, "
                                  f"not {tuple(self.names)}")
        return dataclasses.replace(self, invariant_proofs=self.invariant_proofs + tuple(proofs))

    @property
    def invariants(self) -> tuple:
        """The assumed facts as z3 over the columns, one conjunct each — what a
        claim's obligation may narrow its domain by."""
        out = []
        for p in self.invariant_proofs:
            out += _flatten_and(resolve(self, p.claim.holds)[0])
        return tuple(out)


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
        is ``None``, else its affine piece where ``pattern`` is the active set."""
        if not self.net.units:
            return _affine_value(self.net.out, self.args(s_syms, body))
        acts = self.at(s_syms, body)
        return (exact_value(acts, self.net.out) if pattern is None
                else piece_value(acts, self.net.out, pattern))


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
    conj += list((system.precondition or (lambda st: []))(s_map))
    return z3.And(*conj) if conj else z3.BoolVal(True)


# ---------------------------------------------------------------------------
# The obligation: what the engine proves, and the witnesses that build one
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Obligation:
    """What the engine proves: ``formula`` holds on every round satisfying
    ``domain``, and every formula in ``whole`` is unsatisfiable.

    ``domain`` and ``formula`` are ``(W, S) -> BoolRef`` over the module's wires
    (see :mod:`._property`) and are resolved here; ``whole`` is already z3 over
    the columns, a fact about the run as a whole — the entry state satisfies the
    invariant — being about no round in particular. This is the engine's whole
    contract: paths, regions, the LP and the certificates see nothing else, and
    know nothing of programs, ranks or invariants."""
    domain: object
    formula: object
    whole: tuple = ()             # ((name, must-be-unsat), ...)


_STATE_ONLY = ("an invariant is over the state; a wire belongs in the claim or "
               "the witness's formula")


class Inductive:
    """Discharge a :class:`._property.Safety` claim by an inductive invariant:
    predicates over the state — ``S`` alone — that the entry state satisfies and
    every step preserves, all of them at the next state given all of them at the
    pre-state; and wherever they hold, the claim's predicate does. With no
    invariant the claim must hold outright."""
    def __init__(self, inv=()):
        self.inv = tuple(inv)

    def obligation(self, claim, system) -> Obligation:
        if claim.holds is None:
            raise Unsupported(f"inductive discharges a claim with a predicate; "
                              f"{type(claim).__name__} has none")
        inv = self.inv
        Wi = _WireMap(system, refuse=_STATE_ONLY)          # the clearer refusal first
        for f in inv:
            f(Wi, _StateMap(system, Wi))

        def formula(W, S):
            now = [f(W, S) for f in inv]
            held = z3.And(*now) if inv else z3.BoolVal(True)
            step = (z3.Implies(held, z3.And(*[f(W.next, S.next) for f in inv]))
                    if inv else z3.BoolVal(True))
            holds = claim.holds(W, S)
            parts = holds.children() if z3.is_and(holds) else [holds]
            if inv and len(parts) == len(now) and all(a.eq(b) for a, b in zip(parts, now)):
                return step              # the claim is its own invariant: nothing left to imply
            return z3.And(step, z3.Implies(held, holds))

        Wc = _WireMap(system, refuse=_STATE_ONLY)
        Sc = _StateMap(system, Wc)
        conj = z3.And(*[f(Wc, Sc) for f in inv]) if inv else z3.BoolVal(True)
        return Obligation(claim.domain, formula,
                          (("initiation", z3.And(entry_predicate(system), z3.Not(conj))),))


def inductive(inv=()) -> Inductive:
    return Inductive(inv)


class LexDecrease:
    """Discharge a :class:`._property.Liveness` claim by ranks that drop
    lexicographically: on every round inside the domain some rank drops by ``delta`` while
    every earlier one does not increase — the substrate's ``lexDec``, which one
    rank instantiates as a plain drop. ``ranks`` pairs ``(at s, at s')`` wires, two
    readings of one function; what they must be is :func:`check_ranks`'s to say."""
    def __init__(self, ranks, delta=1):
        self.ranks = tuple(tuple(r) for r in ranks)
        self.delta = int(delta)

    def obligation(self, claim, system) -> Obligation:
        if claim.holds is not None:
            raise Unsupported(f"lex decrease discharges a run claim, one with no predicate; "
                              f"{type(claim).__name__} has one")
        check_ranks(system, self.ranks)
        ranks, d = self.ranks, self.delta

        def formula(W, S):
            alts = []
            for i, (v, vp) in enumerate(ranks):
                held = [W[ranks[j][1]] <= W[ranks[j][0]] for j in range(i)]
                drop = W[v] - W[vp] >= d
                alts.append(z3.And(*held, drop) if held else drop)
            return z3.Or(*alts) if len(alts) > 1 else alts[0]

        return Obligation(claim.domain, formula)


def lex_decrease(ranks, delta=1) -> LexDecrease:
    return LexDecrease(ranks, delta)


def decrease(v_s, v_sp, delta=1) -> LexDecrease:
    """Termination by one rank: ``V(s) - V(s') >= delta`` on every counting round."""
    return LexDecrease(((v_s, v_sp),), delta)


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
    """``S`` as a predicate sees it: a column's latched value by name, ``names``
    the columns, and ``next`` the same columns after the step — each the ``W`` of
    the column's next wire, so naming one is recorded like naming a wire. A
    ``concrete`` map hands out the transition itself for the next value instead,
    which is what a domain wants: the cases are split on the transition."""
    def __init__(self, system, W, ahead: bool = False, concrete: bool = False):
        self.system, self.W, self.ahead, self.concrete = system, W, ahead, concrete

    @property
    def names(self):
        return self.system.names

    def __getitem__(self, name):
        if name not in self.system.names:
            raise Unsupported(f"{name!r} is not a column of this system")
        k = self.system.names.index(name)
        if self.ahead:
            return (self.system.sp_syms[k] if self.concrete
                    else self.W[self.system.pairs[k][1]])
        return self.system.s_syms[k]

    @property
    def next(self):
        if self.ahead:
            raise Unsupported("the state two rounds ahead is not available")
        return _StateMap(self.system, self.W, ahead=True, concrete=self.concrete)


def resolve(system: System, pred):
    """``pred`` — ``(W, S) -> BoolRef`` — as z3 over the columns' symbols and the
    wire symbols ``system.W``, with the wires it named in order. The form the
    engine cuts into rows and substitutes into, and what the proof layer renders."""
    used = []
    W = _WireMap(system, used)
    return lift_ites(pred(W, _StateMap(system, W))), tuple(used)


def resolve_domain(system: System, domain):
    """A claim's domain as z3 over the columns and the transition, ready to be
    split into cases — and, for a client, to be sampled: the trainer draws its
    pairs from the rounds the claim counts. A domain is over the columns; a wire
    belongs in the formula."""
    W = _WireMap(system, refuse="a domain is over the columns and their next values; "
                                "a wire belongs in the formula")
    return lift_ites(domain(W, _StateMap(system, W, concrete=True)))


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


def _check_linear(disjuncts, syms) -> None:
    """Refuse by name a rule atom the LP cannot turn into rows.

    Every atom of the rule's negation must be a linear comparison over the columns
    and the named wires, since that is what a Farkas row is."""
    for atoms in disjuncts:
        for a in atoms:
            if not atom_rows(a, syms):
                raise Unsupported("rule predicate is not a linear comparison over the "
                                  f"columns and the named wires: {a}")


# ---------------------------------------------------------------------------
# Doors
# ---------------------------------------------------------------------------

def check_supported(system: System) -> None:
    """Raise :class:`Unsupported`, naming the reason, if this procedure has no rule
    for something in ``system``'s module. What a claim or a witness may ask is
    theirs to refuse.

    Checked before any work so the interior can assume its preconditions. The
    alternative — proceeding with whatever it happens to understand — reports a
    proof that will not close rather than the thing it could not use."""
    names = [str(s) for s in system.s_syms]
    for name, e in zip(names, system.sp_syms):
        extra = free_symbols(e) - set(names)
        if extra:
            raise Unsupported(
                f"the next value of {name!r} reads {sorted(extra)}, which are not "
                f"columns — nondeterministic inputs are not supported")


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
# The engine: regions, one Farkas system per disjunct
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
    named wires as devices, and the piecewise-linear nodes behind them.

    ``modes`` is what a region enumerates over: every node the named wires read
    through, in device order."""
    s_syms: tuple
    body: tuple
    guard: object
    invariants: tuple
    dom: object
    ok: object
    disjuncts: tuple
    devices: tuple           # _Device, in the order the rule named the wires
    columns: dict            # wire id -> column index, for a column's own next wire
    modes: tuple
    unused: set

    @staticmethod
    def of(system, ok, disjuncts, devices, readings, columns, body, guard,
           invariants) -> "_Path":
        invariants = tuple(invariants)
        dom = z3.And(guard, *invariants) if invariants else guard
        s_syms = tuple(system.s_syms)
        devs, acts_all = [], []
        for spec in devices:
            rd = readings[spec.wire_id]
            args = rd.args(s_syms, body)
            acts = tuple(_pre_activations(rd.net, args))
            devs.append(_Device(spec, system.W[spec.wire_id], rd, tuple(args), acts,
                                rd.net.out))
            acts_all.extend(acts)
        return _Path(s_syms=s_syms, body=tuple(body), guard=guard,
                     invariants=invariants, dom=dom, ok=ok, disjuncts=tuple(disjuncts),
                     devices=tuple(devs), columns=dict(columns),
                     modes=(("relu", tuple(acts_all)),) if acts_all else (),
                     unused=set())

    def _values(self, system, value_of) -> list:
        out = [(system.W[wid], self.body[k]) for wid, k in self.columns.items()]
        return out + [(d.sym, value_of(d)) for d in self.devices]

    def subst(self, system, lam) -> list:
        """``(W symbol, value)`` for every named wire on the region ``lam``: its
        affine piece there, a column's next wire its body on this path, and a
        node-free wire its value outright."""
        def value(d):
            n = len(d.acts)
            if not n:
                return _affine_value(d.weights, d.args)
            return piece_value(d.acts, d.weights,
                               lam[d.spec.offset:d.spec.offset + n])
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
        columns — what the emitter's collapse lemmas state."""
        smap = {str(k): v for k, v in sub}
        return tuple(_affine_pair(z3.simplify(smap[str(d.sym)]), self.s_syms)
                     for d in self.devices)


def _try_cell(system, p: _Path, lam):
    """Farkas-certify one region, named by the mode pattern ``lam``.

    One LP per disjunct of the rule's negation — the rows of the region, the
    guard, the invariants and the disjunct. Returns one :class:`CellCert` per
    disjunct, or ``None`` if some disjunct stays feasible on the region."""
    sub = p.subst(system, lam)
    affines = p.affines(sub)
    certs = []
    for d, atoms in enumerate(p.disjuncts):
        rows = [z3.substitute(a, *sub) for a in atoms] if sub else list(atoms)
        A, b, labels, unused = build_integer_system(
            mode_region(p.modes, lam), p.guard, list(p.invariants), rows, p.s_syms)
        p.unused.update(str(a) for a in unused)
        y = find_infeasibility_certificate(A, b)
        if y is None:
            return None
        certs.append(CellCert(tuple(map(tuple, A)), tuple(b), tuple(y), tuple(labels),
                              tuple(lam), disjunct=d, affines=affines))
    return certs


def _certify_cell(system, p: _Path, lam):
    """Certify the region ``lam`` names, or say why not.

    Over the region every named wire is one affine piece, so a disjunct the LP
    leaves feasible is either a genuine counterexample or an entailment only the
    integers witness. Evaluating the rule *exactly* on the region tells the two
    apart."""
    certs = _try_cell(system, p, lam)
    if certs is not None:
        return certs, "ok"
    region = z3.And(p.dom, *mode_region(p.modes, lam))
    if _feasible(z3.And(region, p.violation(system))):
        return None, "violated"
    return None, "uncertifiable"


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
        lam = modes_at(p.modes, model)
        if z3.is_true(model.eval(p.violation(system), model_completion=True)):
            return False, cells, np.array(s_val, dtype=np.float64), "FAILED(violated)"
        try:
            new, status = _certify_cell(system, p, lam)
        except ValueError as exc:            # non-affine even after splitting
            return (False, cells, np.array(s_val, dtype=np.float64),
                    f"FAILED(non-affine: {exc})")
        if status != "ok":
            return False, cells, np.array(s_val, dtype=np.float64), f"FAILED({status})"
        cells.extend(new)
        block = mode_region(p.modes, lam)
        solver.add(z3.Not(z3.And(*block)) if block else z3.BoolVal(False))
    return False, cells, None, "FAILED(max_iters)"


def check_ranks(system: System, ranks) -> None:
    """What a rank witness needs of its wires, refused by name otherwise.

    A rank is one network read at both ends of a step: its two wires read the same
    network, the first nothing of the next state and the second nothing of the
    latched — what lets the proof state the rank as one ``V`` and apply it to the
    state at either end. And that network's output layer is non-negative, which is
    how the proof knows the rank is bounded below; a rank the decrease alone would
    certify is no use to the well-foundedness theorem without it."""
    for v_s, v_sp in ranks:
        a, b = reading(system, v_s), reading(system, v_sp)
        if (a.net != b.net or not a.net.units
                or any(k == "next" for k, _ in a.inputs)
                or any(k == "latched" for k, _ in b.inputs)):
            raise Unsupported(f"rank ({v_s.id}, {v_sp.id}) is not one network read at "
                              f"the latched state and at the next state")
        c, k = a.net.out
        if not (all(x >= 0 for x in c) and k >= 0):
            raise Unsupported(f"rank ({v_s.id}, {v_sp.id}): the network's output layer "
                              f"is not non-negative, so the rank is not bounded below")


def certify(system: System, claim, witness, max_iters: int = 1000) -> Proof:
    """Certify ``claim`` of ``system`` by ``witness``.

    The witness turns the claim into an :class:`Obligation`. Its formula is
    resolved over the columns and the named wires and its negation cut into
    disjuncts of rows; its whole-run facts are checked; then its domain is expanded
    into cases (:func:`expand_cases`), each a convex region with an affine
    transition, and each certified by the region/CEGAR engine. Nothing here knows
    what a program, a rank or an invariant is."""
    check_supported(system)
    ob = witness.obligation(claim, system)
    formula, wires = resolve(system, ob.formula)
    W_syms = [system.W[w.id] for w in wires]
    disjuncts = _dnf(z3.Not(formula))
    _check_linear(disjuncts, list(system.s_syms) + W_syms)
    nexts = {pr[1].id: k for k, pr in enumerate(system.pairs)}
    columns = {w.id: nexts[w.id] for w in wires if w.id in nexts}
    readings = {w.id: reading(system, w) for w in wires if w.id not in nexts}
    nets, devices, offset = [], [], 0
    for w in wires:
        if w.id in columns:
            continue
        rd = readings[w.id]
        if rd.net not in nets:
            nets.append(rd.net)
        devices.append(Device(w.id, nets.index(rd.net), rd.inputs, offset))
        offset += len(rd.net.units)
    nets, devices = tuple(nets), tuple(devices)

    def result(verified, paths, cex, status, unused=()):
        return Proof(verified, paths, cex, status, tuple(sorted(unused)), claim,
                     witness, formula, devices, nets, columns=tuple(system.names))

    s_syms, sp_syms = system.s_syms, system.sp_syms
    invariants = system.invariants
    for name, bad in ob.whole:
        s = z3.Solver()
        s.add(bad)
        r = s.check()
        if r != z3.unsat:
            cex = None
            if r == z3.sat:
                m = s.model()
                cex = np.array([m.eval(x, model_completion=True).as_long()
                                for x in s_syms], dtype=np.float64)
            return result(False, [], cex, f"FAILED({name})")
    paths: list[PathCert] = []
    unused: set = set()
    dom = resolve_domain(system, ob.domain)
    for pguard, pbody in expand_cases(dom, list(sp_syms)):
        region = z3.And(pguard, *invariants) if invariants else pguard
        if not _feasible(region):
            continue                       # dead case — no state takes this step
        p = _Path.of(system, formula, disjuncts, devices, readings, columns, pbody,
                     pguard, invariants)
        verified, cells, cex, status = _certify_path(system, p, max_iters)
        unused |= p.unused
        if not verified:
            return result(False, [], cex, status, unused)
        paths.append(PathCert(
            pguard, tuple(pbody), tuple(cells),
            device_units=tuple(tuple(_affine_pair(e, s_syms) for e in d.acts)
                               for d in p.devices)))
    if not paths:
        return result(False, [], None, "FAILED(no step in the domain)", unused)
    return result(True, paths, None, "VERIFIED", unused)
