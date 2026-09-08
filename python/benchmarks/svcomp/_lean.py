"""Emit a kernel-checkable Lean proof from a decision procedure's certificates.

Consumes the :class:`._farkas.FarkasResult` a ``certify`` run returns (plus the
:class:`._farkas.System` it was run on) and writes ONE Lean file
against the vendored ``lean/`` substrate (``Coverage``/``Net``/``Termination``).
The file follows the rule's shape:

  * every network a named wire is read through, once, with its structural
    non-negativity ``V_j_nonneg``;
  * per affine path (in-loop branching is *path-split*: the body's nested
    ``ite``s are expanded into affine paths by :func:`._farkas.expand_cases`, one
    namespace each): ``trans`` and ``post_state``; the regions — one per mode
    pattern, partitioning the guard — and ``covered``, the CEGAR coverage
    guarantee, by ``omega`` over a sign-literal decision tree; per device its
    pre-activations and its exact collapse per region; per region and per
    disjunct of the
    rule's negation the Farkas system ``A``/``b``/``y``, its ``farkas_sound``
    infeasibility and the ``refute`` lemma (via the substrate's
    ``refute_bridge``); the rule's formula ``ok`` and ``step_ok``, which puts the
    bounds, collapses and refutations together by ``omega``; and
    ``Step``/``RawStep``/``consecution``;
  * the composition the property needs: ranks conclude ``program_terminates``
    through ``lexDec`` and ``no_infinite_run_lex``; an invariant concludes
    ``always_holds``, with ``initiation`` and per-path ``consecution`` carrying the
    invariant along any run and ``step_ok`` closing the predicate at each state.
"""
from __future__ import annotations

import dataclasses

from pathlib import Path

import z3

from ._farkas import (CellCert, _find_ite_cond, _flatten_and, affine_coeffs,
                      entry_predicate)


def _contains_ite(e) -> bool:
    """True if a z3 term still has an ``ite`` — a path body must have none (all
    in-loop branches are split out by ``expand_cases`` before emission)."""
    if z3.is_app(e) and e.decl().kind() == z3.Z3_OP_ITE:
        return True
    return any(_contains_ite(c) for c in e.children())


# ---------------------------------------------------------------------------
# Literals and Fin indices
# ---------------------------------------------------------------------------

def _fin(i: int) -> str:
    """The ``i``-th index of the substrate's inductive ``Fin``."""
    s = "fzero"
    for _ in range(i):
        s = f"(fsucc {s})"
    return s


def _encode_int(n: int) -> str:
    """An ``Int`` literal in constructor form (reduces definitionally — needed by
    the ``rfl`` / ``Int.negSucc_lt_zero`` proofs of the Farkas side goals)."""
    return f"Int.ofNat {n}" if n >= 0 else f"Int.negOfNat {abs(n)}"


def _vec_def(name: str, vals, render, typ: str = "Int") -> str:
    lines = [f"def {name} : Vector {len(vals)} {typ} := fun"]
    for i, x in enumerate(vals):
        lines.append(f"  | {_fin(i)} => {render(x)}")
    return "\n".join(lines)


def _mat_def(name: str, A, render) -> str:
    m, n = len(A), len(A[0])
    lines = [f"def {name} : Matrix {m} {n} Int := fun"]
    for i in range(m):
        for j in range(n):
            lines.append(f"  | {_fin(i)}, {_fin(j)} => {render(A[i][j])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Affine expressions and rows (omega-friendly plain integers)
# ---------------------------------------------------------------------------

def _affine_terms(coeffs, const: int, terms) -> str:
    """``Σ cⱼ·termsⱼ + const`` over the nonzero coefficients (``const`` if none);
    ``terms[j]`` is the Lean term the ``j``-th symbol stands for."""
    parts = [f"({int(c)} * {t})" for c, t in zip(coeffs, terms) if c]
    if not parts:
        return f"{int(const)}"
    expr = parts[0]
    for t in parts[1:]:
        expr = f"({expr} + {t})"
    if const:
        expr = f"({expr} + {int(const)})"
    return expr


def _state_terms(n: int, var: str = "s") -> list:
    return [f"{var} {_fin(j)}" for j in range(n)]


def _affine_str(coeffs, const: int, var: str = "s") -> str:
    """``Σ cⱼ·(var j) + const`` over the nonzero coefficients (``const`` if none)."""
    return _affine_terms(coeffs, const, _state_terms(len(coeffs), var))


def _affine_z3(coeffs, const: int, syms):
    """``Σ cⱼ·symsⱼ + const`` as a z3 term, the counterpart of :func:`_affine_str`."""
    e = z3.IntVal(int(const))
    for c, sym in zip(coeffs, syms):
        if c:
            e = e + int(c) * sym
    return e


_CMP = {z3.Z3_OP_GE: "≥", z3.Z3_OP_LE: "≤", z3.Z3_OP_GT: ">",
        z3.Z3_OP_LT: "<", z3.Z3_OP_EQ: "="}


class _Drop(Exception):
    """Raised when a guard/invariant atom is not integer-linear (soundly skipped:
    dropping a conjunct only weakens ``trans``, an over-approximation)."""


def _z3_prop(a, syms, terms=None) -> str:
    """Render a z3 boolean expression to a Lean ``Prop`` over the integer symbols
    ``syms`` — the state columns by default, or ``terms[j]`` for each symbol when
    some stand for other Lean terms. Handles ∧ / ∨ / ¬ / → / linear comparisons;
    raises :class:`_Drop` on anything non-linear."""
    if z3.is_true(a):
        return "True"
    if z3.is_false(a):
        return "False"
    if z3.is_and(a):
        return "(" + " ∧ ".join(_z3_prop(c, syms, terms) for c in a.children()) + ")"
    if z3.is_or(a):
        return "(" + " ∨ ".join(_z3_prop(c, syms, terms) for c in a.children()) + ")"
    if z3.is_not(a):
        inner = a.arg(0)
        if z3.is_app(inner) and inner.decl().kind() == z3.Z3_OP_EQ:
            return f"({_cmp_prop(inner, syms, '≠', terms)})"
        return f"(¬ {_z3_prop(inner, syms, terms)})"
    if z3.is_app(a) and a.decl().kind() == z3.Z3_OP_IMPLIES:
        return (f"({_z3_prop(z3.Not(a.arg(0)), syms, terms)} ∨ "   # omega case-splits it
                f"{_z3_prop(a.arg(1), syms, terms)})")
    if z3.is_app(a) and a.decl().kind() in _CMP:
        return _cmp_prop(a, syms, None, terms)
    if z3.is_app(a) and a.decl().kind() == z3.Z3_OP_DISTINCT:
        return _cmp_prop(a, syms, "≠", terms)
    raise _Drop(str(a))


def _cmp_prop(a, syms, op: str | None = None, terms=None) -> str:
    """A comparison ``lhs ⋈ rhs`` as ``(<lhs-rhs> ⋈ 0)`` (linear, omega-ready)."""
    if op is None:
        op = _CMP[a.decl().kind()]
    try:
        coeffs, const = affine_coeffs(a.arg(0) - a.arg(1), syms)
    except ValueError:
        raise _Drop(str(a))
    terms = terms if terms is not None else _state_terms(len(syms))
    return f"({_affine_terms(coeffs, const, terms)} {op} 0)"


def _expand_ites(pred):
    """``pred`` with every ``ite`` case-split away, leaving a Boolean combination of
    linear atoms. An ``ite`` *term* is not linear, so a conjunct holding one (an
    entry fact like ``t = ite(b >= 1, 1, -1)``) would be dropped instead."""
    cond = _find_ite_cond(pred)
    if cond is None:
        return pred
    arms = []
    for truth in (True, False):
        branch = z3.simplify(z3.substitute(pred, (cond, z3.BoolVal(truth))))
        side = cond if truth else z3.Not(cond)
        arms.append(z3.And(side, _expand_ites(branch)))
    return z3.Or(*arms)


def _render_conjuncts(pred, syms) -> str:
    """Flatten ``pred`` (a z3 ∧-tree) to conjuncts, render each, drop the ones
    that are not integer-linear. Empty → ``True``."""
    out = []
    for c in _flatten_and(_expand_ites(pred)):
        try:
            out.append(_z3_prop(c, syms))
        except _Drop:
            continue
    return " ∧ ".join(out) if out else "True"


# ---------------------------------------------------------------------------
# Per-cell Farkas systems (restricted to the certificate's support)
# ---------------------------------------------------------------------------

def _support(cert: CellCert):
    """The rows with a nonzero multiplier — infeasibility of this subset implies
    it for the full system, and the cert is valid verbatim on it. Returns
    ``(A, b, y, labels)``."""
    idx = [i for i, yv in enumerate(cert.y) if yv != 0]
    return ([cert.A[i] for i in idx], [cert.b[i] for i in idx],
            [cert.y[i] for i in idx], [cert.labels[i] for i in idx])


def _pattern_literals(pattern, units):
    """A pattern's region as literals ``(coeffs, const, active)``: ``active`` means
    ``0 < e``, otherwise ``e ≤ 0``. Units the pattern leaves open (``None``)
    contribute nothing.

    This is the one place a region is derived. Both the Lean definition
    (:func:`_lit_str`) and the z3 term the coverage tree reasons over
    (:func:`_lit_z3`) render from it, so the two cannot drift apart."""
    return [(c, k, a) for (c, k), a in zip(units, pattern) if a is not None]


def _lit_str(lit) -> str:
    c, k, active = lit
    return f"(0 < {_affine_str(c, k)})" if active else f"({_affine_str(c, k)} ≤ 0)"


def _lit_z3(lit, syms):
    c, k, active = lit
    e = _affine_z3(c, k, syms)
    return (e > 0) if active else (e <= 0)


def _rule_rows(cert: CellCert):
    """The disjunct's own rows ``(coeffs, const)`` — ``coeffs·s ≤ const`` — as the
    LP saw them, without their gcd-tightened variants (``omega`` re-derives those)."""
    return [(a, b) for a, b, lbl in zip(cert.A, cert.b, cert.labels)
            if lbl.startswith("rule[") and not lbl.endswith("/int")]


def _trivial(cert: CellCert) -> bool:
    """The support reduces to a constant infeasibility ``0·s ≤ c`` (c < 0): the
    disjunct is false on its own, so it needs no Farkas certificate — ``omega``
    refutes it directly."""
    A, *_ = _support(cert)
    return all(all(x == 0 for x in row) for row in A)


def _emit_system(prefix: str, A, b, y, labels) -> str:
    """Data + three side-condition goals + the ``farkas_sound`` conclusion."""
    dot = sum(bi * yi for bi, yi in zip(b, y))
    assert dot < 0, f"certificate bound not negative: b·y = {dot}"
    goal1_cases = "\n  | ".join(f"{_fin(j)} => rfl" for j in range(len(A[0])))
    goal2_cases = "\n  | ".join(
        f"{_fin(i)} => Int.natCast_nonneg {y[i]}" for i in range(len(y)))
    return "\n\n".join([
        f"-- rows: {', '.join(labels)}",
        _mat_def(f"{prefix}_A", A, _encode_int),
        _vec_def(f"{prefix}_b", b, _encode_int),
        _vec_def(f"{prefix}_y", y, _encode_int),
        f"theorem {prefix}_goal₁ : {prefix}_A ᵀ *ᵥ {prefix}_y = 𝟎ᵥ := funext fun\n"
        f"  | {goal1_cases}",
        f"theorem {prefix}_goal₂ : 𝟎ᵥ ≤ᵥ {prefix}_y := fun\n  | {goal2_cases}",
        f"theorem {prefix}_goal₃ : ({prefix}_b ·ᵥ {prefix}_y) < Int.ofNat 0 :=\n"
        f"  Int.negSucc_lt_zero {abs(dot) - 1}",
        f"theorem {prefix}_infeasible : ∀ s, ¬ ({prefix}_A *ᵥ s ≤ᵥ {prefix}_b) :=\n"
        f"  farkas_sound {prefix}_A {prefix}_b {prefix}_y "
        f"{prefix}_goal₁ {prefix}_goal₂ {prefix}_goal₃",
    ])


def _emit_cell(idx: str, cert: CellCert) -> str:
    manifest = f"-- cell {idx}: pattern={list(cert.pattern)}"
    if _trivial(cert):
        return manifest + f"\n-- cell {idx}: constant infeasibility (no Farkas system)"
    A, b, y, labels = _support(cert)
    return manifest + "\n\n" + _emit_system(f"cell{idx}", A, b, y, labels)


# ---------------------------------------------------------------------------
# The networks the named wires are read through
# ---------------------------------------------------------------------------

def _dev(tag: str) -> str:
    """Name suffix for the ``tag``-th network."""
    return f"_{tag}"


def _simp(tags=("0",)) -> str:
    """The `simp only` set for the collapse / bounded-below proofs: structural
    unfoldings, the `Fin` expanders, every named network's defs, the ReLU `ite`
    reduction, and the cast/base reducers that turn `sum`'s base and casts into
    literals so `omega` sees them. The caller appends per-cell pattern defs."""
    nets = ", ".join(f"nrf{_dev(g)}_W0, nrf{_dev(g)}_b0, nrf{_dev(g)}_W1, nrf{_dev(g)}_b1"
                     for g in tags)
    return (
        "affine, addᵥ, mulVec_apply, sum, mask, Function.comp, "
        "forall_fin_succ, forall_fin_zero, ↓reduceIte, "
        f"{nets}, "
        "Nat.zero_eq, Int.ofNat_eq_natCast, Int.cast_ofNat_Int, Int.reduceNeg, "
        "Int.reduceLE, Int.neg_ofNat_le_ofNat, Int.add_zero, Int.zero_add, "
        "Int.mul_one, Int.mul_zero, Int.zero_mul, Int.mul_neg_one, Int.neg_nonneg, "
        "Int.neg_le_zero_iff, Int.zero_le_ofNat, Std.le_refl, and_true, true_and, "
        "and_self, forall_const, imp_self, implies_true, "
        "Bool.true_eq_false, Bool.false_eq_true, false_implies"
    )


def _net_arrays(net):
    """``(W0, b0, W1, b1)`` as integer grids/vectors, read off a :class:`Net`: one
    hidden row per unit, one output row."""
    W0 = tuple(tuple(int(c) for c in coeffs) for coeffs, _ in net.units)
    b0 = tuple(int(k) for _, k in net.units)
    W1 = (tuple(int(c) for c in net.out[0]),)
    b1 = (int(net.out[1]),)
    return W0, b0, W1, b1


def _emit_network(net, tag: str) -> str:
    W0, b0, W1, b1 = _net_arrays(net)
    n_in, n_out = len(W0[0]), len(b1)
    d = _dev(tag)
    blocks = [
        _mat_def(f"nrf{d}_W0", W0, str),
        _vec_def(f"nrf{d}_b0", b0, str),
        _mat_def(f"nrf{d}_W1", W1, str),
        _vec_def(f"nrf{d}_b1", b1, str),
        f"def V{d} (s : Vector {n_in} Int) : Vector {n_out} Int :=\n"
        f"  (affine nrf{d}_W1 nrf{d}_b1 (reluᵥ (affine nrf{d}_W0 nrf{d}_b0 s)))",
    ]
    return "\n\n".join(blocks)


def _emit_out_apply(net, tag: str) -> str:
    """The output layer applied to an arbitrary vector, reduced once per network.

    Every bound's ``heq`` and every collapse's tail reduce
    ``affine nrf_W1 nrf_b1 (mask _ _) fzero``, and that matrix product expands the
    same way every time — only the mask differs. Rewriting by this leaves each site
    with the mask reduction alone."""
    _, _, W1, b1 = _net_arrays(net)
    c, k = W1[0], b1[0]
    d = _dev(tag)
    return (f"theorem out_apply{d} (v : Vector {len(c)} Int) :\n"
            f"    affine nrf{d}_W1 nrf{d}_b1 v fzero = {_affine_str(c, k, 'v')} := by\n"
            f"  simp only [{_simp((tag,))}] <;> omega")


def _emit_out_nonneg(tag: str) -> str:
    """The output layer's non-negativity, proved once per network: the side goals
    of ``affine_nonneg``, facts about ``nrf_W1``/``nrf_b1`` alone."""
    d = _dev(tag)
    return "\n\n".join([
        f"theorem nrf{d}_W1_nonneg : ∀ i j, 0 ≤ nrf{d}_W1 i j := by\n"
        f"  simp only [{_simp((tag,))}]",
        f"theorem nrf{d}_b1_nonneg : ∀ i, 0 ≤ nrf{d}_b1 i := by\n"
        f"  simp only [{_simp((tag,))}]",
    ])


def _emit_nonneg(net, tag: str) -> str:
    W0, _, _, b1 = _net_arrays(net)
    n_in, n_out = len(W0[0]), len(b1)
    d = _dev(tag)
    return (
        f"theorem V{d}_nonneg (s : Vector {n_in} Int) (j : Fin {n_out}) : 0 ≤ V{d} s j := by\n"
        f"  simp only [V{d}]\n"
        f"  exact affine_nonneg nrf{d}_W1 nrf{d}_b1 _\n"
        f"    nrf{d}_W1_nonneg nrf{d}_b1_nonneg\n"
        f"    (fun k => reluᵥ_nonneg _ k) j"
    )


# ---------------------------------------------------------------------------
# Regions, their signs and the coverage of a path
# ---------------------------------------------------------------------------

def _regions(certs):
    """Group a path's certificates into regions — the cells sharing a mode
    pattern — each with its disjuncts in order. A certified region carries one
    certificate per disjunct of the rule's negation."""
    regions, index = [], {}
    for c in certs:
        key = tuple(c.pattern)
        if key not in index:
            index[key] = len(regions)
            regions.append([])
        regions[index[key]].append(c)
    for cs in regions:
        cs.sort(key=lambda c: c.disjunct)
    return regions


def _sign_index(reps):
    """Index a path's distinct mode patterns, and map each region to the index it
    uses. Regions sharing a pattern share one sign definition, emitted once."""
    pins, of_region = {}, []
    for c in reps:
        q = tuple(c.pattern)
        pins.setdefault(q, len(pins))
        of_region.append(pins[q])
    return pins, of_region


def _emit_signs_def(name: str, pattern, units, n: int) -> str:
    """A pattern's region, one condition per hidden unit it pins: ``0 < e`` where
    the pattern says active, ``e ≤ 0`` where inactive, nothing where it leaves the
    unit open (``None`` — a narrowing only pins what the certificate needed).

    The two forms are complementary, so the regions partition the state space and
    a state lies in exactly one. Written in the form :func:`_tiling_tree` splits
    on, so a coverage leaf's hypotheses match the conjuncts it has to produce."""
    conj = [_lit_str(l) for l in _pattern_literals(pattern, units)]
    return (f"def {name} (s : Vector {n} Int) : Prop :=\n"
            f"  {' ∧ '.join(conj) if conj else 'True'}")


def _cell_sign_defs(pat_i: int) -> list[str]:
    """The sign definition ``cellR_signs`` is built from."""
    return [f"signs_{pat_i}"]


def _tiling_tree(pcert, invariants, s_syms):
    """A decision tree over the hidden units' sign literals whose leaves each name
    one region.

    Splitting on ``0 < unitⱼ`` in turn narrows which regions a branch can still be
    in, and a branch is finished as soon as the literals taken so far *entail* some
    region's sign rows — checked here, so the ``omega`` the leaf emits is known to
    succeed. Entailment rather than a pattern match is the leaf test because a
    branch can settle a region before every literal is taken, and because the
    guard may imply signs no literal has fixed.

    z3 also prunes: a branch no guarded state satisfies becomes ``dead``, which
    keeps the tree the size of the region set instead of ``2^units``.

    Returns ``("split", lean_literal, yes, no)``, ``("cell", index)`` or
    ``("dead",)``; raises :class:`_Drop` if a live branch runs out of literals
    without entailing any region, which means the regions do not tile the guard."""
    # One literal per unit: ``0 < e`` and its negation ``e ≤ 0`` are exactly the two
    # sides a pattern names, so a single split settles the unit.
    lits = []
    for coeffs, const in pcert.units:
        lean, e = _affine_str(coeffs, const), _affine_z3(coeffs, const, s_syms)
        lits.append((f"0 < {lean}", e > 0))
    domain = [pcert.guard, *invariants]
    signs = [z3.And(*[_lit_z3(l, s_syms)
                      for l in _pattern_literals(c.pattern, pcert.units)])
             for c in pcert.cells]

    def unsat(*claims) -> bool:
        solver = z3.Solver()
        solver.add(*claims)
        return solver.check() == z3.unsat

    def build(i: int, taken: list, live: list[int]):
        if unsat(*domain, *taken):
            return ("dead",)
        live = [k for k in live if not unsat(*domain, *taken, signs[k])]
        for k in live:                       # entailed => the leaf's omega closes
            if unsat(*domain, *taken, z3.Not(signs[k])):
                return ("cell", k)
        while i < len(lits):                 # skip literals the branch already fixes
            _, z3_lit = lits[i]
            if unsat(*domain, *taken, z3_lit) or unsat(*domain, *taken, z3.Not(z3_lit)):
                i += 1
                continue
            break
        if i == len(lits):
            raise _Drop("regions do not tile the guard")
        lean_lit, z3_lit = lits[i]
        yes = build(i + 1, taken + [z3_lit], live)
        no = build(i + 1, taken + [z3.Not(z3_lit)], live)
        if yes == ("dead",) and no == ("dead",):
            return ("dead",)
        return ("split", lean_lit, yes, no)

    return build(0, [], list(range(len(pcert.cells))))


def _disjunct(i: int, total: int) -> tuple[str, str]:
    """``Or`` injections wrapping the ``i``-th of ``total`` disjuncts."""
    return "Or.inr (" * i + ("Or.inl " if i < total - 1 else ""), ")" * i


def _emit_tiling_tree(reps, tree, of_region, depth: int = 1) -> str:
    """The coverage case split, as a decision tree over the hidden units' sign
    literals rather than one ``omega`` over the whole disjunction.

    ``omega`` decides a conjunctive goal in time linear in its facts, but the flat
    tiling goal ``⋁ᵢ cellᵢ_signs`` negates into a clause per region, and the case
    split across those clauses grows exponentially in the number of regions — 12
    already exhaust the elaborator's budget. Splitting on the sign literals
    instead reaches, at each leaf, an assignment that names one region, so every
    ``omega`` sees a conjunction: the accumulated literals entailing that region's
    sign rows. The tree is built with the guard in hand, so a branch no state can
    satisfy is closed rather than explored (see :func:`_tiling_tree`)."""
    pad = "  " * depth
    kind = tree[0]
    if kind == "cell":
        i = tree[1]
        inj, close = _disjunct(i, len(reps))
        halves = ", ".join(_cell_sign_defs(of_region[i]))
        return (f"{pad}have hc : cell{i}_signs s := by\n"
                f"{pad}  simp only [trans, invariants, cell{i}_signs, {halves}] at *\n"
                f"{pad}  omega\n"
                f"{pad}exact {inj}hc{close}")
    if kind == "dead":
        return (f"{pad}exfalso\n"
                f"{pad}simp only [trans, invariants] at *\n"
                f"{pad}omega")
    _, lit, yes, no = tree
    return (f"{pad}by_cases hs{depth} : {lit}\n"
            f"{pad}· {_emit_tiling_tree(reps, yes, of_region, depth + 1).lstrip()}\n"
            f"{pad}· {_emit_tiling_tree(reps, no, of_region, depth + 1).lstrip()}")


def _emit_covered(reps, n: int, tree, of_region, pcert) -> str:
    """Every guarded state lies in some region: ``⋁ᵣ cellR_signs s``, by the
    sign-literal decision tree — the CEGAR coverage guarantee, re-proved."""
    hyps = "(hg : trans s) (hinv : invariants s)"
    disj = "\n      ∨ ".join(f"cell{i}_signs s" for i in range(len(reps)))
    if len(reps) == 1:
        if not _pattern_literals(reps[0].pattern, pcert.units):
            body = "  exact trivial"           # nothing pinned: the one region is everything
        else:
            unfold = ", ".join(["trans", "invariants", "cell0_signs"]
                               + _cell_sign_defs(of_region[0]))
            body = f"  simp only [{unfold}] at *\n  omega"
    else:
        body = _emit_tiling_tree(reps, tree, of_region)
    return (f"theorem covered (s : Vector {n} Int)\n"
            f"    {hyps} :\n"
            f"    {disj} := by\n"
            f"{body}")


# ---------------------------------------------------------------------------
# Devices on a path: activations and their collapse per region
# ---------------------------------------------------------------------------

def _emit_post_state(body_affines, n: int) -> str:
    """``post_state s = body(s)`` — the loop body's next state as an affine map."""
    arms = "\n".join(
        f"    | {_fin(k)} => {_affine_str(c, kk)}"
        for k, (c, kk) in enumerate(body_affines))
    return (f"def post_state (s : Vector {n} Int) : Vector {n} Int := fun i =>\n"
            f"  match i with\n{arms}")


def _bool_vec(name: str, p) -> str:
    return _vec_def(name, p, lambda x: "true" if x else "false", typ="Bool")


def _input(k: int, dev, n: int):
    """The Lean term a device's reading is applied to — the state, the successor,
    or a per-device vector mixing both ends — as ``(term, definition | None,
    names to unfold)``."""
    kinds = {kind for kind, _ in dev.inputs if kind != "unread"}
    if kinds <= {"latched"}:
        return "s", None, []
    if kinds == {"next"}:
        return "post_state s", None, ["post_state"]
    arms = "\n".join(
        f"    | {_fin(j)} => {'post_state s' if kind == 'next' else 's'} {_fin(j)}"
        for kind, j in dev.inputs)
    return (f"in{k} s",
            f"def in{k} (s : Vector {n} Int) : Vector {n} Int := fun i =>\n"
            f"  match i with\n{arms}",
            [f"in{k}", "post_state"])


def _arg(inp: str) -> str:
    """``inp`` as a function argument: parenthesised only when compound."""
    return inp if " " not in inp else f"({inp})"


def _device_term(dev, net, inp: str) -> str:
    """The Lean term for a device's value: its network applied to its input, or
    the affine form over the input for a ReLU-free reading."""
    if net.units:
        return f"V{_dev(str(dev.net))} {_arg(inp)} fzero"
    coeffs, const = net.out
    return _affine_terms(coeffs, const, [f"{inp} {_fin(j)}" for j in range(len(coeffs))])


def _emit_activations(name: str, units, input_term: str, extra_unfold: list[str],
                      n: int, tag: str) -> str:
    """The hidden layer's pre-activations at ``input_term``, as an explicit vector
    plus the lemma identifying it with ``affine nrf_W0 nrf_b0 <input>``.

    Unfolding that matrix product is the single most expensive step in a proof, and
    it is the same work for every region of a path. Doing it once here leaves each
    region's collapse side-goal to be read off an explicit affine form."""
    arms = "\n".join(f"  | {_fin(j)} => {_affine_str(c, k)}"
                     for j, (c, k) in enumerate(units))
    d = _dev(tag)
    cases = "\n".join(
        f"  | {_fin(j)} => simp only [{', '.join([_simp((tag,)), name] + extra_unfold)}] <;> omega"
        for j in range(len(units)))
    return (
        f"def {name} (s : Vector {n} Int) : Vector {len(units)} Int := fun\n{arms}\n\n"
        f"theorem {name}_eq (s : Vector {n} Int) :\n"
        f"    affine nrf{d}_W0 nrf{d}_b0 {_arg(input_term)} = {name} s := by\n"
        f"  funext i\n"
        f"  match i with\n{cases}"
    )


def _emit_collapse(k: int, r: int, affine, n: int, dev, inp: str, signs: str,
                   unfold_signs: tuple) -> str:
    """``V (input) fzero = <affine piece>`` wherever region ``r`` holds: rewrite
    the ReLU layer by the pattern's mask (``reluᵥ_eq_mask``, its side goal read
    off the device's activations), unfold the output layer, and ``omega``."""
    coeffs, const = affine
    d = _dev(str(dev.net))
    pat, acts = f"pat{k}_{r}", f"act{k}"
    small = ", ".join([acts, pat, "forall_fin_succ", "forall_fin_zero",
                       "↓reduceIte", "and_true", "true_and", "implies_true",
                       "Bool.true_eq_false", "Bool.false_eq_true", "false_implies",
                       "forall_const", "imp_self"])
    return (
        f"theorem c0{k}_{r} (s : Vector {n} Int)\n"
        f"    (hs : {signs} s) : V{d} {_arg(inp)} fzero = {_affine_str(coeffs, const)} := by\n"
        f"  simp only [{', '.join([signs, *unfold_signs])}] at hs\n"
        f"  have hmask : reluᵥ (affine nrf{d}_W0 nrf{d}_b0 {_arg(inp)})\n"
        f"             = mask {pat} (affine nrf{d}_W0 nrf{d}_b0 {_arg(inp)}) := by\n"
        f"    rw [{acts}_eq]\n"
        f"    apply reluᵥ_eq_mask\n"
        f"    simp only [{small}] <;> omega\n"
        f"  simp only [V{d}]\n"
        f"  rw [hmask, {acts}_eq, out_apply{d}]\n"
        f"  simp only [mask, {pat}, Bool.true_eq_false, Bool.false_eq_true,\n"
        f"             ↓reduceIte, {acts}] <;> omega"
    )


# ---------------------------------------------------------------------------
# The rule on a path: refuting each disjunct on each region, then `step_ok`
# ---------------------------------------------------------------------------

def _emit_refute(r: int, cert: CellCert, n: int, sign_defs: list[str]) -> str:
    """``cellR_signs s → trans s → invariants s → ¬(disjunct)``: the disjunct's
    rows, taken as hypotheses, complete the region's Farkas system (the
    substrate's ``refute_bridge`` rebuilds the rows and feeds
    ``cellRdD_infeasible``, closed by ``omega``). A constant infeasibility needs
    no system: ``omega`` refutes the disjunct on its own."""
    name = f"cell{r}d{cert.disjunct}"
    rows = _rule_rows(cert)
    conj = " ∧ ".join(f"({_affine_str(a, 0)} ≤ {int(b)})" for a, b in rows) or "True"
    head = (f"theorem {name}_refute (s : Vector {n} Int)\n"
            f"    (hg : trans s) (hinv : invariants s) (hs : cell{r}_signs s) :\n"
            f"    ¬ ({conj}) := by\n")
    if _trivial(cert):
        return head + "  omega"
    unfold = ", ".join(["trans", "invariants", f"cell{r}_signs"] + sign_defs
                       + [f"{name}_A", f"{name}_b"])
    return head + f"  refute_bridge ({name}_infeasible s) with {unfold}"


def _formula_prop(formula, s_syms, wire_terms: dict) -> str:
    """A rule or property formula, z3 over the columns and wire symbols, as a
    Lean ``Prop`` over ``s`` — each wire symbol standing for the Lean term
    ``wire_terms`` gives it."""
    syms = list(s_syms) + list(wire_terms)
    terms = _state_terms(len(s_syms)) + list(wire_terms.values())
    try:
        return _z3_prop(formula, syms, terms)
    except _Drop as exc:
        raise ValueError(f"formula is not linear over the columns and wires: {exc}")


def _emit_path(path: str, pcert, res, system, s_syms, trivial_inv: bool,
               invariants) -> str:
    """One affine path: its Farkas systems, ``trans`` and ``post_state``, the
    regions' signs and ``covered``, each device's activations with its bounds or
    collapses, the ``refute`` lemmas, ``ok`` and ``step_ok``, ``Step`` (and
    ``lex_step`` under a ranking rule), ``RawStep`` and ``consecution``."""
    n = len(s_syms)
    regions = _regions(pcert.cells)
    reps = [cs[0] for cs in regions]
    tree = (_tiling_tree(dataclasses.replace(pcert, cells=tuple(reps)), invariants, s_syms)
            if len(reps) > 1 else None)
    body = list(pcert.body)
    assert not any(_contains_ite(e) for e in body), \
        "path body must be affine (expand_cases should have split every ite)"
    body_affines = [affine_coeffs(e, s_syms) for e in body]
    pins, of_region = _sign_index(reps)

    parts = []
    for r, cs in enumerate(regions):
        for c in cs:
            parts.append(_emit_cell(f"{r}d{c.disjunct}", c))
    parts.append(f"def trans (s : Vector {n} Int) : Prop :=\n"
                 f"  {_render_conjuncts(pcert.guard, s_syms)}")
    parts.append(_emit_post_state(body_affines, n))
    for pattern, k in pins.items():
        parts.append(_emit_signs_def(f"signs_{k}", pattern, pcert.units, n))
    for r, pats in enumerate(of_region):
        parts.append(f"def cell{r}_signs (s : Vector {n} Int) : Prop :=\n"
                     f"  {' ∧ '.join(f'{d} s' for d in _cell_sign_defs(pats))}")
    if reps:
        parts.append(_emit_covered(reps, n, tree, of_region, pcert))

    # per device: its input, activations, and a bound per distinct mask slice or a
    # collapse per region; `haves[r]` collects what `step_ok` brings in on region r
    wire_terms = {system.W[pr[1].id]: _affine_str(c, k)
                  for pr, (c, k) in zip(system.pairs, body_affines)}
    haves = [[] for _ in regions]
    for k, dev in enumerate(res.devices):
        net, units = res.nets[dev.net], pcert.device_units[k]
        inp, in_def, unfold = _input(k, dev, n)
        if in_def:
            parts.append(in_def)
        wire_terms[system.W[dev.wire_id]] = _device_term(dev, net, inp)
        if not net.units:
            continue
        parts.append(_emit_activations(f"act{k}", units, inp, unfold, n, str(dev.net)))
        m = len(units)
        for r, c in enumerate(reps):
            sl = tuple(c.pattern)[dev.offset:dev.offset + m]
            parts.append(_bool_vec(f"pat{k}_{r}", sl))
            parts.append(_emit_collapse(k, r, c.affines[k], n, dev, inp,
                                        f"cell{r}_signs",
                                        tuple(_cell_sign_defs(of_region[r]))))
            haves[r].append(f"have hc{k} := c0{k}_{r} s hs")

    for r, cs in enumerate(regions):
        for c in cs:
            parts.append(_emit_refute(r, c, n, _cell_sign_defs(of_region[r])))

    parts.append(f"/-- The rule on this path's round. -/\n"
                 f"def ok (s : Vector {n} Int) : Prop :=\n"
                 f"  {_formula_prop(res.ok, s_syms, wire_terms)}")
    head = (f"theorem step_ok (s : Vector {n} Int) (hg : trans s) (hinv : invariants s) :\n"
            f"    ok s := by\n")
    if not regions:                          # the rule holds outright: no disjunct to refute
        parts.append(head + "  unfold ok\n  trivial")
    else:
        branches = " | ".join("hs" for _ in regions)
        bullets = []
        for r, cs in enumerate(regions):
            lines = haves[r] + [f"have hr{c.disjunct} := cell{r}d{c.disjunct}_refute s hg hinv hs"
                                for c in cs] + ["unfold ok", "omega"]
            bullets.append("  · " + "\n    ".join(lines))
        parts.append(head + f"  rcases covered s hg hinv with {branches}\n" + "\n".join(bullets))

    # The transition is functional (b = post_state a on the guard), so Step needs
    # no SSA witness: the pre-state *is* a.
    parts.append(f"def Step (a b : Vector {n} Int) : Prop :=\n"
                 f"  trans a ∧ invariants a ∧ post_state a = b")
    if res.rule.ranks:
        ranks = ", ".join(f"R{d}" for d in range(len(res.rule.ranks)))
        parts.append(
            f"/-- lex step of this path: the ranks drop lexicographically. -/\n"
            f"theorem lex_step (a b : Vector {n} Int) (h : Step a b) :\n"
            f"    lexDec [{ranks}] a b := by\n"
            f"  obtain ⟨hg, hinv, hpost⟩ := h\n"
            f"  subst hpost\n"
            f"  have hok := step_ok a hg hinv\n"
            f"  unfold ok at hok\n"
            f"  simp only [lexDec, {ranks}]\n"
            f"  omega")
    parts += _emit_rawstep_consecution(n, trivial_inv)
    return f"namespace {path}\n\n" + "\n\n".join(parts) + f"\n\nend {path}"


def _inv_proof(trivial_inv: bool, unfold: str) -> str:
    """Tactic closing an invariant lemma (``initiation`` / ``consecution``):
    ``trivial`` for the ``True`` invariant, else ``omega`` on the linear
    entailment left by unfolding ``unfold``."""
    if trivial_inv:
        return "  trivial"
    return f"  simp only [{unfold}] at *\n  omega"


def _emit_rawstep_consecution(n: int, trivial_inv: bool) -> list[str]:
    """The path's transition relation and its consecution lemma — what any
    property's proof needs of a path, with no certificate involved."""
    return [
        f"/-- One iteration of this path: the guard and the body. -/\n"
        f"def RawStep (a b : Vector {n} Int) : Prop :=\n"
        f"  trans a ∧ post_state a = b",
        f"/-- Consecution: the body preserves the invariant on this path. -/\n"
        f"theorem consecution (s : Vector {n} Int)\n"
        f"    (hg : trans s) (hinv : invariants s) :\n"
        f"    invariants (post_state s) := by\n"
        f"{_inv_proof(trivial_inv, 'trans, invariants, post_state')}",
    ]


# ---------------------------------------------------------------------------
# Whole-program compositions
# ---------------------------------------------------------------------------

def _hinv_induction(path_names) -> str:
    """``have hInv : ∀ i, invariants (f i)`` along a run of ``RawStep``:
    ``initiation`` at the entry state, the taken path's ``consecution`` at each
    step. Shared by every whole-program theorem, whatever the property."""
    if len(path_names) == 1:
        p = path_names[0]
        cons_case = (f"      obtain ⟨hg, hp⟩ := hstep k\n"
                     f"      rw [← hp]\n"
                     f"      exact {p}.consecution (f k) hg ih")
    else:
        rc = " | ".join("h" for _ in path_names)
        cons_case = (f"      rcases hstep k with {rc}\n" + "\n".join(
            f"      · obtain ⟨hg, hp⟩ := h\n"
            f"        rw [← hp]\n"
            f"        exact {p}.consecution (f k) hg ih" for p in path_names))
    return (f"  have hInv : ∀ i, invariants (f i) := by\n"
            f"    intro i\n"
            f"    induction i with\n"
            f"    | zero => rw [hf0]; exact initiation s0 hinit\n"
            f"    | succ k ih =>\n"
            f"{cons_case}")


def _emit_init_and_initiation(n: int, init_lean: str, trivial_inv: bool) -> str:
    return (f"def Init (s : Vector {n} Int) : Prop :=\n  {init_lean}\n\n"
            f"/-- Initiation: the loop is entered in an invariant-satisfying state. -/\n"
            f"theorem initiation (s : Vector {n} Int) (h : Init s) : invariants s := by\n"
            f"{_inv_proof(trivial_inv, 'Init, invariants')}")


def _emit_termination_composition(path_names, n: int, init_lean: str, trivial_inv: bool,
                                  rank_nets) -> str:
    """The whole-program theorem under a ranking rule: ``Step`` as the union of
    the paths, ``no_infinite_run_lex [R0, …]`` fed each rank's non-negativity and
    each path's ``lex_step``, and ``program_terminates`` — from any ``Init``
    state there is no infinite run of ``RawStep``, since the invariant derived
    along the run upgrades every ``RawStep`` to a ``Step``."""
    K = len(rank_nets)
    ranks = ", ".join(f"R{d}" for d in range(K))
    step = " ∨ ".join(f"{p}.Step a b" for p in path_names)
    rawstep = " ∨ ".join(f"{p}.RawStep a b" for p in path_names)
    pos_pat = " | ".join("rfl" for _ in range(K))
    pos = "\n".join(f"    · exact V{_dev(str(j))}_nonneg s fzero" for j in rank_nets)
    pat = " | ".join("h" for _ in path_names)
    dec = "\n".join(f"    · exact {p}.lex_step a b h" for p in path_names)
    if len(path_names) == 1:
        step_run = f"  obtain ⟨hg, hp⟩ := hstep i\n  exact ⟨hg, hInv i, hp⟩"
    else:
        rc = " | ".join("h" for _ in path_names)
        step_bul = []
        for i, p in enumerate(path_names):
            inj, close = _disjunct(i, len(path_names))
            step_bul.append(f"  · obtain ⟨hg, hp⟩ := h\n    exact {inj}⟨hg, hInv i, hp⟩{close}")
        step_run = f"  rcases hstep i with {rc}\n" + "\n".join(step_bul)
    return (
        f"/-- The program's step relation: one iteration of the loop (any path). -/\n"
        f"def Step (a b : Vector {n} Int) : Prop := {step}\n\n"
        f"{_emit_init_and_initiation(n, init_lean, trivial_inv)}\n\n"
        f"/-- One iteration of the loop on any path: the guard and the body. -/\n"
        f"def RawStep (a b : Vector {n} Int) : Prop := {rawstep}\n\n"
        f"theorem no_inf_step :\n"
        f"    ¬ ∃ f : Nat → Vector {n} Int, ∀ m, Step (f m) (f (m + 1)) := by\n"
        f"  apply no_infinite_run_lex [{ranks}] Step\n"
        f"  · intro W hW s\n"
        f"    simp only [List.mem_cons, List.not_mem_nil, or_false] at hW\n"
        f"    rcases hW with {pos_pat}\n"
        f"{pos}\n"
        f"  · rintro a b ({pat})\n"
        f"{dec}\n\n"
        f"/-- The program terminates: from any loop-entry state there is no\n"
        f"    infinite run of guarded steps. -/\n"
        f"theorem program_terminates (s0 : Vector {n} Int) (hinit : Init s0) :\n"
        f"    ¬ ∃ f : Nat → Vector {n} Int, f 0 = s0 ∧ ∀ i, RawStep (f i) (f (i + 1)) := by\n"
        f"  rintro ⟨f, hf0, hstep⟩\n"
        f"{_hinv_induction(path_names)}\n"
        f"  apply no_inf_step\n"
        f"  refine ⟨f, fun i => ?_⟩\n"
        f"{step_run}"
    )


def _emit_safety_composition(path_names, n: int, init_lean: str, trivial_inv: bool) -> str:
    """The whole-program safety theorem: along any run of ``RawStep`` from an
    ``Init`` state, ``pred`` holds at every step. The invariant carries the
    proof — :func:`_hinv_induction` derives it along the run — and the taken
    path's ``step_ok`` closes the predicate at each state."""
    rawstep = " ∨ ".join(f"{p}.RawStep a b" for p in path_names)

    def close(p: str, pad: str) -> str:
        return (f"{pad}have hok := {p}.step_ok (f i) hg (hInv i)\n"
                f"{pad}have hi := hInv i\n"
                f"{pad}unfold {p}.ok at hok\n"
                f"{pad}unfold pred\n"
                f"{pad}unfold invariants at hi\n"
                f"{pad}omega")

    if len(path_names) == 1:
        body = f"  obtain ⟨hg, _⟩ := hstep i\n{close(path_names[0], '  ')}"
    else:
        rc = " | ".join("h" for _ in path_names)
        body = f"  rcases hstep i with {rc}\n" + "\n".join(
            f"  · obtain ⟨hg, _⟩ := h\n{close(p, '    ')}" for p in path_names)
    return (
        f"{_emit_init_and_initiation(n, init_lean, trivial_inv)}\n\n"
        f"/-- One iteration of the loop on any path: the guard and the body. -/\n"
        f"def RawStep (a b : Vector {n} Int) : Prop := {rawstep}\n\n"
        f"/-- The property holds: along any run from a loop-entry state, every\n"
        f"    state satisfies ``pred``. -/\n"
        f"theorem always_holds (s0 : Vector {n} Int) (hinit : Init s0)\n"
        f"    (f : Nat → Vector {n} Int) (hf0 : f 0 = s0)\n"
        f"    (hstep : ∀ i, RawStep (f i) (f (i + 1))) : ∀ i, pred (f i) := by\n"
        f"  intro i\n"
        f"{_hinv_induction(path_names)}\n"
        f"{body}"
    )


_HEADER = (
    "import Coverage\nimport Net\nimport Termination\n"
    "set_option linter.unusedVariables false\n"
    "set_option linter.unusedSimpArgs false\n"
    "set_option maxHeartbeats 1000000\n"
    "namespace Matrix\nopen Fin\n"
)
_FOOTER = "\nend Matrix\n"


def emit_program(name: str, system, result) -> str:
    """The whole ``program.lean`` for ``name``, from ``system`` — the module as
    read, with what is known of its states — and ``result``, the
    :class:`._farkas.FarkasResult` a ``certify`` run on it returned: the
    certificates, the resolved formula and predicate, the rule and the devices.

    A rule with ranks proves termination, concluding ``program_terminates``; a
    rule without proves the :class:`._property.Always` property, concluding
    ``always_holds``. The entry state is read off the system, the networks off
    the result's devices — never the weights."""
    rule = result.rule
    paths = result.certificates
    if not paths:
        raise ValueError(f"{name}: no certified paths to emit")
    s_syms = list(system.s_syms)
    n = len(s_syms)
    path_names = [f"loop0_path{i}" for i in range(len(paths))]
    cols = ", ".join(f"s {j} = {nm}" for j, nm in enumerate(system.names))
    npaths = f" ({len(paths)} paths)" if len(paths) > 1 else ""
    init_lean = _render_conjuncts(entry_predicate(system), s_syms)
    inv_all = list(result.inv) + list(system.invariants)
    inv_lean = _render_conjuncts(z3.And(*inv_all), s_syms) if inv_all else "True"
    trivial_inv = inv_lean == "True"
    by_id = {d.wire_id: d for d in result.devices}
    rank_nets = [by_id[v_s.id].net for v_s, _ in rule.ranks]

    if rank_nets:
        what = ("terminates via a ranking function" if len(rank_nets) == 1 else
                f"terminates via a lexicographic rank of {len(rank_nets)} networks")
        pred_lean = None
    else:
        terms = {system.W[d.wire_id]: _device_term(d, result.nets[d.net], "s")
                 for d in result.devices}
        pred_lean = _formula_prop(result.pred, s_syms, terms)
        what = f"`{pred_lean}` holds on every run"
    parts = [f"/- ──── program: {name} — {what}{npaths}.\n   Columns: {cols}. ──── -/"]
    for j, net in enumerate(result.nets):
        if net.units:
            parts += [_emit_network(net, str(j)), _emit_out_apply(net, str(j)),
                      _emit_out_nonneg(str(j)), _emit_nonneg(net, str(j))]
    for d, j in enumerate(rank_nets):
        parts.append(f"def R{d} : Vector {n} Int → Int := fun s => V{_dev(str(j))} s fzero")
    parts.append(f"def invariants (s : Vector {n} Int) : Prop :=\n  {inv_lean}")
    if pred_lean is not None:
        parts.append(f"def pred (s : Vector {n} Int) : Prop :=\n  {pred_lean}")
    for pname, pcert in zip(path_names, paths):
        parts.append(_emit_path(pname, pcert, result, system, s_syms, trivial_inv,
                                inv_all))
    if rank_nets:
        parts.append(_emit_termination_composition(path_names, n, init_lean, trivial_inv,
                                                   rank_nets))
    else:
        parts.append(_emit_safety_composition(path_names, n, init_lean, trivial_inv))
    return _HEADER + "\n\n".join(parts) + _FOOTER


def write_program_proof(name: str, system, result, out_dir: Path) -> Path:
    """Write ``<out_dir>/<name>/program.lean`` and return its path."""
    target = Path(out_dir) / name
    target.mkdir(parents=True, exist_ok=True)
    out = target / "program.lean"
    out.write_text(emit_program(name, system, result))
    return out
