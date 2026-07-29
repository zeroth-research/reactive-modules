"""Emit a kernel-checkable Lean termination proof from Farkas certificates.

Consumes the :class:`._farkas.CellCert` set a ``farkas_cell`` run captures (plus
the :class:`._verify_ranking.Obligation` it was run on) and writes ONE Lean file
proving the program terminates, against the vendored ``lean/`` substrate
(``Coverage``/``Net``/``Termination``). The file contains:

  * the integer ranking network ``V`` and its structural ``V_nonneg``;
  * per cell: the Farkas system ``A``/``b``/``y``, the three side-condition
    goals, the ``farkas_sound`` infeasibility, and the scalar ``decrease``
    entailment (via the substrate's ``decrease_bridge`` tactic);
  * ``trans``/``invariants`` (the loop guard, and the Houdini invariants), the
    per-cell sign regions, and ``covered`` (the cells tile the guard — the CEGAR
    coverage guarantee, discharged by ``omega``);
  * the V-collapse lemmas (``V`` equals its affine piece on each cell) and, from
    them, ``lex_step`` (``V`` strictly drops on every guarded step);
  * ``initiation`` and per-path ``consecution``, proving the invariants inductive;
  * ``program_terminates`` — from any loop-entry (``Init``) state there is no
    infinite run of guarded steps — via ``no_infinite_run_lex``.

Scope: scalar (single-component) ranking function, single loop with a single
hidden layer. In-loop branching is handled by *path-splitting*: the body's nested
``ite``s are expanded into affine paths (:func:`._farkas.enumerate_paths`), one
namespace each, and ``Step`` is the union of the per-path relations.

The emitter is pure formatting: every number comes verbatim from the captured
certificate or the network weights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import z3

from ._farkas import CellCert, _flatten_and, affine_coeffs


def _contains_ite(e) -> bool:
    """True if a z3 term still has an ``ite`` — a path body must have none (all
    in-loop branches are split out by ``enumerate_paths`` before emission)."""
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

def _affine_str(coeffs, const: int) -> str:
    """``Σ cⱼ·(s j) + const`` over the nonzero coefficients (``const`` if none)."""
    terms = [f"({int(c)} * s {_fin(j)})" for j, c in enumerate(coeffs) if c]
    if not terms:
        return f"{int(const)}"
    expr = terms[0]
    for t in terms[1:]:
        expr = f"({expr} + {t})"
    if const:
        expr = f"({expr} + {int(const)})"
    return expr


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


def _z3_prop(a, syms) -> str:
    """Render a z3 boolean expression to a Lean ``Prop`` over ``s`` (integer
    columns ``syms``). Handles ∧ / ∨ / ¬ / linear comparisons; raises
    :class:`_Drop` on anything non-linear."""
    if z3.is_and(a):
        return "(" + " ∧ ".join(_z3_prop(c, syms) for c in a.children()) + ")"
    if z3.is_or(a):
        return "(" + " ∨ ".join(_z3_prop(c, syms) for c in a.children()) + ")"
    if z3.is_not(a):
        inner = a.arg(0)
        if z3.is_app(inner) and inner.decl().kind() == z3.Z3_OP_EQ:
            return f"({_cmp_prop(inner, syms, op='≠')})"
        return f"(¬ {_z3_prop(inner, syms)})"
    if z3.is_app(a) and a.decl().kind() in _CMP:
        return _cmp_prop(a, syms)
    if z3.is_app(a) and a.decl().kind() == z3.Z3_OP_DISTINCT:
        return _cmp_prop(a, syms, op="≠")
    raise _Drop(str(a))


def _cmp_prop(a, syms, op: str | None = None) -> str:
    """A comparison ``lhs ⋈ rhs`` as ``(<lhs-rhs> ⋈ 0)`` (linear, omega-ready)."""
    if op is None:
        op = _CMP[a.decl().kind()]
    try:
        coeffs, const = affine_coeffs(a.arg(0) - a.arg(1), syms)
    except ValueError:
        raise _Drop(str(a))
    return f"({_affine_str(coeffs, const)} {op} 0)"


def _render_conjuncts(pred, syms) -> str:
    """Flatten ``pred`` (a z3 ∧-tree) to conjuncts, render each, drop the ones
    that are not integer-linear. Empty → ``True``."""
    out = []
    for c in _flatten_and(pred):
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


def _sign_rows(cert: CellCert):
    """The full cell's sign rows (both pre ``cell_s[..]`` and post ``cell_sp[..]``
    activation constraints) — these pin the cell for coverage and the V-collapse."""
    return [(a, b) for a, b, lbl in zip(cert.A, cert.b, cert.labels)
            if lbl.startswith("cell_s")]


def _neg_decrease(cert: CellCert):
    for a, b, lbl in zip(cert.A, cert.b, cert.labels):
        if lbl == "neg_decrease":
            return a, b
    raise ValueError("certificate has no neg_decrease row")


def _trivial(cert: CellCert) -> bool:
    """The support reduces to a constant infeasibility ``0·s ≤ c`` (c < 0): the
    decrease holds *unconditionally* on this cell, so it needs no Farkas
    certificate — ``omega`` closes ``(c+1) ≤ 0`` directly."""
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


# ---------------------------------------------------------------------------
# The ranking network V (the integer net the V-lift reasons about)
# ---------------------------------------------------------------------------

# `simp only` set for the collapse / bounded-below proofs: structural unfoldings,
# the `Fin` expanders, the network defs, the ReLU `ite` reduction, and the
# cast/base reducers that turn `sum`'s base and casts into literals so `omega`
# sees them. Network-independent; the caller appends per-cell pattern/slice defs.
_SIMP = (
    "affine, addᵥ, mulVec_apply, sum, mask, Function.comp, "
    "forall_fin_succ, forall_fin_zero, ↓reduceIte, "
    "nrf_W0, nrf_b0, nrf_W1, nrf_b1, "
    "Nat.zero_eq, Int.ofNat_eq_natCast, Int.cast_ofNat_Int, Int.reduceNeg, "
    "Int.reduceLE, Int.neg_ofNat_le_ofNat, Int.add_zero, Int.zero_add, "
    "Int.mul_one, Int.mul_zero, Int.zero_mul, Int.mul_neg_one, Int.neg_nonneg, "
    "Int.neg_le_zero_iff, Int.zero_le_ofNat, Std.le_refl, and_true, true_and, "
    "and_self, forall_const, imp_self, implies_true, "
    "Bool.true_eq_false, Bool.false_eq_true, false_implies"
)


def _int_grid(W):
    return tuple(tuple(int(round(float(x))) for x in row) for row in np.asarray(W))


def _int_vec(b):
    return tuple(int(round(float(x))) for x in np.asarray(b))


def _emit_network(layers) -> str:
    (W0, b0), (W1, b1) = layers
    n_in, n_out = _int_grid(W0)[0].__len__(), len(_int_vec(b1))
    blocks = [
        _mat_def("nrf_W0", _int_grid(W0), str),
        _vec_def("nrf_b0", _int_vec(b0), str),
        _mat_def("nrf_W1", _int_grid(W1), str),
        _vec_def("nrf_b1", _int_vec(b1), str),
        f"def V (s : Vector {n_in} Int) : Vector {n_out} Int :=\n"
        f"  (affine nrf_W1 nrf_b1 (reluᵥ (affine nrf_W0 nrf_b0 s)))",
    ]
    return "\n\n".join(blocks)


def _emit_nonneg(layers) -> str:
    (W0, _), (W1, b1) = layers
    n_in, n_out = _int_grid(W0)[0].__len__(), len(_int_vec(b1))
    return (
        f"theorem V_nonneg (s : Vector {n_in} Int) (j : Fin {n_out}) : 0 ≤ V s j := by\n"
        f"  simp only [V]\n"
        f"  exact affine_nonneg nrf_W1 nrf_b1 _\n"
        f"    (by simp only [{_SIMP}])\n"
        f"    (by simp only [{_SIMP}])\n"
        f"    (fun k => reluᵥ_nonneg _ k) j"
    )


# ---------------------------------------------------------------------------
# Per-path proof (single affine path): predicates, decrease, coverage, collapse
# ---------------------------------------------------------------------------


def _decrease_goal(cert: CellCert) -> str:
    """``(b_nd + 1) ≤ Σ aⱼ·sⱼ`` — the negation of the ``neg_decrease`` row
    ``Σ aⱼ·sⱼ ≤ b_nd``; over the integers this is the strict decrease."""
    a, b = _neg_decrease(cert)
    return f"({int(b) + 1} ≤ {_affine_str(a, 0)})"


def _emit_cell(idx: int, cert: CellCert) -> str:
    manifest = (f"-- cell {idx}: pattern_s={list(cert.pattern_s)} "
                f"pattern_sp={list(cert.pattern_sp)}")
    if _trivial(cert):
        return manifest + f"\n-- cell {idx}: unconditional decrease (no Farkas system)"
    A, b, y, labels = _support(cert)
    return manifest + "\n\n" + _emit_system(f"cell{idx}", A, b, y, labels)


def _emit_signs_def(idx: int, cert: CellCert, n: int, units) -> str:
    """The cell's region, one condition per hidden unit: ``0 ≤ e`` where its
    pattern says active, ``e ≤ 0`` where inactive. Same inequalities as the
    certificate's sign rows, written in the form :func:`_tiling_tree` splits on so
    a coverage leaf's hypotheses match the conjuncts it has to produce."""
    pattern = tuple(cert.pattern_s) + tuple(cert.pattern_sp)
    conj = [f"(0 ≤ {_affine_str(c, k)})" if active else f"({_affine_str(c, k)} ≤ 0)"
            for (c, k), active in zip(units, pattern)]
    return (f"def cell{idx}_signs (s : Vector {n} Int) : Prop :=\n"
            f"  {' ∧ '.join(conj) if conj else 'True'}")


def _emit_decrease(idx: int, cert: CellCert, n: int) -> str:
    """``cellK_signs s → trans s → invariants s → decrease`` via the substrate's
    one-line ``decrease_bridge`` (it rebuilds the row system and feeds
    ``cellK_infeasible``, closed by ``omega``). The full cell signs are always
    taken as a hypothesis so every support row is available."""
    hyps = (f"(hg : trans s) (hinv : invariants s) (hs : cell{idx}_signs s)")
    goal = _decrease_goal(cert)
    if _trivial(cert):
        # unconditional: goal is ``(c+1) ≤ 0`` with c < 0 — omega, no certificate.
        return (f"theorem cell{idx}_decrease (s : Vector {n} Int)\n"
                f"    {hyps} :\n    {goal} := by omega")
    unfold = ", ".join(["trans", "invariants",
                        f"cell{idx}_signs", f"cell{idx}_A", f"cell{idx}_b"])
    return (
        f"theorem cell{idx}_decrease (s : Vector {n} Int)\n"
        f"    {hyps} :\n"
        f"    {goal} := by\n"
        f"  decrease_bridge (cell{idx}_infeasible s) with {unfold}"
    )


def _tiling_tree(pcert, invariants, s_syms):
    """A decision tree over the hidden units' sign literals whose leaves each name
    one certified cell.

    Splitting on ``0 ≤ unitⱼ`` in turn narrows which cells a branch can still be
    in, and a branch is finished as soon as the literals taken so far *entail* some
    cell's sign rows — checked here, so the ``omega`` the leaf emits is known to
    succeed. Entailment is the test rather than a match against the cell's
    activation pattern: the cells are closed (``pre ≥ 0`` and ``pre ≤ 0`` both hold
    at ``pre = 0``) so they overlap on their boundaries, and a state on one is
    covered by a cell whose pattern names the opposite sign.

    z3 also prunes: a branch no guarded state satisfies becomes ``dead``, which
    keeps the tree the size of the certificate set instead of ``2^units``.

    Returns ``("split", lean_literal, yes, no)``, ``("cell", index)`` or
    ``("dead",)``; raises :class:`_Drop` if a live branch runs out of literals
    without entailing any cell, which means the cells do not tile the guard."""
    # Both signs of each unit are offered as literals. A cell is closed on either
    # side, so a state where a pre-activation is exactly zero lies in cells of both
    # polarities, and pinning only ``0 ≤ e`` leaves the covering cell's ``e ≤ 0``
    # unproved; offering both lets such a branch settle the unit to zero.
    lits = []
    for coeffs, const in pcert.units:
        lean, e = _affine_str(coeffs, const), _affine_z3(coeffs, const, s_syms)
        lits.append((f"0 ≤ {lean}", e >= 0))
        lits.append((f"{lean} ≤ 0", e <= 0))
    domain = [pcert.guard, *invariants]
    signs = [z3.And(*[_affine_z3(a, 0, s_syms) <= int(b)
                      for a, b in _sign_rows(c)]) for c in pcert.cells]

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
            raise _Drop("cells do not tile the guard")
        lean_lit, z3_lit = lits[i]
        yes = build(i + 1, taken + [z3_lit], live)
        no = build(i + 1, taken + [z3.Not(z3_lit)], live)
        if yes == ("dead",) and no == ("dead",):
            return ("dead",)
        return ("split", lean_lit, yes, no)

    return build(0, [], list(range(len(pcert.cells))))


def _emit_covered(certs, n: int, tree) -> str:
    """Every guarded state lies in a certified cell that decreases, from two
    separate ingredients:

      * **coverage** — ``omega`` proves the cells' sign regions tile the guard
        (``guard ∧ inv → ⋁ᵢ cellᵢ_signs``), a linear fact over the sign rows;
      * **decrease** — the covering cell's ``cellᵢ_decrease`` supplies the drop,
        backed by its Farkas certificate ``cellᵢ_infeasible``.

    Deleting a certificate therefore breaks this proof."""
    hyps = "(hg : trans s) (hinv : invariants s)"
    dec_args = "s hg hinv"
    disj = "\n      ∨ ".join(
        f"(cell{i}_signs s ∧ {_decrease_goal(c)})" for i, c in enumerate(certs))
    ncerts = len(certs)
    if ncerts == 1:
        unfold = ", ".join(["trans", "invariants", "cell0_signs"])
        body = (f"  have h : cell0_signs s := by\n"
                f"    simp only [{unfold}] at *\n"
                f"    omega\n"
                f"  exact ⟨h, cell0_decrease {dec_args} h⟩")
    else:
        body = _emit_tiling_tree(certs, tree, dec_args)
    return (
        f"theorem covered (s : Vector {n} Int)\n"
        f"    {hyps} :\n"
        f"    {disj} := by\n"
        f"{body}"
    )


def _disjunct(i: int, total: int) -> tuple[str, str]:
    """``Or`` injections wrapping the ``i``-th of ``total`` disjuncts."""
    return "Or.inr (" * i + ("Or.inl " if i < total - 1 else ""), ")" * i


def _emit_tiling_tree(certs, tree, dec_args: str, depth: int = 1) -> str:
    """The coverage case split, as a decision tree over the hidden units' sign
    literals rather than one ``omega`` over the whole disjunction.

    ``omega`` decides a conjunctive goal in time linear in its facts, but the flat
    tiling goal ``⋁ᵢ cellᵢ_signs`` negates into a clause per cell, and the case
    split across those clauses grows exponentially in the number of cells — 12
    cells already exhaust the elaborator's budget. Splitting on the sign literals
    instead reaches, at each leaf, an assignment that names one cell, so every
    ``omega`` sees a conjunction: the accumulated literals entailing that cell's
    sign rows. The tree is built with the guard in hand, so a branch no state can
    satisfy is closed rather than explored (see :func:`_tiling_tree`)."""
    pad = "  " * depth
    kind = tree[0]
    if kind == "cell":
        i = tree[1]
        inj, close = _disjunct(i, len(certs))
        return (f"{pad}have hc : cell{i}_signs s := by\n"
                f"{pad}  simp only [trans, invariants, cell{i}_signs] at *\n"
                f"{pad}  omega\n"
                f"{pad}exact {inj}⟨hc, cell{i}_decrease {dec_args} hc⟩{close}")
    if kind == "dead":
        return (f"{pad}exfalso\n"
                f"{pad}simp only [trans, invariants] at *\n"
                f"{pad}omega")
    _, lit, yes, no = tree
    return (f"{pad}by_cases hs{depth} : {lit}\n"
            f"{pad}· {_emit_tiling_tree(certs, yes, dec_args, depth + 1).lstrip()}\n"
            f"{pad}· {_emit_tiling_tree(certs, no, dec_args, depth + 1).lstrip()}")


def _emit_post_state(body_affines, n: int) -> str:
    """``post_state s = body(s)`` — the loop body's next state as an affine map
    (V is evaluated here for the successor rank)."""
    arms = "\n".join(
        f"    | {_fin(k)} => {_affine_str(c, kk)}"
        for k, (c, kk) in enumerate(body_affines))
    return (f"def post_state (s : Vector {n} Int) : Vector {n} Int := fun i =>\n"
            f"  match i with\n{arms}")


def _bool_vec(name: str, p) -> str:
    return _vec_def(name, p, lambda x: "true" if x else "false", typ="Bool")


def _emit_activations(name: str, units, input_term: str, extra_unfold: list[str],
                      n: int) -> str:
    """The hidden layer's pre-activations at ``input_term``, as an explicit vector
    plus the lemma identifying it with ``affine nrf_W0 nrf_b0 <input>``.

    Unfolding that matrix product is the single most expensive step in a proof, and
    it is the same work for every cell of a path. Doing it once here leaves each
    cell's mask side-goal to be read off an explicit affine form."""
    arms = "\n".join(f"  | {_fin(j)} => {_affine_str(c, k)}"
                     for j, (c, k) in enumerate(units))
    cases = "\n".join(
        f"  | {_fin(j)} => simp only [{', '.join([_SIMP, name] + extra_unfold)}] <;> omega"
        for j in range(len(units)))
    return (
        f"def {name} (s : Vector {n} Int) : Vector {len(units)} Int := fun\n{arms}\n\n"
        f"theorem {name}_eq (s : Vector {n} Int) :\n"
        f"    affine nrf_W0 nrf_b0 ({input_term}) = {name} s := by\n"
        f"  funext i\n"
        f"  match i with\n{cases}"
    )


def _emit_collapse(idx: int, side: str, input_term: str, extra_unfold: list[str],
                   affine, n: int, acts: str) -> str:
    """``V (<input>) fzero = <affine piece>`` on the cell: rewrite the ReLU layer
    by the cell's mask (``reluᵥ_eq_mask``, its side goal read off ``acts``), unfold
    the output layer, and ``omega``."""
    coeffs, const = affine
    pat = f"cell{idx}_pat_{side}"
    small = ", ".join([acts, pat, "forall_fin_succ", "forall_fin_zero",
                       "↓reduceIte", "and_true", "true_and", "implies_true",
                       "Bool.true_eq_false", "Bool.false_eq_true", "false_implies",
                       "forall_const", "imp_self"])
    simp_tail = ", ".join([_SIMP, pat, acts] + extra_unfold)
    return (
        f"theorem cell{idx}_{side}_c0 (s : Vector {n} Int)\n"
        f"    (hs : cell{idx}_signs s) : V ({input_term}) fzero = {_affine_str(coeffs, const)} := by\n"
        f"  simp only [cell{idx}_signs] at hs\n"
        f"  have hmask : reluᵥ (affine nrf_W0 nrf_b0 ({input_term}))\n"
        f"             = mask {pat} (affine nrf_W0 nrf_b0 ({input_term})) := by\n"
        f"    rw [{acts}_eq]\n"
        f"    apply reluᵥ_eq_mask\n"
        f"    simp only [{small}] <;> omega\n"
        f"  simp only [V]\n"
        f"  rw [hmask, {acts}_eq]\n"
        f"  simp only [{simp_tail}] <;> omega"
    )


def _inv_proof(trivial_inv: bool, unfold: str) -> str:
    """Tactic closing an invariant lemma (``initiation`` / ``consecution``):
    ``trivial`` for the ``True`` invariant, else ``omega`` on the linear
    entailment left by unfolding ``unfold``."""
    if trivial_inv:
        return "  trivial"
    return f"  simp only [{unfold}] at *\n  omega"


def _emit_path(path: str, pcert, s_syms, trivial_inv: bool, invariants) -> str:
    """One affine path: its Farkas cells, ``trans``, ``covered``, ``post_state``,
    the V-collapse lemmas, ``Step``/``lex_step``, and ``RawStep``/``consecution``."""
    n = len(s_syms)
    certs = pcert.cells
    tree = _tiling_tree(pcert, invariants, s_syms) if len(certs) > 1 else None
    trans_lean = _render_conjuncts(pcert.guard, s_syms)
    body = list(pcert.body)
    assert not any(_contains_ite(e) for e in body), \
        "path body must be affine (enumerate_paths should have split every ite)"
    body_affines = [affine_coeffs(e, s_syms) for e in body]

    parts = []
    for idx, cert in enumerate(certs):
        parts.append(_emit_cell(idx, cert))
    parts.append(f"def trans (s : Vector {n} Int) : Prop :=\n  {trans_lean}")
    for idx, cert in enumerate(certs):
        parts.append(_emit_signs_def(idx, cert, n, pcert.units))
    for idx, cert in enumerate(certs):
        parts.append(_emit_decrease(idx, cert, n))
    parts.append(_emit_covered(certs, n, tree))
    parts.append(_emit_post_state(body_affines, n))
    m = len(pcert.units) // 2
    parts.append(_emit_activations("pre_act", pcert.units[:m], "s", [], n))
    parts.append(_emit_activations("post_act", pcert.units[m:], "post_state s",
                                   ["post_state"], n))
    for idx, cert in enumerate(certs):
        parts.append(_bool_vec(f"cell{idx}_pat_pre", cert.pattern_s))
        parts.append(_bool_vec(f"cell{idx}_pat_post", cert.pattern_sp))
        parts.append(_emit_collapse(idx, "pre", "s", [], cert.pre_affine, n,
                                    "pre_act"))
        parts.append(_emit_collapse(idx, "post", "post_state s", ["post_state"],
                                    cert.post_affine, n, "post_act"))

    # The transition is functional (b = post_state a on the guard), so Step needs
    # no SSA witness: the pre-state *is* a.
    branches = " | ".join("⟨hs, hd⟩" for _ in certs)
    bullets = "\n".join(
        f"  · rw [cell{idx}_pre_c0 a hs, cell{idx}_post_c0 a hs]; omega"
        for idx in range(len(certs)))
    parts.append(
        f"def Step (a b : Vector {n} Int) : Prop :=\n"
        f"  trans a ∧ invariants a ∧ post_state a = b")
    parts.append(
        f"/-- lex step of this path (strict component 0). -/\n"
        f"theorem lex_step (a b : Vector {n} Int) (h : Step a b) :\n"
        f"    V b fzero < V a fzero := by\n"
        f"  obtain ⟨hg, hinv, hpost⟩ := h\n"
        f"  subst hpost\n"
        f"  rcases covered a hg hinv with {branches}\n"
        f"{bullets}")
    parts.append(
        f"/-- One iteration of this path: the guard and the body. -/\n"
        f"def RawStep (a b : Vector {n} Int) : Prop :=\n"
        f"  trans a ∧ post_state a = b")
    parts.append(
        f"/-- Consecution: the body preserves the invariant on this path. -/\n"
        f"theorem consecution (s : Vector {n} Int)\n"
        f"    (hg : trans s) (hinv : invariants s) :\n"
        f"    invariants (post_state s) := by\n"
        f"{_inv_proof(trivial_inv, 'trans, invariants, post_state')}")
    return f"namespace {path}\n\n" + "\n\n".join(parts) + f"\n\nend {path}"


# ---------------------------------------------------------------------------
# Composition: program_terminates via no_infinite_run_lex
# ---------------------------------------------------------------------------

def _v0_and_step(path_names, n: int) -> str:
    """The ranking projection ``V0`` and the whole-program step relation ``Step``
    (the union of the per-path relations)."""
    step = " ∨ ".join(f"{p}.Step a b" for p in path_names)
    return (f"def V0 : Vector {n} Int → Int := fun s => V s fzero\n\n"
            f"/-- The program's step relation: one iteration of the loop (any path). -/\n"
            f"def Step (a b : Vector {n} Int) : Prop := {step}")


def _no_inf_run_body(path_names) -> str:
    """Tactic body proving ``¬ ∃ f, ∀ _, Step ..``: ``no_infinite_run_lex`` needs
    V ≥ 0 (``V_nonneg``) and a strict drop on every step, which each path's
    ``lex_step`` supplies after the union is case-split."""
    pat = " | ".join("h" for _ in path_names)
    bullets = "\n".join(
        f"    · simp only [lexDec, V0]\n"
        f"      have hx := {p}.lex_step a b h\n"
        f"      exact Or.inl (hx)" for p in path_names)
    return (
        f"  apply no_infinite_run_lex [V0] Step\n"
        f"  · intro W hW s\n"
        f"    simp only [List.mem_cons, List.not_mem_nil, or_false] at hW\n"
        f"    rcases hW with rfl\n"
        f"    · exact V_nonneg s fzero\n"
        f"  · rintro a b ({pat})\n"
        f"{bullets}")


def _emit_composition(path_names, n: int, init_lean: str, trivial_inv: bool) -> str:
    """The whole-program theorem: from any ``Init`` state there is no infinite run
    of ``RawStep`` (the guard and body, no invariant). The proof derives
    ``invariants (f i)`` along the run by induction — ``initiation`` at the entry
    state, the taken path's ``consecution`` at each step — which upgrades every
    ``RawStep`` to a ``Step`` and contradicts :func:`no_inf_step`."""
    rawstep = " ∨ ".join(f"{p}.RawStep a b" for p in path_names)

    if len(path_names) == 1:
        p = path_names[0]
        cons_case = (f"      obtain ⟨hg, hp⟩ := hstep k\n"
                     f"      rw [← hp]\n"
                     f"      exact {p}.consecution (f k) hg ih")
        step_run = (f"  obtain ⟨hg, hp⟩ := hstep i\n"
                    f"  exact ⟨hg, hInv i, hp⟩")
    else:
        rc = " | ".join("h" for _ in path_names)
        cons_case = (f"      rcases hstep k with {rc}\n" + "\n".join(
            f"      · obtain ⟨hg, hp⟩ := h\n"
            f"        rw [← hp]\n"
            f"        exact {p}.consecution (f k) hg ih" for p in path_names))
        step_bul = []
        for i, p in enumerate(path_names):
            inj = "Or.inr (" * i + ("Or.inl " if i < len(path_names) - 1 else "")
            step_bul.append(f"  · obtain ⟨hg, hp⟩ := h\n"
                            f"    exact {inj}⟨hg, hInv i, hp⟩{')' * i}")
        step_run = f"  rcases hstep i with {rc}\n" + "\n".join(step_bul)

    return (
        f"{_v0_and_step(path_names, n)}\n\n"
        f"def Init (s : Vector {n} Int) : Prop :=\n  {init_lean}\n\n"
        f"/-- Initiation: the loop is entered in an invariant-satisfying state. -/\n"
        f"theorem initiation (s : Vector {n} Int) (h : Init s) : invariants s := by\n"
        f"{_inv_proof(trivial_inv, 'Init, invariants')}\n\n"
        f"/-- One iteration of the loop on any path: the guard and the body. -/\n"
        f"def RawStep (a b : Vector {n} Int) : Prop := {rawstep}\n\n"
        f"theorem no_inf_step :\n"
        f"    ¬ ∃ f : Nat → Vector {n} Int, ∀ m, Step (f m) (f (m + 1)) := by\n"
        f"{_no_inf_run_body(path_names)}\n\n"
        f"/-- The program terminates: from any loop-entry state there is no\n"
        f"    infinite run of guarded steps. -/\n"
        f"theorem program_terminates (s0 : Vector {n} Int) (hinit : Init s0) :\n"
        f"    ¬ ∃ f : Nat → Vector {n} Int, f 0 = s0 ∧ ∀ i, RawStep (f i) (f (i + 1)) := by\n"
        f"  rintro ⟨f, hf0, hstep⟩\n"
        f"  have hInv : ∀ i, invariants (f i) := by\n"
        f"    intro i\n"
        f"    induction i with\n"
        f"    | zero => rw [hf0]; exact initiation s0 hinit\n"
        f"    | succ k ih =>\n"
        f"{cons_case}\n"
        f"  apply no_inf_step\n"
        f"  refine ⟨f, fun i => ?_⟩\n"
        f"{step_run}"
    )


_HEADER = (
    "import Coverage\nimport Net\nimport Termination\n"
    "set_option linter.unusedVariables false\n"
    "set_option linter.unusedSimpArgs false\n"
    "set_option maxHeartbeats 1000000\n"
    "namespace Matrix\nopen Fin\n"
)
_FOOTER = "\nend Matrix\n"


def emit_program(name: str, ob, paths) -> str:
    """The whole ``program.lean`` proving ``name`` terminates, from the per-path
    Farkas certificates (:class:`._farkas.PathCert`) captured on ``ob``. Emits the
    network, the invariant, one namespace per affine path (``loop0_path{i}``), and
    the composition that discharges the invariant and concludes termination."""
    if not paths:
        raise ValueError(f"{name}: no certified paths to emit")
    n = len(ob.s_syms)
    path_names = [f"loop0_path{i}" for i in range(len(paths))]
    cols = ", ".join(f"s {j} = {nm}" for j, nm in enumerate(ob.state))
    npaths = f" ({len(paths)} paths)" if len(paths) > 1 else ""
    # Shared top-level invariant, proved inductive below by `initiation` and the
    # per-path `consecution`; `True` when none were inferred.
    inv_lean = (_render_conjuncts(z3.And(*ob.invariants), ob.s_syms)
                if ob.invariants else "True")
    trivial_inv = inv_lean == "True"
    init_lean = ("True" if ob.init is None
                 else _render_conjuncts(ob.init, ob.s_syms))
    parts = [
        f"/- ──── program: {name} — terminates via a ranking function{npaths}.\n"
        f"   Columns: {cols}. ──── -/",
        _emit_network(ob.layers),
        _emit_nonneg(ob.layers),
        f"def invariants (s : Vector {n} Int) : Prop :=\n  {inv_lean}",
    ]
    for pname, pcert in zip(path_names, paths):
        parts.append(_emit_path(pname, pcert, ob.s_syms, trivial_inv,
                                ob.invariants))
    parts.append(_emit_composition(path_names, n, init_lean, trivial_inv))
    return _HEADER + "\n\n".join(parts) + _FOOTER


def write_program_proof(name: str, ob, paths, out_dir: Path) -> Path:
    """Write ``<out_dir>/<name>/program.lean`` and return its path."""
    target = Path(out_dir) / name
    target.mkdir(parents=True, exist_ok=True)
    out = target / "program.lean"
    out.write_text(emit_program(name, ob, paths))
    return out
