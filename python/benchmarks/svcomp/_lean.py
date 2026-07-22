"""Emit a kernel-checkable Lean termination proof from Farkas certificates.

Consumes the :class:`._farkas.CellCert` set a ``farkas_cell`` run captures (plus
the :class:`._verify_ranking.Obligation` it was run on) and writes ONE Lean file
proving the program terminates, against the vendored ``lean/`` substrate
(``Coverage``/``Net``/``Termination``). The file contains:

  * the integer ranking network ``V`` and its structural ``V_nonneg``;
  * per cell: the Farkas system ``A``/``b``/``y``, the three side-condition
    goals, the ``farkas_sound`` infeasibility, and the scalar ``decrease``
    entailment (via the substrate's ``decrease_bridge`` tactic);
  * ``trans``/``invariants`` (the loop guard, and the trusted Houdini
    invariants), the per-cell sign regions, and ``covered`` (the cells tile the
    guard — the CEGAR coverage guarantee, discharged by ``omega``);
  * the V-collapse lemmas (``V`` equals its affine piece on each cell) and, from
    them, ``lex_step`` (``V`` strictly drops on every guarded step);
  * ``program_terminates`` — no infinite run — via ``no_infinite_run_lex``.

Scope (v1): scalar single loop, single affine path, single hidden layer. Unlike
nuTerm's relational SSA encoding (packed ``(pre ‖ post)`` state, ``pre_slice`` /
``post_slice`` projections), our reactive-module transition is *functional*
(``s' = body(s)``), so the proof is over the pre-state columns with
``post_state`` an affine map. Invariants are emitted as trusted hypotheses (their
initiation/consecution is checked Python-side, not re-proven here).

The emitter is pure formatting: every number comes verbatim from the captured
certificate or the network weights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import z3

from ._farkas import CellCert, _flatten_and, _on_guard_body, affine_coeffs


class UnsupportedProgram(Exception):
    """The program is outside the emitter's v1 scope (scalar single loop, single
    affine path) — e.g. an in-loop branch makes the transition non-affine. We
    refuse to emit rather than silently linearise it into an unsound proof."""


def _contains_ite(e) -> bool:
    """True if a z3 term still has an ``ite`` — for a next-state expression that
    means in-loop branching (the guard ``ite`` has already been peeled off)."""
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

def _signs_conj(cert: CellCert) -> str:
    rows = [f"({_affine_str(a, 0)} ≤ {int(b)})" for a, b in _sign_rows(cert)]
    return " ∧ ".join(rows) if rows else "True"


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


def _emit_signs_def(idx: int, cert: CellCert, n: int) -> str:
    return (f"def cell{idx}_signs (s : Vector {n} Int) : Prop :=\n"
            f"  {_signs_conj(cert)}")


def _emit_decrease(idx: int, cert: CellCert, n: int, has_inv: bool) -> str:
    """``cellK_signs s → trans s → invariants s → decrease`` via the substrate's
    one-line ``decrease_bridge`` (it rebuilds the row system and feeds
    ``cellK_infeasible``, closed by ``omega``). The full cell signs are always
    taken as a hypothesis so every support row is available."""
    hyps = "(hg : trans s)"
    if has_inv:
        hyps += " (hinv : invariants s)"
    hyps += f" (hs : cell{idx}_signs s)"
    goal = _decrease_goal(cert)
    if _trivial(cert):
        # unconditional: goal is ``(c+1) ≤ 0`` with c < 0 — omega, no certificate.
        return (f"theorem cell{idx}_decrease (s : Vector {n} Int)\n"
                f"    {hyps} :\n    {goal} := by omega")
    unfold = ", ".join(["trans"] + (["invariants"] if has_inv else [])
                       + [f"cell{idx}_signs", f"cell{idx}_A", f"cell{idx}_b"])
    return (
        f"theorem cell{idx}_decrease (s : Vector {n} Int)\n"
        f"    {hyps} :\n"
        f"    {goal} := by\n"
        f"  decrease_bridge (cell{idx}_infeasible s) with {unfold}"
    )


def _emit_covered(certs, n: int, has_inv: bool) -> str:
    """Every guarded state lies in a certified cell that decreases. The two
    ingredients are kept strictly separate, matching the certificate architecture:

      * **coverage** — the cells' sign regions tile the guard
        (``guard ∧ inv → ⋁ᵢ cellᵢ_signs``): a purely combinatorial linear fact,
        discharged by ``omega``. This is the *only* thing ``omega`` proves here.
      * **decrease** — that ``V`` drops on the covering cell is discharged by that
        cell's ``cellᵢ_decrease``, which is backed by the Farkas certificate
        (``cellᵢ_infeasible``); ``omega`` never re-derives it.

    So the emitted proof genuinely rests on the per-cell certificates: delete them
    and coverage no longer closes."""
    hyps = "(hg : trans s)" + (" (hinv : invariants s)" if has_inv else "")
    dec_args = "s hg hinv" if has_inv else "s hg"
    disj = "\n      ∨ ".join(
        f"(cell{i}_signs s ∧ {_decrease_goal(c)})" for i, c in enumerate(certs))
    tiling = "\n      ∨ ".join(f"cell{i}_signs s" for i in range(len(certs)))
    unfold = ", ".join(
        ["trans"] + (["invariants"] if has_inv else [])
        + [f"cell{i}_signs" for i in range(len(certs))])
    tiling_block = (
        f"  have tiling : {tiling} := by\n"
        f"    simp only [{unfold}] at *\n"
        f"    omega")
    ncerts = len(certs)
    if ncerts == 1:
        # single cell: tiling *is* cell0_signs s; no case split needed.
        body = (f"{tiling_block}\n"
                f"  exact ⟨tiling, cell0_decrease {dec_args} tiling⟩")
    else:
        bullets = []
        for i in range(ncerts):
            inj = "Or.inr (" * i + ("Or.inl " if i < ncerts - 1 else "")
            close = ")" * i
            bullets.append(
                f"  · exact {inj}⟨h, cell{i}_decrease {dec_args} h⟩{close}")
        cases = " | ".join("h" for _ in range(ncerts))
        body = (f"{tiling_block}\n"
                f"  rcases tiling with {cases}\n"
                + "\n".join(bullets))
    return (
        f"theorem covered (s : Vector {n} Int)\n"
        f"    {hyps} :\n"
        f"    {disj} := by\n"
        f"{body}"
    )


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


def _emit_collapse(idx: int, side: str, input_term: str, extra_unfold: list[str],
                   affine, n: int) -> str:
    """``V (<input>) fzero = <affine piece>`` on the cell: rewrite the ReLU layer
    by the cell's mask (``reluᵥ_eq_mask``, its side goal closed from the signs),
    unfold, and ``omega``."""
    coeffs, const = affine
    pat = f"cell{idx}_pat_{side}"
    simp_tail = ", ".join([_SIMP, pat] + extra_unfold)
    return (
        f"theorem cell{idx}_{side}_c0 (s : Vector {n} Int)\n"
        f"    (hs : cell{idx}_signs s) : V ({input_term}) fzero = {_affine_str(coeffs, const)} := by\n"
        f"  simp only [cell{idx}_signs] at hs\n"
        f"  have hmask : reluᵥ (affine nrf_W0 nrf_b0 ({input_term}))\n"
        f"             = mask {pat} (affine nrf_W0 nrf_b0 ({input_term})) := by\n"
        f"    apply reluᵥ_eq_mask\n"
        f"    simp only [{simp_tail}] <;> omega\n"
        f"  simp only [V]\n"
        f"  rw [hmask]\n"
        f"  simp only [{simp_tail}] <;> omega"
    )


def _emit_path(path: str, certs, layers, guard, invariants, s_syms, sp_syms) -> str:
    n = len(s_syms)
    trans_lean = _render_conjuncts(guard, s_syms)
    inv_lean = _render_conjuncts(z3.And(*invariants) if invariants else z3.BoolVal(True),
                                 s_syms)
    has_inv = inv_lean not in ("", "True")
    body = _on_guard_body(sp_syms, s_syms)
    if any(_contains_ite(e) for e in body):
        raise UnsupportedProgram(
            "in-loop branching: the next-state is not a single affine map")
    body_affines = [affine_coeffs(e, s_syms) for e in body]

    parts = []
    for idx, cert in enumerate(certs):
        parts.append(_emit_cell(idx, cert))
    parts.append(f"def trans (s : Vector {n} Int) : Prop :=\n  {trans_lean}")
    if has_inv:
        parts.append(f"def invariants (s : Vector {n} Int) : Prop :=\n  {inv_lean}")
    for idx, cert in enumerate(certs):
        parts.append(_emit_signs_def(idx, cert, n))
    for idx, cert in enumerate(certs):
        parts.append(_emit_decrease(idx, cert, n, has_inv))
    parts.append(_emit_covered(certs, n, has_inv))
    parts.append(_emit_post_state(body_affines, n))
    for idx, cert in enumerate(certs):
        parts.append(_bool_vec(f"cell{idx}_pat_pre", cert.pattern_s))
        parts.append(_bool_vec(f"cell{idx}_pat_post", cert.pattern_sp))
        parts.append(_emit_collapse(idx, "pre", "s", [], cert.pre_affine, n))
        parts.append(_emit_collapse(idx, "post", "post_state s", ["post_state"],
                                    cert.post_affine, n))

    # The transition is functional (b = post_state a on the guard), so Step needs
    # no SSA witness: the pre-state *is* a.
    inv_conj = " ∧ invariants a" if has_inv else ""
    obtain = "⟨hg, hinv, hpost⟩" if has_inv else "⟨hg, hpost⟩"
    cov_args = "a hg hinv" if has_inv else "a hg"
    branches = " | ".join("⟨hs, hd⟩" for _ in certs)
    bullets = "\n".join(
        f"  · rw [cell{idx}_pre_c0 a hs, cell{idx}_post_c0 a hs]; omega"
        for idx in range(len(certs)))
    parts.append(
        f"def Step (a b : Vector {n} Int) : Prop :=\n"
        f"  trans a{inv_conj} ∧ post_state a = b")
    parts.append(
        f"/-- lex step of this path (strict component 0). -/\n"
        f"theorem lex_step (a b : Vector {n} Int) (h : Step a b) :\n"
        f"    V b fzero < V a fzero := by\n"
        f"  obtain {obtain} := h\n"
        f"  subst hpost\n"
        f"  rcases covered {cov_args} with {branches}\n"
        f"{bullets}")
    return f"namespace {path}\n\n" + "\n\n".join(parts) + f"\n\nend {path}"


# ---------------------------------------------------------------------------
# Composition: program_terminates via no_infinite_run_lex
# ---------------------------------------------------------------------------

def _emit_composition(path: str, n: int) -> str:
    return (
        f"def V0 : Vector {n} Int → Int := fun s => V s fzero\n\n"
        f"/-- The program's step relation: one iteration of the loop. -/\n"
        f"def Step (a b : Vector {n} Int) : Prop := {path}.Step a b\n\n"
        f"/-- The program terminates: no infinite sequence of loop iterations. -/\n"
        f"theorem program_terminates :\n"
        f"    ¬ ∃ f : Nat → Vector {n} Int, ∀ n, Step (f n) (f (n + 1)) := by\n"
        f"  apply no_infinite_run_lex [V0] Step\n"
        f"  · intro W hW s\n"
        f"    simp only [List.mem_cons, List.not_mem_nil, or_false] at hW\n"
        f"    rcases hW with rfl\n"
        f"    · exact V_nonneg s fzero\n"
        f"  · rintro a b (h)\n"
        f"    · simp only [lexDec, V0]\n"
        f"      have hx := {path}.lex_step a b h\n"
        f"      exact Or.inl (hx)"
    )


_HEADER = (
    "import Coverage\nimport Net\nimport Termination\n"
    "set_option linter.unusedVariables false\n"
    "set_option linter.unusedSimpArgs false\n"
    "set_option maxHeartbeats 1000000\n"
    "namespace Matrix\nopen Fin\n"
)
_FOOTER = "\nend Matrix\n"


def emit_program(name: str, ob, certs) -> str:
    """The whole ``program.lean`` proving ``name`` terminates, from the Farkas
    ``certs`` captured on the obligation ``ob`` (scalar single loop, single path)."""
    if not certs:
        raise ValueError(f"{name}: no certified cells to emit")
    n = len(ob.s_syms)
    path = "loop0_path0"
    cols = ", ".join(f"s {j} = {nm}" for j, nm in enumerate(ob.state))
    parts = [
        f"/- ──── program: {name} — terminates via a ranking function.\n"
        f"   Columns: {cols}. ──── -/",
        _emit_network(ob.layers),
        _emit_nonneg(ob.layers),
        _emit_path(path, certs, ob.layers, ob.guard, ob.invariants,
                   ob.s_syms, ob.sp_syms),
        _emit_composition(path, n),
    ]
    return _HEADER + "\n\n".join(parts) + _FOOTER


def write_program_proof(name: str, ob, certs, out_dir: Path) -> Path:
    """Write ``<out_dir>/<name>/program.lean`` and return its path."""
    target = Path(out_dir) / name
    target.mkdir(parents=True, exist_ok=True)
    out = target / "program.lean"
    out.write_text(emit_program(name, ob, certs))
    return out
