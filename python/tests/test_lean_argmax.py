"""Argmax semantics must agree across the three implementations.

`Argmax` exists three times over: `torch.argmax` in the evaluator (the
runtime reference), `_argmax_flat` in the SMT encoder (what CEGAR reasons
about), and `argmax_1d` / `argmax` in Core/Mat.lean (what the certificate
states). If they disagree, CEGAR can prove an invariant that is false of the
system the Lean proof is about — so these tests pin all three to torch.

Two historical bugs, one per Lean definition:

* `argmax_1d` seeded its fold with `(0, default)` and updated on a
  non-strict `<=`, so ties resolved to the *last* index and every
  all-negative row reported index 0. The generated scalar variant mirrored
  the same fold, so `argmax1d_scalar_n_eq` stayed provable while both sides
  were wrong.
* `argmax` had the same mis-seeding *and* returned the wrong kind of answer:
  the `[i, j]` pair of the maximum packed as `Mat Nat 1 2`, where torch
  flattens row-major and returns the single index `i * n + j`. The SMT
  encoder had no 2-D path at all — it rejected `m != 1` outright.
"""

import cvc5
import pytest
import torch

from zrth import Int, Term, Module, LIA, Var, X
from zrth.lean import ModuleToLean4
from zrth.lean.smt_encode import _argmax_flat, MatShape, elem_sort


# (row, expected) — expected is torch's answer, computed below to keep the
# reference explicit rather than hand-copied.
_ROWS = [
    [1, 5, 3],           # plain maximum, interior
    [5, 1, 3],           # maximum first
    [1, 3, 5],           # maximum last
    [3, 3],              # tie -> first index
    [3, 3, 3],           # all tied -> first index
    [-5, -3],            # all negative: 0 must not be a candidate
    [-5, -3, -9],        # all negative, maximum interior
    [-1],                # single negative element
    [0, -1],             # zero present, still first-index-wins
    [-1, 0, 0],          # tie at the maximum, both non-first
]


def _torch_argmax(rows):
    """torch's own answer: a single index into the row-major flattening."""
    return int(torch.argmax(torch.tensor(rows, dtype=torch.int64)).item())


def _smt_argmax(rows):
    """Evaluate the SMT encoder's argmax on a concrete m x n matrix."""
    flat = [v for row in rows for v in row]
    m, n = len(rows), len(rows[0])
    tm = cvc5.TermManager()
    solver = cvc5.Solver(tm)
    solver.setOption("produce-models", "true")
    shape = MatShape(m, n)
    elem = elem_sort(tm, Int([m, n]))
    tup = tm.mkTupleSort(*([elem] * len(flat)))
    const = tm.mkConst(tup, "x")
    solver.assertFormula(
        tm.mkTerm(
            cvc5.Kind.EQUAL,
            const,
            tm.mkTuple([tm.mkInteger(v) for v in flat]),
        )
    )
    idx = _argmax_flat(tm, const, shape)
    assert solver.checkSat().isSat()
    return int(str(solver.getValue(idx)))


# 2-D matrices, as row lists. torch flattens row-major, so the expected
# answer is a single index into the concatenation of these rows.
_MATRICES = [
    [[1, 9, 3], [4, 5, 6]],        # maximum in the first row
    [[1, 2, 3], [4, 5, 9]],        # maximum in the last position
    [[9, 2, 3], [4, 5, 6]],        # maximum first
    [[1, 2, 3], [4, 5, 6]],        # increasing: maximum last
    [[9, 9, 3], [4, 5, 6]],        # tie within a row
    [[1, 2, 3], [9, 5, 9]],        # tie within the second row
    [[5, 5], [5, 5]],              # everything tied -> flat index 0
    [[-9, -2, -3], [-4, -5, -6]],  # all negative
    [[-9, -8, -7], [-6, -5, -4]],  # all negative, maximum last
    [[3], [1], [2]],               # single column
    [[7]],                         # 1x1
    [[0, -1], [-2, -3]],           # zero is the genuine maximum
]


@pytest.mark.parametrize("row", _ROWS, ids=lambda r: ",".join(map(str, r)))
def test_smt_argmax_matches_torch_1d(row):
    assert _smt_argmax([row]) == _torch_argmax([row])


@pytest.mark.parametrize("rows", _MATRICES, ids=lambda m: ";".join(
    ",".join(map(str, r)) for r in m))
def test_smt_argmax_matches_torch_2d(rows):
    assert _smt_argmax(rows) == _torch_argmax(rows)


@pytest.mark.parametrize("row", _ROWS, ids=lambda r: ",".join(map(str, r)))
def test_single_row_flat_index_is_the_column_index(row):
    """What makes `argmax` and `argmax_1d` interchangeable on one row."""
    assert _smt_argmax([row]) < len(row)


def test_torch_reference_conventions():
    """Spell out the conventions all three implementations must share."""
    assert _torch_argmax([[3, 3]]) == 0, "ties resolve to the first index"
    assert _torch_argmax([[-5, -3]]) == 1, "a neutral 0 is not a candidate"
    assert _torch_argmax([[1, 9], [3, 4]]) == 1, "2-D returns one row-major flat index"
    assert _torch_argmax([[1, 2], [9, 4]]) == 2, "flat index crosses row boundaries"


# ──────────────────────────────────────────────────────────────
# Emission: one flat index out, in every encoding
# ──────────────────────────────────────────────────────────────

_ENCODINGS = ["to_lean_functional", "to_lean_circ", "to_lean_scalar"]


def _argmax_module(in_shape, out_shape):
    """Sequential module whose update takes the Argmax of its own state."""
    s = Var(Int(in_shape))
    out = Var(Int(out_shape))
    init = [
        Term(LIA.Int(torch.zeros(*in_shape, dtype=torch.int64)), [X(s)]),
        Term(LIA.Int(torch.zeros(*out_shape, dtype=torch.int64)), [X(out)]),
    ]
    update = [
        Term(LIA.Id(), [X(s)], [s]),
        Term(LIA.Argmax(), [X(out)], [s]),
    ]
    return Module.sequential([s, out], init, update)


@pytest.mark.parametrize("encoding", _ENCODINGS)
def test_argmax_emits_for_multi_row_input(encoding):
    """A genuinely 2-D input reaches Lean as the general `argmax`."""
    lean = getattr(ModuleToLean4(_argmax_module([3, 4], [1, 1])), encoding)()
    assert "argmax" in lean


def test_multi_row_input_selects_the_general_variant():
    """1-row input uses argmax_1d; more rows use argmax (a `Mat _ 1 1` either way)."""
    one_row = ModuleToLean4(_argmax_module([1, 4], [1, 1])).to_lean_functional()
    many_row = ModuleToLean4(_argmax_module([3, 4], [1, 1])).to_lean_functional()
    assert "argmax_1d" in one_row
    assert "(Mat Int 1 1) := (fun i j => ((argmax ctrl" in many_row, (
        "multi-row Argmax should use the general variant with a 1x1 result"
    )


@pytest.mark.parametrize("out_shape", [[1, 4], [4, 1], [3, 4]])
def test_theory_rejects_wider_argmax_output(out_shape):
    """Argmax yields one index, so the theory admits only a [1, 1] output.

    A vector output would have to mean per-column argmax, which nothing
    implements. This used to be accepted and then emitted a `Mat _ 1 1` body
    ascribed to the wider wire, which does not elaborate.
    """
    s = Var(Int([3, 4]))
    out = Var(Int(out_shape))
    with pytest.raises(Exception, match="exactly one index|vector|matrix"):
        Term(LIA.Argmax(), [X(out)], [s])


def test_codegen_guard_still_states_the_one_index_contract():
    """Defence in depth behind the theory check, which now catches this first.

    Kept so the contract is stated where the Lean is emitted: both variants
    produce a `Mat _ 1 1`, so a wider wire cannot be served.
    """
    from zrth.lean.native import _check_argmax_output

    _check_argmax_output([1, 1])  # the only shape argmax can fill
    _check_argmax_output(None)  # opt out
    for bad in ([1, 4], [4, 1], [3, 4]):
        with pytest.raises(ValueError, match="single flat index"):
            _check_argmax_output(bad)


def test_scalar_argmax_variants_are_named_per_element_type():
    """Variants are collected per (elem_ty, n), so the name needs both.

    Naming by `n` alone emitted two `def argmax1d_scalar_4` and two `_eq`
    theorems into one file for an Int and a Real Argmax of equal width.
    """
    from zrth.lean.native import _argmax_scalar_name

    names = {
        _argmax_scalar_name(ety, 4)
        for ety in ("Int", "Real", "Bool", "(BitVec 8)")
    }
    assert len(names) == 4, f"variant names collide: {sorted(names)}"
    assert _argmax_scalar_name("Int", 4) != _argmax_scalar_name("Real", 4)
    # widths stay distinguished too
    assert _argmax_scalar_name("Int", 4) != _argmax_scalar_name("Int", 8)
    # and the names are Lean identifiers
    assert all(n.replace("_", "").isalnum() for n in names), sorted(names)


def test_same_width_variants_emit_distinct_definitions():
    """Two element types at one width must not produce one name twice."""
    import re
    from zrth.lean.translate.scalar import _argmax_scalar_def_lines

    src = "\n".join(
        _argmax_scalar_def_lines("Int", 4) + _argmax_scalar_def_lines("Real", 4)
    )
    defs = re.findall(r"^(?:noncomputable )?def (\w+)", src, re.M)
    assert len(defs) == len(set(defs)), f"duplicate definitions: {defs}"
    theorems = re.findall(r"^@\[simp\] theorem (\w+)", src, re.M)
    assert len(theorems) == len(set(theorems)), f"duplicate theorems: {theorems}"
