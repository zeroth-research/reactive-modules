"""Argmax semantics must agree across the three implementations.

`Argmax` exists three times over: `torch.argmax` in the evaluator (the
runtime reference), `_argmax_1d` in the SMT encoder (what CEGAR reasons
about), and `argmax_1d` in Core/Mat.lean (what the certificate states). If
they disagree, CEGAR can prove an invariant that is false of the system the
Lean proof is about — so these tests pin all three to torch.

The historical bug: the Lean fold seeded its accumulator with
`(0, default)` and updated on a non-strict `<=`, which made ties resolve to
the *last* index and made every all-negative row report index 0. The
generated scalar variant mirrored the same fold, so `argmax1d_scalar_n_eq`
stayed provable while both sides were wrong.
"""

import cvc5
import pytest
import torch

from zrth import Int
from zrth.lean.smt_encode import _argmax_1d, MatShape, elem_sort


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


def _torch_argmax(row):
    return int(torch.argmax(torch.tensor(row, dtype=torch.int64)).item())


def _smt_argmax(row):
    """Evaluate the SMT encoder's argmax on a concrete row."""
    tm = cvc5.TermManager()
    solver = cvc5.Solver(tm)
    solver.setOption("produce-models", "true")
    shape = MatShape(1, len(row))
    elem = elem_sort(tm, Int([1, len(row)]))
    tup = tm.mkTupleSort(*([elem] * len(row)))
    const = tm.mkConst(tup, "x")
    solver.assertFormula(
        tm.mkTerm(
            cvc5.Kind.EQUAL,
            const,
            tm.mkTuple([tm.mkInteger(v) for v in row]),
        )
    )
    idx = _argmax_1d(tm, const, shape)
    assert solver.checkSat().isSat()
    return int(str(solver.getValue(idx)))


@pytest.mark.parametrize("row", _ROWS, ids=lambda r: ",".join(map(str, r)))
def test_smt_argmax_matches_torch(row):
    assert _smt_argmax(row) == _torch_argmax(row)


def test_torch_reference_conventions():
    """Spell out the two conventions the other two implementations must share."""
    assert _torch_argmax([3, 3]) == 0, "ties resolve to the first index"
    assert _torch_argmax([-5, -3]) == 1, "a neutral 0 is not a candidate"
