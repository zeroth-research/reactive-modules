"""Tests for `zrth.builder` (the term builder used by the gym / SMV / analyzer front-ends)."""

import pytest
import torch

from zrth import LRA, LIA, Real, Int, Wire
from zrth.builder import builder_for
from zrth.eval import eval_itype


@pytest.mark.parametrize("theory,sort,dtype", [
    (LRA, Real, torch.float32),
    (LIA, Int, torch.int64),
])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_mul_scales_a_column_vector(theory, sort, dtype, n):
    builder = builder_for(theory)
    x = Wire(sort([n, 1]))
    const = builder.const(torch.tensor([[3]], dtype=dtype))

    term = builder.mul(const, x)

    vals = torch.arange(1, n + 1, dtype=dtype).reshape(n, 1)
    out = eval_itype(term.itype, [vals], term.write[0].dtype)[0]
    assert out.flatten().tolist() == [3 * v for v in range(1, n + 1)]
