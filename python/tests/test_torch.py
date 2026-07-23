"""Tests for zrth.torch.Module: theory-parameterized interface sorts (LRA/LIA/BV)
and weight-dtype validation."""

import pytest
import torch
import torch.nn as nn

from zrth import LRA, LIA, BV, Sort
from zrth.torch import Module


class Net(nn.Module):
    """Linear -> ReLU -> Linear (obs_size=2, qval=1)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _to_int_weights(net, dtype=torch.int64):
    """Replace a net's Linear weights/biases with integer tensors (values rounded)."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight = nn.Parameter(m.weight.detach().round().to(dtype), requires_grad=False)
            m.bias = nn.Parameter(m.bias.detach().round().to(dtype), requires_grad=False)
    return net


# --- interface sorts follow the theory ---------------------------------------


def _sorts(m):
    return list(m.extl)[0][0].dtype, list(m.intf)[0][0].dtype


def test_default_theory_is_real():
    e, i = _sorts(Module(Net()))
    assert isinstance(e, Sort.Real) and isinstance(i, Sort.Real)


def test_lra_is_real():
    e, i = _sorts(Module(Net(), theory=LRA))
    assert isinstance(e, Sort.Real) and isinstance(i, Sort.Real)


def test_lia_is_int():
    e, i = _sorts(Module(_to_int_weights(Net()), theory=LIA))
    assert isinstance(e, Sort.Int) and isinstance(i, Sort.Int)


def test_bv_unsupported_for_nn():
    # BV has no Transpose/Linear ops, so a neural module can't compile to it (yet).
    with pytest.raises(NotImplementedError, match="does not support neural modules"):
        Module(_to_int_weights(Net()), theory=BV)


# --- weight dtype must match the theory (used as-is, no coercion) ------------


def test_float_weights_under_lia_raise():
    with pytest.raises(TypeError, match="integer"):
        Module(Net(), theory=LIA)


def test_int_weights_under_lra_raise():
    with pytest.raises(TypeError, match="floating-point"):
        Module(_to_int_weights(Net()), theory=LRA)


# --- combinatorial (V(s')) vs sequential (V(s)) ------------------------------


def _read_ids(m):
    return {w.id for w in m.atoms[0].read}


def _wait_ids(m):
    return {w.id for w in m.atoms[0].wait}


def test_default_is_combinatorial_reads_next_only():
    # V(s'): memoryless — does not read the latched input, only awaits the next one.
    m = Module(Net())
    latched, nxt = list(m.extl)[0]
    assert latched.id not in _read_ids(m)
    assert nxt.id in _wait_ids(m)


def test_sequential_reads_latched_input():
    # V(s): sequential atom reads the latched input (and still awaits next for init).
    m = Module(Net(), sequential=True)
    latched, nxt = list(m.extl)[0]
    assert latched.id in _read_ids(m)
    assert nxt.id in _wait_ids(m)


def test_sequential_with_lia_builds():
    m = Module(_to_int_weights(Net()), theory=LIA, sequential=True)
    latched, _ = list(m.extl)[0]
    assert isinstance(latched.dtype, Sort.Int)
    assert latched.id in _read_ids(m)
