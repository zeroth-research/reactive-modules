"""Low-level interpreter test for the 2-bit digital circuit counter.

Circuit equations:
    b0_next = b0 XOR enable         = Ite(enable, NOT b0, b0)
    b1_next = b1 XOR (b0 AND enable) = Ite(b0 AND enable, NOT b1, b1)

Counting sequence (enable=1 each step):
    (b1, b0): (F,F)=0 -> (F,T)=1 -> (T,F)=2 -> (T,T)=3 -> (F,F)=0 -> ...
"""
import torch
from zrth import Wire, Term, Module, DType as dt, IType as it
from zrth.examples import Interpreter


def _make_twobitcounter():
    """Build the 2-bit counter module.

    Returns (module, b0_wires, b1_wires, enable_wires).
    enable is an external input: pass via step({enable[1].id(): tensor}).
    """
    b0 = (Wire(dt.Bool()), Wire(dt.Bool()))  # (current, next)
    b1 = (Wire(dt.Bool()), Wire(dt.Bool()))
    enable = (Wire(dt.Bool()), Wire(dt.Bool()))

    # init: b0=False, b1=False
    init = [
        Term(it.Tensor(torch.tensor([False])), [b0[1]]),
        Term(it.Tensor(torch.tensor([False])), [b1[1]]),
    ]

    # Intermediate wires
    not_b0 = Wire(dt.Bool())
    not_b1 = Wire(dt.Bool())
    b0_and_enable = Wire(dt.Bool())

    update = [
        # not_b0 = NOT b0
        Term(it.Not(), [not_b0], [b0[0]]),
        # b0' = Ite(enable, NOT b0, b0)  ->  b0 XOR enable
        Term(it.Ite(), [b0[1]], [enable[1], not_b0, b0[0]]),
        # b0_and_enable = b0 AND enable  (carry for b1)
        Term(it.And(), [b0_and_enable], [b0[0], enable[1]]),
        # not_b1 = NOT b1
        Term(it.Not(), [not_b1], [b1[0]]),
        # b1' = Ite(b0 AND enable, NOT b1, b1)  ->  b1 XOR (b0 AND enable)
        Term(it.Ite(), [b1[1]], [b0_and_enable, not_b1, b1[0]]),
    ]

    m = Module.sequential(init, update, obs=[b0, b1, enable])
    return m, b0, b1, enable


def _enable(wire):
    return {wire[1].id(): torch.tensor([True])}


def _hold(wire):
    return {wire[1].id(): torch.tensor([False])}


def _bits(interp, b0, b1):
    """Return (b1_val, b0_val) as booleans — (MSB, LSB)."""
    return (
        bool(interp.get(b1[0].id()).item()),
        bool(interp.get(b0[0].id()).item()),
    )


def test_initial_state():
    """After init, counter is at (b1=F, b0=F) = 0."""
    m, b0, b1, enable = _make_twobitcounter()
    interp = Interpreter(m)
    interp.initialize()

    assert _bits(interp, b0, b1) == (False, False)


def test_count_sequence():
    """Stepping with enable=True cycles 00->01->10->11->00."""
    m, b0, b1, enable = _make_twobitcounter()
    interp = Interpreter(m)
    interp.initialize()

    expected = [
        (False, False),  # 0: initial state
        (False, True),   # 1: after step 1
        (True,  False),  # 2: after step 2
        (True,  True),   # 3: after step 3
        (False, False),  # 0: after step 4 (wraps)
    ]

    assert _bits(interp, b0, b1) == expected[0]

    for step_num, exp in enumerate(expected[1:], 1):
        interp.step(_enable(enable))
        assert _bits(interp, b0, b1) == exp, f"step {step_num}: expected {exp}"


def test_hold():
    """Stepping with enable=False leaves state unchanged."""
    m, b0, b1, enable = _make_twobitcounter()
    interp = Interpreter(m)
    interp.initialize()

    # Advance to state 2 (b1=T, b0=F)
    interp.step(_enable(enable))
    interp.step(_enable(enable))
    assert _bits(interp, b0, b1) == (True, False)

    # Hold for 3 steps — state must not change
    for _ in range(3):
        interp.step(_hold(enable))
    assert _bits(interp, b0, b1) == (True, False)


def test_mixed_enable_and_hold():
    """Interleaved enable/hold steps produce correct bit sequence."""
    m, b0, b1, enable = _make_twobitcounter()
    interp = Interpreter(m)
    interp.initialize()

    interp.step(_enable(enable))   # 0 -> 1
    assert _bits(interp, b0, b1) == (False, True)

    interp.step(_hold(enable))     # hold at 1
    assert _bits(interp, b0, b1) == (False, True)

    interp.step(_enable(enable))   # 1 -> 2
    assert _bits(interp, b0, b1) == (True, False)

    interp.step(_hold(enable))     # hold at 2
    interp.step(_hold(enable))     # hold at 2
    assert _bits(interp, b0, b1) == (True, False)

    interp.step(_enable(enable))   # 2 -> 3
    assert _bits(interp, b0, b1) == (True, True)

    interp.step(_enable(enable))   # 3 -> 0 (wrap)
    assert _bits(interp, b0, b1) == (False, False)
