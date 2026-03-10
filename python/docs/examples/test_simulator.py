"""Minimal reactive module simulator — companion to docs/simulator-tutorial.md.

This file is a pytest-discoverable test that keeps the tutorial examples
up-to-date. Run with: just pytest docs/examples
"""
import torch
from zrth import Wire, Term, Module, Int, Bool, IType as it


# ---------------------------------------------------------------------------
# A small evaluator — maps IType variants to tensor operations
# ---------------------------------------------------------------------------

def evaluate(itype, inputs):
    """Evaluate a single instruction on its input tensors, return output list."""
    dispatch = {
        type(it.Tensor(torch.zeros(1))): lambda: [itype._0.clone()],
        type(it.Id()):       lambda: [inputs[0].clone()],
        type(it.Add()):      lambda: [inputs[0] + inputs[1]],
        type(it.Sub()):      lambda: [inputs[0] - inputs[1]],
        type(it.Mul()):      lambda: [inputs[0] * inputs[1]],
        type(it.Not()):      lambda: [inputs[0].logical_not()],
        type(it.And()):      lambda: [inputs[0].logical_and(inputs[1])],
        type(it.Or()):       lambda: [inputs[0].logical_or(inputs[1])],
        type(it.Ite()):      lambda: [torch.where(inputs[0], inputs[1], inputs[2])],
        type(it.Eq()):       lambda: [inputs[0].eq(inputs[1])],
        type(it.Lt()):       lambda: [inputs[0].lt(inputs[1])],
        type(it.ReLU()):     lambda: [inputs[0].relu()],
        type(it.ConstBool(False)): lambda: [torch.tensor([itype._0], dtype=torch.bool)],
        type(it.ConstInt(0)):  lambda: [torch.tensor([itype._0], dtype=torch.long)],
    }

    fn = dispatch.get(type(itype))
    if fn is None:
        raise RuntimeError(f"unsupported instruction: {type(itype).__name__}")
    return fn()


# ---------------------------------------------------------------------------
# Helper: build a counter module
# ---------------------------------------------------------------------------

def make_counter():
    """Counter module: x starts at 0, incremented by 1 each step."""
    x = (Wire(Int()), Wire(Int()))

    init = [Term(it.Tensor(torch.tensor([0], dtype=torch.int64)), [x[1]])]

    one = Wire(Int())
    update = [
        Term(it.Tensor(torch.tensor([1], dtype=torch.int64)), [one]),
        Term(it.Add(), [x[1]], [x[0], one]),
    ]

    m = Module.sequential(init, update, [x])
    return m, x


# ---------------------------------------------------------------------------
# The simulation loop
# ---------------------------------------------------------------------------

def simulate_init(module):
    """Run the init block and latch — returns the initial state dict."""
    state = {}
    for term in (t for atom in module.atoms for t in atom.init):
        inputs = [state[w] for w in term.read]
        outputs = evaluate(term.itype, inputs)
        state.update(zip(term.write, outputs))

    # Latch: copy next-wire values into latched-wire slots
    state = {ltc: state[nxt] for (ltc, nxt) in module.ctrl}
    return state


def simulate_step(module, state):
    """Run one update step and latch — mutates and returns state."""
    for term in (t for atom in module.atoms for t in atom.update):
        inputs = [state[w] for w in term.read]
        outputs = evaluate(term.itype, inputs)
        state.update(zip(term.write, outputs))

    # Latch
    for ltc, nxt in module.ctrl:
        if nxt in state:
            state[ltc] = state[nxt].clone()
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_counter():
    m, x = make_counter()
    assert m.closed()

    state = simulate_init(m)
    assert int(state[x[0]].item()) == 0

    state = simulate_step(m, state)
    assert int(state[x[0]].item()) == 1

    state = simulate_step(m, state)
    assert int(state[x[0]].item()) == 2


def test_multi_step():
    m, x = make_counter()

    state = simulate_init(m)
    for _ in range(10):
        state = simulate_step(m, state)

    assert int(state[x[0]].item()) == 10


def test_boolean_toggle():
    """A boolean that flips every step: init=True, update=NOT(prev)."""
    flag = (Wire(Bool()), Wire(Bool()))

    init = [Term(it.Tensor(torch.tensor([True])), [flag[1]])]
    update = [Term(it.Not(), [flag[1]], [flag[0]])]

    m = Module.sequential(init, update, [flag])

    state = simulate_init(m)
    assert bool(state[flag[0]].item()) is True

    state = simulate_step(m, state)
    assert bool(state[flag[0]].item()) is False

    state = simulate_step(m, state)
    assert bool(state[flag[0]].item()) is True
