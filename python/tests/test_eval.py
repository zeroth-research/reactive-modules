"""Tests for the symbolic term evaluation engine.

Tests IType operations (Add, And, Or, Not, Ite, ReLU, etc.) and
multi-step execution of hand-built modules.
"""
import pytest
import torch
from zrth import Wire, Term, Module, DType as dt, IType as it, set_theory
from zrth.eval import eval_itype, execute_init, execute_update


@pytest.fixture(autouse=True)
def _theory():
    set_theory(it.LIA)


# ── helpers ──────────────────────────────────────────────────────────────────

def _run_module(module, n_steps, env_inputs_fn=None):
    """Run a hand-built module for n_steps, returning state after each step.

    env_inputs_fn(step) returns a dict of {wire: tensor} or None.
    """
    state = {}

    # Init
    execute_init(state, module.atoms)
    for ltc, nxt in module.ctrl:
        if nxt in state:
            state[ltc] = state[nxt].clone()

    history = [dict(state)]

    # Steps
    for step in range(n_steps):
        if env_inputs_fn:
            inputs = env_inputs_fn(step)
            if inputs:
                for wire, tensor in inputs.items():
                    state[wire] = tensor
        execute_update(state, module.atoms)
        for ltc, nxt in module.ctrl:
            if nxt in state:
                state[ltc] = state[nxt].clone()
        history.append(dict(state))

    return state, history


def _get(state, wire):
    """Read a wire value from state."""
    return state[wire]


# ── counter ──────────────────────────────────────────────────────────────────

def _make_counter():
    """Simple counter: init x=0, update x'=x+1."""
    x = (Wire(dt.Int([1])), Wire(dt.Int([1])))

    init = [Term(it.Tensor(torch.tensor([0], dtype=torch.int64)), [x[1]])]
    one = Wire(dt.Int([1]))
    update = [
        Term(it.Tensor(torch.tensor([1], dtype=torch.int64)), [one]),
        Term(it.Add(), [x[1]], [x[0], one]),
    ]
    m = Module.sequential(init, update, [x])
    return m, x


def test_counter():
    m, x = _make_counter()
    state, history = _run_module(m, 10)

    assert int(_get(history[0], x[0]).item()) == 0
    assert int(_get(history[1], x[0]).item()) == 1
    assert int(_get(history[2], x[0]).item()) == 2
    assert int(_get(history[10], x[0]).item()) == 10


def test_basic_eval_counter():
    """Raw term evaluation: run init then one update step manually."""
    m, x = _make_counter()
    assert m.closed()

    state = {}
    for t in (t for a in m.atoms for t in a.init):
        state.update(zip(t.write, eval_itype(t.itype, [state[w] for w in t.read])))

    state = {ltc: state[nxt] for (ltc, nxt) in m.ctrl}
    for t in (t for a in m.atoms for t in a.update):
        state.update(zip(t.write, eval_itype(t.itype, [state[w] for w in t.read])))

    assert state[x[1]] == 1


# ── boolean logic ────────────────────────────────────────────────────────────

def test_boolean_logic():
    """AND/OR/NOT: state wires computed from each other."""
    a = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))
    b = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))
    c = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))

    init = [
        Term(it.Tensor(torch.tensor([True])), [a[1]]),
        Term(it.Tensor(torch.tensor([False])), [b[1]]),
        Term(it.Not(), [c[1]], [a[1]]),
    ]
    update = [
        Term(it.And(), [a[1]], [a[0], b[0]]),
        Term(it.Or(), [b[1]], [a[0], b[0]]),
        Term(it.Not(), [c[1]], [c[0]]),
    ]
    m = Module.sequential(init, update, [a, b, c])
    state, history = _run_module(m, 2)

    # After init: a=True, b=False, c=not(True)=False
    assert bool(_get(history[0], a[0]).item()) is True
    assert bool(_get(history[0], b[0]).item()) is False
    assert bool(_get(history[0], c[0]).item()) is False

    # Step 1: a'=and(T,F)=F, b'=or(T,F)=T, c'=not(F)=T
    assert bool(_get(history[1], a[0]).item()) is False
    assert bool(_get(history[1], b[0]).item()) is True
    assert bool(_get(history[1], c[0]).item()) is True

    # Step 2: a'=and(F,T)=F, b'=or(F,T)=T, c'=not(T)=F
    assert bool(_get(history[2], a[0]).item()) is False
    assert bool(_get(history[2], b[0]).item()) is True
    assert bool(_get(history[2], c[0]).item()) is False


# ── ite branching ────────────────────────────────────────────────────────────

def test_ite():
    """Ite branching: cond toggles, x depends on previous x."""
    cond = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))
    x = (Wire(dt.Int([1])), Wire(dt.Int([1])))

    init = [
        Term(it.Tensor(torch.tensor([True])), [cond[1]]),
        Term(it.Tensor(torch.tensor([0], dtype=torch.int64)), [x[1]]),
    ]

    one = Wire(dt.Int([1]))
    two = Wire(dt.Int([1]))
    tmp1 = Wire(dt.Int([1]))
    tmp2 = Wire(dt.Int([1]))

    update = [
        Term(it.Not(), [cond[1]], [cond[0]]),
        Term(it.Tensor(torch.tensor([1], dtype=torch.int64)), [one]),
        Term(it.Tensor(torch.tensor([2], dtype=torch.int64)), [two]),
        Term(it.Add(), [tmp1], [x[0], one]),
        Term(it.Add(), [tmp2], [x[0], two]),
        Term(it.Ite(), [x[1]], [cond[0], tmp1, tmp2]),
    ]
    m = Module.sequential(init, update, [cond, x])
    state, history = _run_module(m, 2)

    # After init: cond=True, x=0
    assert bool(_get(history[0], cond[0]).item()) is True
    assert int(_get(history[0], x[0]).item()) == 0

    # Step 1: cond'=F, x'=ite(T, 0+1, 0+2)=1
    assert bool(_get(history[1], cond[0]).item()) is False
    assert int(_get(history[1], x[0]).item()) == 1

    # Step 2: cond'=T, x'=ite(F, 1+1, 1+2)=3
    assert bool(_get(history[2], cond[0]).item()) is True
    assert int(_get(history[2], x[0]).item()) == 3


# ── tensor ops ───────────────────────────────────────────────────────────────

def test_tensor_ops():
    """ReLU on Float vector state."""
    data = (Wire(dt.Float([4])), Wire(dt.Float([4])))

    init = [
        Term(it.Tensor(torch.tensor([-1.0, 2.0, 3.0, -4.0])), [data[1]]),
    ]
    update = [
        Term(it.ReLU(), [data[1]], [data[0]]),
    ]
    m = Module.sequential(init, update, [data])
    state, history = _run_module(m, 2)

    expected = torch.tensor([-1.0, 2.0, 3.0, -4.0])
    assert torch.equal(_get(history[0], data[0]), expected)

    # Step 1: relu([-1,2,3,-4]) = [0,2,3,0]
    assert torch.equal(_get(history[1], data[0]), expected.relu())

    # Step 2: relu([0,2,3,0]) = [0,2,3,0] (fixed point)
    assert torch.equal(_get(history[2], data[0]), expected.relu())


def test_tensor_reductions():
    """TensorSum, TensorMean, TensorMax, Argmax as temp wires."""
    data = (Wire(dt.Float([4])), Wire(dt.Float([4])))
    sum_wire = Wire(dt.Float([1]))
    mean_wire = Wire(dt.Float([1]))
    max_wire = Wire(dt.Float([1]))
    argmax_wire = Wire(dt.Int([1]))

    init = [
        Term(it.Tensor(torch.tensor([-1.0, 2.0, 3.0, -4.0])), [data[1]]),
        Term(it.TensorSum(), [sum_wire], [data[1]]),
        Term(it.TensorMean(), [mean_wire], [data[1]]),
        Term(it.TensorMax(), [max_wire], [data[1]]),
        Term(it.Argmax(), [argmax_wire], [data[1]]),
    ]
    update = [
        Term(it.Id(), [data[1]], [data[0]]),
    ]
    m = Module.sequential(init, update, [data])
    state, _ = _run_module(m, 0)

    expected = torch.tensor([-1.0, 2.0, 3.0, -4.0])
    assert float(state[sum_wire].item()) == float(expected.sum().item())
    assert float(state[mean_wire].item()) == float(expected.mean().item())
    assert float(state[max_wire].item()) == float(expected.max().item())
    assert int(state[argmax_wire].item()) == int(expected.argmax().item())


# ── comparisons ──────────────────────────────────────────────────────────────

def test_comparisons():
    """Eq and Lt comparisons over multiple steps."""
    a = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    b = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    eq_wire = Wire(dt.Bool([1]))
    lt_wire = Wire(dt.Bool([1]))

    init = [
        Term(it.Tensor(torch.tensor([3], dtype=torch.int64)), [a[1]]),
        Term(it.Tensor(torch.tensor([5], dtype=torch.int64)), [b[1]]),
        Term(it.Eq(), [eq_wire], [a[1], b[1]]),
        Term(it.Lt(), [lt_wire], [a[1], b[1]]),
    ]

    one = Wire(dt.Int([1]))
    eq_wire2 = Wire(dt.Bool([1]))
    lt_wire2 = Wire(dt.Bool([1]))
    update = [
        Term(it.Tensor(torch.tensor([1], dtype=torch.int64)), [one]),
        Term(it.Add(), [a[1]], [a[0], one]),
        Term(it.Id(), [b[1]], [b[0]]),
        Term(it.Eq(), [eq_wire2], [a[0], b[0]]),
        Term(it.Lt(), [lt_wire2], [a[0], b[0]]),
    ]
    m = Module.sequential(init, update, [a, b])
    state, history = _run_module(m, 3)

    # After init: a=3, b=5, eq(3,5)=F, lt(3,5)=T
    assert bool(history[0][eq_wire].item()) is False
    assert bool(history[0][lt_wire].item()) is True

    # Step 1: a'=4; eq(3,5)=F, lt(3,5)=T
    assert int(_get(history[1], a[0]).item()) == 4
    assert bool(history[1][eq_wire2].item()) is False
    assert bool(history[1][lt_wire2].item()) is True

    # Step 2: a'=5; eq(4,5)=F, lt(4,5)=T
    assert int(_get(history[2], a[0]).item()) == 5
    assert bool(history[2][eq_wire2].item()) is False
    assert bool(history[2][lt_wire2].item()) is True

    # Step 3: a'=6; eq(5,5)=T, lt(5,5)=F
    assert int(_get(history[3], a[0]).item()) == 6
    assert bool(history[3][eq_wire2].item()) is True
    assert bool(history[3][lt_wire2].item()) is False


# ── env inputs ───────────────────────────────────────────────────────────────

def test_env_inputs():
    """Module with external inputs: counter that adds env input each step."""
    x = (Wire(dt.Int([1])), Wire(dt.Int([1])))
    env = (Wire(dt.Int([1])), Wire(dt.Int([1])))

    init = [
        Term(it.Tensor(torch.tensor([0], dtype=torch.int64)), [x[1]]),
    ]
    update = [
        Term(it.Add(), [x[1]], [x[0], env[1]]),
    ]
    m = Module.sequential(init, update, obs=[x, env])

    inputs_seq = [
        {env[1]: torch.tensor([5], dtype=torch.int64)},
        {env[1]: torch.tensor([3], dtype=torch.int64)},
    ]
    state, history = _run_module(m, 2, env_inputs_fn=lambda step: inputs_seq[step])

    assert int(_get(history[0], x[0]).item()) == 0
    assert int(_get(history[1], x[0]).item()) == 5
    assert int(_get(history[2], x[0]).item()) == 8


# ── 2-bit counter circuit ───────────────────────────────────────────────────

def _make_twobitcounter():
    b0 = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))
    b1 = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))
    enable = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))

    init = [
        Term(it.Tensor(torch.tensor([False])), [b0[1]]),
        Term(it.Tensor(torch.tensor([False])), [b1[1]]),
    ]

    not_b0 = Wire(dt.Bool([1]))
    not_b1 = Wire(dt.Bool([1]))
    b0_and_enable = Wire(dt.Bool([1]))

    update = [
        Term(it.Not(), [not_b0], [b0[0]]),
        Term(it.Ite(), [b0[1]], [enable[1], not_b0, b0[0]]),
        Term(it.And(), [b0_and_enable], [b0[0], enable[1]]),
        Term(it.Not(), [not_b1], [b1[0]]),
        Term(it.Ite(), [b1[1]], [b0_and_enable, not_b1, b1[0]]),
    ]

    m = Module.sequential(init, update, obs=[b0, b1, enable])
    return m, b0, b1, enable


def _bits(state, b0, b1):
    return (bool(state[b1[0]].item()), bool(state[b0[0]].item()))


def test_twobitcounter_initial_state():
    m, b0, b1, enable = _make_twobitcounter()
    state, _ = _run_module(m, 0)
    assert _bits(state, b0, b1) == (False, False)


def test_twobitcounter_count_sequence():
    """Stepping with enable=True cycles 00->01->10->11->00."""
    m, b0, b1, enable = _make_twobitcounter()
    EN = {enable[1]: torch.tensor([True])}

    state, history = _run_module(m, 4, env_inputs_fn=lambda _: EN)

    expected = [
        (False, False),  # 0
        (False, True),   # 1
        (True, False),   # 2
        (True, True),    # 3
        (False, False),  # 0 (wrap)
    ]
    for i, exp in enumerate(expected):
        assert _bits(history[i], b0, b1) == exp, f"step {i}: expected {exp}"


def test_twobitcounter_hold():
    """Enable=False leaves state unchanged."""
    m, b0, b1, enable = _make_twobitcounter()
    EN = {enable[1]: torch.tensor([True])}
    HLD = {enable[1]: torch.tensor([False])}

    # Advance to state 2, then hold 3 steps
    inputs = [EN, EN, HLD, HLD, HLD]
    state, history = _run_module(m, 5, env_inputs_fn=lambda s: inputs[s])

    assert _bits(history[2], b0, b1) == (True, False)  # state 2
    assert _bits(history[3], b0, b1) == (True, False)   # hold
    assert _bits(history[4], b0, b1) == (True, False)   # hold
    assert _bits(history[5], b0, b1) == (True, False)   # hold


def test_twobitcounter_mixed():
    """Interleaved enable/hold steps."""
    m, b0, b1, enable = _make_twobitcounter()
    EN = {enable[1]: torch.tensor([True])}
    HLD = {enable[1]: torch.tensor([False])}

    inputs = [EN, HLD, EN, HLD, HLD, EN, EN]
    state, history = _run_module(m, 7, env_inputs_fn=lambda s: inputs[s])

    assert _bits(history[1], b0, b1) == (False, True)   # 0->1
    assert _bits(history[2], b0, b1) == (False, True)   # hold
    assert _bits(history[3], b0, b1) == (True, False)    # 1->2
    assert _bits(history[4], b0, b1) == (True, False)    # hold
    assert _bits(history[5], b0, b1) == (True, False)    # hold
    assert _bits(history[6], b0, b1) == (True, True)     # 2->3
    assert _bits(history[7], b0, b1) == (False, False)   # 3->0
