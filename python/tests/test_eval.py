"""Tests for the symbolic term evaluation engine.

Tests IType operations (Add, And, Or, Not, Ite, ReLU, etc.) and
multi-step execution of hand-built modules.
"""
import pytest
import torch
from zrth import Wire, Term, Module, LIA, BV, Bool, Real, Int, BitVec, Var, X
from zrth.eval import eval_itype, execute_init, execute_update


# ── helpers ──────────────────────────────────────────────────────────────────

def _run_module(module, n_steps, env_inputs_fn=None):
    """Run a hand-built module for n_steps, returning state after each step.

    env_inputs_fn(step) returns a dict of {wire: tensor} or None.
    """
    state = {}

    # Init
    execute_init(state, module.atoms)
    for var in module.ctrl:
        if X(var) in state:
            state[var] = state[X(var)].clone()

    history = [dict(state)]

    # Steps
    for step in range(n_steps):
        if env_inputs_fn:
            inputs = env_inputs_fn(step)
            if inputs:
                for wire, tensor in inputs.items():
                    state[wire] = tensor
        execute_update(state, module.atoms)
        for var in module.ctrl:
            if X(var) in state:
                state[var] = state[X(var)].clone()
        history.append(dict(state))

    return state, history


def _get(state, wire):
    """Read a wire value from state."""
    return state[wire]


# ── counter ──────────────────────────────────────────────────────────────────

def _make_counter():
    """Simple counter: init x=0, update x'=x+1."""
    x = Var(Int([1, 1]))

    init = [Term(LIA.Const(torch.tensor([[0]], dtype=torch.int64)), [X(x)])]
    one = Wire(Int([1, 1]))
    update = [
        Term(LIA.Const(torch.tensor([[1]], dtype=torch.int64)), [one]),
        Term(LIA.Add(), [X(x)], [x, one]),
    ]
    m = Module.sequential([x], init, update)
    return m, x


def test_counter():
    m, x = _make_counter()
    state, history = _run_module(m, 10)

    assert int(_get(history[0], x).item()) == 0
    assert int(_get(history[1], x).item()) == 1
    assert int(_get(history[2], x).item()) == 2
    assert int(_get(history[10], x).item()) == 10


def test_basic_eval_counter():
    """Raw term evaluation: run init then one update step manually."""
    m, x = _make_counter()
    assert m.closed()

    state = {}
    for t in (t for a in m.atoms for t in a.init):
        state.update(zip(t.write, eval_itype(t.itype, [state[w] for w in t.read])))

    state = {var: state[X(var)] for var in m.ctrl}
    for t in (t for a in m.atoms for t in a.update):
        state.update(zip(t.write, eval_itype(t.itype, [state[w] for w in t.read])))

    assert state[X(x)] == 1


# ── boolean logic ────────────────────────────────────────────────────────────

def test_boolean_logic():
    """AND/OR/NOT: state wires computed from each other."""
    a = Var(Bool([1, 1]))
    b = Var(Bool([1, 1]))
    c = Var(Bool([1, 1]))

    init = [
        Term(LIA.Const(torch.tensor([[True]])), [X(a)]),
        Term(LIA.Const(torch.tensor([[False]])), [X(b)]),
        Term(LIA.Not(), [X(c)], [X(a)]),
    ]
    update = [
        Term(LIA.And(), [X(a)], [a, b]),
        Term(LIA.Or(), [X(b)], [a, b]),
        Term(LIA.Not(), [X(c)], [c]),
    ]
    m = Module.sequential([a, b, c], init, update)
    state, history = _run_module(m, 2)

    # After init: a=True, b=False, c=not(True)=False
    assert bool(_get(history[0], a).item()) is True
    assert bool(_get(history[0], b).item()) is False
    assert bool(_get(history[0], c).item()) is False

    # Step 1: a'=and(T,F)=F, b'=or(T,F)=T, c'=not(F)=T
    assert bool(_get(history[1], a).item()) is False
    assert bool(_get(history[1], b).item()) is True
    assert bool(_get(history[1], c).item()) is True

    # Step 2: a'=and(F,T)=F, b'=or(F,T)=T, c'=not(T)=F
    assert bool(_get(history[2], a).item()) is False
    assert bool(_get(history[2], b).item()) is True
    assert bool(_get(history[2], c).item()) is False


# ── ite branching ────────────────────────────────────────────────────────────

def test_ite():
    """Ite branching: cond toggles, x depends on previous x."""
    cond = Var(Bool([1, 1]))
    x = Var(Int([1, 1]))

    init = [
        Term(LIA.Const(torch.tensor([[True]])), [X(cond)]),
        Term(LIA.Const(torch.tensor([[0]], dtype=torch.int64)), [X(x)]),
    ]

    one = Wire(Int([1, 1]))
    two = Wire(Int([1, 1]))
    tmp1 = Wire(Int([1, 1]))
    tmp2 = Wire(Int([1, 1]))

    update = [
        Term(LIA.Not(), [X(cond)], [cond]),
        Term(LIA.Const(torch.tensor([[1]], dtype=torch.int64)), [one]),
        Term(LIA.Const(torch.tensor([[2]], dtype=torch.int64)), [two]),
        Term(LIA.Add(), [tmp1], [x, one]),
        Term(LIA.Add(), [tmp2], [x, two]),
        Term(LIA.Ite(), [X(x)], [cond, tmp1, tmp2]),
    ]
    m = Module.sequential([cond, x], init, update)
    state, history = _run_module(m, 2)

    # After init: cond=True, x=0
    assert bool(_get(history[0], cond).item()) is True
    assert int(_get(history[0], x).item()) == 0

    # Step 1: cond'=F, x'=ite(T, 0+1, 0+2)=1
    assert bool(_get(history[1], cond).item()) is False
    assert int(_get(history[1], x).item()) == 1

    # Step 2: cond'=T, x'=ite(F, 1+1, 1+2)=3
    assert bool(_get(history[2], cond).item()) is True
    assert int(_get(history[2], x).item()) == 3


# ── tensor ops ───────────────────────────────────────────────────────────────

def test_tensor_ops():
    """ReLU on an Int vector state."""
    data = Var(Int([1, 4]))

    init = [
        Term(LIA.Const(torch.tensor([[-1, 2, 3, -4]], dtype=torch.int64)), [X(data)]),
    ]
    update = [
        Term(LIA.ReLU(), [X(data)], [data]),
    ]
    m = Module.sequential([data], init, update)
    state, history = _run_module(m, 2)

    # Wire data is stored as 2-D `(1, N)`; reshape to match.
    expected = torch.tensor([[-1, 2, 3, -4]], dtype=torch.int64)
    assert torch.equal(_get(history[0], data), expected)

    # Step 1: relu([-1,2,3,-4]) = [0,2,3,0]
    assert torch.equal(_get(history[1], data), expected.relu())

    # Step 2: relu([0,2,3,0]) = [0,2,3,0] (fixed point)
    assert torch.equal(_get(history[2], data), expected.relu())


# ── bit-vectors ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b,width,expected", [
    (3, 4, 8, 7),  # ordinary add
    (200, 100, 8, 44),  # wraps modulo 2^8
])
def test_bv_add_masks_to_width(a, b, width, expected):
    """BV results are masked to the wire's bit-width, which needs the width read off
    the output Sort."""
    out = Wire(BitVec(width, [1, 1]))
    term = Term(BV.Add(), [out], [Wire(BitVec(width, [1, 1])), Wire(BitVec(width, [1, 1]))])
    got = eval_itype(term.itype, [torch.tensor([[a]]), torch.tensor([[b]])], out.dtype)
    assert got[0].flatten().tolist() == [expected]


# ── comparisons ──────────────────────────────────────────────────────────────

def test_comparisons():
    """Eq and Lt comparisons over multiple steps."""
    a = Var(Int([1, 1]))
    b = Var(Int([1, 1]))
    eq_wire = Wire(Bool([1, 1]))
    lt_wire = Wire(Bool([1, 1]))

    init = [
        Term(LIA.Const(torch.tensor([[3]], dtype=torch.int64)), [X(a)]),
        Term(LIA.Const(torch.tensor([[5]], dtype=torch.int64)), [X(b)]),
        Term(LIA.Eq(), [eq_wire], [X(a), X(b)]),
        Term(LIA.Lt(), [lt_wire], [X(a), X(b)]),
    ]

    one = Wire(Int([1, 1]))
    eq_wire2 = Wire(Bool([1, 1]))
    lt_wire2 = Wire(Bool([1, 1]))
    update = [
        Term(LIA.Const(torch.tensor([[1]], dtype=torch.int64)), [one]),
        Term(LIA.Add(), [X(a)], [a, one]),
        Term(LIA.Id(), [X(b)], [b]),
        Term(LIA.Eq(), [eq_wire2], [a, b]),
        Term(LIA.Lt(), [lt_wire2], [a, b]),
    ]
    m = Module.sequential([a, b], init, update)
    state, history = _run_module(m, 3)

    # After init: a=3, b=5, eq(3,5)=F, lt(3,5)=T
    assert bool(history[0][eq_wire].item()) is False
    assert bool(history[0][lt_wire].item()) is True

    # Step 1: a'=4; eq(3,5)=F, lt(3,5)=T
    assert int(_get(history[1], a).item()) == 4
    assert bool(history[1][eq_wire2].item()) is False
    assert bool(history[1][lt_wire2].item()) is True

    # Step 2: a'=5; eq(4,5)=F, lt(4,5)=T
    assert int(_get(history[2], a).item()) == 5
    assert bool(history[2][eq_wire2].item()) is False
    assert bool(history[2][lt_wire2].item()) is True

    # Step 3: a'=6; eq(5,5)=T, lt(5,5)=F
    assert int(_get(history[3], a).item()) == 6
    assert bool(history[3][eq_wire2].item()) is True
    assert bool(history[3][lt_wire2].item()) is False


# ── env inputs ───────────────────────────────────────────────────────────────

def test_env_inputs():
    """Module with external inputs: counter that adds env input each step."""
    x = Var(Int([1, 1]))
    env = Var(Int([1, 1]))

    init = [
        Term(LIA.Const(torch.tensor([[0]], dtype=torch.int64)), [X(x)]),
    ]
    update = [
        Term(LIA.Add(), [X(x)], [x, X(env)]),
    ]
    m = Module.sequential([x, env], init, update)

    inputs_seq = [
        {X(env): torch.tensor([[5]], dtype=torch.int64)},
        {X(env): torch.tensor([[3]], dtype=torch.int64)},
    ]
    state, history = _run_module(m, 2, env_inputs_fn=lambda step: inputs_seq[step])

    assert int(_get(history[0], x).item()) == 0
    assert int(_get(history[1], x).item()) == 5
    assert int(_get(history[2], x).item()) == 8


# ── 2-bit counter circuit ───────────────────────────────────────────────────

def _make_twobitcounter():
    b0 = Var(Bool([1, 1]))
    b1 = Var(Bool([1, 1]))
    enable = Var(Bool([1, 1]))

    init = [
        Term(LIA.Const(torch.tensor([[False]])), [X(b0)]),
        Term(LIA.Const(torch.tensor([[False]])), [X(b1)]),
    ]

    not_b0 = Wire(Bool([1, 1]))
    not_b1 = Wire(Bool([1, 1]))
    b0_and_enable = Wire(Bool([1, 1]))

    update = [
        Term(LIA.Not(), [not_b0], [b0]),
        Term(LIA.Ite(), [X(b0)], [X(enable), not_b0, b0]),
        Term(LIA.And(), [b0_and_enable], [b0, X(enable)]),
        Term(LIA.Not(), [not_b1], [b1]),
        Term(LIA.Ite(), [X(b1)], [b0_and_enable, not_b1, b1]),
    ]

    m = Module.sequential([b0, b1, enable], init, update)
    return m, b0, b1, enable


def _bits(state, b0, b1):
    return (bool(state[b1].item()), bool(state[b0].item()))


def test_twobitcounter_initial_state():
    m, b0, b1, enable = _make_twobitcounter()
    state, _ = _run_module(m, 0)
    assert _bits(state, b0, b1) == (False, False)


def test_twobitcounter_count_sequence():
    """Stepping with enable=True cycles 00->01->10->11->00."""
    m, b0, b1, enable = _make_twobitcounter()
    EN = {X(enable): torch.tensor([[True]])}

    state, history = _run_module(m, 4, env_inputs_fn=lambda _: EN)

    expected = [
        (False, False),  # 0
        (False, True),  # 1
        (True, False),  # 2
        (True, True),  # 3
        (False, False),  # 0 (wrap)
    ]
    for i, exp in enumerate(expected):
        assert _bits(history[i], b0, b1) == exp, f"step {i}: expected {exp}"


def test_twobitcounter_hold():
    """Enable=False leaves state unchanged."""
    m, b0, b1, enable = _make_twobitcounter()
    EN = {X(enable): torch.tensor([[True]])}
    HLD = {X(enable): torch.tensor([[False]])}

    # Advance to state 2, then hold 3 steps
    inputs = [EN, EN, HLD, HLD, HLD]
    state, history = _run_module(m, 5, env_inputs_fn=lambda s: inputs[s])

    assert _bits(history[2], b0, b1) == (True, False)  # state 2
    assert _bits(history[3], b0, b1) == (True, False)  # hold
    assert _bits(history[4], b0, b1) == (True, False)  # hold
    assert _bits(history[5], b0, b1) == (True, False)  # hold


def test_twobitcounter_mixed():
    """Interleaved enable/hold steps."""
    m, b0, b1, enable = _make_twobitcounter()
    EN = {X(enable): torch.tensor([[True]])}
    HLD = {X(enable): torch.tensor([[False]])}

    inputs = [EN, HLD, EN, HLD, HLD, EN, EN]
    state, history = _run_module(m, 7, env_inputs_fn=lambda s: inputs[s])

    assert _bits(history[1], b0, b1) == (False, True)  # 0->1
    assert _bits(history[2], b0, b1) == (False, True)  # hold
    assert _bits(history[3], b0, b1) == (True, False)  # 1->2
    assert _bits(history[4], b0, b1) == (True, False)  # hold
    assert _bits(history[5], b0, b1) == (True, False)  # hold
    assert _bits(history[6], b0, b1) == (True, True)  # 2->3
    assert _bits(history[7], b0, b1) == (False, False)  # 3->0
