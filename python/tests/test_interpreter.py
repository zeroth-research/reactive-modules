import torch
from zrth import Wire, Term, Module, DType as dt, IType as it
from zrth.examples import Interpreter


def _make_counter():
    """Simple counter: init x=0, update x'=x+1."""
    x = (Wire(dt.Int()), Wire(dt.Int()))

    init = [Term(it.Tensor(torch.tensor([0], dtype=torch.int64)), [x[1]])]
    one = Wire(dt.Int())
    update = [
        Term(it.Tensor(torch.tensor([1], dtype=torch.int64)), [one]),
        Term(it.Add(), [x[1]], [x[0], one]),
    ]
    m = Module.sequential(init, update, [x])
    return m, x


def test_counter():
    m, x = _make_counter()
    interp = Interpreter(m)
    interp.initialize()

    val = interp.get(x[0].id())
    assert int(val.item()) == 0

    interp.step()
    val = interp.get(x[0].id())
    assert int(val.item()) == 1

    interp.step()
    val = interp.get(x[0].id())
    assert int(val.item()) == 2

    for _ in range(8):
        interp.step()
    val = interp.get(x[0].id())
    assert int(val.item()) == 10


def test_boolean_logic():
    """AND/OR/NOT: state wires computed from each other."""
    a = (Wire(dt.Bool()), Wire(dt.Bool()))
    b = (Wire(dt.Bool()), Wire(dt.Bool()))
    c = (Wire(dt.Bool()), Wire(dt.Bool()))

    # init: a=True, b=False, c=not(a)=False
    init = [
        Term(it.Tensor(torch.tensor([True])), [a[1]]),
        Term(it.Tensor(torch.tensor([False])), [b[1]]),
        Term(it.Not(), [c[1]], [a[1]]),
    ]
    # update: a'=and(a,b), b'=or(a,b), c'=not(c)
    update = [
        Term(it.And(), [a[1]], [a[0], b[0]]),
        Term(it.Or(), [b[1]], [a[0], b[0]]),
        Term(it.Not(), [c[1]], [c[0]]),
    ]
    m = Module.sequential(init, update, [a, b, c])
    interp = Interpreter(m)
    interp.initialize()

    # After init: a=True, b=False, c=False
    assert bool(interp.get(a[0].id()).item()) is True
    assert bool(interp.get(b[0].id()).item()) is False
    assert bool(interp.get(c[0].id()).item()) is False

    interp.step()
    # a'=and(T,F)=F, b'=or(T,F)=T, c'=not(F)=T
    assert bool(interp.get(a[0].id()).item()) is False
    assert bool(interp.get(b[0].id()).item()) is True
    assert bool(interp.get(c[0].id()).item()) is True

    interp.step()
    # a'=and(F,T)=F, b'=or(F,T)=T, c'=not(T)=F
    assert bool(interp.get(a[0].id()).item()) is False
    assert bool(interp.get(b[0].id()).item()) is True
    assert bool(interp.get(c[0].id()).item()) is False


def test_ite():
    """Ite branching: cond toggles, x depends on previous x."""
    cond = (Wire(dt.Bool()), Wire(dt.Bool()))
    x = (Wire(dt.Int()), Wire(dt.Int()))

    init = [
        Term(it.Tensor(torch.tensor([True])), [cond[1]]),
        Term(it.Tensor(torch.tensor([0], dtype=torch.int64)), [x[1]]),
    ]

    one = Wire(dt.Int())
    two = Wire(dt.Int())
    tmp1 = Wire(dt.Int())
    tmp2 = Wire(dt.Int())

    # update: cond'=not(cond), x'=ite(cond, x+1, x+2)
    update = [
        Term(it.Not(), [cond[1]], [cond[0]]),
        Term(it.Tensor(torch.tensor([1], dtype=torch.int64)), [one]),
        Term(it.Tensor(torch.tensor([2], dtype=torch.int64)), [two]),
        Term(it.Add(), [tmp1], [x[0], one]),
        Term(it.Add(), [tmp2], [x[0], two]),
        Term(it.Ite(), [x[1]], [cond[0], tmp1, tmp2]),
    ]
    m = Module.sequential(init, update, [cond, x])
    interp = Interpreter(m)
    interp.initialize()

    # After init: cond=True, x=0
    assert bool(interp.get(cond[0].id()).item()) is True
    assert int(interp.get(x[0].id()).item()) == 0

    interp.step()
    # cond'=not(T)=F, x'=ite(T, 0+1, 0+2)=1
    assert bool(interp.get(cond[0].id()).item()) is False
    assert int(interp.get(x[0].id()).item()) == 1

    interp.step()
    # cond'=not(F)=T, x'=ite(F, 1+1, 1+2)=3
    assert bool(interp.get(cond[0].id()).item()) is True
    assert int(interp.get(x[0].id()).item()) == 3


def test_tensor_ops():
    """TensorSum, ReLU using data as state."""
    data = (Wire(dt.Float([4])), Wire(dt.Float([4])))

    init = [
        Term(it.Tensor(torch.tensor([-1.0, 2.0, 3.0, -4.0])), [data[1]]),
    ]
    # update: data' = relu(data)
    update = [
        Term(it.ReLU(), [data[1]], [data[0]]),
    ]
    m = Module.sequential(init, update, [data])
    interp = Interpreter(m)
    interp.initialize()

    expected = torch.tensor([-1.0, 2.0, 3.0, -4.0])
    assert torch.equal(interp.get(data[0].id()), expected)

    interp.step()
    # data' = relu([-1,2,3,-4]) = [0,2,3,0]
    assert torch.equal(interp.get(data[0].id()), expected.relu())

    interp.step()
    # data' = relu([0,2,3,0]) = [0,2,3,0] (fixed point)
    assert torch.equal(interp.get(data[0].id()), expected.relu())


def test_tensor_reductions():
    """TensorSum, TensorMean, TensorMax, Argmax as temp wires computed from state."""
    # Only data is state; reduction outputs are temp wires
    data = (Wire(dt.Float([4])), Wire(dt.Float([4])))
    sum_wire = Wire(dt.Float())
    mean_wire = Wire(dt.Float())
    max_wire = Wire(dt.Float())
    argmax_wire = Wire(dt.Int())

    init = [
        Term(it.Tensor(torch.tensor([-1.0, 2.0, 3.0, -4.0])), [data[1]]),
        Term(it.TensorSum(), [sum_wire], [data[1]]),
        Term(it.TensorMean(), [mean_wire], [data[1]]),
        Term(it.TensorMax(), [max_wire], [data[1]]),
        Term(it.Argmax(), [argmax_wire], [data[1]]),
    ]
    # update: data stays the same; reductions are recomputed
    update = [
        Term(it.Id(), [data[1]], [data[0]]),
    ]
    m = Module.sequential(init, update, [data])
    interp = Interpreter(m)
    interp.initialize()

    expected = torch.tensor([-1.0, 2.0, 3.0, -4.0])
    # Check temp wire values persisted in state after init
    assert float(interp.get(sum_wire.id()).item()) == float(expected.sum().item())
    assert float(interp.get(mean_wire.id()).item()) == float(expected.mean().item())
    assert float(interp.get(max_wire.id()).item()) == float(expected.max().item())
    assert int(interp.get(argmax_wire.id()).item()) == int(expected.argmax().item())


def test_comparisons():
    """Eq and Lt comparisons."""
    # a and b are state; comparison results are temp wires
    a = (Wire(dt.Int()), Wire(dt.Int()))
    b = (Wire(dt.Int()), Wire(dt.Int()))
    eq_wire = Wire(dt.Bool())
    lt_wire = Wire(dt.Bool())

    init = [
        Term(it.Tensor(torch.tensor([3], dtype=torch.int64)), [a[1]]),
        Term(it.Tensor(torch.tensor([5], dtype=torch.int64)), [b[1]]),
        Term(it.Eq(), [eq_wire], [a[1], b[1]]),
        Term(it.Lt(), [lt_wire], [a[1], b[1]]),
    ]

    one = Wire(dt.Int())
    eq_wire2 = Wire(dt.Bool())
    lt_wire2 = Wire(dt.Bool())
    # update: a'=a+1, b'=b, compute fresh comparison temps
    update = [
        Term(it.Tensor(torch.tensor([1], dtype=torch.int64)), [one]),
        Term(it.Add(), [a[1]], [a[0], one]),
        Term(it.Id(), [b[1]], [b[0]]),
        Term(it.Eq(), [eq_wire2], [a[0], b[0]]),
        Term(it.Lt(), [lt_wire2], [a[0], b[0]]),
    ]
    m = Module.sequential(init, update, [a, b])
    interp = Interpreter(m)
    interp.initialize()

    # After init: a=3, b=5, eq(3,5)=F, lt(3,5)=T
    assert bool(interp.get(eq_wire.id()).item()) is False
    assert bool(interp.get(lt_wire.id()).item()) is True

    interp.step()
    # a'=3+1=4, b'=5; eq(3,5)=F, lt(3,5)=T (computed from latched a=3, b=5)
    assert int(interp.get(a[0].id()).item()) == 4
    assert bool(interp.get(eq_wire2.id()).item()) is False
    assert bool(interp.get(lt_wire2.id()).item()) is True

    interp.step()
    # a'=5, b'=5; eq(4,5)=F, lt(4,5)=T
    assert int(interp.get(a[0].id()).item()) == 5
    assert bool(interp.get(eq_wire2.id()).item()) is False
    assert bool(interp.get(lt_wire2.id()).item()) is True

    interp.step()
    # a'=6, b'=5; eq(5,5)=T, lt(5,5)=F
    assert int(interp.get(a[0].id()).item()) == 6
    assert bool(interp.get(eq_wire2.id()).item()) is True
    assert bool(interp.get(lt_wire2.id()).item()) is False


def test_state_dict():
    """Verify state_dict returns all wire values."""
    m, x = _make_counter()
    interp = Interpreter(m)
    interp.initialize()

    sd = interp.state_dict()
    assert isinstance(sd, dict)
    assert x[0].id() in sd


def test_step_before_init_raises():
    """Calling step() before initialize() should raise."""
    m, _ = _make_counter()
    interp = Interpreter(m)
    try:
        interp.step()
        assert False, "should have raised"
    except RuntimeError:
        pass


def test_env_inputs():
    """Module with external inputs: counter that adds env input each step."""
    x = (Wire(dt.Int()), Wire(dt.Int()))
    env = (Wire(dt.Int()), Wire(dt.Int()))

    init = [
        Term(it.Tensor(torch.tensor([0], dtype=torch.int64)), [x[1]]),
    ]
    update = [
        Term(it.Add(), [x[1]], [x[0], env[1]]),
    ]
    m = Module.sequential(init, update, obs=[x, env])
    interp = Interpreter(m)
    interp.initialize()

    assert int(interp.get(x[0].id()).item()) == 0

    # Step with env input = 5
    interp.step({env[1].id(): torch.tensor([5], dtype=torch.int64)})
    assert int(interp.get(x[0].id()).item()) == 5

    # Step with env input = 3
    interp.step({env[1].id(): torch.tensor([3], dtype=torch.int64)})
    assert int(interp.get(x[0].id()).item()) == 8
