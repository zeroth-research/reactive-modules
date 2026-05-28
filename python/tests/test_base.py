import pytest
import torch
from zrth import Wire, Term, Module, DType as dt, IType as it, Int, Float, Bool, set_theory
from torch import Tensor


@pytest.fixture(autouse=True)
def _theory():
    set_theory(it.LIA)


def _bool_t(v):
    """Helper: build a bool tensor (Tensor([True]) collapses to float32)."""
    return torch.tensor([v], dtype=torch.bool)


def test_wire_new():
    Wire(Bool())
    Wire(Int())
    Wire(Int(2, 3))


def test_term_new():
    x = Wire(Int(2, 3))
    y = Wire(Int(2, 3))
    xn = Wire(Int(2, 3))
    w4 = Wire(Int(3))
    w5 = Wire(Int(3))

    # test `function` ctor
    _ = Term.function(it.Add(), [xn], [x, y])
    _ = Term.function(it.Tensor(torch.tensor([3, 4, 5])), [w4], [])
    _ = Term.function(it.Tensor(torch.tensor([3, 4, 6])), [w5], [])

    # test `new` ctor
    Term(it.Lt(), [Wire(Bool(3))], [w4, Wire(Int(3))])
    Term(it.Tensor(torch.tensor([3, 2, 1])), [Wire(Int(3))])


def test_module_sequential():
    x = (Wire(Bool()), Wire(Bool()))
    init = [Term(it.Tensor(_bool_t(True)), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    _ = Module.sequential(init, update, [x])


def test_module_combinatorial():
    x = (Wire(Bool()), Wire(Bool()))

    assign = [Term(it.Tensor(_bool_t(False)), [x[1]])]
    _ = Module.combinatorial(assign, [x])


def test_module_parallel():
    x = (Wire(Bool()), Wire(Bool()))
    y = (Wire(Bool()), Wire(Bool()))
    z = (Wire(Bool()), Wire(Bool()))
    w = (Wire(Bool()), Wire(Bool()))
    v = (Wire(Bool()), Wire(Bool()))

    init = [Term(it.Tensor(_bool_t(False)), [x[1]])]
    update = [Term(it.And(), [x[1]], [x[0], y[1]])]
    p = Module.sequential(init, update, obs=[x, y])

    init = [
        Term(it.Tensor(_bool_t(False)), [v[1]]),
        Term(it.Tensor(_bool_t(False)), [y[1]]),
    ]
    update = [Term(it.And(), [v[1]], [v[0], x[0]]), Term(it.Id(), [y[1]], [x[0]])]
    q = Module.sequential(init, update, obs=[x, y], prvt=[v])

    assign = [Term(it.Or(), [z[1]], [y[1], w[1]])]
    r = Module.combinatorial(assign, obs=(z, y, w))

    m = Module.parallel(p, q, r)

    c = m.ctrl
    print(c)

    for ltc, nxt in c:
        print(f"({ltc}, {nxt})")

    for atom in m.atoms:
        print(atom)

    print(m.intf)
    assert m.intf == [x, y, z]
    assert m.extl == [w]
    assert m.prvt == [v]


def test_interface():
    x = Wire(Bool())
    y = Wire(Bool())
    xn = Wire(Bool())
    f = Term(it.And(), [xn], [x, y])
    f2 = Term(it.And(), [xn], [x, y])

    w = f.write
    r = f.read
    r2 = f.read
    assert r is not r2
    assert r is not w
    assert r2 == r
    assert f2.read == r
    assert r == [x, y]
    assert [x, y] == r

    for wire in r:
        print("-->", wire)

    for i in range(len(w)):
        print("-->", w[i])
