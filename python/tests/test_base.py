from zrth import Wire, Term, Module, DType as dt, IType, Int, Float, Bool
import torch
from torch import Tensor


def test_wire_new():
    Wire(Bool())
    Wire(Int())
    Wire(Int(2, 3))


def test_term_new():
    x = Wire(Int(4, 3))
    y = Wire(Int(4, 3))
    xn = Wire(Int(4, 3))
    w4 = Wire(Float(3))
    w5 = Wire(Int(3))

    # test `function` ctor
    _ = Term.function(IType.Int.Add, [xn], [x, y])
    _ = Term.function(IType.from_tensor(Tensor([3, 4, 5])), [w4], [])
    _ = Term.function(IType.from_tensor(torch.tensor([3, 4, 6])), [w5], [])

    # test `new` ctor
    Term(IType.Cmp.Lt, [Wire(Bool(3))], [w4, Wire(Float(3))])
    Term(IType.from_tensor(torch.tensor([3, 2, 1])), [Wire(Int(3))])


def test_module_sequential():
    x = (Wire(Bool()), Wire(Bool()))
    init = [Term(IType.from_tensor(torch.tensor([True])), [x[1]])]
    update = [Term(IType.Id, [x[1]], [x[0]])]
    _ = Module.sequential(init, update, [x])


def test_module_combinatorial():
    x = (Wire(Bool()), Wire(Bool()))

    assign = [Term(IType.from_tensor(torch.tensor([False])), [x[1]])]
    _ = Module.combinatorial(assign, [x])


def test_module_parallel():
    x = (Wire(Bool()), Wire(Bool()))
    y = (Wire(Bool()), Wire(Bool()))
    z = (Wire(Bool()), Wire(Bool()))
    w = (Wire(Bool()), Wire(Bool()))
    v = (Wire(Bool()), Wire(Bool()))

    init = [Term(IType.from_tensor(torch.tensor([False])), [x[1]])]
    update = [Term(IType.Bool.And, [x[1]], [x[0], y[1]])]
    p = Module.sequential(init, update, obs=[x, y])

    init = [
        Term(IType.from_tensor(torch.tensor([False])), [v[1]]),
        Term(IType.from_tensor(torch.tensor([False])), [y[1]]),
    ]
    update = [Term(IType.Bool.And, [v[1]], [v[0], x[0]]), Term(IType.Id, [y[1]], [x[0]])]
    q = Module.sequential(init, update, obs=[x, y], prvt=[v])

    assign = [Term(IType.Bool.Or, [z[1]], [y[1], w[1]])]
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
    f = Term(IType.Bool.And, [xn], [x, y])
    f2 = Term(IType.Bool.And, [xn], [x, y])

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
