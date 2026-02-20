from zrth import Wire, Term, Module, DType as dt, IType as it, mk_term
from torch import Tensor


def test_wire_new():
    Wire(dt.Bool())
    Wire(dt.Int())
    Wire(dt.TensorInt([1, 2, 3]))


def test_term_new():
    x = Wire(dt.TensorInt([2, 2, 3]))
    y = Wire(dt.TensorInt([2, 2, 3]))
    xn = Wire(dt.TensorInt([2, 2, 3]))
    w4 = Wire(dt.TensorFloat([3]))
    w5 = Wire(dt.TensorInt([3]))

    # test `function` ctor
    _ = Term.function(it.Add(), [xn], [x, y])
    _ = Term.function(it.Tensor(Tensor([3, 4, 5])), [w4], [])
    _ = Term.function(it.Tensor(Tensor([3, 4, 6])), [w5], [])

    # test `new` ctor
    Term(it.Lt(), [Wire(dt.Bool())], [w4, w5])
    Term(it.Tensor(Tensor([3, 2, 1])), [Wire(dt.TensorInt([3]))])


def test_mk_term():
    x = Wire(dt.TensorInt([2, 2, 3]))
    y = Wire(dt.TensorInt([2, 2, 3]))
    xn = Wire(dt.TensorInt([2, 2, 3]))

    _ = mk_term(it.Add(), [xn], [x, y])
    g = mk_term(it.Tensor(Tensor([3, 4, 5])), [Wire(dt.TensorInt([3]))], [])
    h = mk_term(it.Tensor(Tensor([3, 4, 6])), [Wire(dt.TensorInt([3]))], [])

    # test `new` ctor
    _ = mk_term(it.Lt(), [Wire(dt.Bool())], [g, h])
    _ = mk_term(it.Tensor(Tensor([3, 2, 1])), [Wire(dt.TensorInt([3]))])


def test_module_sequential():
    x = (Wire(dt.Bool()), Wire(dt.Bool()))
    init = [Term(it.Tensor(Tensor([True])), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    _ = Module.sequential(init, update, [x])


def test_module_combinatorial():
    x = (Wire(dt.Bool()), Wire(dt.Bool()))

    assign = [Term(it.Tensor(Tensor([False])), [x[1]])]
    _ = Module.combinatorial(assign, [x])


def test_module_parallel():
    x = (Wire(dt.Bool()), Wire(dt.Bool()))
    y = (Wire(dt.Bool()), Wire(dt.Bool()))
    z = (Wire(dt.Bool()), Wire(dt.Bool()))
    w = (Wire(dt.Bool()), Wire(dt.Bool()))
    v = (Wire(dt.Bool()), Wire(dt.Bool()))

    init = [Term(it.Tensor(Tensor([False])), [x[1]])]
    update = [Term(it.And(), [x[1]], [x[0], y[1]])]
    p = Module.sequential(init, update, obs=[x, y])

    init = [
        Term(it.Tensor(Tensor([False])), [v[1]]),
        Term(it.Tensor(Tensor([False])), [y[1]]),
    ]
    update = [Term(it.And(), [v[1]], [v[0], x[0]]), Term(it.Id(), [y[1]], [x[0]])]
    q = Module.sequential(init, update, obs=[x, y], prvt=[v])

    assign = [Term(it.Or(), [z[1]], [y[1], w[1]])]
    r = Module.combinatorial(assign, obs=(z, y, w))

    m = Module.parallel(p, q, r)

    c = m.ctrl()
    print(c)

    for ltc, nxt in c:
        print(f"({ltc}, {nxt})")

    for atom in m.atoms():
        print(atom)

    print(m.intf())
    assert m.intf() == [x, y, z]
    assert m.extl() == [w]
    assert m.prvt() == [v]


def test_interface():
    x = Wire(dt.Bool())
    y = Wire(dt.Bool())
    xn = Wire(dt.Bool())
    f = Term(it.Id(), [xn], [x, y])
    f2 = Term(it.Id(), [xn], [x, y])

    w = f.write()
    r = f.read()
    r2 = f.read()
    assert r is not r2
    assert r is not w
    assert r2 == r
    assert f2.read() == r
    assert r == [x, y]
    assert [x, y] == r

    for wire in r:
        print("-->", wire)

    for i in range(len(w)):
        print("-->", w[i])
