from zrth.torch.ll import Wire, Term, Module, DType as dt, IType as it
from zrth.torch import mk_term

from torch import Tensor


def test_wire_new():
    x = Wire(0, dt.bool())
    y = Wire(1, dt.bool())


def test_term_new():
    x = Wire(1, dt.tensor([2, 2, 3]))
    y = Wire(2, dt.tensor([2, 2, 3]))
    xn = Wire(3, dt.tensor([2, 2, 3]))
    w4 = Wire(4, dt.tensor([3]))
    w5 = Wire(5, dt.tensor([3]))

    # test `function` ctor
    f = Term.function(it.mk_add(), [xn], [x, y])
    g = Term.function(it.mk_const_tensor(Tensor([3, 4, 5])), [w4], [])
    h = Term.function(it.mk_const_tensor(Tensor([3, 4, 6])), [w5], [])

    # test `new` ctor
    Term(it.mk_lt(), [Wire(6, dt.bool())], [w4, w5])
    Term(it.mk_const_tensor(Tensor([3, 2, 1])), [Wire(7, dt.tensor([3]))])


def test_mk_term():
    x = Wire(1, dt.tensor([2, 2, 3]))
    y = Wire(2, dt.tensor([2, 2, 3]))
    xn = Wire(3, dt.tensor([2, 2, 3]))

    f = mk_term(it.mk_add(), [xn], [x, y])
    g = mk_term(it.mk_const_tensor(Tensor([3, 4, 5])), [Wire(4, dt.tensor([3]))], [])
    h = mk_term(it.mk_const_tensor(Tensor([3, 4, 6])), [Wire(5, dt.tensor([3]))], [])

    # test `new` ctor
    t1 = mk_term(it.mk_lt(), [Wire(6, dt.bool())], [g, h])
    t2 = mk_term(it.mk_const_tensor(Tensor([3, 2, 1])), [Wire(7, dt.tensor([3]))])


def test_module_sequential():
    x = (Wire(0, dt.bool()), Wire(1, dt.bool()))
    init = [Term(it.mk_const_bool(True), [x[1]])]
    update = [Term(it.mk_id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])


def test_module_combinatorial():
    x = (Wire(0, dt.bool()), Wire(1, dt.bool()))

    assign = [Term(it.mk_const_bool(False), [x[1]])]
    m = Module.combinatorial(assign, [x])


def test_module_parallel():
    x = (Wire(0, dt.bool()), Wire(1, dt.bool()))
    y = (Wire(2, dt.bool()), Wire(3, dt.bool()))
    z = (Wire(4, dt.bool()), Wire(5, dt.bool()))
    w = (Wire(6, dt.bool()), Wire(7, dt.bool()))
    v = (Wire(8, dt.bool()), Wire(9, dt.bool()))

    init = [Term(it.mk_const_bool(False), [x[1]])]
    update = [Term(it.mk_and(), [x[1]], [x[0], y[1]])]
    p = Module.sequential(init, update, obs=[x, y])

    init = [
        Term(it.mk_const_bool(False), [v[1]]),
        Term(it.mk_const_bool(False), [y[1]]),
    ]
    update = [Term(it.mk_and(), [v[1]], [v[0], x[0]]), Term(it.mk_id(), [y[1]], [x[0]])]
    q = Module.sequential(init, update, obs=[x, y], prvt=[v])

    assign = [Term(it.mk_or(), [z[1]], [y[1], w[1]])]
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
    x = Wire(0, dt.bool())
    y = Wire(1, dt.bool())
    xn = Wire(2, dt.bool())
    f = Term(it.mk_id(), [xn], [x, y])
    f2 = Term(it.mk_id(), [xn], [x, y])

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


if __name__ == "__main__":
    test_module_parallel()
    test_interface()
