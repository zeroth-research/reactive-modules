from zrth.torch.ll import Wire, Term, Module, mk_term
from zrth.torch.ll import DType as dt
from zrth.torch.ll import IType as it

from torch import Tensor


def test_wire_new():
    x = Wire(dt.bool(), 0)
    y = Wire(dt.bool(), 1)


def test_term_new():
    x = Wire(dt.tensor([2, 2, 3]), 1)
    y = Wire(dt.tensor([2, 2, 3]), 2)
    xn = Wire(dt.tensor([2, 2, 3]), 3)
    w4 = Wire(dt.tensor([3]), 4)
    w5 = Wire(dt.tensor([3]), 5)

    # test `function` ctor
    f = Term.function(it.mk_add(), [xn], [x, y])
    g = Term.function(it.mk_const_tensor(Tensor([3, 4, 5])), [w4], [])
    h = Term.function(it.mk_const_tensor(Tensor([3, 4, 6])), [w5], [])

    # test `new` ctor
    Term(it.mk_lt(), [Wire(dt.bool(), 6)], [w4, w5])
    Term(it.mk_const_tensor(Tensor([3, 2, 1])), [Wire(dt.tensor([3]), 7)])


def test_mk_term():
    x = Wire(dt.tensor([2, 2, 3]), 1)
    y = Wire(dt.tensor([2, 2, 3]), 2)
    xn = Wire(dt.tensor([2, 2, 3]), 3)

    f = mk_term(it.mk_add(), [xn], [x, y])
    g = mk_term(it.mk_const_tensor(Tensor([3, 4, 5])), [Wire(dt.tensor([3]), 4)], [])
    h = mk_term(it.mk_const_tensor(Tensor([3, 4, 6])), [Wire(dt.tensor([3]), 5)], [])

    # test `new` ctor
    t1 = mk_term(it.mk_lt(), [Wire(dt.bool(), 6)], [g, h])
    t2 = mk_term(it.mk_const_tensor(Tensor([3, 2, 1])), [Wire(dt.tensor([3]), 7)])


def test_module_sequential():
    x = (Wire(dt.bool(), 0), Wire(dt.bool(), 1))
    init = [Term(it.mk_const_bool(True), [x[1]])]
    update = [Term(it.mk_id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])


def test_module_combinatorial():
    x = (Wire(dt.bool(), 0), Wire(dt.bool(), 1))

    assign = [Term(it.mk_const_bool(False), [x[1]])]
    m = Module.combinatorial(assign, [x])


def test_module_parallel():
    x = (Wire(dt.bool(), 0), Wire(dt.bool(), 1))
    y = (Wire(dt.bool(), 2), Wire(dt.bool(), 3))
    z = (Wire(dt.bool(), 4), Wire(dt.bool(), 5))
    w = (Wire(dt.bool(), 6), Wire(dt.bool(), 7))
    v = (Wire(dt.bool(), 8), Wire(dt.bool(), 9))

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
    x = Wire(dt.bool(), 0)
    y = Wire(dt.bool(), 1)
    xn = Wire(dt.bool(), 2)
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
