import pytest
import torch
from zrth import Wire, Term, Module, LIA, Bool, Int, LRA, Real


def _bool_t(v):
    """Helper: build a 2-D bool tensor."""
    return torch.tensor([[bool(v)]], dtype=torch.bool)


def test_wire_new():
    Wire(Bool([1, 1]))
    Wire(Int([1, 1]))
    Wire(Int([2, 3]))


def test_term_new():
    x = Wire(Int([2, 3]))
    y = Wire(Int([2, 3]))
    xn = Wire(Int([2, 3]))
    w4 = Wire(Int([1, 3]))
    w5 = Wire(Int([1, 3]))

    # test `function` ctor
    _ = Term.function(LIA.Add(), [xn], [x, y])
    _ = Term.function(LIA.Const(torch.tensor([[3, 4, 5]])), [w4], [])
    _ = Term.function(LIA.Const(torch.tensor([[3, 4, 6]])), [w5], [])

    # comparisons are pointwise -> Bool of the operand shape
    Term(LIA.Lt(), [Wire(Bool([1, 3]))], [w4, Wire(Int([1, 3]))])
    Term.constant(LIA.Const(torch.tensor([[3, 2, 1]])), [Wire(Int([1, 3]))])


def test_module_sequential():
    x = (Wire(Bool([1, 1])), Wire(Bool([1, 1])))
    init = [Term.constant(LIA.Const(_bool_t(True)), [x[1]])]
    update = [Term(LIA.Id(), [x[1]], [x[0]])]
    _ = Module.sequential(init, update, [x])


def test_module_combinatorial():
    x = (Wire(Bool([1, 1])), Wire(Bool([1, 1])))

    assign = [Term.constant(LIA.Const(_bool_t(False)), [x[1]])]
    _ = Module.combinatorial(assign, [x])


def test_module_parallel():
    x = (Wire(Bool([1, 1])), Wire(Bool([1, 1])))
    y = (Wire(Bool([1, 1])), Wire(Bool([1, 1])))
    z = (Wire(Bool([1, 1])), Wire(Bool([1, 1])))
    w = (Wire(Bool([1, 1])), Wire(Bool([1, 1])))
    v = (Wire(Bool([1, 1])), Wire(Bool([1, 1])))

    init = [Term.constant(LIA.Const(_bool_t(False)), [x[1]])]
    update = [Term(LIA.And(), [x[1]], [x[0], y[1]])]
    p = Module.sequential(init, update, obs=[x, y])

    init = [
        Term.constant(LIA.Const(_bool_t(False)), [v[1]]),
        Term.constant(LIA.Const(_bool_t(False)), [y[1]]),
    ]
    update = [Term(LIA.And(), [v[1]], [v[0], x[0]]), Term(LIA.Id(), [y[1]], [x[0]])]
    q = Module.sequential(init, update, obs=[x, y], prvt=[v])

    assign = [Term(LIA.Or(), [z[1]], [y[1], w[1]])]
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
    x = Wire(Bool([1, 1]))
    y = Wire(Bool([1, 1]))
    xn = Wire(Bool([1, 1]))
    f = Term(LIA.And(), [xn], [x, y])
    f2 = Term(LIA.And(), [xn], [x, y])

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


def test_heterogeneous_composition():
    x = (Wire(Real([1, 1])), Wire(Real([1, 1])))
    y = (Wire(Real([1, 1])), Wire(Real([1, 1])))
    z = (Wire(Real([1, 1])), Wire(Real([1, 1])))

    zero = torch.tensor([[0]])
    init = [Term(LRA.Const(zero), [x[1]])]
    update = [Term(LRA.Id(), [x[1]], [x[0]])]
    P = Module.sequential(init, update, [x])

    init = [Term(LRA.Const(zero), [y[1]])]
    flow = [Term(LRA.Const(zero), [y[1]])]
    Q = Module.differential(init, flow, [y])

    comb = [Term(LRA.Add(), [z[1]], [x[1], y[1]])]
    R = Module.combinatorial(comb, [x, y, z])

    S = Module.parallel(P, Q, R)
