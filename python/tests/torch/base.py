import unittest

from zrth.torch.ll import Wire, Term, Module
from zrth.torch.ll import DType as dt
from zrth.torch.ll import IType as it

from torch import Tensor


class MyTestCase(unittest.TestCase):
    def test_wire_new(self):
        x = Wire(dt.Bool(), 0)
        y = Wire(dt.Bool(), 1)

    def test_term_new(self):
        x = Wire(dt.Bool(), 0)
        y = Wire(dt.Tensor([2, 2, 3]), 1)
        f = Term.function(it.A(), [x, y], [x])
        t = Tensor([3, 4, 5])
        g = Term.function(it.C(t), [x], [y])
        h = Term(it.A(), [x], [y])
        h = Term(it.A(), [x])

    def test_module_sequential(self):
        x = (Wire(dt.Bool(), 0), Wire(dt.Bool(), 1))
        init = [Term(it.A(), [x[1]])]
        update = [Term(it.A(), [x[1]], [x[0]])]
        m = Module.sequential(init, update, [x])

    def test_module_combinatorial(self):
        x = (Wire(dt.Bool(), 0), Wire(dt.Bool(), 1))

        assign = [Term(it.A(), [x[1]])]
        m = Module.combinatorial(assign, [x])

    def test_module_parallel(self):
        x = (Wire(dt.Tensor([2, 2, 3]), 0), Wire(dt.Tensor([2, 2, 3]), 1))
        y = (Wire(dt.Bool(), 2), Wire(dt.Bool(), 3))
        z = (Wire(dt.Bool(), 4), Wire(dt.Bool(), 5))
        w = (Wire(dt.Bool(), 6), Wire(dt.Bool(), 7))
        v = (Wire(dt.Bool(), 8), Wire(dt.Bool(), 9))

        init = [Term(it.A(), [x[1]])]
        update = [Term(it.A(), [x[1]], [x[0], y[1]])]
        p = Module.sequential(init, update, obs=[x, y])

        init = [Term(it.A(), [y[1], v[1]])]
        update = [Term(it.A(), [y[1], v[1]], [x[0], v[0]])]
        q = Module.sequential(init, update, obs=[x, y], prvt=[v])

        assign = [Term(it.A(), [z[1]], [y[1], w[1]])]
        r = Module.combinatorial(assign, obs=(z, y, w))

        m = Module.parallel(p, q, r)

        c = m.ctrl()
        print(c)

        for (ltc, nxt) in c:
            print(f'({ltc}, {nxt})')

        for atom in m.atoms():
            print(atom)

        print(m.intf())
        assert (m.intf() == [x, y, z])
        assert (m.extl() == [w])
        assert (m.prvt() == [v])

    def test_interface(self):
        x = Wire(dt.Bool(), 0)
        y = Wire(dt.Bool(), 1)
        f = Term(it.A(), [x, y], [x, y])

        w = f.write()
        r = f.read()
        r2 = f.read()
        assert (r is not r2)
        assert (r is not w)
        assert (w == r)
        assert (w == [x, y])
        assert ([x, y] == w)

        for wire in r:
            print('-->', wire)

        for i in range(len(w)):
            print('-->', w[i])


if __name__ == '__main__':
    unittest.main()
