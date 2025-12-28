import unittest

from zrth.torch.ll import Wire, Term, Module
from zrth.torch.ll import DType as dt
from zrth.torch.ll import IType as it

from torch import Tensor


class MyTestCase(unittest.TestCase):
    def test_wire_new(self):
        x = Wire(dt.C, 0)
        y = Wire(dt.D, 1)

    def test_term_new(self):
        x = Wire(dt.C, 0)
        y = Wire(dt.D, 1)
        f = Term.function(it.A(), [x, y], [x])
        t = Tensor([3, 4, 5])
        g = Term.function(it.C(t), [x], [y])
        h = Term(it.A(), [x], [y])
        h = Term(it.A(), [x])

    def test_module_sequential(self):
        x = (Wire(dt.C, 0), Wire(dt.C, 1))
        init = Term(it.A(), [x[1]])
        update = Term(it.A(), [x[1]], [x[0]])
        m = Module.sequential([x], [init], [update])

    def test_module_combinatorial(self):
        x = (Wire(dt.C, 0), Wire(dt.C, 1))

        assign = Term(it.A(), [x[1]])
        m = Module.combinatorial([x], [assign])

    def test_module_parallel(self):
        x = (Wire(dt.C, 0), Wire(dt.C, 1))
        y = (Wire(dt.C, 2), Wire(dt.C, 3))
        z = (Wire(dt.C, 4), Wire(dt.C, 5))

        init = [Term(it.A(), [x[1]])]
        update = [Term(it.A(), [x[1]], [x[0], y[1]])]
        p = Module.sequential([x, y], init, update)

        init = [Term(it.A(), [y[1]])]
        update = [Term(it.A(), [y[1]], [x[0]])]
        q = Module.sequential([x, y], init, update)

        assign = [Term(it.A(), [z[1]], [y[1]])]
        r = Module.combinatorial([y, z], assign)

        m = Module.parallel(p, q, r)


if __name__ == '__main__':
    unittest.main()
