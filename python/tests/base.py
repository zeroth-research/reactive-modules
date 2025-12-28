import unittest
import zrth

from zrth import Wire, Term, MyTensor, Module
from zrth import DType as dt
from zrth import IType as it


class MyTestCase(unittest.TestCase):
    def test_wire_new(self):
        x = Wire(dt.C, 0)
        y = Wire(dt.D, 1)

    def test_term_new(self):
        x = Wire(dt.C, 0)
        y = Wire(dt.D, 1)
        f = Term.function(it.A(), [x, y], [x])
        t = MyTensor([3, 4, 5])
        g = Term.function(it.C(t), [x], [y])
        h = Term(it.A(), [x], [y])
        h = Term(it.A(), [x])

    def test_module_sequential(self):
        x = Wire(dt.C, 0)
        y = Wire(dt.C, 1)
        init = Term(it.A(), [y])
        update = Term(it.A(), [y], [x])
        m = Module.sequential([(x, y)], [init], [update])

    def test_module_combinatorial(self):
        x = Wire(dt.C, 0)
        y = Wire(dt.C, 1)
        assign = Term(it.A(), [y])
        m = Module.combinatorial([(x, y)], [assign])


if __name__ == '__main__':
    unittest.main()
