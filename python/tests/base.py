import unittest
import zrth

from zrth import Wire, Term

from zrth import Wire, Term, MyTensor
from zrth import DType as dt
from zrth import IType as it


class MyTestCase(unittest.TestCase):
    def test_something(self):
        x = Wire.new(0, dt.C)
        y = Wire.new(1, dt.D)
        f = Term.function(it.A(), [x, y], [x])
        t = MyTensor([3, 4, 5])
        g = Term.function(it.C(t), [x], [y])


if __name__ == '__main__':
    unittest.main()
