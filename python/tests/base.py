import unittest
import zrth

from zrth import Wire, Term


class MyTestCase(unittest.TestCase):
    def test_something(self):
        x = Wire.new(0, zrth.DType.C)
        y = Wire.new(1, zrth.DType.D)
        f = Term.function(zrth.IType.A, [x, y], [x])
        print(f)


if __name__ == '__main__':
    unittest.main()
